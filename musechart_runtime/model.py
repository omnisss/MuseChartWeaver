from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .audio import LogMelSpectrogram
from .schema import (BOSS_CONTROL_IDS, EVENT_BUNDLES, IBMS_IDS, SEMANTIC_TYPES,
                     STATE_CONTROLS)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            padding=2 * dilation,
            dilation=dilation,
            groups=channels,
        )
        self.expand = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.contract = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((1, channels, 1), 0.1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.norm(values)
        residual = self.depthwise(residual)
        residual = nn.functional.glu(self.expand(residual), dim=1)
        residual = self.contract(self.dropout(residual))
        return values + self.scale * residual


class MuseChartModel(nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int,
        hop_length: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        d_model = int(config["d_model"])
        self.frame_seconds = hop_length / sample_rate
        dropout = float(config["dropout"])
        self.features = LogMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=int(config["n_fft"]),
            hop_length=hop_length,
            n_mels=int(config["n_mels"]),
            f_min=float(config["f_min"]),
            f_max=float(config["f_max"]),
        )
        self.input_projection = nn.Sequential(
            nn.Conv1d(int(config["n_mels"]), d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.difficulty_embedding = nn.Embedding(
            int(config.get("max_difficulty", 8)) + 2,
            d_model,
        )
        self.bpm_projection = nn.Sequential(
            nn.Linear(2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        dilations = [2 ** (index % 5) for index in range(int(config["conv_layers"]))]
        self.temporal_blocks = nn.ModuleList(
            TemporalResidualBlock(d_model, dilation, dropout) for dilation in dilations
        )
        gru_hidden = int(config["gru_hidden"])
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=int(config["gru_layers"]),
            dropout=dropout if int(config["gru_layers"]) > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.gru_projection = nn.Linear(gru_hidden * 2, d_model)
        self.output_norm = nn.GroupNorm(1, d_model)
        self.onset_head = nn.Conv1d(d_model, 1, kernel_size=1)
        self.position_projection = nn.Conv1d(10, d_model, 1)
        self.song_encoder = nn.GRU(6, 32, bidirectional=True, batch_first=True)
        self.song_projection = nn.Conv1d(64, d_model, 1)
        self.bundle_head = nn.Conv1d(d_model, len(EVENT_BUNDLES), 1)
        self.pair_context_head = nn.Conv1d(d_model, 16, 1)
        self.pair_embeddings = nn.Parameter(
            torch.zeros(len(IBMS_IDS), len(IBMS_IDS), 16))
        nn.init.normal_(self.pair_embeddings, std=0.01)
        self.duration_ratio_head = nn.Conv1d(d_model, 2, 1)
        self.occupancy_head = nn.Conv1d(d_model, 2, 1)
        self.impact_head = nn.Conv1d(d_model, 6, 1)
        self.density_head = nn.Conv1d(d_model, 1, 1)
        self.boss_state_head = nn.Conv1d(d_model, 4, 1)
        self.boss_boundary_head = nn.Conv1d(d_model, 1, 1)
        self.boss_transitions = nn.Parameter(torch.eye(4) * 4.0)
        allowed = torch.eye(4, dtype=torch.bool)
        for left, right in STATE_CONTROLS:
            allowed[left, right] = True
        self.register_buffer("allowed_transitions", allowed)
        self.lane_head = nn.Conv1d(d_model, 3, kernel_size=1)
        self.semantic_head = nn.Conv1d(d_model, 2 * len(SEMANTIC_TYPES), kernel_size=1)
        self.hold_head = nn.Conv1d(d_model, 2, kernel_size=1)
        self.duration_head = nn.Conv1d(d_model, 2, kernel_size=1)
        self.ibms_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, 2 * len(IBMS_IDS), 1),
        )
        self.boss_control_head = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(d_model, len(BOSS_CONTROL_IDS), 1),
        )

        nn.init.constant_(self.onset_head.bias, -4.0)
        nn.init.zeros_(self.duration_head.bias)
        nn.init.constant_(self.boss_control_head[-1].bias, -4.0)
        nn.init.constant_(self.boss_boundary_head.bias, -4.0)

    @staticmethod
    def _bpm_features(bpm: torch.Tensor) -> torch.Tensor:
        known = (bpm > 0).to(bpm.dtype)
        safe = bpm.clamp(30.0, 640.0)
        ratio = torch.log2(safe / 160.0).clamp(-2.0, 2.0) * known
        return torch.stack((ratio, known), dim=-1)

    def forward(
        self,
        waveform: torch.Tensor,
        difficulty: torch.Tensor,
        bpm: torch.Tensor,
        *,
        segment_start: torch.Tensor | None = None,
        audio_duration: torch.Tensor | None = None,
        beat_offset: torch.Tensor | None = None,
        beat_known: torch.Tensor | None = None,
        audio_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        values = self.input_projection(self.features(waveform))
        batch, _, frames = values.shape
        start = torch.zeros_like(bpm) if segment_start is None else segment_start
        length = (torch.full_like(bpm, (frames - 1) * self.frame_seconds)
                  if audio_duration is None else audio_duration)
        offset = torch.zeros_like(bpm) if beat_offset is None else beat_offset
        known = (torch.zeros_like(bpm) if beat_known is None
                 else beat_known * (bpm > 0))
        absolute = (start[:, None]
                    + torch.arange(frames, device=values.device)[None]
                    * self.frame_seconds)
        position = (absolute / length[:, None].clamp_min(1e-3)).clamp(0, 1)
        beat = (absolute - offset[:, None]) * bpm[:, None] / 60
        phases = [function(beat * (2 * torch.pi / period)) * known[:, None]
                  for period in (1, 4, 16)
                  for function in (torch.sin, torch.cos)]
        position_features = torch.stack([
            position, 1 - position,
            torch.log1p(length)[:, None].expand(-1, frames) / 6,
            known[:, None].expand(-1, frames), *phases,
        ], dim=1)
        values = values + self.position_projection(position_features.to(values.dtype))
        if audio_context is None:
            audio_context = waveform.new_zeros((batch, 6, 256))
        global_values, _ = self.song_encoder(audio_context.transpose(1, 2))
        global_values = self.song_projection(global_values.transpose(1, 2))
        grid = torch.stack((2 * position - 1, torch.zeros_like(position)), dim=-1)[:, None]
        context_values = nn.functional.grid_sample(
            global_values.float().unsqueeze(2), grid.float(), mode="bilinear",
            align_corners=True, padding_mode="border").squeeze(2)
        values = values + context_values.to(values.dtype)
        maximum_index = self.difficulty_embedding.num_embeddings - 1
        difficulty = difficulty.clamp(0, maximum_index)
        condition = self.difficulty_embedding(difficulty)
        condition = condition + self.bpm_projection(self._bpm_features(bpm))
        values = values + condition.unsqueeze(-1)
        for block in self.temporal_blocks:
            values = block(values)
        recurrent, _ = self.gru(values.transpose(1, 2))
        values = values + self.gru_projection(recurrent).transpose(1, 2)
        values = self.output_norm(values)
        return {
            "bundle_logits": self.bundle_head(values),
            "pair_context": self.pair_context_head(values),
            "pair_embeddings": self.pair_embeddings,
            "duration_ratio_logits": self.duration_ratio_head(values),
            "occupancy_logits": self.occupancy_head(values),
            "impact_logits": self.impact_head(values).view(batch, 2, 3, frames),
            "density_log": self.density_head(values),
            "boss_state_logits": self.boss_state_head(values),
            "boss_boundary_logits": self.boss_boundary_head(values),
            "boss_transition_logits": self.boss_transitions.masked_fill(
                ~self.allowed_transitions, -1e4),
            "onset_logits": self.onset_head(values),
            "lane_logits": self.lane_head(values),
            "semantic_logits": self.semantic_head(values).view(values.shape[0], 2, len(SEMANTIC_TYPES), -1),
            "hold_logits": self.hold_head(values),
            "duration_log": self.duration_head(values),
            "ibms_logits": self.ibms_head(values).view(
                values.shape[0], 2, len(IBMS_IDS), -1),
            "boss_control_logits": self.boss_control_head(values),
        }


def joint_pair_logits(outputs, batch_indices, frame_indices):
    """Compute the 28x28 joint distribution only at decoded onsets."""
    fine = outputs["ibms_logits"].float()
    left = fine[batch_indices, 0, :, frame_indices]
    right = fine[batch_indices, 1, :, frame_indices]
    context = outputs["pair_context"][batch_indices, :, frame_indices].float()
    residual = torch.einsum(
        "nr,ijr->nij", context, outputs["pair_embeddings"].float()) / 4
    return left[:, :, None] + right[:, None, :] + residual
