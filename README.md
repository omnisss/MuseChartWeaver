# MuseChartWeaver v1.0

将 MP3/WAV/FLAC/OGG 通过模型转换为可用的 MDM，同时输出可选的 `events.json` 和推理报告。

## 启动

首先双击`setup_venv.bat`安装环境，首次直接双击 `launch_gui.bat`，也会自动运行，但是如果环境安装失败后再次使用`launch_gui.bat`就会直接开启UI导致运行报错。

环境安装完成后双击 `launch_gui.bat`。

系统还需安装 `ffmpeg` 和 `ffprobe`，并确保它们位于 `PATH`。如需 NVIDIA 加速，可以在 `.venv` 中安装与PC驱动匹配的 CUDA 版 PyTorch。

## 模型

还没有提供，不要急（

## GUI 选项

- 左侧在同一紧凑页面中放置“曲目素材”和“谱面设计”，不使用滚动条；右侧上方
  是选曲效果预览，下方是“生成与结果”，不显示原始控制台输出。
- 窗口会依据当前屏幕尺寸自动收缩；紧凑预览与短输入框布局可在 1280×720
  下保留底部生成、取消和输出按钮。
- 生成进度来自模型的五个真实处理阶段；核心结果会显示 BPM、有效事件、
  长按、Boss 控制、修复项、耗时和实际封面来源。
- 完整 stdout/stderr 会保存到 `inference/logs/`，仅在排查问题时点击
  “任务日志”打开。
- 模型难度：`1/2/3`，输出名会自动拼接为 `_mapN.mdm`。
- 事件密度：`casual` 较稀、`balanced` 默认、`aggressive` 较密。
- 场景：以下拉列表选择 `scene_01～10、scene_12`，右侧预览会使用
  `assets/backgrounds` 中对应的游戏场景并进行模糊和暗化。`scene_11/13`
  属于特殊玩法，不作为普通 MDM 场景提供。
- 封面按实际优先级使用显式封面、音频内嵌封面或 `image` 目录兜底图；所有封面
  视觉统一显示在右侧预览区，左侧只保留紧凑的选择和恢复按钮。
  右侧按原游戏比例组合唱片底图和圆形封面，并实时显示场景、标题、作者、
  难度颜色与等级星星；等级数字使用 `assets/fonts` 中的游戏字体。
  MDM 中仍会使用圆形透明封面。
- 事件秩序化：整理局部突兀怪物和异常半个 `0E`。
- 双轨修复：将普通怪和 Boss 双轨整理为经典双轨。
- 明显错误修复：处理双齿轮、双音符/红心、音符/红心参与双轨，以及异轨长按期间出现音符/红心等持续事件冲突。
- 三种优化均默认开启，也可以分别关闭；它们不会移动踩点时间。

GUI 使用的唱片、难度、字体和场景预览素材均位于 `inference/assets`，不依赖
项目外围的 `global` 目录。复制或单独打包 `inference` 时应保留 `assets`；
素材缺失时推理功能仍可使用，但对应图片预览不会显示。


## 命令行运行

一般用不上，给喜欢用的人参考

```powershell
cd .\inference
.\.venv\Scripts\python.exe -X utf8 .\generate_mdm.py "D:\Music\song.mp3" `
  --checkpoint ".\checkpoints\MuseChart_v1.0.pt" `
  --difficulty 2 `
  --play-level 7 `
  --output ".\output\song_map2.mdm"
```

也可以把相同参数传给：

```bat
generate_mdm.bat "D:\Music\song.mp3" --difficulty 2 --play-level 7
```

## 常用参数

```text
--checkpoint #模型路径
--profile #casual|balanced|aggressive
--threshold #0-1
--event-ordering / --no-event-ordering  #是否开启事件秩序化
--double-optimization / --no-double-optimization #是否开启双轨修复
--obvious-error-repair / --no-obvious-error-repair #是否开启错误修复
--optimization-matrix #以上三种修复排列组合全部输出，一共八种
--bpm #音乐bpm
--difficulty #这个不是音乐难度，是萌新、高手、大触：1|2|3
--play-level #这个是音乐难度，仅用于谱面显示，不决定推理时的难度
--scene scene_01 #场景
--cover 封面路径 / --no-cover #封面
--no-demo
--no-sidecars
```

最终 MDM 可放入 `Muse Dash\Custom_Albums\`。

## 测试

```powershell
cd .\inference
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
```
