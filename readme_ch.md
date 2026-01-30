简体中文 | [English](readme.md) | [Russian](readme_ru.md)

# VidSubX

![Python 版本](https://img.shields.io/badge/Python-3.12-blue)
![支持的操作系统](https://img.shields.io/badge/OS-Windows-green.svg)

一款免费的程序，可从视频中提取硬编码字幕并生成外挂字幕文件。

<img src="docs/images/gui%20screenshot.png" width="500">

**功能**

- 通过搜索常见区域来检测字幕区域。
- 手动调整或更改字幕区域（点击并拖动鼠标进行操作）。
- 支持单文件和批量字幕检测与提取。
- 可选择字幕提取的开始和结束位置（使用方向键进行精确选择）。
- 可调整视频显示尺寸（放大（Ctrl+加号），缩小（Ctrl+减号））。
- 可以隐藏视频的非字幕区域以减少剧透。
- **多GPU支持**：CUDA（英伟达）、DirectML（AMD）、OpenVINO（英特尔）
- **自动检测**：系统检测可用的GPU提供商
- **GUI选择**：用户可以在设置中选择GPU提供商
- **性能监控**：针对CPU和GPU分别进行优化
- **GPU信息记录**：详细记录显卡信息
- **后备兼容性**：如果GPU不可用，自动回退到CPU
- 针对CPU配置自动优化以获得最佳性能。
- 发布的版本无需安装Python即可运行。
- 在Windows系统上，字幕检测和提取完成后会显示Toast通知。
- 可通过[偏好设置文档](docs/Preferences.md)修改提取字幕时的选项。
- 通过[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)支持多语言。它们将在需要时自动下载。

**[支持的语言及缩写](docs/Supported_Languages.md)**

### 生成的字幕可以使用此[脚本](https://github.com/voun7/Subtitle_Translator)或[SubtitleEdit](https://github.com/SubtitleEdit/subtitleedit)进行翻译。

### 下载

必须先安装[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist)。没有它程序将无法启动。

- [最新 CPU 和 GPU 版本](https://github.com/voun7/VidSubX/releases/latest)

## 演示视频

[![演示视频](docs/images/demo%20screenshot.png)](https://youtu.be/i_VNDN7AMP4 "演示视频")

## 安装说明

### 下载并安装：

[最新版本的 Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist)

**安装包**

GPU版

```
pip install -r requirements-gpu.txt
```

CPU版

```
pip install -r requirements-cpu.txt
```

**_注意：_** `onnxruntime` 和 `onnxruntime-gpu` 不应同时安装。

运行 `gui.py` 使用图形界面，运行 `main.py` 使用终端。

### 编译说明

运行 `compiler.py` 来构建编译后的程序。

如果出现病毒警告，可以将文件提交[此处](https://www.microsoft.com/en-us/wdsi/filesubmission)进行分析。
