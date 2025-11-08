# VidSubX

![python version](https://img.shields.io/badge/Python-3.12-blue)
![support os](https://img.shields.io/badge/OS-Windows-green.svg)

A free program that extracts hard coded subtitles from a video and generates an external subtitle file.

<img src="docs/images/gui%20screenshot.png" width="500">


**Features**

- Detect subtitle area by searching common area.
- Manual resize or change of subtitle area (click and drag mouse to perform).
- Single and Batch subtitle detection and extraction.
- Start and Stop subtitle extraction positions can be selected (use arrow keys for precise selection).
- Resize video display (Zoom In (Ctrl+Plus), Zoom Out (Ctrl+Minus)).
- Non subtitle area of the video can be hidden to limit spoilers.
- Toast Notification available on Windows upon completion of subtitle detection and extraction.
- [Preferences docs](docs/Preferences.md) available for modification of options when extraction subtitles.
- Multiple languages supported through [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). They will be
  automatically downloaded as needed.

**[Supported languages and Abbreviations](docs/Supported_Languages.md)**

### Generated subtitles can be translated with this [script](https://github.com/voun7/Subtitle_Translator).

### Download

[Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist) must be
installed. The program will not start without it.

- [Latest CPU & GPU Version](https://github.com/voun7/VidSubX/releases/latest)

## Demo Video

[![Demo Video](docs/images/demo%20screenshot.png)](https://youtu.be/i_VNDN7AMP4 "Demo Video")

## Setup Instructions

### Download and Install:

[Latest Version of Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist)

Install packages

For GPU

```
pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

For CPU

```
pip install paddlepaddle==3.2.1
```

Other packages

```commandline
pip install -r requirements.txt
```

Run `gui.py` to use Graphical interface and `main.py` to use Terminal.

### Compile Instructions

Run `compiler.py` to build compiled program

For virus warning the file can be submitted [here](https://www.microsoft.com/en-us/wdsi/filesubmission) for analysis.
