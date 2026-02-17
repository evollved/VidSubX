```markdown
# VidSubX Command Line Interface (CLI)

Command-line interface for extracting hardcoded subtitles from video files. Provides all the functionality of VidSubX without the graphical interface.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/voun7/VidSubX.git
cd VidSubX

# Install dependencies (choose based on GPU availability)
pip install -r requirements-cpu.txt  # for CPU
# OR
pip install -r requirements-gpu.txt  # for GPU with CUDA support
```

## 🚀 Quick Start

```bash
# Basic usage
python cmd.py video.mp4

# With subtitle area specified
python cmd.py video.mp4 --sub-area 100 800 1800 1000

# With language and performance settings
python cmd.py video.mp4 --lang en --use-gpu --output result.log
```

## 📖 Complete Argument Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `video_file` | Path to the video file to process |

### Basic Options

| Argument | Description | Example |
|----------|-------------|---------|
| `-o, --output FILE` | Output log file path (default: `video_filename.log` in same directory) | `--output subtitles.log` |
| `--sub-area X1 Y1 X2 Y2` | Subtitle area coordinates: x1 y1 x2 y2 | `--sub-area 100 800 1800 1000` |
| `--start-frame N` | Starting frame number for processing | `--start-frame 500` |
| `--stop-frame N` | Stopping frame number for processing | `--stop-frame 5000` |

### Subtitle Detection Settings

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `--split-start FLOAT` | Relative start position for detection | 0.25 | 0.0 - 0.5 |
| `--split-stop FLOAT` | Relative stop position for detection | 0.50 | 0.5 - 1.0 |
| `--no-of-frames INT` | Number of frames to analyze | 200 | any > 0 |
| `--sub-area-x-padding FLOAT` | Relative X-axis padding | 0.9 | 0.5 - 1.0 |
| `--sub-area-y-padding INT` | Absolute Y-axis padding (pixels) | 25 | any > 0 |
| `--bbox-drop-score FLOAT` | Bounding box drop score threshold | 0.7 | 0.0 - 1.0 |
| `--[no-]use-search-area` | Use default search area | Yes | On/Off |

### Frame Extraction Settings

| Argument | Description | Default |
|----------|-------------|---------|
| `--frame-frequency INT` | Extract every Nth frame | 2 |
| `--frame-batch-size INT` | Frame batch size | 250 |

### Text Extraction Settings

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `--text-batch-size INT` | Text batch size | 100 | any > 0 |
| `--text-drop-score FLOAT` | Text recognition drop score | 0.6 | 0.0 - 1.0 |
| `--[no-]line-break` | Use line breaks in extracted text | Yes | On/Off |

### Subtitle Generator Settings

| Argument | Description | Default | Range |
|----------|-------------|---------|-------|
| `--similarity-threshold FLOAT` | Text similarity threshold | 0.85 | 0.0 - 1.0 |
| `--min-consecutive-dur FLOAT` | Min consecutive subtitle duration (ms) | 500.0 | any > 0 |
| `--max-consecutive-short INT` | Max consecutive short durations | 4 | 2 - 10 |
| `--min-sub-duration FLOAT` | Minimum subtitle duration (ms) | 120.0 | any > 0 |

### OCR Engine Settings

| Argument | Description | Default |
|----------|-------------|---------|
| `--lang, --language CODE` | Recognition language code | ch |
| `--[no-]use-mobile-model` | Use mobile OCR model | Yes |
| `--[no-]use-text-ori` | Use text line orientation model | No |

**Supported languages:** ab, ady, af, ang, ar, av, az, ba, bal, be, bg, bgc, bh, bho, bs, bua, ca, ce, ch, chinese_cht, cs, cv, cy, da, dar, de, el, en, es, et, eu, fa, fi, fr, ga, gl, gom, hi, hr, hu, id, inh, is, it, japan, kaa, kbd, kk, korean, ku, kv, ky, la, lb, lez, lki, lt, lv, mah, mai, mhr, mi, mk, mn, mo, mr, ms, mt, ne, new, nl, no, oc, os, pi, pl, ps, pt, qu, rm, ro, rs_latin, ru, sa, sah, sck, sd, sk, sl, sq, sr, sv, sw, ta, tab, te, tg, th, tl, tr, tt, tyv, udm, ug, uk, ur, uz, vi, xal

### OCR Performance Settings

| Argument | Description | Default |
|----------|-------------|---------|
| `--cpu-processes INT` | Number of CPU OCR processes | half of cores |
| `--cpu-threads INT` | Number of ONNX intra threads | max(4, cores/2.5) |
| `--gpu-processes INT` | Number of GPU OCR processes | max(2, cores/3) |
| `--[no-]use-gpu` | Use GPU if available | Yes |
| `--[no-]auto-optimize` | Auto-optimize performance | Yes |

### Notification Settings (Windows only)

| Argument | Description | Default |
|----------|-------------|---------|
| `--notify-sound SOUND` | Notification sound | Default |
| `--[no-]loop-sound` | Loop notification sound | Yes |

**Available sounds:** Default, IM, Mail, Reminder, SMS, Silent

### Debug Options

| Argument | Description |
|----------|-------------|
| `-v, --verbose` | Enable verbose logging (DEBUG level) |
| `--keep-cache` | Keep cache files after extraction |

## 📋 Usage Examples

### Basic Examples

```bash
# Simple extraction with default settings
python cmd.py movie.mp4

# Extraction with language specification
python cmd.py anime.mkv --lang japan

# Extract specific segment
python cmd.py series.mp4 --start-frame 1000 --stop-frame 5000
```

### Advanced Examples

```bash
# Full configuration for optimal extraction
python cmd.py video.mp4 \
    --sub-area 200 850 1700 1050 \
    --lang en \
    --use-mobile-model \
    --frame-frequency 3 \
    --text-drop-score 0.7 \
    --similarity-threshold 0.9 \
    --min-sub-duration 200 \
    --use-gpu \
    --cpu-processes 8 \
    --output detailed.log \
    --verbose

# Batch processing (via shell script)
for video in *.mp4; do
    python cmd.py "$video" --lang en --output "logs/${video}.log"
done

# High-performance mode for powerful PCs
python cmd.py 4k_video.mkv \
    --use-gpu \
    --cpu-processes 16 \
    --frame-frequency 1 \
    --text-batch-size 200 \
    --auto-optimize

# Economy mode for low-end PCs
python cmd.py old_movie.avi \
    --no-use-gpu \
    --cpu-processes 2 \
    --use-mobile-model \
    --frame-frequency 5 \
    --text-batch-size 50
```

### Integration with Other Tools

```bash
# Extract and convert subtitles
python cmd.py video.mp4 --lang en
ffmpeg -i video.srt video.ass  # convert to another format

# Automatic processing of new files (Linux/macOS)
inotifywait -m /watch/folder -e create -e moved_to |
    while read path action file; do
        if [[ "$file" =~ \.(mp4|mkv|avi)$ ]]; then
            python cmd.py "$path$file" --lang auto
        fi
    done

# Integration with download managers (after download completes)
python cmd.py downloaded_video.mp4 --output subtitles.srt
```

## 📊 Output Files

### Files Created by the Program:

1. **Subtitle file** - `.srt` file in the same directory as the video
2. **Log file** - detailed process log (if `--output` is specified)

### Console Output Example:

```
$ python cmd.py video.mp4 --lang en --use-gpu
Processing video: video.mp4
Output log: /home/user/video.log
Setting up OCR...
Checking for required models...
Starting subtitle extraction...
File Path: /home/user/video.mp4
Frame Total: 123,456, Frame Rate: 30.0
Resolution: 1920 X 1080
Subtitle Area: (0, 810, 1920, 1080)
Start Frame No: 0, Stop Frame No: 123456
Starting Multiprocess Frame Extraction from video... Batches: 494.
Frame Extraction |#########################| 100.0% Complete
Starting Multiprocess Text Extraction from frames on GPU, Batches: 494.
Text Extraction |#########################| 100.0% Complete
Generating subtitle...
Subtitle file saved. Path: /home/user/video.srt
Subtitle Extraction Done! Duration: 0:02:34

✅ Subtitle file created: /home/user/video.srt
```

## ⚙️ Configuration

The CLI uses the same configuration files as the GUI version:
- `config.ini` - in working directory or `%APPDATA%/VidSubX/` (Windows)

You can combine settings from the configuration file with command-line arguments - CLI arguments take precedence.

## 🐛 Debugging

```bash
# Detailed output for debugging
python cmd.py problem_video.mp4 --verbose --keep-cache

# Check specific section
python cmd.py video.mp4 --start-frame 10000 --stop-frame 10100 --verbose
```

## 🔄 Compatibility

- **Windows**: full support (including notifications)
- **Linux/macOS**: full support (except notifications)

## 📝 Notes

1. GPU usage requires a CUDA-compatible graphics card and properly installed dependencies
2. Required OCR models will be downloaded on first run
3. Temporary files are automatically deleted after completion (except with `--keep-cache`)
4. For batch processing, shell scripts are recommended

## 🆘 Getting Help

```bash
# Display help
python cmd.py -h
python cmd.py --help

# Display version
python cmd.py --version
```

## 🤝 Contributing

Found a bug or have a suggestion for improvement? Create an issue or pull request on GitHub.

## 📄 License

MIT License
```
