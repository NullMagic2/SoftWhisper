<h1 align="center">SoftWhisper 🎤✨</h1>

<p align="center">
   <a href="README_JP.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white.svg" alt="JA doc"/></a>
   <a href="README_EN.md"><img src="https://img.shields.io/badge/english-document-white.svg" alt="EN doc"></a>
</p>

<p align="center">
   <img src="docs/header.png" alt="SoftWhisper Header">
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg" alt="Python 3.7+"/>
   <img src="https://img.shields.io/badge/FFmpeg-Required-green.svg" alt="FFmpeg"/>
   <img src="https://img.shields.io/badge/VLC-Required-orange.svg" alt="VLC"/>
</p>

A software that makes speech recognition and speaker diarization easy!

## Requirements

- Python 3.7 or higher
- FFmpeg
- VLC Media Player

## Installation

1. Clone this repository:
```bash
git clone https://github.com/NullMagic2/SoftWhisper .
```

2. Install required software:
   - [Python](https://www.python.org/downloads/) (3.7 or higher)
   - [FFmpeg](https://ffmpeg.org/download.html)
   - [VLC Media Player](https://www.videolan.org/vlc/)

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run SoftWhisper.bat:
```bash
.\SoftWhisper.bat
```

2. When the GUI launches, follow these steps for transcription:
   - Select an audio/video file
   - Choose a model size (tiny, base, small, medium, large)
   - Enable speaker diarization if needed
   - Click the "Start" button

<p align="center">
   <img src="docs/demo.png" alt="SoftWhisper Demo" width="600px">
</p>

## Features

- 🎯 High-accuracy transcription (using Whisper model)
- 👥 Speaker diarization (identify who is speaking)
- 🌍 Multi-language support
- 🎮 User-friendly GUI interface

## Troubleshooting

### Common Issues

1. `libvlc.dll not found` error
   - Make sure VLC Media Player is installed
   - Restart the program after installation

2. FFmpeg error
   - Ensure FFmpeg is properly installed and added to PATH

## License

[MIT License](LICENSE)

## Acknowledgments

This project uses the following open-source projects:
- [Whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter)
- [FFmpeg](https://ffmpeg.org/)
