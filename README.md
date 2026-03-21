# What is SoftWhisper?

SoftWhisper simplifies audio and video transcription using the powerful Whisper model. 
You can easily select custom models, languages, and tasks, fine-tune transcription with beam size adjustment, and specify start and end times for targeted segments.

## Features
🎯 High-accuracy transcription (using Whisper model)<br>
👥 Speaker identification<br>
🌍 Supports all languages supported by the Whisper model (+30)<br>
🎮 User-friendly GUI interface<br>

## Usage

1. Run SoftWhisper.bat:
.\SoftWhisper.bat
When the GUI launches, follow these steps for transcription (screenshot credits: [Sunwood-ai-labs](https://github.com/user-attachments/assets/d28b227a-0ae3-4336-a655-abfbf35ef3e9)): 

![Softwhisper interface – Credits to Sunwood-ai-labs](https://github.com/user-attachments/assets/d28b227a-0ae3-4336-a655-abfbf35ef3e9)

2. Select an audio/video file.
3. Choose a model size (tiny, base, small, medium, large).
4. Enable speaker diarization if needed.
5. Click the "Start" button.

## Common issues and how to solve them
1. ```libvlc.dll not found``` error
    - Please check if VLC Media Player is installed. Please download it here: https://www.videolan.org/
    - Restart the program after installation
      
2. FFmpeg or corresponding library not found
   - Ensure FFmpeg is properly installed and added to PATH. Here is one example it can be downloaded from: https://github.com/BtbN/FFmpeg-Builds/releases
