SoftWhisper Installation and Setup Tutorial
This tutorial will guide you through the steps to install and run SoftWhisper.

PREREQUISITES:

* Python 3.7 or higher: 
You can download it from python.org, from your distro (if you run Linux) or from the Microsoft Store (if you run Windows).

REQUIRED LIBRARIES: SoftWhisper relies on several Python packages, including:

- Pyannote.audio (for speaker diarization)
- Pydub (for audio processing)
- Torch (for Whisper's AI model)
- Whisper (the core transcription library)
- Tkinter (for the graphical interface)
- Vlc (for media playback)
- psutil (for system resource management)

A requirements.txt file is provided for your convenience. 
You can download all the required libraries at once by typing on console:

pip install -r requirements.txt

Installation Steps:

1. Download the SoftWhisper release: Download the latest release of SoftWhisper from the [GitHub releases page](Your GitHub Release Page Link). 
Extract the contents of the zip file to a location of your choice.

2. Install Python: Verify you have Python 3.7 or higher installed.

3. Install Dependencies:
Open your terminal or command prompt, navigate to the extracted SoftWhisper directory, and run: 

pip install -r requirements.txt.

