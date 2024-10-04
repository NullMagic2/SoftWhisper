pip -r requirements@echo off

echo ** SoftWhisper dependency installer **
echo.
echo This script will install the required Python packages for SoftWhisper.
echo It uses pip, the Python package installer. Make sure you have Python 3.7 or higher installed.
echo WARNING: Make sure you have the requirements.txt file in the same folder as this script!
echo.
echo The following packages will be installed:
echo pyannote.audio (for speaker diarization)
echo pydub (for audio processing)
echo torch (for Whisper's AI model - may take some time and require a lot of disk space)
echo whisper (the core transcription library)
echo tkinter (for the graphical interface - usually included with Python)
echo vlc (for media playback)
echo psutil (for system resource management)
echo.


choice /M "Do you want to proceed with the installation? (Y/N)"
if errorlevel 2 goto :cancel
if errorlevel 1 goto :install


:install
echo Installing... Please wait. This may take a few minutes...

pip install -r requirements.txt

if %errorlevel% == 0 (
    echo.
    echo Installation successful! You can now run SoftWhisper.
    pause
) else (
    echo.
    echo Error during installation. Please check your internet connection and Python installation.
    echo Check the console output for more details.
    pause
)
goto :end


:cancel
echo Installation cancelled.
goto :end


:end