
:: Place contents of VLC Player into VLC subfolder, FFMPEG to ffmpeg subfolder, and PythonXXX into Python subfolder.
:: Get the path of the folder where this batch file is located
set "SCRIPT_DIR=%~dp0"
:: Set PATH to include folders (which are next to the batch file)
set "PATH=%SCRIPT_DIR%VLC;%PATH%"
set "PATH=%SCRIPT_DIR%ffmpeg;%PATH%"
set "PATH=%SCRIPT_DIR%Python;%PATH%"
set "PATH=%SCRIPT_DIR%Python\Scripts;%PATH%"

python softwhisper.py
