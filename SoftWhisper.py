"""
SoftWhisper - Audio/Video Transcription Application using Whisper.cpp
"""

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import os
import tempfile
import sys
import queue
import time
import json
import psutil
import urllib.request
import re
from diarization_gui import DiarizationOption

# Import media player module
from media_player import MediaPlayer, MediaPlayerUI

# Import our simplest SRT functions
from subtitles import save_whisper_as_srt, whisper_to_srt

# Import export button creation from file_export.py
from file_export import create_export_button

import subprocess
import io
import signal
from pydub import AudioSegment

_app = None

# Debug helper: Write messages to the original stdout
def debug_print(msg):
    # Write to original stdout (for terminal debugging)
    sys.__stdout__.write(f"DEBUG: {msg}\n")
    sys.__stdout__.flush()
    # Also write to the console queue if available
    global _app
    if _app is not None and hasattr(_app, 'console_queue'):
        _app.console_queue.put({'type': 'append', 'content': f"DEBUG: {msg}\n"})

def get_default_whisper_cpp_path():
    program_dir = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":
        # Default Windows path: a directory; we'll later append the executable name.
        return os.path.join(program_dir, "Whisper_win-x64")
    else:
        return os.path.join(program_dir, "Whisper_lin-x64")

# Allowed model filenames for automatic download.
ALLOWED_MODELS = [
    "ggml-tiny.bin", "ggml-tiny.en.bin",
    "ggml-base.bin", "ggml-base.en.bin",
    "ggml-small.bin", "ggml-small.en.bin",
    "ggml-medium.bin", "ggml-medium.en.bin",
    "ggml-large.bin", "ggml-large-v2.bin",
    "ggml-large-v3.bin", "ggml-large-v3-turbo.bin"
]


CONFIG_FILE = 'config.json'

# Maximum duration per chunk in seconds
MAX_CHUNK_DURATION = 120  

# Redirect stdout and stderr to the UI console
def set_console_redirect(console_queue):
    sys.stdout = ConsoleRedirector(console_queue)
    sys.stderr = ConsoleRedirector(console_queue)

# The transcribe_audio function with built-in logic to parse timestamps from
# Whisper.cpp JSON lines.
def transcribe_audio(file_path, options, progress_callback=None, status_callback=None, stop_event=None):
    file_path = os.path.abspath(file_path)
    debug_print(f"transcribe_audio() => Processing file: {file_path}")

    # Opt-in cloud backend: route to TwelveLabs Pegasus when selected. The
    # default ('whisper.cpp') path below is unchanged.
    if options.get('engine') == 'twelvelabs-pegasus':
        from twelvelabs_backend import transcribe_with_pegasus
        debug_print("Using TwelveLabs Pegasus transcription backend.")
        return transcribe_with_pegasus(
            file_path, options, progress_callback, status_callback, stop_event
        )

    model_name = options.get('model_name', 'base')
    model_path = os.path.abspath(os.path.join("models", "whisper", f"ggml-{model_name}.bin"))
    language = options.get('language', 'auto')
    beam_size = min(int(options.get('beam_size', 5)), 8)
    task = options.get('task', 'transcribe')

    audio = AudioSegment.from_file(file_path)
    audio_length = len(audio) / 1000.0

    # Parse start/end directly from options and clamp to audio duration
    start_text = options.get('start_time', '').strip()
    end_text = options.get('end_time', '').strip()

    def _s_to_sec(txt):
        # Minimal inline parser supporting "HH:MM:SS", "MM:SS", or "SS"
        if not txt:
            return None
        parts = txt.split(':')
        try:
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            else:
                return float(parts[0])
        except Exception:
            return None

    start_override = _s_to_sec(start_text) or 0.0
    end_override = _s_to_sec(end_text) if end_text else None
    if end_override is None or end_override <= 0:
        end_override = audio_length

    start_sec = max(0.0, min(start_override, audio_length))
    end_sec = max(start_sec, min(end_override, audio_length))

    # Zero-length guard to avoid loops/hangs on empty intervals
    if end_sec - start_sec <= 0.001:
        return {
            'raw': "",
            'text': "",
            'segments': [],
            'audio_length': audio_length,
            'stderr': "",
            'cancelled': bool(stop_event and stop_event.is_set())
        }

    trimmed_audio = audio[int(start_sec * 1000):int(end_sec * 1000)]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        trimmed_audio.export(tmp.name, format="wav")
        temp_audio_path = tmp.name

    cmd = [
        options['whisper_executable'], "-m", model_path,
        "-f", temp_audio_path, "-bs", str(beam_size), "-pp",
        "-l", language, "-oj", "--prompt", "Always use punctuation. Do not use dashes to indicate dialog. Do not censor any words."
    ]
    if task == "translate":
        cmd.append("-translate")

    debug_print(f"Running Whisper.cpp with command: {' '.join(cmd)}")
    env = os.environ.copy()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
        env=env
    )

    stderr_data = []
    threading.Thread(
        target=lambda: [stderr_data.append(line) for line in iter(process.stderr.readline, "")],
        daemon=True
    ).start()

    stdout_lines = []
    last_output_time = time.time()
    STALL_TIMEOUT = 60  # seconds without output -> consider stalled

    while process.poll() is None:
        if stop_event and stop_event.is_set():
            try:
                psutil.Process(process.pid).kill()
            except Exception:
                pass
            break

        line = process.stdout.readline()
        if line:
            last_output_time = time.time()
            stdout_lines.append(line)
            match = re.search(r'\[(\d{2}:\d{2}:\d{2}\.\d{3}) -->', line)
            if match and progress_callback:
                h, m, s_ms = match.group(1).split(':')
                s, ms = s_ms.split('.')
                current = int(h) * 3600 + int(m) * 60 + int(s) + float(ms) / 1000
                den = max(0.001, (end_sec - start_sec))
                progress = int(max(0.0, min(100.0, (current / den) * 100)))
                progress_callback(progress, f"Transcribing: {progress}%")
        else:
            if time.time() - last_output_time > STALL_TIMEOUT:
                debug_print("Whisper.cpp appears stalled; terminating process to avoid hang.")
                try:
                    psutil.Process(process.pid).kill()
                except Exception:
                    pass
                break
            time.sleep(0.05)

    if progress_callback:
        progress_callback(100, "Transcribing: 100%")

    raw = "".join(stdout_lines).strip()
    segments, plain_text = [], ""
    if raw:
        try:
            data = json.loads(raw)
            segments = data.get("segments", [])
            plain_text = " ".join(seg.get("text", "").strip() for seg in segments)
        except json.JSONDecodeError as e:
            debug_print(f"JSON parse error (fallback): {e}")
            lines = raw.splitlines()
            cleaned = [re.sub(r'^\[[^\]]+\]\s*', "", l) for l in lines if l.strip()]
            plain_text = " ".join(cleaned)

    result = {
        'raw': raw,
        'text': plain_text,
        'segments': segments,
        'audio_length': audio_length,
        'stderr': "".join(stderr_data),
        'cancelled': bool(stop_event and stop_event.is_set()),
        'temp_audio_path': temp_audio_path
    }

    try:
        os.remove(temp_audio_path)
    except:
        pass

    return result


class CustomProgressBar(tk.Canvas):
    def __init__(self, master, width, height, bg_color="#E0E0E0", fill_color="#4CAF50"):
        super().__init__(master, width=width, height=height, bg=bg_color, highlightthickness=0)
        self.fill_color = fill_color
        self.width = width
        self.height = height
        self.bar = self.create_rectangle(0, 0, 0, height, fill=fill_color, width=0)

    def set_progress(self, percentage):
        fill_width = int(self.width * percentage / 100)
        self.coords(self.bar, 0, 0, fill_width, self.height)
        self.update_idletasks()

class ConsoleRedirector:
    def __init__(self, console_queue):
        self.console_queue = console_queue

    def write(self, message):
        if message and message.strip():
            # Queue the message for display in the UI
            self.console_queue.put({'type': 'append', 'content': message})
            # Also write to the original stderr for debugging in terminal
            sys.__stderr__.write(f"REDIRECT: {message}")
            sys.__stderr__.flush()

    def flush(self):
        pass

class SoftWhisper:
    def __init__(self, root):
        global _app
        _app = self
        debug_print("Initializing SoftWhisper")
        self.root = root
        self.root.title("SoftWhisper")
        self.set_window_centered(1000, 800)
        self.root.resizable(False, False)

        # Make debug_print available as an instance method
        self.debug_print = debug_print

        # Initialize variables
        self.setup_variables()
        self.setup_queues()
        self.create_widgets()
        self.load_config()
        self.setup_callbacks()
        debug_print("SoftWhisper initialization complete.")

    def setup_variables(self):
        debug_print("Setting up variables")
        self.model_loaded = False
        self.previous_model = "base"
        self.model_var = tk.StringVar(value="base")
        self.task_var = tk.StringVar(value="transcribe")
        # Transcription engine: "whisper.cpp" (default, local) or
        # "twelvelabs-pegasus" (opt-in cloud backend).
        self.engine_var = tk.StringVar(value="whisper.cpp")
        self.twelvelabs_api_key_var = tk.StringVar(value=os.environ.get("TWELVELABS_API_KEY", ""))
        self.language_var = tk.StringVar(value="auto")
        self.beam_size_var = tk.IntVar(value=5)
        self.start_time_var = tk.StringVar(value="00:00:00")
        self.end_time_var = tk.StringVar(value="")
        self.srt_var = tk.BooleanVar(value=False)
        self.file_path = None
        self.transcription_thread = None
        self.model_loading_thread = None
        self.transcription_stop_event = threading.Event()
        self.model_stop_event = threading.Event()
        self.slider_dragging = False

        # Store final segments & text
        self.current_segments = None
        self.current_text = None

        self.WHISPER_CPP_PATH = tk.StringVar(value=get_default_whisper_cpp_path())

        # Ensure last_dir is always initialized
        self.last_dir = os.getcwd()

        num_cores = psutil.cpu_count(logical=True)
        self.num_threads = max(1, int(num_cores * 0.8))
        debug_print(f"Using {self.num_threads} threads (logical cores * 0.8)")

    def setup_queues(self):
        debug_print("Setting up queues")
        self.console_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        # Redirect stdout and stderr immediately and keep it redirected
        set_console_redirect(self.console_queue)

    def create_widgets(self):
        debug_print("Creating widgets")
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left side: Media controls (using pack)
        media_frame = tk.Frame(main_frame)
        media_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Right side: Settings & output (using pack)
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # ---------------------------
        # Media Controls
        # ---------------------------
        self.video_frame = tk.Frame(media_frame, width=300, height=200, bg="black")
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(0)

        playback_frame = tk.Frame(media_frame)
        playback_frame.pack(pady=10)
        self.play_button = tk.Button(playback_frame, text="Play", font=("Arial", 12), state=tk.DISABLED)
        self.play_button.grid(row=0, column=0, padx=5)
        self.pause_button = tk.Button(playback_frame, text="Pause", font=("Arial", 12), state=tk.DISABLED)
        self.pause_button.grid(row=0, column=1, padx=5)
        self.stop_media_button = tk.Button(playback_frame, text="Stop", font=("Arial", 12), state=tk.DISABLED)
        self.stop_media_button.grid(row=0, column=2, padx=5)

        self.slider = ttk.Scale(playback_frame, from_=0, to=100, orient="horizontal", length=300)
        self.slider.grid(row=1, column=0, columnspan=3, pady=10)
        self.time_label = tk.Label(playback_frame, text="00:00:00 / 00:00:00", font=("Arial", 10))
        self.time_label.grid(row=2, column=0, columnspan=3)

        from media_player import MediaPlayerUI
        self.media_player_ui = MediaPlayerUI(
            parent_frame=self.video_frame,
            play_button=self.play_button,
            pause_button=self.pause_button,
            stop_button=self.stop_media_button,
            slider=self.slider,
            time_label=self.time_label,
            error_callback=lambda msg: messagebox.showerror("Media Error", msg)
        )
        self.play_button.config(command=lambda: (debug_print("Play media requested"), self.media_player_ui.play()))
        self.pause_button.config(command=lambda: (debug_print("Pause media requested"), self.media_player_ui.pause()))
        self.stop_media_button.config(command=lambda: (debug_print("Stop media requested"), self.media_player_ui.stop()))

        self.select_file_button = tk.Button(media_frame, text="Select Audio/Video File",
                                            command=self.select_file, font=("Arial", 12))
        self.select_file_button.pack(pady=10)

        buttons_frame = tk.Frame(media_frame)
        buttons_frame.pack(pady=5)
        self.start_button = tk.Button(buttons_frame, text="Start Transcription",
                                      command=self.start_transcription, font=("Arial", 12), state=tk.DISABLED)
        self.start_button.grid(row=0, column=0, padx=10, pady=5)
        self.stop_button = tk.Button(buttons_frame, text="Stop Transcription",
                                     command=self.stop_processing, font=("Arial", 12), state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=10, pady=5)

        # ---------------------------
        # Status and Console Output
        # ---------------------------
        self.status_label = tk.Label(right_frame, text="Checking Whisper.cpp model...", fg="blue",
                                     font=("Arial", 12), wraplength=700, justify="left")
        self.status_label.pack(pady=10)
        self.progress_bar = CustomProgressBar(right_frame, width=700, height=20)
        self.progress_bar.pack(pady=10)

        console_frame = tk.LabelFrame(right_frame, text="Console Output", padx=10, pady=10, font=("Arial", 12))
        console_frame.pack(padx=10, pady=10, fill="x")
        self.console_output_box = scrolledtext.ScrolledText(console_frame, wrap="word", width=80, height=5,
                                                            state=tk.DISABLED, font=("Courier New", 10))
        self.console_output_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        console_frame.rowconfigure(0, weight=1)
        console_frame.columnconfigure(0, weight=1)

        # ---------------------------
        # Settings Section (Using ONLY GRID)
        # ---------------------------
        settings_frame = tk.LabelFrame(right_frame, text="Optional Settings", padx=10, pady=10, font=("Arial", 12))
        settings_frame.pack(padx=10, pady=10, fill="x")

        # Row 0: Engine (transcription backend)
        tk.Label(settings_frame, text="Engine:", font=("Arial", 10)).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        engine_options = ["whisper.cpp", "twelvelabs-pegasus"]
        self.engine_menu = ttk.Combobox(settings_frame, textvariable=self.engine_var, values=engine_options,
                                        state="readonly", width=20, font=("Arial", 10))
        self.engine_menu.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.engine_menu.bind("<<ComboboxSelected>>", self.on_engine_change)

        # Row 1: Model
        tk.Label(settings_frame, text="Model:", font=("Arial", 10)).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        model_options = ["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                         "medium", "medium.en", "large", "large-v2", "large-v3", "large-v3-turbo"]
        self.model_menu = ttk.Combobox(settings_frame, textvariable=self.model_var, values=model_options,
                                       state="readonly", width=20, font=("Arial", 10))
        self.model_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.model_menu.bind("<<ComboboxSelected>>", self.on_model_change)

        # Row 2: Task
        tk.Label(settings_frame, text="Task:", font=("Arial", 10)).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        task_options = ["transcribe", "translate"]
        self.task_menu = ttk.Combobox(settings_frame, textvariable=self.task_var, values=task_options,
                                      state="readonly", width=20, font=("Arial", 10))
        self.task_menu.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Row 3: Language
        tk.Label(settings_frame, text="Language:", font=("Arial", 10)).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        lang_container = tk.Frame(settings_frame)
        lang_container.grid(row=3, column=1, sticky="w", padx=5, pady=5, columnspan=2)

        self.language_entry = tk.Entry(lang_container, textvariable=self.language_var, width=20, font=("Arial", 10))
        self.language_entry.pack(side="top", anchor="w")

        tk.Label(lang_container, text="(Use \"auto\" for auto-detection)", font=("Arial", 8)).pack(side="top", anchor="w")

        # Row 4: Beam Size
        tk.Label(settings_frame, text="Beam Size:", font=("Arial", 10)).grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.beam_size_spinbox = tk.Spinbox(settings_frame, from_=1, to=10, textvariable=self.beam_size_var,
                                            width=5, font=("Arial", 10))
        self.beam_size_spinbox.grid(row=4, column=1, sticky="w", padx=5, pady=5)

        # Row 5: Start Time
        tk.Label(settings_frame, text="Start Time [hh:mm:ss]:", font=("Arial", 10)).grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.start_time_entry = tk.Entry(settings_frame, textvariable=self.start_time_var, width=10, font=("Arial", 10))
        self.start_time_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)

        # Row 6: End Time
        tk.Label(settings_frame, text="End Time [hh:mm:ss]:", font=("Arial", 10)).grid(row=6, column=0, sticky="w", padx=5, pady=5)
        endtime_container = tk.Frame(settings_frame)
        endtime_container.grid(row=6, column=1, sticky="w", padx=5, pady=5, columnspan=2)

        self.end_time_entry = tk.Entry(endtime_container, textvariable=self.end_time_var, width=10, font=("Arial", 10))
        self.end_time_entry.pack(side="top", anchor="w")

        tk.Label(endtime_container, text="(Leave empty for full duration)", font=("Arial", 8)).pack(side="top", anchor="w")

        # Row 7: Generate SRT Subtitles Checkbox
        self.srt_checkbox = tk.Checkbutton(settings_frame, text="Generate SRT Subtitles", variable=self.srt_var, anchor="w")
        self.srt_checkbox.grid(row=7, column=1, sticky="w", padx=5, pady=2)

        # Row 8: Enable Diarization Checkbox
        self.diarization_option = DiarizationOption(settings_frame)
        self.diarization_option.checkbox.grid(row=8, column=1, sticky="w", padx=5, pady=2)

        # Row 9: Whisper.cpp Executable
        tk.Label(settings_frame, text="Whisper.cpp Executable:", font=("Arial", 10)).grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.whisper_location_entry = tk.Entry(settings_frame, textvariable=self.WHISPER_CPP_PATH, width=40, font=("Arial", 10))
        self.whisper_location_entry.grid(row=9, column=1, sticky="w", padx=5, pady=5)
        self.whisper_browse_button = tk.Button(settings_frame, text="Browse", command=self.browse_whisper_executable, font=("Arial", 10))
        self.whisper_browse_button.grid(row=9, column=2, sticky="w", padx=5, pady=5)

        # Row 10: TwelveLabs API key (used only by the Pegasus engine)
        tk.Label(settings_frame, text="TwelveLabs API Key:", font=("Arial", 10)).grid(row=10, column=0, sticky="w", padx=5, pady=5)
        self.twelvelabs_key_entry = tk.Entry(settings_frame, textvariable=self.twelvelabs_api_key_var,
                                             width=40, show="*", font=("Arial", 10))
        self.twelvelabs_key_entry.grid(row=10, column=1, sticky="w", padx=5, pady=5)
        tk.Label(settings_frame, text="(or set TWELVELABS_API_KEY env var)", font=("Arial", 8)).grid(
            row=11, column=1, sticky="w", padx=5)

        # ---------------------------
        # Transcription Frame
        # ---------------------------
        transcription_frame = tk.LabelFrame(right_frame, text="Transcription", padx=10, pady=10, font=("Arial", 12))
        transcription_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.transcription_box = scrolledtext.ScrolledText(transcription_frame, wrap="word", width=80, height=10, font=("Arial", 10))
        self.transcription_box.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        transcription_frame.rowconfigure(0, weight=1)
        transcription_frame.columnconfigure(0, weight=1)
        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.transcription_box.bind(key, lambda e: "break")

        spacer_frame = tk.Frame(media_frame, height=460)
        spacer_frame.pack(pady=5)
        export_frame, self.export_button = create_export_button(media_frame, self)
        export_frame.pack(side="bottom", pady=10, before=spacer_frame)

        debug_print("Widgets created.")

    def _set_buttons_state(self, select=True, start=False, stop=False, play=False, pause=False, stop_media=False):
        # Centralized method to set button states
        self.select_file_button.config(state=tk.NORMAL if select else tk.DISABLED)
        self.start_button.config(state=tk.NORMAL if start else tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL if stop else tk.DISABLED)
        self.play_button.config(state=tk.NORMAL if play else tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL if pause else tk.DISABLED)
        self.stop_media_button.config(state=tk.NORMAL if stop_media else tk.DISABLED)

    def _reset_for_new_operation(self, status_message, status_color="orange"):
        # Reset UI and state for a new operation (transcription, model change, etc.)
        self.progress_bar.set_progress(0)
        self.clear_transcription_box()
        self.clear_console_output()
        self.export_button.config(state=tk.DISABLED)
        self.current_segments = None
        self.current_text = None
        self.update_status(status_message, status_color)

    def _resolve_whisper_executable(self, path_value):
        # Ensure executable path is absolute and points to the binary
        exe_path = path_value
        if os.path.isdir(exe_path):
            if os.name == "nt":
                exe_path = os.path.join(exe_path, "whisper-cli.exe")
            else:
                exe_path = os.path.join(exe_path, "whisper-cli")
        return os.path.abspath(exe_path)

    def _ensure_model_file(self, selected_model, progress_queue):
        # Ensure model file exists, download if allowed and missing
        model_filename = f"ggml-{selected_model}.bin"
        model_path = os.path.join("models", "whisper", model_filename)
        debug_print(f"Looking for model file: {model_path}")
        if not os.path.exists(model_path):
            if model_filename in ALLOWED_MODELS:
                debug_print(f"Model file not found, attempting download for {model_filename}...")
                whisper_folder = os.path.join("models", "whisper")
                if not os.path.exists(whisper_folder):
                    os.makedirs(whisper_folder)
                url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model_filename}"
                debug_print(f"Downloading model from {url}")

                def download_reporthook(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percentage = int(downloaded / total_size * 100) if total_size > 0 else 0
                    progress_message = f"Downloading {model_filename}: {percentage}%"
                    progress_queue.put((percentage, progress_message))

                urllib.request.urlretrieve(url, model_path, reporthook=download_reporthook)
                debug_print("Download successful.")
                progress_queue.put((100, f"Download of {model_filename} complete"))
            else:
                raise Exception(f"Model file {model_path} not found and automatic download is not supported for this model.")
        return model_path

    def _format_and_display_transcription(self, result):
        # Format and display transcription based on options and diarization
        if self.transcription_stop_event.is_set() or result.get('cancelled', False):
            return

        raw_output = result.get('raw', '')

        if hasattr(self, 'diarization_option') and self.diarization_option.is_enabled():
            self.current_text = raw_output
            debug_print("Converting to SRT format for diarization")
            srt_content = whisper_to_srt(self.current_text)
            from diarization_gui import merge_diarization

            def diarization_progress_callback(progress, message):
                if self.transcription_stop_event.is_set():
                    return
                self.progress_queue.put((progress, message))

            diarized_text = merge_diarization(
                self.file_path,
                srt_content,
                remove_timestamps=not self.srt_var.get(),
                progress_callback=diarization_progress_callback
            )
            self.current_text = diarized_text
            self.display_transcription(diarized_text)
        elif self.srt_var.get():
            self.current_text = raw_output
            debug_print("Converting to proper SRT format for display")
            srt_content = whisper_to_srt(self.current_text)
            self.current_text = srt_content
            self.display_transcription(srt_content)
        else:
            lines = raw_output.splitlines()
            plain_lines = [re.sub(r'^\[[^\]]+\]\s*', '', line) for line in lines if line.strip()]
            plain_text = " ".join(plain_lines)
            self.current_text = plain_text
            self.display_transcription(plain_text)

        self.current_segments = result.get('segments', [])
        if len(self.current_text.strip()) > 0:
            self.export_button.config(state=tk.NORMAL)

        self.update_status("Transcription completed.", "green")
        self.console_queue.put({'type': 'append', 'content': "Transcription process complete.\n"})

    def browse_whisper_executable(self):
        current_path = self.WHISPER_CPP_PATH.get()
        init_dir = os.path.dirname(current_path) if current_path else os.getcwd()
        file_path = filedialog.askopenfilename(
            title="Select Whisper.cpp Executable",
            initialdir=init_dir,
            filetypes=[("Executable Files", "*.exe" if os.name == "nt" else "*.*")]
        )
        if file_path:
            self.WHISPER_CPP_PATH.set(file_path)
            debug_print(f"User selected Whisper.cpp executable: {file_path}")

    def load_config(self):
        debug_print("Loading configuration")
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                # Restore beam size
                self.beam_size_var.set(config.get('beam_size', 5))
                # Restore Whisper path
                self.WHISPER_CPP_PATH.set(config.get('WHISPER_CPP_PATH', get_default_whisper_cpp_path()))
                # Restore transcription engine (default: local Whisper.cpp)
                self.engine_var.set(config.get('engine', 'whisper.cpp'))
                # Restore last‑opened folder (fallback to already-set self.last_dir)
                self.last_dir = config.get('last_dir', self.last_dir)
                debug_print(f"Configuration loaded: {config}")
            except Exception as e:
                debug_print(f"Error loading config: {e}")
        else:
            debug_print("No configuration file found; using defaults.")

    def save_config(self, *args, **kwargs):
        config = {
            'beam_size': self.beam_size_var.get(),
            'WHISPER_CPP_PATH': self.WHISPER_CPP_PATH.get(),
            'last_dir': self.last_dir,
            'engine': self.engine_var.get()
        }
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            debug_print(f"Error saving config: {e}")

    def setup_callbacks(self):
        debug_print("Setting up callbacks")
        self.model_var.trace("w", self.save_config)
        self.beam_size_var.trace("w", self.save_config)
        self.check_queues()

    def load_model(self):
        debug_print("Entering load_model()")
        # The TwelveLabs Pegasus engine is cloud-based; no local model file.
        if self.engine_var.get() == "twelvelabs-pegasus":
            self.model_loaded = True
            self.progress_queue.put((100, "TwelveLabs Pegasus engine ready (cloud)"))
            self.root.after(0, self.enable_buttons)
            return
        selected_model = self.model_var.get()
        try:
            self.progress_queue.put((0, f"Checking model '{selected_model}'..."))
            # Ensure model file exists (and download if necessary)
            self._ensure_model_file(selected_model, self.progress_queue)
            self.model_loaded = True
            self.previous_model = selected_model
            self.progress_queue.put((100, f"Model '{selected_model}' is ready (Whisper.cpp)"))
            debug_print("Model loaded successfully.")
            self.root.after(0, self.enable_buttons)
        except Exception as e:
            self.progress_queue.put((0, f"Error: {str(e)}"))
            self.console_queue.put({'type': 'append', 'content': f"Error loading model: {str(e)}\n"})
            messagebox.showerror("Model Loading Error", f"Failed to load model '{selected_model}'.\nError: {str(e)}")
            self.model_var.set(self.previous_model if hasattr(self, 'previous_model') else "base")
            self.root.after(0, self.enable_buttons)
            debug_print("load_model() encountered an error.")

    def on_engine_change(self, event):
        debug_print(f"Engine change requested: {self.engine_var.get()}")
        self.save_config()
        if self.engine_var.get() == "twelvelabs-pegasus":
            # Cloud engine needs no local model file; allow starting immediately.
            self.model_loaded = True
            self.update_status("TwelveLabs Pegasus engine selected (cloud).", "blue")
            self.root.after(0, self.enable_buttons)
        else:
            # Switching back to local Whisper.cpp: re-check the model file.
            self.model_loaded = False
            self.update_status("Checking selected Whisper.cpp model...", "blue")
            self.disable_buttons()
            self.model_loading_thread = threading.Thread(target=self.load_model, daemon=True)
            self.model_loading_thread.start()

    def on_model_change(self, event):
        debug_print("Model change requested")
        selected_model = self.model_var.get()
        if self.model_loaded and selected_model != self.previous_model:
            response = messagebox.askyesno("Change Model", "Changing the model will require a check. Continue?")
            if response:
                self.transcription_stop_event.set()
                self.model_loaded = False
                self._reset_for_new_operation("Checking selected Whisper.cpp model...", "blue")
                self.disable_buttons()
                self.transcription_stop_event.clear()
                self.model_loading_thread = threading.Thread(target=self.load_model, daemon=True)
                self.model_loading_thread.start()
            else:
                self.model_var.set(self.previous_model)
        elif not self.model_loaded:
            self.update_status("Checking selected Whisper.cpp model...", "blue")
            self.disable_buttons()
            self.model_loading_thread = threading.Thread(target=self.load_model, daemon=True)
            self.model_loading_thread.start()

    def select_file(self):
        debug_print("User requested file selection")
        
        # Use the simple approach that works
        file_path = filedialog.askopenfilename(
            title="Select Audio/Video File",
            initialdir=self.last_dir,
            filetypes=[("Audio/Video Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.wma *.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")],
            multiple=False
        )
        
        if file_path:
            self.last_dir = os.path.dirname(file_path)
            self.save_config()

            self.file_path = file_path
            filename = os.path.basename(file_path)
            if len(filename) > 50:
                filename = filename[:47] + '...'
            self.update_status(f"Selected file: {filename}", "blue")
            debug_print(f"File selected: {file_path}")

            # Clear old transcription data
            self.clear_transcription_box()
            self.clear_console_output()
            self.current_segments = None
            self.current_text = None
            self.export_button.config(state=tk.DISABLED)

            if self.model_loaded:
                self.start_button.config(state=tk.NORMAL)

            self.root.update_idletasks()
            self.media_player_ui.load_media(file_path)

            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_media_button.config(state=tk.NORMAL)
            self.root.update_idletasks()
        else:
            # Do NOT reset self.file_path if no new file is selected.
            debug_print("No file selected, keeping previous file.")

    def start_transcription(self):
        debug_print("Start transcription requested")
        if not self.file_path:
            messagebox.showwarning("No File Selected", "Please select an audio/video file to transcribe.")
            return
        if not self.model_loaded:
            messagebox.showwarning("Model Not Ready", "Please wait until the Whisper.cpp model is ready.")
            return

        self.disable_buttons()
        self.stop_button.config(state=tk.NORMAL)
        self._reset_for_new_operation("Preparing for transcription...", "orange")
        self.transcription_stop_event.clear()

        self.transcription_thread = threading.Thread(target=self.transcribe_file, args=(self.file_path,), daemon=True)
        self.transcription_thread.start()
        debug_print("Transcription thread started.")

    def stop_processing(self):
        debug_print("Stop transcription requested")
        self.transcription_stop_event.set()
        self.update_status("Stopping transcription...", "red")
        self.stop_button.config(state=tk.DISABLED)
        self.root.after(0, self.enable_buttons)

    def transcribe_file(self, file_path: str):
        debug_print(f"transcribe_file() => {file_path}")
        set_console_redirect(self.console_queue)

        try:
            lang = self.language_var.get().strip().lower() or "auto"
            debug_print(f"Language setting: '{lang}'")

            options = {
                'engine': self.engine_var.get(),
                'model_name': self.model_var.get(),
                'task': self.task_var.get(),
                'language': lang,
                'beam_size': self.beam_size_var.get(),
                'start_time': self.start_time_var.get().strip(),
                'end_time': self.end_time_var.get().strip(),
                'generate_srt': self.srt_var.get(),
                'parent_window': self.root,
                'whisper_executable': self.WHISPER_CPP_PATH.get(),
                'twelvelabs_api_key': self.twelvelabs_api_key_var.get().strip(),
            }

            # Absolute path for the input file
            file_path = os.path.abspath(file_path)

            # Whisper.cpp-only setup. The TwelveLabs Pegasus engine is
            # cloud-based and needs no local executable, so skip it.
            if options['engine'] != 'twelvelabs-pegasus':
                # Resolve executable path
                executable_abs = self._resolve_whisper_executable(options['whisper_executable'])
                options['whisper_executable'] = executable_abs
                debug_print(f"Using Whisper executable: {executable_abs}")

                # Build a rough command template for debugging purposes
                model_abs = os.path.abspath(os.path.join("models", "whisper", f"ggml-{options['model_name']}.bin"))
                whisper_cmd = f"{executable_abs} -m {model_abs} -f {file_path} -l {options['language']} -bs {options['beam_size']}"
                if options['task'] == 'translate':
                    whisper_cmd += " -translate"
                whisper_cmd += " -oj"
                debug_print(f"Command template: {whisper_cmd}")

            # Define callbacks for progress and status updates
            def progress_callback(progress, message):
                if self.transcription_stop_event.is_set():
                    return
                self.progress_queue.put((progress, message))

            def status_callback(message, color):
                self.update_status(message, color)

            debug_print("Calling transcribe_audio()...")
            if not self.transcription_stop_event.is_set():
                result = transcribe_audio(
                    file_path=file_path,
                    options=options,
                    progress_callback=progress_callback,
                    status_callback=status_callback,
                    stop_event=self.transcription_stop_event
                )
                debug_print("Transcription completed or cancelled")

                if (not self.transcription_stop_event.is_set()) and not result.get('cancelled', False):
                    self._format_and_display_transcription(result)
                elif result.get('cancelled', False):
                    self.update_status("Transcription cancelled by user.", "red")
                else:
                    self.update_status("Transcription aborted.", "red")

                if result.get('temp_audio_path') and os.path.exists(result['temp_audio_path']):
                    try:
                        os.remove(result['temp_audio_path'])
                    except Exception:
                        pass
            else:
                self.update_status("Transcription aborted.", "red")
        except Exception as e:
            import traceback
            error_msg = str(e)
            stack_trace = traceback.format_exc()
            self.console_queue.put({'type': 'append', 'content': f"Error during transcription: {error_msg}\n{stack_trace}\n"})
            self.progress_queue.put((0, f"Error during transcription: {error_msg}"))
            messagebox.showerror("Transcription Error", f"Failed to transcribe the audio/video file.\nError: {error_msg}")
            debug_print(f"Transcription error: {error_msg}")
        finally:
            self.root.after(100, self.enable_buttons)

    def disable_buttons(self):
        self._set_buttons_state(
            select=False, start=False, stop=False, play=False, pause=False, stop_media=False
        )

    def enable_buttons(self):
        self._set_buttons_state(
            select=True,
            start=bool(self.file_path and self.model_loaded),
            stop=False,
            play=bool(self.file_path and self.model_loaded),
            pause=bool(self.file_path and self.model_loaded),
            stop_media=bool(self.file_path and self.model_loaded)
        )

    def update_status(self, message, color):
        def update():
            self.status_label.config(text=message, fg=color)
        self.root.after(0, update)

    def check_queues(self):
        # Flag to determine if we need to update the UI
        needs_update = False

        # Process progress queue
        try:
            while True:
                progress, status_message = self.progress_queue.get_nowait()
                self.progress_bar.set_progress(progress)
                if status_message:
                    self.update_status(status_message, "blue")
                needs_update = True
        except queue.Empty:
            pass

        # Process console queue
        try:
            while True:
                message_data = self.console_queue.get_nowait()
                self.console_output_box.config(state=tk.NORMAL)

                if message_data['type'] == 'append':
                    self.console_output_box.insert(tk.END, message_data['content'])
                    if float(self.console_output_box.index(tk.END)) > 1000:
                        self.console_output_box.delete(1.0, "end-500l")
                    self.console_output_box.see(tk.END)
                elif message_data['type'] == 'clear':
                    self.console_output_box.delete(1.0, tk.END)

                self.console_output_box.config(state=tk.DISABLED)
                needs_update = True
        except queue.Empty:
            pass

        # Process transcription queue
        try:
            while True:
                action = self.transcription_queue.get_nowait()
                if action['type'] == 'set_text':
                    self.transcription_box.delete(1.0, tk.END)
                    self.transcription_box.insert(tk.END, action['text'])
                elif action['type'] == 'clear':
                    self.transcription_box.delete(1.0, tk.END)
                needs_update = True
        except queue.Empty:
            pass

        # Force an update if needed
        if needs_update:
            self.root.update_idletasks()

        # Schedule the next check
        self.root.after(50, self.check_queues)

    def clear_transcription_box(self):
        # Queue a request to clear the transcription textbox
        self.transcription_queue.put({'type': 'clear'})

    def clear_console_output(self):
        # Clear the console output textbox
        self.console_output_box.config(state=tk.NORMAL)
        self.console_output_box.delete(1.0, tk.END)
        self.console_output_box.config(state=tk.DISABLED)

    def display_transcription(self, text):
        # Queue transcription text to be displayed in the textbox
        self.transcription_queue.put({'type': 'set_text', 'text': text})

    def set_window_centered(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        geometry_str = f"{width}x{height}+{x}+{y}"
        self.root.geometry(geometry_str)
        debug_print(f"Window geometry set to: {geometry_str}")

    def on_closing(self):
        debug_print("Closing application")
        self.transcription_stop_event.set()
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join()
        if hasattr(self, 'media_player_ui'):
            self.media_player_ui.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    debug_print("Starting SoftWhisper using Whisper.cpp...")
    root = tk.Tk()
    app = SoftWhisper(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.load_model()
    app.check_queues()
    debug_print("Entering mainloop")
    root.mainloop()
