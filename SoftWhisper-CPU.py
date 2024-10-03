'''
SoftWhisper Functions
├── GUI Setup
│   ├── __init__(self, root: tk.Tk)
│   │   └── Initializes the SoftWhisper app with a main window.
│   │       └── Calls set_window_centered(1000, 800).
│   │       └── Sets title: "SoftWhisper".
│   │       └── Loads configurations from config.json.
│   │       └── Creates UI components via create_widgets().
│   │       └── Initializes VLC media player.
│   │       └── Initializes speaker diarization pipeline (if enabled).
│   │       └── Starts loading the default Whisper model in a background thread.
│   ├── set_window_centered(self, width: int, height: int)
│   │   └── Centers the window on the screen with given dimensions.

├── Widget Creation
│   ├── create_widgets(self)
│   │   └── Creates the main frame and media player controls.
│   │   └── Adds play, pause, stop buttons, and playback slider.
│   │   └── Adds a file selection button.
│   │   └── Adds start/stop transcription buttons.
│   │   └── Adds status label, custom progress bar, and console output box.
│   │   └── Adds model selection, task selection, language entry, beam size, start/end time entries.
│   │   └── Adds speaker diarization checkbox and HuggingFace API token input.
│   │   └── Adds transcription text box for displaying results.

├── Model Loading
│   ├── load_model(self)
│   │   └── Loads the selected Whisper model in a background thread.
│   │   └── Adjusts process priority and loads the model.
│   │   └── Updates progress bar and status label during model loading.
│   │   └── Handles errors and resets the model to previous if loading fails.
│   │   └── Enables file selection after loading completes.
│   ├── _load_model_internal(self, selected_model: str)
│   │   └── Internal method for loading the Whisper model.

├── Transcription Process
│   ├── select_file(self)
│   │   └── Opens file dialog to select an audio/video file.
│   │   └── Prepares the media for playback via VLC.
│   │   └── Enables transcription start button if the model is loaded.
│   ├── start_transcription(self)
│   │   └── Starts transcription for the selected audio/video file in a separate thread.
│   │   └── Prepares the file and optional settings for transcription.
│   ├── transcribe_file(self, file_path: str)
│   │   └── Extracts audio using pydub.
│   │   └── Trims audio based on start/end time settings.
│   │   └── Processes the transcription via the Whisper model.
│   │   └── Runs speaker diarization if enabled.
│   │   └── Displays the transcription in the text box and updates the progress bar.
│   │   └── Generates SRT subtitles if requested and saves them.

├── Media Playback
│   ├── prepare_media_playback(self)
│   │   └── Loads selected media into VLC player.
│   │   └── Configures VLC player window based on platform.
│   ├── play_media(self)
│   │   └── Plays the loaded media.
│   ├── pause_media(self)
│   │   └── Pauses the media playback.
│   ├── stop_media(self)
│   │   └── Stops the media playback.
│   ├── update_slider(self)
│   │   └── Updates the slider based on current media playback position.
│   ├── on_slider_press(self, event)
│   │   └── Tracks slider interaction for manual position adjustment.
│   ├── on_slider_release(self, event)
│   │   └── Adjusts media playback position after slider interaction.

├── Utility Functions
│   ├── update_status(self, message: str, color: str)
│   │   └── Updates the status label with a message in the specified color.
│   ├── parse_time(self, time_str: str) -> int
│   │   └── Converts a time string in hh:mm:ss format to total seconds.
│   ├── format_time(self, seconds: int) -> str
│   │   └── Formats total seconds into hh:mm:ss format.
│   ├── format_timestamp(self, seconds: float) -> str
│   │   └── Formats seconds into SRT subtitle timestamp format.
│   ├── display_transcription(self, text: str)
│   │   └── Displays the transcribed text in the scrolled text box.
│   ├── generate_srt_text_with_speakers(self, segments: list) -> str
│   │   └── Generates SRT formatted text with speaker labels.
│   ├── save_srt_file(self, segments: list, original_file_path: str)
│   │   └── Saves the SRT file for the transcription with speaker labels.
│   ├── write_srt_file(self, save_path: str, segments: list)
│   │   └── Writes the transcription to an SRT file.

├── Configuration Management
│   ├── load_config(self)
│   │   └── Loads speaker diarization and HuggingFace token from config.json.
│   ├── save_config(self, event)
│   │   └── Saves speaker diarization and HuggingFace token to config.json.

├── Diarization Handling
│   ├── on_speaker_id_toggle(self)
│   │   └── Toggles speaker diarization and manages HuggingFace token entry.
│   ├── initialize_diarization_pipeline(self)
│   │   └── Initializes the speaker diarization pipeline from HuggingFace.

├── Event Handling
│   ├── check_queues(self)
│   │   └── Monitors queues for progress and console output updates.
│   ├── on_model_change(self, event)
│   │   └── Handles model change and reloads the selected Whisper model.
│   ├── stop_processing(self)
│   │   └── Stops the transcription process if requested by the user.
│   ├── on_closing(self)
│   │   └── Handles application close, stops transcription, and cleans up.
'''

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import whisper
import os
from pydub import AudioSegment
import tempfile
import re
import sys
import queue
import time
import vlc  # For VLC media player integration
import torch  # For setting the number of threads
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil  # For adjusting process priority
from pyannote.audio import Pipeline

# Maximum duration per chunk in seconds when transcribing segments.
MAX_CHUNK_DURATION = 60  


import json

CONFIG_FILE = 'config.json'

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
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        while '\n' in self.buffer or '\r' in self.buffer:
            if '\r' in self.buffer:
                line, self.buffer = self.buffer.split('\r', 1)
                self.console_queue.put({'type': 'overwrite', 'content': self.buffer})
            elif '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                self.console_queue.put({'type': 'append', 'content': line + '\n'})

    def flush(self):
        if self.buffer:
            self.console_queue.put({'type': 'append', 'content': self.buffer})
            self.buffer = ""

class SoftWhisper:
    def __init__(self, root):
        self.root = root
        self.root.title("SoftWhisper")
        # Adjusted window size to better fit desktop
        self.set_window_centered(1000, 800)  # Modified from 1100x900 to 900x700
        self.root.resizable(False, False)

        # Initialize variables
        self.model = None
        self.model_loaded = False
        self.previous_model = "base"
        self.model_var = tk.StringVar(value="base")
        self.task_var = tk.StringVar(value="transcribe")
        self.language_var = tk.StringVar(value="auto")
        self.beam_size_var = tk.IntVar(value=5)  # Default beam size
        self.start_time_var = tk.StringVar(value="00:00:00")
        self.end_time_var = tk.StringVar(value="")
        self.srt_var = tk.BooleanVar(value=False)
        self.speaker_id_var = tk.BooleanVar(value=False)
        self.hf_token_var = tk.StringVar(value="")

        self.file_path = None
        self.transcription_thread = None
        self.model_loading_thread = None
        self.transcription_stop_event = threading.Event()
        self.model_stop_event = threading.Event()
        self.slider_dragging = False  # Flag to track slider interaction

        # Lock for model access to ensure thread safety
        self.model_lock = threading.Lock()

        # Determine number of threads based on CPU cores (80% of available cores)
        num_cores = psutil.cpu_count(logical=True)
        self.num_threads = max(1, int(num_cores * 0.8))
        torch.set_num_threads(self.num_threads)

        # VLC player instance
        self.vlc_instance = vlc.Instance()
        self.player = self.vlc_instance.media_player_new()

        # Create queues for progress and console output
        self.progress_queue = queue.Queue()
        self.console_queue = queue.Queue()

        # Load configuration
        self.load_config()

        # Create UI components
        self.create_widgets()

        # Bind beam size changes to save_config
        self.beam_size_var.trace("w", self.save_config)

        # Start checking for progress and console output updates
        self.check_queues()

        # Initialize speaker diarization pipeline (if enabled)
        self.diarization_pipeline = None
        if self.speaker_id_var.get():
            self.on_speaker_id_toggle()  # Ensure API token textbox state is set correctly

        # Start loading the default Whisper model in a background thread
        self.model_loading_thread = threading.Thread(target=self.load_model, daemon=True)
        self.model_loading_thread.start()

    def set_window_centered(self, width, height):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def create_widgets(self):
        # Main frame to hold all widgets
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)

        # Left frame for media player and controls
        media_frame = tk.Frame(main_frame)
        media_frame.pack(side="left", fill="y", padx=10, pady=10)

        # Right frame for controls and text boxes
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # VLC Video Frame
        self.video_frame = tk.Frame(media_frame, width=300, height=200)  # Adjusted width
        self.video_frame.pack(pady=10)
        self.video_frame.configure(bg='black')

        # Ensure the video frame size is fixed
        self.video_frame.pack_propagate(0)

        # Playback Controls Frame
        playback_frame = tk.Frame(media_frame)
        playback_frame.pack(pady=10)

        # Play Button
        self.play_button = tk.Button(
            playback_frame,
            text="Play",
            command=self.play_media,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.play_button.grid(row=0, column=0, padx=5)

        # Pause Button
        self.pause_button = tk.Button(
            playback_frame,
            text="Pause",
            command=self.pause_media,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.pause_button.grid(row=0, column=1, padx=5)

        # Stop Button
        self.stop_media_button = tk.Button(
            playback_frame,
            text="Stop",
            command=self.stop_media,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.stop_media_button.grid(row=0, column=2, padx=5)

        # Playback Slider
        self.slider = ttk.Scale(
            playback_frame,
            from_=0,
            to=100,
            orient='horizontal',
            length=300  # Adjusted length
        )
        self.slider.grid(row=1, column=0, columnspan=3, pady=10)
        self.slider.bind('<ButtonPress-1>', self.on_slider_press)
        self.slider.bind('<ButtonRelease-1>', self.on_slider_release)

        # Label to show current time and total duration
        self.time_label = tk.Label(
            playback_frame,
            text="00:00:00 / 00:00:00",
            font=("Arial", 10)
        )
        self.time_label.grid(row=2, column=0, columnspan=3)

        # File Selection Button
        self.select_file_button = tk.Button(
            media_frame,
            text="Select Audio/Video File",
            command=self.select_file,
            font=("Arial", 12)
        )
        self.select_file_button.pack(pady=10)

        # Start and Stop Transcription Buttons
        buttons_frame = tk.Frame(media_frame)
        buttons_frame.pack(pady=5)

        self.start_button = tk.Button(
            buttons_frame,
            text="Start Transcription",
            command=self.start_transcription,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.start_button.grid(row=0, column=0, padx=10, pady=5)

        self.stop_button = tk.Button(
            buttons_frame,
            text="Stop Transcription",
            command=self.stop_processing,
            font=("Arial", 12),
            state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1, padx=10, pady=5)

        # Status Label
        self.status_label = tk.Label(
            right_frame,
            text="Loading Whisper model...",
            fg="blue",
            font=("Arial", 12),
            wraplength=700,
            justify="left"
        )
        self.status_label.pack(pady=10)

        # Custom Progress Bar
        self.progress_bar = CustomProgressBar(right_frame, width=700, height=20)
        self.progress_bar.pack(pady=10)

        # Console Output Frame
        console_frame = tk.LabelFrame(
            right_frame,
            text="Console Output",
            padx=10,
            pady=10,
            font=("Arial", 12)
        )
        console_frame.pack(padx=10, pady=10, fill="x")

        # Console Output Text Box
        self.console_output_box = scrolledtext.ScrolledText(
            console_frame,
            wrap=tk.WORD,
            width=80,
            height=5,
            state=tk.DISABLED,
            font=("Courier New", 10)
        )
        self.console_output_box.pack(pady=5, fill="x", expand=False)

        # Frame for Optional Settings
        settings_frame = tk.LabelFrame(
            right_frame,
            text="Optional Settings",
            padx=10,
            pady=10,
            font=("Arial", 12)
        )
        settings_frame.pack(padx=10, pady=10, fill="x")

        # Model Selection
        tk.Label(
            settings_frame,
            text="Model:",
            font=("Arial", 10)
        ).grid(row=0, column=0, sticky="w", pady=5)
        model_options = [
            "tiny", "tiny.en", "base", "base.en",
            "small", "small.en", "medium", "medium.en",
            "large", "turbo"
        ]
        self.model_menu = ttk.Combobox(
            settings_frame,
            textvariable=self.model_var,
            values=model_options,
            state="readonly",
            width=20,
            font=("Arial", 10)
        )
        self.model_menu.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.model_menu.bind("<<ComboboxSelected>>", self.on_model_change)

        # Task Selection
        tk.Label(
            settings_frame,
            text="Task:",
            font=("Arial", 10)
        ).grid(row=1, column=0, sticky="w", pady=5)
        task_options = ["transcribe", "translate"]
        self.task_menu = ttk.Combobox(
            settings_frame,
            textvariable=self.task_var,
            values=task_options,
            state="readonly",
            width=20,
            font=("Arial", 10)
        )
        self.task_menu.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Language Selection
        tk.Label(
            settings_frame,
            text="Language:",
            font=("Arial", 10)
        ).grid(row=2, column=0, sticky="w", pady=5)
        self.language_entry = tk.Entry(
            settings_frame,
            textvariable=self.language_var,
            width=20,
            font=("Arial", 10)
        )
        self.language_entry.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        tk.Label(
            settings_frame,
            text="(Use 'auto' for auto-detection)",
            font=("Arial", 8)
        ).grid(row=2, column=2, sticky="w", pady=5)

        # Beam Size
        tk.Label(
            settings_frame,
            text="Beam Size:",
            font=("Arial", 10)
        ).grid(row=3, column=0, sticky="w", pady=5)
        self.beam_size_spinbox = tk.Spinbox(
            settings_frame,
            from_=1,
            to=10,
            textvariable=self.beam_size_var,
            width=5,
            font=("Arial", 10)
        )
        self.beam_size_spinbox.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        # Bind beam size change to save_config
        self.beam_size_spinbox.bind("<FocusOut>", self.save_config)

        # Start and End Time Selection
        tk.Label(
            settings_frame,
            text="Start Time [hh:mm:ss]:",
            font=("Arial", 10)
        ).grid(row=4, column=0, sticky="w", pady=5)
        self.start_time_entry = tk.Entry(
            settings_frame,
            textvariable=self.start_time_var,
            width=10,
            font=("Arial", 10)
        )
        self.start_time_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        # Bind start_time_entry change to save_config
        self.start_time_entry.bind("<FocusOut>", self.save_config)

        tk.Label(
            settings_frame,
            text="End Time [hh:mm:ss]:",
            font=("Arial", 10)
        ).grid(row=5, column=0, sticky="w", pady=5)
        self.end_time_entry = tk.Entry(
            settings_frame,
            textvariable=self.end_time_var,
            width=10,
            font=("Arial", 10)
        )
        self.end_time_entry.grid(row=5, column=1, sticky="w", padx=5, pady=5)
        tk.Label(
            settings_frame,
            text="(Leave empty for full duration)",
            font=("Arial", 8)
        ).grid(row=5, column=2, sticky="w", pady=5)
        # Bind end_time_entry change to save_config
        self.end_time_entry.bind("<FocusOut>", self.save_config)

        # **Align "Enable Speaker Diarization" with "HuggingFace API Token"**
        # Create a separate frame for alignment
        alignment_frame = tk.Frame(settings_frame)
        alignment_frame.grid(row=6, column=0, columnspan=3, sticky="w", pady=5)

        # Enable Speaker Diarization Checkbox
        self.speaker_id_checkbox = tk.Checkbutton(
            alignment_frame,
            text="Enable Speaker Diarization",
            variable=self.speaker_id_var,
            command=self.on_speaker_id_toggle
        )
        self.speaker_id_checkbox.pack(side="left", padx=(0, 50))  # Added padding to separate from API token

        # HuggingFace API Token Label and Entry
        tk.Label(
            alignment_frame,
            text="HuggingFace API Token:",
            font=("Arial", 10)
        ).pack(side="left")
        self.hf_token_entry = tk.Entry(
            alignment_frame,
            textvariable=self.hf_token_var,
            width=30,
            font=("Arial", 10),
            show="*"  # Masked input
        )
        self.hf_token_entry.pack(side="left", padx=5, pady=5)
        # **Updated: Set the state based on speaker_id_var**
        if self.speaker_id_var.get():
            self.hf_token_entry.config(state=tk.NORMAL)
        else:
            self.hf_token_entry.config(state=tk.DISABLED)
        self.hf_token_entry.bind("<FocusOut>", self.save_config)

        # Generate SRT Subtitles Checkbox
        self.srt_checkbox = tk.Checkbutton(
            settings_frame,
            text="Generate SRT Subtitles",
            variable=self.srt_var
        )
        self.srt_checkbox.grid(row=7, column=0, sticky="w", pady=5)

        # Transcription Frame
        transcription_frame = tk.LabelFrame(
            right_frame,
            text="Transcription",
            padx=10,
            pady=10,
            font=("Arial", 12)
        )
        transcription_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Transcription Text Box
        self.transcription_box = scrolledtext.ScrolledText(
            transcription_frame,
            wrap=tk.WORD,
            width=80,
            height=10,
            state=tk.DISABLED,
            font=("Arial", 10)
        )
        self.transcription_box.pack(pady=5, fill="both", expand=True)



    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.speaker_id_var.set(config.get('speaker_identification', False))
                    self.hf_token_var.set(config.get('huggingface_token', ''))
                    self.beam_size_var.set(config.get('beam_size', 5))  # **Added Beam Size**
            except Exception as e:
                print(f"Error loading config: {e}")

    def save_config(self, *args, **kwargs):  # **Modified to accept arbitrary arguments**
        config = {
            'speaker_identification': self.speaker_id_var.get(),
            'huggingface_token': self.hf_token_var.get() if self.speaker_id_var.get() else '',
            'beam_size': self.beam_size_var.get()  # **Added Beam Size**
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)  # **Added indent for readability**
        except Exception as e:
            print(f"Error saving config: {e}")


    def on_speaker_id_toggle(self):
        if self.speaker_id_var.get():
            self.hf_token_entry.config(state=tk.NORMAL)
            self.initialize_diarization_pipeline()
        else:
            self.hf_token_entry.config(state=tk.DISABLED)
            self.diarization_pipeline = None
        self.save_config() 

    def initialize_diarization_pipeline(self):
        if not self.hf_token_var.get():
            messagebox.showwarning("HuggingFace Token Required", "Please enter your HuggingFace API token to enable speaker diarization.")
            # Keep the checkbox selected but inform the user to enter the token
            return
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token_var.get()
            )
            self.update_status("Speaker diarization model loaded.", "green")
        except Exception as e:
            self.update_status(f"Failed to load diarization model: {str(e)}", "red")
            messagebox.showerror(
                "Diarization Model Error",
                f"Failed to load diarization model.\nError: {str(e)}\n\n"
                f"Please ensure your HuggingFace API token is correct and has the necessary permissions.\n"
                f"If the pipeline is gated, visit https://hf.co/pyannote/speaker-diarization to accept the user conditions."
            )
            # Do not deselect the checkbox; allow the user to correct the token
            self.hf_token_entry.config(state=tk.NORMAL)

    def check_queues(self):
        # Update progress bar and status label
        try:
            while True:
                progress, status_message = self.progress_queue.get_nowait()
                # Update progress bar if needed
                self.progress_bar.set_progress(progress)
                if status_message:
                    self.update_status(status_message, "blue")
        except queue.Empty:
            pass

        # Update console output
        try:
            while True:
                message_data = self.console_queue.get_nowait()
                self.console_output_box.config(state=tk.NORMAL)
                if message_data['type'] == 'overwrite':
                    # Delete the last line
                    self.console_output_box.delete("end-2l", "end-1l")
                    self.console_output_box.insert(tk.END, message_data['content'])
                else:
                    self.console_output_box.insert(tk.END, message_data['content'])
                self.console_output_box.see(tk.END)
                self.console_output_box.config(state=tk.DISABLED)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queues)

    def on_model_change(self, event):
        selected_model = self.model_var.get()
        if self.model_loaded and selected_model != self.previous_model:
            response = messagebox.askyesno("Change Model", "Changing the model will unload the current model and load the new one. Continue?")
            if response:
                # Stop any ongoing transcription
                self.transcription_stop_event.set()  # Corrected from self.stop_event.set()
                # Stop any ongoing model loading
                self.model_stop_event.set()
                self.model = None
                self.model_loaded = False
                self.clear_transcription_box()
                self.clear_console_output()
                self.progress_bar.set_progress(0)
                self.update_status("Loading selected Whisper model...", "blue")
                self.disable_buttons()
                # Reset stop events
                self.transcription_stop_event.clear()
                self.model_stop_event.clear()
                self.model_loading_thread = threading.Thread(target=self.load_model, daemon=True)
                self.model_loading_thread.start()
            else:
                self.model_var.set(self.previous_model)
        elif not self.model_loaded:
            self.update_status("Loading selected Whisper model...", "blue")
            self.disable_buttons()
            # Reset model stop event before loading a new model
            self.model_stop_event.clear()
            self.model_loading_thread = threading.Thread(target=self.load_model, daemon=True)
            self.model_loading_thread.start()

    def disable_buttons(self):
        self.select_file_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.DISABLED)
        self.stop_media_button.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.select_file_button.config(state=tk.NORMAL)
        if self.file_path and self.model_loaded:
            self.start_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_media_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def load_model(self):
        selected_model = self.model_var.get()
        try:
            self.progress_queue.put((0, f"Loading model '{selected_model}'..."))

            # Redirect stdout and stderr
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = ConsoleRedirector(self.console_queue)
            sys.stderr = ConsoleRedirector(self.console_queue)

            # Lower process priority
            process = psutil.Process(os.getpid())
            original_nice = process.nice()
            if sys.platform.startswith('win'):
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(10)  # Increase nice value to lower priority

            # Load the model
            load_thread = threading.Thread(target=self._load_model_internal, args=(selected_model,), daemon=True)
            load_thread.start()

            # Monitor model loading for stop event
            while load_thread.is_alive():
                if self.model_stop_event.is_set():
                    # If stop event is set, terminate the loading thread if possible
                    # Note: Python threads cannot be killed, so we rely on cooperative checking
                    self.update_status("Model loading stopped by user.", "red")
                    return
                time.sleep(0.1)  # Check periodically

            load_thread.join()

            # Restore process priority
            process.nice(original_nice)

            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr

            if self.model_stop_event.is_set():
                self.progress_queue.put((0, "Model loading stopped."))
                self.model = None
                self.model_loaded = False
                self.model_stop_event.clear()
            else:
                self.model_loaded = True
                self.previous_model = selected_model
                self.progress_queue.put((100, f"Model '{selected_model}' loaded."))
                self.root.after(0, self.enable_buttons)

        except Exception as e:
            # Restore stdout and stderr in case of error
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.progress_queue.put((0, f"Error loading model '{selected_model}': {str(e)}"))
            messagebox.showerror("Model Loading Error", f"Failed to load model '{selected_model}'.\nError: {str(e)}")
            if hasattr(self, 'previous_model'):
                self.model_var.set(self.previous_model)
            else:
                self.model_var.set("base")
            self.previous_model = self.model_var.get()
            self.root.after(0, self.enable_buttons)

    def _load_model_internal(self, selected_model):
        """Internal method to load the Whisper model."""
        self.model = whisper.load_model(selected_model)


    def update_status(self, message, color):
        def update():
            self.status_label.config(text=message, fg=color)
        self.root.after(0, update)

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio/Video File",
            filetypes=[
                ("Audio/Video Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.wma *.mp4 *.mov *.avi *.mkv"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            # Truncate filename if it's too long
            if len(filename) > 50:
                filename = filename[:47] + '...'
            self.update_status(f"Selected file: {filename}", "blue")
            if self.model_loaded:
                self.start_button.config(state=tk.NORMAL)
            self.prepare_media_playback()
        else:
            self.file_path = None
            self.start_button.config(state=tk.DISABLED)

    def prepare_media_playback(self):
        try:
            media = self.vlc_instance.media_new(self.file_path)
            self.player.set_media(media)
            if sys.platform.startswith('linux'):
                self.player.set_xwindow(self.video_frame.winfo_id())
            elif sys.platform == "win32":
                self.player.set_hwnd(self.video_frame.winfo_id())
            elif sys.platform == "darwin":
                from ctypes import c_void_p
                self.player.set_nsobject(c_void_p(self.video_frame.winfo_id()))
            self.play_button.config(state=tk.NORMAL)
            self.pause_button.config(state=tk.NORMAL)
            self.stop_media_button.config(state=tk.NORMAL)
            self.slider.set(0)
            self.time_label.config(text="00:00:00 / 00:00:00")
        except Exception as e:
            messagebox.showerror("Media Playback Error", f"Failed to load media file for playback.\nError: {str(e)}")
            self.play_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.DISABLED)
            self.stop_media_button.config(state=tk.DISABLED)

    def play_media(self):
        try:
            self.player.play()
            self.update_slider()  # Start updating the slider
        except Exception as e:
            messagebox.showerror("Media Playback Error", f"Failed to play media.\nError: {str(e)}")

    def pause_media(self):
        try:
            self.player.pause()
        except Exception as e:
            messagebox.showerror("Media Playback Error", f"Failed to pause media.\nError: {str(e)}")

    def stop_media(self):
        try:
            self.player.stop()
            self.slider.set(0)
            self.time_label.config(text="00:00:00 / 00:00:00")
        except Exception as e:
            messagebox.showerror("Media Playback Error", f"Failed to stop media.\nError: {str(e)}")

    def on_slider_press(self, event):
        self.slider_dragging = True

    def on_slider_release(self, event):
        self.slider_dragging = False
        value = self.slider.get()
        self.set_media_position(value)

    def set_media_position(self, value):
        if self.player is not None and self.player.get_length() > 0:
            position = value / 100
            self.player.set_position(position)

    def update_slider(self):
        if self.player is not None:
            if not self.slider_dragging:
                length = self.player.get_length()
                if length > 0:
                    current_time = self.player.get_time()
                    slider_position = self.player.get_position() * 100
                    self.slider.set(slider_position)
                    # Update time label
                    current_time_str = self.format_time(current_time // 1000)
                    total_time_str = self.format_time(length // 1000)
                    self.time_label.config(text=f"{current_time_str} / {total_time_str}")
            else:
                # Update time label while dragging
                length = self.player.get_length()
                position = self.slider.get() / 100
                new_time = position * length
                current_time_str = self.format_time(new_time // 1000)
                total_time_str = self.format_time(length // 1000)
                self.time_label.config(text=f"{current_time_str} / {total_time_str}")
        # Schedule the next update
        self.root.after(200, self.update_slider)  # Reduced interval for smoother updates

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    def start_transcription(self):
        if not self.file_path:
            messagebox.showwarning("No File Selected", "Please select an audio/video file to transcribe.")
            return
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please load a Whisper model before starting transcription.")
            return
        if self.speaker_id_var.get() and not self.diarization_pipeline:
            messagebox.showwarning("Diarization Not Available", "Speaker diarization is enabled but the pipeline is not loaded.")
            return
        if self.transcription_thread and self.transcription_thread.is_alive():
            messagebox.showwarning("Transcription in Progress", "A transcription is already in progress.")
            return

        self.disable_buttons()
        self.stop_button.config(state=tk.NORMAL)
        self.progress_bar.set_progress(0)
        self.clear_transcription_box()
        self.clear_console_output()
        self.update_status("Preparing for transcription...", "orange")
        self.transcription_stop_event.clear()  # Ensure the stop event is cleared before starting
        self.transcription_thread = threading.Thread(target=self.transcribe_file, args=(self.file_path,), daemon=True)
        self.transcription_thread.start()

    def stop_processing(self):
        self.transcription_stop_event.set()
        self.update_status("Stopping transcription...", "red")
        self.stop_button.config(state=tk.DISABLED)


    def clear_transcription_box(self):
        self.transcription_box.config(state=tk.NORMAL)
        self.transcription_box.delete(1.0, tk.END)
        self.transcription_box.config(state=tk.DISABLED)

    def clear_console_output(self):
        self.console_output_box.config(state=tk.NORMAL)
        self.console_output_box.delete(1.0, tk.END)
        self.console_output_box.config(state=tk.DISABLED)

    def transcribe_file(self, file_path: str):
        # Global variable defining the maximum chunk duration.
        global MAX_CHUNK_DURATION
        
        # Redirect stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = ConsoleRedirector(self.console_queue)
        sys.stderr = ConsoleRedirector(self.console_queue)

        try:
            task = self.task_var.get()
            language = self.language_var.get().strip().lower()
            language = language if language != "auto" else None
            beam_size = self.beam_size_var.get()
            start_time = self.start_time_var.get().strip()
            end_time = self.end_time_var.get().strip()

            # Parse start and end times
            start_sec = self.parse_time(start_time)
            if start_sec < 0:
                raise ValueError("Start time cannot be negative.")

            audio = AudioSegment.from_file(self.file_path)
            audio_length = len(audio) / 1000  # Duration in seconds

            if end_time != "":
                end_sec = self.parse_time(end_time)
                if end_sec <= start_sec:
                    raise ValueError("End time must be greater than start time.")
                if end_sec > audio_length:
                    end_sec = audio_length
            else:
                end_sec = audio_length

            if start_sec >= audio_length:
                raise ValueError("Start time exceeds audio duration.")

            # Trim the audio
            trimmed_audio = audio[start_sec * 1000:end_sec * 1000]
            self.trimmed_audio = trimmed_audio  # Make it accessible to transcribe_segment
            trimmed_duration = len(trimmed_audio) / 1000  # Duration in seconds

            if trimmed_duration <= 0:
                raise ValueError("Trimmed audio has zero duration.")

            # Export trimmed audio to a temporary WAV file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                trimmed_audio.export(tmp_file.name, format="wav")
                temp_file_path = tmp_file.name

            # Perform speaker diarization if enabled
            if self.speaker_id_var.get() and self.diarization_pipeline:
                self.progress_queue.put((10, "Performing speaker diarization..."))
                diarization = self.diarization_pipeline(temp_file_path)

                # **Check if transcription stop event is set after diarization**
                if self.transcription_stop_event.is_set():
                    raise Exception("Transcription stopped by user.")

                # Parse diarization results
                speaker_segments = []
                speaker_map = {}
                speaker_count = 1
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speaker_map:
                        speaker_map[speaker] = f"Speaker {speaker_count}"
                        speaker_count += 1
                    speaker_segments.append({
                        "speaker": speaker_map[speaker],
                        "start": turn.start,
                        "end": turn.end
                    })

                if not speaker_segments:
                    raise ValueError("No speaker segments found during diarization.")

                # Combine smaller adjacent segments to form larger chunks
                combined_segments = []
                current_chunk = {
                    "speaker": speaker_segments[0]["speaker"],
                    "start": speaker_segments[0]["start"],
                    "end": speaker_segments[0]["end"]
                }

                for segment in speaker_segments[1:]:
                    potential_end = segment["end"]
                    potential_duration = potential_end - current_chunk["start"]
                    if potential_duration <= MAX_CHUNK_DURATION:
                        current_chunk["end"] = potential_end
                    else:
                        combined_segments.append(current_chunk)
                        current_chunk = {
                            "speaker": segment["speaker"],
                            "start": segment["start"],
                            "end": segment["end"]
                        }
                combined_segments.append(current_chunk)

                speaker_segments = combined_segments  # Update speaker_segments with combined chunks

                # **Debugging: Log combined segments**
                self.console_queue.put({'type': 'append', 'content': f"Number of combined diarization segments: {len(speaker_segments)}\n"})
            else:
                # If diarization is not enabled, divide the audio into chunks of MAX_CHUNK_DURATION
                self.progress_queue.put((10, "Dividing audio into chunks..."))
                speaker_segments = []
                num_full_chunks = int(trimmed_duration // MAX_CHUNK_DURATION)
                remainder = trimmed_duration % MAX_CHUNK_DURATION

                for i in range(num_full_chunks):
                    chunk_start = i * MAX_CHUNK_DURATION
                    chunk_end = (i + 1) * MAX_CHUNK_DURATION
                    speaker_segments.append({
                        "speaker": "Speaker 1",
                        "start": chunk_start,
                        "end": chunk_end
                    })

                if remainder > 0:
                    chunk_start = num_full_chunks * MAX_CHUNK_DURATION
                    chunk_end = trimmed_duration
                    speaker_segments.append({
                        "speaker": "Speaker 1",
                        "start": chunk_start,
                        "end": chunk_end
                    })

                # **Debugging: Log number of segments when diarization is off**
                self.console_queue.put({'type': 'append', 'content': f"Number of transcription segments: {len(speaker_segments)}\n"})

            # **Check if transcription stop event is set before proceeding**
            if self.transcription_stop_event.is_set():
                raise Exception("Transcription stopped by user.")

            # Initialize transcription
            transcription = ""
            all_segments = []
            transcription_lock = threading.Lock()

            # Transcribe each speaker segment using ThreadPoolExecutor
            num_workers = self.num_threads  # Number of threads based on CPU cores
            self.progress_queue.put((20, f"Transcribing {len(speaker_segments)} segments with {num_workers} threads..."))

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_segment = {
                    executor.submit(
                        self.transcribe_segment,
                        segment,
                        task,
                        language,
                        beam_size,
                        start_sec
                    ): segment for segment in speaker_segments
                }

                for future in as_completed(future_to_segment):
                    if self.transcription_stop_event.is_set():
                        raise Exception("Transcription stopped by user.")

                    segment = future_to_segment[future]
                    try:
                        segment_text = future.result()
                        if segment_text:
                            with transcription_lock:
                                transcription += f"{segment['speaker']}: {segment_text}\n"
                                all_segments.append({
                                    "speaker": segment['speaker'],
                                    "start": segment['start'] + start_sec,
                                    "end": segment['end'] + start_sec,
                                    "text": segment_text
                                })
                    except Exception as e:
                        self.console_queue.put({'type': 'append', 'content': f"Error transcribing segment: {str(e)}\n"})

                    # Update progress
                    completed = len(all_segments)
                    progress_percentage = int(20 + 80 * completed / len(speaker_segments))
                    self.progress_queue.put((progress_percentage, f"Transcribed {completed} of {len(speaker_segments)} segments..."))

            if self.transcription_stop_event.is_set():
                self.progress_queue.put((0, "Transcription stopped."))
                self.transcription_stop_event.clear()
            else:
                self.progress_queue.put((100, "Transcription completed."))

                if self.srt_var.get():
                    # Generate and display SRT with speaker labels
                    srt_text = self.generate_srt_text_with_speakers(all_segments)
                    self.root.after(0, self.display_transcription, srt_text)
                    # Save SRT file
                    self.root.after(0, self.save_srt_file, all_segments, self.file_path)
                else:
                    # Display transcription with speaker labels
                    self.root.after(0, self.display_transcription, transcription.strip())

        except Exception as e:
            if str(e) == "Transcription stopped by user.":
                self.progress_queue.put((0, "Transcription stopped."))
            else:
                self.progress_queue.put((0, f"Error during transcription: {str(e)}"))
                messagebox.showerror("Transcription Error", f"Failed to transcribe the audio/video file.\nError: {str(e)}")
        finally:
            # Restore stdout and stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            # Clean up temporary file
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            # Reset trimmed_audio
            self.trimmed_audio = None
            self.root.after(0, self.enable_buttons)


    def generate_srt_text_with_speakers(self, segments):
        srt_entries = []
        for i, segment in enumerate(segments, start=1):
            start_time = self.format_timestamp(segment['start'])
            end_time = self.format_timestamp(segment['end'])
            speaker = segment['speaker']
            text = segment['text'].strip()
            srt_entry = f"{i}\n{start_time} --> {end_time}\n{speaker}: {text}\n"
            srt_entries.append(srt_entry)
        srt_text = "\n".join(srt_entries)
        return srt_text

    def save_srt_file(self, segments, original_file_path):
        filetypes = [('SubRip Subtitle', '*.srt')]
        initial_filename = os.path.splitext(os.path.basename(original_file_path))[0] + '.srt'
        save_path = filedialog.asksaveasfilename(
            title="Save SRT Subtitle File",
            defaultextension=".srt",
            initialfile=initial_filename,
            filetypes=filetypes
        )
        if save_path:
            threading.Thread(target=self.write_srt_file, args=(save_path, segments), daemon=True).start()

    def write_srt_file(self, save_path, segments):
        try:
            with open(save_path, 'w', encoding='utf-8') as srt_file:
                for i, segment in enumerate(segments, start=1):
                    start_time = self.format_timestamp(segment['start'])
                    end_time = self.format_timestamp(segment['end'])
                    speaker = segment['speaker']
                    text = segment['text'].strip()
                    srt_file.write(f"{i}\n{start_time} --> {end_time}\n{speaker}: {text}\n\n")
            self.update_status(f"SRT file saved to {save_path}", "green")
        except Exception as e:
            self.update_status(f"Error saving SRT file: {str(e)}", "red")
            messagebox.showerror("SRT Saving Error", f"Failed to save SRT file.\nError: {str(e)}")

    def format_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
        
    def transcribe_segment(self, segment, task, language, beam_size, start_sec):
        """
        Transcribe a single speaker segment.
        """
        if self.transcription_stop_event.is_set():
            raise Exception("Transcription stopped by user.")

        segment_start = segment["start"]
        segment_end = segment["end"]
        speaker = segment["speaker"]

        # Extract the audio segment from the trimmed audio
        try:
            segment_audio = self.trimmed_audio[segment_start * 1000:segment_end * 1000]
        except Exception as e:
            self.console_queue.put({'type': 'append', 'content': f"Error extracting audio segment: {str(e)}\n"})
            return ""

        # Export the segment to a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_file:
            try:
                segment_audio.export(seg_file.name, format="wav")
                seg_file_path = seg_file.name
                self.console_queue.put({'type': 'append', 'content': f"Exported segment {segment_start}-{segment_end} to {seg_file_path}\n"})
            except Exception as e:
                self.console_queue.put({'type': 'append', 'content': f"Error exporting audio segment: {str(e)}\n"})
                return ""

        # Transcribe the audio segment
        try:
            with self.model_lock:
                result = self.model.transcribe(seg_file_path, task=task, language=language, beam_size=beam_size)
            segment_text = result.get("text", "").strip()
            self.console_queue.put({'type': 'append', 'content': f"Transcribed segment {segment_start}-{segment_end}: {segment_text}\n"})
        except Exception as e:
            self.console_queue.put({'type': 'append', 'content': f"Error transcribing segment: {str(e)}\n"})
            segment_text = ""
        finally:
            # Clean up temporary segment file
            if os.path.exists(seg_file_path):
                os.remove(seg_file_path)
                self.console_queue.put({'type': 'append', 'content': f"Deleted temporary file {seg_file_path}\n"})

        return segment_text

    def parse_time(self, time_str):
        pattern = r'^(\d{1,2}):([0-5]?\d):([0-5]?\d)$'
        match = re.match(pattern, time_str)
        if not match:
            raise ValueError("Time must be in [hh:mm:ss] format.")
        hours, minutes, seconds = map(int, match.groups())
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    def display_transcription(self, text):
        def display():
            self.transcription_box.config(state=tk.NORMAL)
            self.transcription_box.delete(1.0, tk.END)
            self.transcription_box.insert(tk.END, text)
            self.transcription_box.config(state=tk.DISABLED)
        self.root.after(0, display)

    def on_closing(self):
        self.transcription_stop_event.set()  # Corrected from self.stop_event.set()
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join()
        self.player.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SoftWhisper(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
