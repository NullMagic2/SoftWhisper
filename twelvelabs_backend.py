"""
Optional TwelveLabs Pegasus transcription backend for SoftWhisper.

This is an opt-in alternative to the default local Whisper.cpp engine. When the
user selects the "TwelveLabs (Pegasus)" engine, transcription is performed by
the TwelveLabs Pegasus video-understanding model in the cloud instead of by the
local whisper-cli executable.

It returns the exact same result dictionary shape as transcribe_audio() in
SoftWhisper.py, so the rest of the application (SRT export, diarization,
display) works unchanged:

    {
        'raw':          <whisper.cpp-style bracketed text>,
        'text':         <plain transcript>,
        'segments':     [{'text', 'start', 'end'} ...],
        'audio_length': <seconds>,
        'stderr':       <diagnostic string>,
        'cancelled':    <bool>,
    }

Requires the `twelvelabs` package (see requirements.txt) and an API key, read
from the TWELVELABS_API_KEY environment variable or passed via options. Grab a
free key at https://twelvelabs.io.
"""

import os
import time

# Default Pegasus model. 'pegasus1.5' is the current general-analysis model.
DEFAULT_PEGASUS_MODEL = "pegasus1.5"

# Prompt that asks Pegasus to return a verbatim, timestamped transcript.
_TRANSCRIBE_PROMPT = (
    "Transcribe all spoken words in this video verbatim. Output one line per "
    "spoken segment in exactly this format, with no extra commentary:\n"
    "[HH:MM:SS.mmm --> HH:MM:SS.mmm] text\n"
    "Use punctuation. Do not censor any words. Do not describe visuals."
)

_TRANSLATE_PROMPT = (
    "Transcribe all spoken words in this video and translate them into English. "
    "Output one line per spoken segment in exactly this format, with no extra "
    "commentary:\n"
    "[HH:MM:SS.mmm --> HH:MM:SS.mmm] text\n"
    "Use punctuation. Do not censor any words. Do not describe visuals."
)

# How long to wait for a freshly uploaded asset to finish server-side
# processing before giving up (seconds).
_ASSET_READY_TIMEOUT = 600


def _resolve_api_key(options):
    """Return the API key from options or the environment, or raise."""
    key = (options or {}).get("twelvelabs_api_key") or os.environ.get("TWELVELABS_API_KEY")
    if not key:
        raise RuntimeError(
            "TwelveLabs API key not found. Set the TWELVELABS_API_KEY environment "
            "variable or enter a key in the settings. Get a free key at "
            "https://twelvelabs.io."
        )
    return key


def _parse_ts(text):
    """Parse 'HH:MM:SS.mmm' (or MM:SS / SS) into seconds, or None."""
    text = text.strip()
    if not text:
        return None
    try:
        parts = text.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(parts[0])
    except (ValueError, TypeError):
        return None


def _segments_from_pegasus_text(raw):
    """Turn Pegasus' bracketed output into a list of segment dicts.

    Mirrors the {'text', 'start', 'end'} shape SoftWhisper already builds from
    whisper.cpp JSON. Lines Pegasus emits without a recognizable timestamp are
    kept as text-only segments so nothing is silently dropped.
    """
    import re

    segments = []
    line_re = re.compile(
        r"^\[(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*-->\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*(.*)$"
    )
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = line_re.match(line)
        if m:
            start, end, text = m.groups()
            segments.append(
                {"text": text.strip(), "start": _parse_ts(start), "end": _parse_ts(end)}
            )
        else:
            segments.append({"text": line, "start": None, "end": None})
    return segments


def _normalize_raw(data):
    """Ensure the model output is bracketed line-per-segment text.

    If Pegasus already returned the requested '[start --> end] text' format we
    pass it through; otherwise we emit a single bracket-less line so downstream
    plain-text display still works.
    """
    text = (data or "").strip()
    if not text:
        return ""
    if "-->" in text:
        return text
    # Model returned a paragraph; keep it as one untimestamped line.
    return text


def transcribe_with_pegasus(
    file_path,
    options,
    progress_callback=None,
    status_callback=None,
    stop_event=None,
):
    """Transcribe a media file with TwelveLabs Pegasus.

    Drop-in replacement for SoftWhisper.transcribe_audio() with the same
    signature and return shape. Network/cloud based and opt-in.
    """
    from twelvelabs import TwelveLabs
    from twelvelabs.types.video_context import VideoContext_AssetId

    file_path = os.path.abspath(file_path)
    options = options or {}
    api_key = _resolve_api_key(options)
    model_name = options.get("pegasus_model", DEFAULT_PEGASUS_MODEL)
    task = options.get("task", "transcribe")
    max_tokens = int(options.get("max_tokens", 2048))

    def _cancelled():
        return bool(stop_event and stop_event.is_set())

    def _status(msg, color="blue"):
        if status_callback:
            status_callback(msg, color)

    client = TwelveLabs(api_key=api_key)

    # 1) Upload the local file as an asset.
    if progress_callback:
        progress_callback(5, "Uploading to TwelveLabs...")
    _status("Uploading media to TwelveLabs...", "blue")
    with open(file_path, "rb") as fh:
        asset = client.assets.create(
            method="direct", file=fh, filename=os.path.basename(file_path)
        )

    if _cancelled():
        return _result("", "", [], 0.0, "Cancelled before analysis.", True)

    # 2) Wait for server-side processing to complete.
    deadline = time.time() + _ASSET_READY_TIMEOUT
    status = getattr(asset, "status", None)
    while status == "processing":
        if _cancelled():
            return _result("", "", [], 0.0, "Cancelled while uploading.", True)
        if time.time() > deadline:
            raise RuntimeError("TwelveLabs asset processing timed out.")
        if progress_callback:
            progress_callback(15, "Processing upload...")
        time.sleep(3)
        asset = client.assets.retrieve(asset.id)
        status = getattr(asset, "status", None)
    if status == "failed":
        raise RuntimeError("TwelveLabs failed to process the uploaded file.")

    audio_length = float(getattr(asset, "duration", 0.0) or 0.0)

    # 3) Run Pegasus analysis. start/end mirror whisper.cpp trimming.
    if progress_callback:
        progress_callback(40, "Analyzing with Pegasus...")
    _status(f"Analyzing with Pegasus ({model_name})...", "blue")

    prompt = _TRANSLATE_PROMPT if task == "translate" else _TRANSCRIBE_PROMPT
    analyze_kwargs = {
        "model_name": model_name,
        "video": VideoContext_AssetId(asset_id=asset.id),
        "prompt": prompt,
        "max_tokens": max_tokens,
    }
    start_sec = _parse_ts(options.get("start_time", ""))
    end_sec = _parse_ts(options.get("end_time", ""))
    if start_sec:
        analyze_kwargs["start_time"] = start_sec
    if end_sec:
        analyze_kwargs["end_time"] = end_sec

    response = client.analyze(**analyze_kwargs)

    if progress_callback:
        progress_callback(95, "Formatting transcript...")

    if _cancelled():
        return _result("", "", [], audio_length, "Cancelled after analysis.", True)

    raw = _normalize_raw(getattr(response, "data", "") or "")
    segments = _segments_from_pegasus_text(raw)
    plain_text = " ".join(seg["text"] for seg in segments if seg.get("text")).strip()

    if progress_callback:
        progress_callback(100, "Transcribing: 100%")

    return _result(raw, plain_text, segments, audio_length, "", False)


def _result(raw, text, segments, audio_length, stderr, cancelled):
    return {
        "raw": raw,
        "text": text,
        "segments": segments,
        "audio_length": audio_length,
        "stderr": stderr,
        "cancelled": cancelled,
    }
