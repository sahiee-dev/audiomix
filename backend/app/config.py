"""
config.py — Application configuration.

All settings are read from environment variables with documented defaults so
the app can be configured without touching source code.
"""

import os

# Root of the backend package directory (two levels up from this file)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Directory where uploaded audio files and session sub-directories are stored.
# Override via the UPLOAD_DIR environment variable.
UPLOAD_DIR: str = os.environ.get("UPLOAD_DIR", "/tmp/smart_mix_uploads")

# Maximum allowed upload size per file, in megabytes.
# Override via MAX_UPLOAD_MB environment variable.
MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "200"))

# Supported audio MIME types for the upload endpoint.
ALLOWED_AUDIO_TYPES: tuple[str, ...] = (
    "audio/mpeg",       # .mp3
    "audio/wav",        # .wav
    "audio/x-wav",
    "audio/flac",       # .flac
    "audio/aac",        # .aac
    "audio/ogg",        # .ogg
    "audio/mp4",        # .m4a
    "audio/x-m4a",
)

# Ensure the upload directory exists at import time so the app never crashes on
# first use if the path does not yet exist.
os.makedirs(UPLOAD_DIR, exist_ok=True)
