"""
audio_utils.py — Low-level audio feature extraction utilities.

All public functions operate on NumPy arrays or file paths.
Temporary WAV files are always cleaned up in a finally block so no temp
file is left on disk even when an exception is raised mid-analysis.
"""

import os
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment
from typing import Dict


# ─── Format conversion ────────────────────────────────────────────────────────

def to_wav_mono(input_path: str, output_path: str, sr: int = 44100) -> str:
    """Convert any audio file to a mono 44.1 kHz WAV at *output_path*."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path


def load_audio_for_librosa(path: str, sr: int = 44100) -> np.ndarray:
    """Load *path* as a mono NumPy array resampled to *sr* Hz."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


# ─── Individual feature estimators ───────────────────────────────────────────

def estimate_bpm(y: np.ndarray, sr: int = 44100) -> float:
    """Return the estimated tempo in BPM."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, units="time")
    return float(tempo)


def estimate_energy(y: np.ndarray) -> float:
    """Return the mean RMS energy of the signal."""
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms))


def estimate_spectral_centroid(y: np.ndarray, sr: int = 44100) -> float:
    """Return the mean spectral centroid in Hz."""
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(sc))


def estimate_key(y: np.ndarray, sr: int = 44100) -> str:
    """Return the dominant pitch class (e.g. 'C', 'F#')."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    pitch_idx = int(chroma_mean.argmax())
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return pitch_names[pitch_idx]


def estimate_mood(y: np.ndarray, sr: int = 44100) -> str:
    """Estimate valence from major/minor chroma weighting."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    major_weight = chroma_mean[[0, 4, 7]].sum()   # C, E, G
    minor_weight = chroma_mean[[0, 3, 7]].sum()   # C, Eb, G
    return "upbeat" if major_weight > minor_weight else "melancholic"


def estimate_danceability(y: np.ndarray, sr: int = 44100) -> float:
    """Estimate danceability (0–1) from onset strength."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm_strength = float(np.mean(onset_env))
    return min(rhythm_strength / 10.0, 1.0)


def estimate_loudness(y: np.ndarray) -> float:
    """Return mean loudness in dB (relative to peak)."""
    rms = librosa.feature.rms(y=y)
    db = librosa.amplitude_to_db(rms, ref=np.max)
    return float(np.mean(db))


# ─── Full feature extraction ──────────────────────────────────────────────────

def extract_features(file_path: str) -> Dict:
    """
    Extract all audio features from *file_path*.

    Converts to a temporary WAV for librosa compatibility.  The temp file is
    always removed in a ``finally`` block — no files are leaked on error.
    """
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        to_wav_mono(file_path, tmp)
        y, sr = librosa.load(tmp, sr=44100, mono=True)
        return {
            "filename": os.path.basename(file_path),
            "bpm": estimate_bpm(y, sr),
            "energy": estimate_energy(y),
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "spectral_centroid": estimate_spectral_centroid(y, sr),
            "key": estimate_key(y, sr),
            "mood": estimate_mood(y, sr),
            "danceability": estimate_danceability(y, sr),
            "loudness": estimate_loudness(y),
        }
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
