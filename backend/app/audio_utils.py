import os
import numpy as np
import librosa
from pydub import AudioSegment
from typing import Dict
import tempfile

def to_wav_mono(input_path: str, output_path: str, sr=44100):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(sr).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path

def load_audio_for_librosa(path: str, sr=44100):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def estimate_bpm(y, sr=44100):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    return float(tempo)

def estimate_energy(y):
    rms = librosa.feature.rms(y=y)
    return float(np.mean(rms))

def estimate_spectral_centroid(y, sr=44100):
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    return float(np.mean(sc))

def estimate_key(y, sr=44100):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    pitch_idx = chroma_mean.argmax()
    pitch_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return pitch_names[pitch_idx]

def estimate_mood(y, sr=44100):
    """Estimate mood based on valence (major/minor tonality)"""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    
    # Major triads (C, E, G) vs Minor triads (C, Eb, G)
    major_weight = chroma_mean[[0, 4, 7]].sum()  # C, E, G
    minor_weight = chroma_mean[[0, 3, 7]].sum()  # C, Eb, G
    
    if major_weight > minor_weight:
        return "upbeat"
    else:
        return "melancholic"

def estimate_danceability(y, sr=44100):
    """Estimate danceability based on rhythm strength"""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    rhythm_strength = float(np.mean(onset_env))
    return min(rhythm_strength / 10.0, 1.0)  # Normalize to 0-1

def estimate_loudness(y):
    """Estimate loudness in dB"""
    rms = librosa.feature.rms(y=y)
    db = librosa.amplitude_to_db(rms, ref=np.max)
    return float(np.mean(db))


def extract_features(file_path: str) -> Dict:
    tmp = tempfile.mktemp(suffix=".wav")
    to_wav_mono(file_path, tmp)
    y, sr = librosa.load(tmp, sr=44100, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    bpm = estimate_bpm(y, sr)
    energy = estimate_energy(y)
    sc = estimate_spectral_centroid(y, sr)
    key = estimate_key(y, sr)
    mood = estimate_mood(y, sr)
    danceability = estimate_danceability(y, sr)
    loudness = estimate_loudness(y)
    os.remove(tmp)
    return {
        "filename": os.path.basename(file_path),
        "bpm": float(bpm),
        "energy": float(energy),
        "duration": float(duration),
        "spectral_centroid": float(sc),
        "key": key,
        "mood": mood,
        "danceability": float(danceability),
        "loudness": float(loudness)
    }
