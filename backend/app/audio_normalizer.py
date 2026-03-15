import numpy as np
import pyloudnorm as pyln
import librosa

def normalize_loudness(audio, sr, target_lufs=-14.0):
    """
    Normalize audio to target LUFS
    
    Args:
        audio: numpy array of audio samples
        sr: sample rate
        target_lufs: target loudness in LUFS (default -14.0)
    
    Returns:
        normalized audio
    """
    # Create loudness meter
    meter = pyln.Meter(sr)
    
    # Measure current loudness
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError:
        # If audio too short, return as-is
        return audio
    
    # Normalize to target
    normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
    
    # Prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    
    print(f"   LUFS: {loudness:.1f} → {target_lufs:.1f}")
    
    return normalized

def normalize_tracks_before_mixing(tracks_audio, sr):
    """
    Normalize all tracks to same LUFS before mixing
    
    Args:
        tracks_audio: list of audio numpy arrays
        sr: sample rate
    
    Returns:
        list of normalized audio
    """
    normalized_tracks = []
    
    for i, audio in enumerate(tracks_audio):
        print(f"   Normalizing track {i+1}/{len(tracks_audio)}...")
        normalized = normalize_loudness(audio, sr, target_lufs=-14.0)
        normalized_tracks.append(normalized)
    
    return normalized_tracks
