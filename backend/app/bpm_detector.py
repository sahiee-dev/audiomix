import numpy as np
import librosa
import madmom


def detect_bpm_hybrid(audio_path_or_array, sr=44100):
    """
    Hybrid BPM detection using Librosa + Madmom.

    Args:
        audio_path_or_array: Either a file path (str) or a pre-loaded mono
                             NumPy audio array. Passing a pre-loaded array
                             avoids a redundant disk read.
        sr: Sample rate, used only when audio_path_or_array is an array.

    Returns: (bpm, confidence)
    """
    if isinstance(audio_path_or_array, np.ndarray):
        y = audio_path_or_array
        audio_path = None  # No file path available; madmom will use the array
    else:
        audio_path = audio_path_or_array
        y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Method 1: Librosa
    tempo_librosa, beats_librosa = librosa.beat.beat_track(y=y, sr=sr)
    tempo_librosa = float(tempo_librosa) if not isinstance(tempo_librosa, np.ndarray) else float(tempo_librosa[0])

    # Method 2: Madmom
    try:
        proc = madmom.features.beats.RNNBeatProcessor()
        # Madmom can accept a raw audio signal (array + sr) or a file path.
        if audio_path is not None:
            act = proc(audio_path)
        else:
            act = proc(y.astype(np.float32), sample_rate=sr)
        beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(act)

        if len(beat_times) > 1:
            tempo_madmom = 60 / np.median(np.diff(beat_times))
        else:
            tempo_madmom = tempo_librosa

        tempo_madmom = float(tempo_madmom)
    except Exception as e:
        print(f"⚠️ Madmom failed: {e}, using Librosa only")
        tempo_madmom = tempo_librosa

    # Hybrid decision
    if abs(tempo_librosa - tempo_madmom) < 3:
        bpm = (tempo_librosa + tempo_madmom) / 2
        confidence = 0.95
    elif abs(tempo_librosa - tempo_madmom) < 10:
        bpm = tempo_madmom
        confidence = 0.85
    else:
        bpm = tempo_madmom  # Madmom generally better
        confidence = 0.70

    # Half-time / double-time correction
    if bpm > 180:
        bpm = bpm / 2
        confidence *= 0.9
    elif bpm < 70:
        bpm = bpm * 2
        confidence *= 0.9

    return round(bpm, 1), round(confidence, 2)
