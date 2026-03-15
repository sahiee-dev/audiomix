import numpy as np
import librosa
import madmom

def detect_beats_advanced(y, sr):
    """
    Advanced beat detection with double/half-time correction
    """
    # Use madmom for initial detection
    processor = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    activations = madmom.features.beats.RNNBeatProcessor()(y)
    beats_seconds = processor(activations)
    beats = librosa.time_to_samples(beats_seconds, sr=sr)
    
    # Detect downbeats
    downbeat_processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
        beats_per_bar=[4], fps=100
    )
    downbeat_activations = madmom.features.downbeats.RNNDownBeatProcessor()(y)
    downbeats_seconds = downbeat_processor(downbeat_activations)
    downbeats = librosa.time_to_samples(downbeats_seconds[:, 0], sr=sr)
    
    # Calculate BPM from beat intervals
    if len(beats) > 1:
        beat_intervals = np.diff(beats) / sr
        median_interval = np.median(beat_intervals)
        bpm = 60.0 / median_interval
        
        # FIX: Detect and correct double/half-time
        # For hip-hop, prefer 60-100 BPM range
        if bpm > 110:
            # Likely double-time, halve it
            bpm = bpm / 2
            print(f"   Corrected double-time: {bpm*2:.1f} → {bpm:.1f} BPM")
        elif bpm < 50:
            # Likely half-time, double it
            bpm = bpm * 2
            print(f"   Corrected half-time: {bpm/2:.1f} → {bpm:.1f} BPM")
    else:
        bpm = 120.0  # fallback
    
    return beats, downbeats, bpm

def compute_beat_phase_offset(beats_a, beats_b, transition_point_a):
    """
    Calculate phase offset to align B's first beat with A's beat grid
    """
    if len(beats_a) == 0 or len(beats_b) == 0:
        return 0
    
    # Find beats near transition point
    a_beats_near = beats_a[beats_a < transition_point_a]
    if len(a_beats_near) == 0:
        return 0
    
    last_beat_a = a_beats_near[-1]
    
    # Calculate beat period (average time between beats)
    if len(a_beats_near) > 3:
        beat_period_a = np.mean(np.diff(a_beats_near[-8:]))  # Use last 8 beats
    else:
        beat_period_a = 0.5 * 44100  # Default fallback
    
    first_beat_b = beats_b[0] if len(beats_b) > 0 else 0
    
    # Calculate phase alignment
    phase_a = (transition_point_a - last_beat_a) % beat_period_a
    offset = phase_a - first_beat_b
    
    return int(offset)

def align_tracks_by_beats(y_a, y_b, sr, bpm_a, bpm_b, beats_a, beats_b, 
                         downbeats_a, downbeats_b, transition_duration=8.0):
    """
    Align two tracks by tempo and beat phase for perfect DJ-style transition
    
    Args:
        y_a, y_b: Audio signals
        sr: Sample rate
        bpm_a, bpm_b: BPMs of both tracks
        beats_a, beats_b: Beat positions in samples
        downbeats_a, downbeats_b: Downbeat positions
        transition_duration: Crossfade duration in seconds
    
    Returns:
        y_b_aligned: Track B time-stretched and phase-aligned
        fade_start_a: Sample position in A where crossfade starts
        entry_point_b: Sample position in B where it enters
    """
    
    # 1. Time-stretch B to match A's tempo
    if abs(bpm_b - bpm_a) > 2:  # Only stretch if BPM differs significantly
        stretch_factor = bpm_b / bpm_a
        print(f"Time-stretching track B: {bpm_b:.1f} BPM → {bpm_a:.1f} BPM (factor: {stretch_factor:.3f})")
        y_b_stretched = librosa.effects.time_stretch(y_b, rate=stretch_factor)
        
        # Recalculate beats for stretched version
        beats_b_stretched = (beats_b * stretch_factor).astype(int)
        downbeats_b_stretched = (downbeats_b * stretch_factor).astype(int)
    else:
        y_b_stretched = y_b
        beats_b_stretched = beats_b
        downbeats_b_stretched = downbeats_b
    
    # 2. Find transition point in A (last N seconds)
    fade_samples = int(transition_duration * sr)
    fade_start_a = max(0, len(y_a) - fade_samples)
    
    # Try to align on downbeat (bar boundary) for smoother transition
    if len(downbeats_a) > 0:
        # Find closest downbeat before fade start
        valid_downbeats = downbeats_a[downbeats_a <= fade_start_a + fade_samples]
        if len(valid_downbeats) > 0:
            fade_start_a = valid_downbeats[-1]
            print(f"Aligning transition on downbeat at {fade_start_a/sr:.2f}s")
    
    # 3. Calculate beat phase alignment
    offset = compute_beat_phase_offset(beats_a, beats_b_stretched, fade_start_a)
    print(f"Phase alignment offset: {offset/sr*1000:.1f}ms")
    
    # 4. Apply phase offset to B
    if offset > 0:
        y_b_aligned = np.pad(y_b_stretched, (offset, 0), mode='constant')
    elif offset < 0:
        y_b_aligned = y_b_stretched[-offset:]
    else:
        y_b_aligned = y_b_stretched
    
    return y_b_aligned, fade_start_a, 0

def find_compatible_key_shift(key_a, key_b):
    """
    Calculate semitone shift for harmonic compatibility (Camelot wheel)
    """
    key_map = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    if key_a not in key_map or key_b not in key_map:
        return 0
    
    diff = (key_map[key_b] - key_map[key_a]) % 12
    
    # Compatible intervals (Camelot wheel)
    compatible = [0, 2, 5, 7]  # Same, whole step, fourth, fifth
    
    if diff in compatible:
        return 0  # Already compatible
    elif diff == 1:
        return -1  # Shift down 1 semitone
    elif diff == 11:
        return 1  # Shift up 1 semitone
    elif diff in [3, 4]:
        return -diff  # Shift down to nearest compatible
    elif diff in [8, 9, 10]:
        return 12 - diff  # Shift up to nearest compatible
    else:
        return 0  # Default: no shift
