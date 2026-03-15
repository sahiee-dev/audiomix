"""
pitch_engine.py — Studio-quality pitch shifting and time stretching.

Uses the `pyrubberband` Python wrapper around the Rubber Band Library,
which implements a high-quality phase-vocoder with transient preservation.
This is the same algorithm used in professional audio workstations.

Fallback: If pyrubberband is not installed, the engine transparently falls
back to librosa's phase-vocoder (lower quality but always available).

Core operations:
  - pitch_shift()     : Change pitch without altering tempo
  - time_stretch()    : Change tempo without altering pitch
  - force_key()       : Pitch-shift a track to the nearest Camelot-compatible key
"""

import numpy as np
from typing import Tuple, Optional

# Camelot Wheel: minor keys and their semitone offsets relative to C
# Used to find the smallest pitch shift that achieves key compatibility
_KEY_SEMITONES = {
    'C':  0,  'C#': 1,  'Db': 1,
    'D':  2,  'D#': 3,  'Eb': 3,
    'E':  4,
    'F':  5,  'F#': 6,  'Gb': 6,
    'G':  7,  'G#': 8,  'Ab': 8,
    'A':  9,  'A#': 10, 'Bb': 10,
    'B':  11,
}

# Compatible intervals on the Camelot Wheel (semitones from root):
# Same key (0), whole step (2), perfect fourth (5), perfect fifth (7)
_COMPATIBLE_OFFSETS = {0, 2, 5, 7}


def _try_rubberband() -> bool:
    """Return True if pyrubberband is importable."""
    try:
        import pyrubberband  # noqa: F401
        return True
    except ImportError:
        return False


_HAS_RUBBERBAND = _try_rubberband()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pitch_shift(y: np.ndarray, sr: int, n_semitones: float) -> np.ndarray:
    """
    Shift the pitch of `y` by `n_semitones` without changing its duration.

    - Positive n_semitones  → higher pitch
    - Negative n_semitones  → lower pitch

    Uses Rubber Band Library when available (phase-correct, transient-aware).
    Falls back to librosa's phase vocoder otherwise.

    Args:
        y:           Mono audio array (float32 or float64)
        sr:          Sample rate
        n_semitones: Number of semitones to shift (can be fractional)

    Returns:
        Pitch-shifted audio array of the same length as y.
    """
    if n_semitones == 0.0:
        return y

    if _HAS_RUBBERBAND:
        import pyrubberband as pyrb
        return pyrb.pitch_shift(y, sr, n_semitones)
    else:
        import librosa
        print(f"   ⚠️  pyrubberband not found — using librosa pitch shift (lower quality)")
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones)


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """
    Change the tempo of `y` by `rate` without altering its pitch.

    rate > 1.0  → faster (shorter)
    rate < 1.0  → slower  (longer)

    Args:
        y:    Mono audio array
        sr:   Sample rate
        rate: Speed ratio (e.g. 1.05 = 5% faster)

    Returns:
        Time-stretched audio array.
    """
    if abs(rate - 1.0) < 0.001:
        return y

    if _HAS_RUBBERBAND:
        import pyrubberband as pyrb
        return pyrb.time_stretch(y, sr, rate)
    else:
        import librosa
        print(f"   ⚠️  pyrubberband not found — using librosa time stretch (lower quality)")
        return librosa.effects.time_stretch(y, rate=rate)


def force_key(y: np.ndarray, sr: int,
              source_key: str, target_key: str,
              source_mode: str = 'major', target_mode: str = 'major') -> Tuple[np.ndarray, float]:
    """
    Pitch-shift `y` so that `source_key` becomes harmonically compatible
    with `target_key` using the minimum possible semitone movement.

    The function searches the Camelot Wheel for the closest compatible key
    and applies the required shift.  Shifts are intentionally capped at
    ±4 semitones — larger shifts are audible artefacts and indicate
    fundamentally incompatible tracks that should hard-cut instead.

    Args:
        y:           Mono audio array of the track to shift
        sr:          Sample rate
        source_key:  Key name of the track to be shifted (e.g. 'A#')
        target_key:  Key name of the reference track (e.g. 'F')
        source_mode: 'major' or 'minor'
        target_mode: 'major' or 'minor'

    Returns:
        (shifted_audio, semitones_applied)
        If no shift is needed, returns (y, 0.0) without copying.
    """
    source_semi = _KEY_SEMITONES.get(source_key, -1)
    target_semi = _KEY_SEMITONES.get(target_key, -1)

    if source_semi == -1 or target_semi == -1:
        print(f"   ⚠️  Unknown key: {source_key!r} or {target_key!r} — skipping pitch force")
        return y, 0.0

    # Raw semitone difference (mod 12 for circular key arithmetic)
    raw_diff = (target_semi - source_semi) % 12

    # Check if already compatible
    if raw_diff in _COMPATIBLE_OFFSETS:
        return y, 0.0

    # Find the smallest shift to reach a compatible key
    # Search all compatible offsets in both directions and pick the shortest
    best_shift = None
    best_abs = float('inf')
    for offset in _COMPATIBLE_OFFSETS:
        # How many semitones would we need to add to source to reach this offset from target?
        needed = (target_semi + offset - source_semi) % 12
        # Convert to signed value in [-6, 6]
        signed = needed if needed <= 6 else needed - 12
        if abs(signed) < best_abs:
            best_abs = abs(signed)
            best_shift = signed

    # Safety cap: don't apply shifts larger than 4 semitones
    if best_shift is None or abs(best_shift) > 4:
        print(f"   ⚠️  Key shift too large ({best_shift} semitones) — skipping pitch force")
        return y, 0.0

    print(f"   🎹 Harmonic forcing: {source_key} {source_mode} → {target_key} {target_mode} "
          f"({best_shift:+d} semitones)")

    shifted = pitch_shift(y, sr, float(best_shift))
    return shifted, float(best_shift)


def detect_key_simple(y: np.ndarray, sr: int) -> Tuple[str, str]:
    """
    Lightweight chroma-based key detection (uses essentia if available,
    falls back to a librosa chroma correlation approach).

    Returns:
        (key_name, mode) e.g. ('A#', 'minor')
    """
    try:
        import essentia.standard as es
        audio = y.astype(np.float32)
        key_extractor = es.KeyExtractor()
        key, scale, _ = key_extractor(audio)
        return key, scale
    except Exception:
        pass

    # Fallback: Krumhansl-Schmuckler profiles via librosa chroma
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Major / minor KS profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                               2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                               2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
            'F#', 'G', 'G#', 'A', 'A#', 'B']

    best_score = -2.0
    best_key = 'C'
    best_mode = 'major'
    for i, k in enumerate(keys):
        rotated_chroma = np.roll(chroma_mean, -i)
        major_score = np.corrcoef(rotated_chroma, major_profile)[0, 1]
        minor_score = np.corrcoef(rotated_chroma, minor_profile)[0, 1]
        if major_score > best_score:
            best_score, best_key, best_mode = major_score, k, 'major'
        if minor_score > best_score:
            best_score, best_key, best_mode = minor_score, k, 'minor'

    return best_key, best_mode
