"""
segmenter.py — Audio segmentation and semantic song structure detection.

Includes:
  - find_optimal_entry_point() : Original entry-point heuristics (preserved)
  - detect_song_structure()    : Structural segment detection (preserved)
  - classify_sections()        : NEW — semantic section classification
                                  returning SongSection objects labelled as
                                  Intro / Verse / Pre-Chorus / Chorus-Drop /
                                  Bridge / Outro.
"""

import numpy as np
import librosa
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

SECTION_LABELS = ('intro', 'verse', 'pre_chorus', 'chorus_drop', 'bridge', 'outro')


@dataclass
class SongSection:
    """A labelled structural segment of a track."""
    label: str            # One of SECTION_LABELS
    start_sample: int
    end_sample: int
    start_time: float     # seconds
    end_time: float       # seconds
    energy: float         # Mean RMS energy (0–1)
    brightness: float     # Mean spectral centroid (Hz)
    novelty: float        # Structural novelty score at this boundary (0–1)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self):
        return (f"SongSection({self.label!r}, "
                f"{self.start_time:.1f}s–{self.end_time:.1f}s, "
                f"energy={self.energy:.3f})")


# ---------------------------------------------------------------------------
# Original helpers (unchanged public API)
# ---------------------------------------------------------------------------

def find_optimal_entry_point(y, sr, beats, method='high_energy'):
    """
    Find the best point in a track to start mixing using heuristic analysis.
    """
    if method == 'skip_intro':
        return int(len(y) * 0.15)

    elif method == 'high_energy':
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        window = 86
        rms_smooth = np.convolve(rms, np.ones(window) / window, mode='same')

        start = int(len(rms_smooth) * 0.2)
        end = int(len(rms_smooth) * 0.8)
        peak_frame = start + np.argmax(rms_smooth[start:end])
        peak_sample = peak_frame * hop_length

        if len(beats) > 0:
            nearest_beat_idx = np.argmin(np.abs(beats - peak_sample))
            return int(beats[nearest_beat_idx])
        return peak_sample

    elif method == 'first_drop':
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)

        skip_samples = int(10 * sr)
        onsets_samples = librosa.frames_to_samples(onsets)
        valid_onsets = onsets_samples[onsets_samples > skip_samples]

        if len(valid_onsets) > 0:
            onset_frames = librosa.samples_to_frames(valid_onsets)
            onset_strengths = onset_env[onset_frames]
            strongest_idx = np.argmax(onset_strengths)
            return int(valid_onsets[strongest_idx])

        return skip_samples

    elif method == 'structural':
        try:
            import essentia.standard as es
            y_essentia = y.astype(np.float32)
            segmentation = es.SBic()(y_essentia)

            energies = []
            for i in range(len(segmentation) - 1):
                start = int(segmentation[i] * sr)
                end = int(segmentation[i + 1] * sr)
                segment_energy = np.mean(librosa.feature.rms(y=y[start:end]))
                energies.append((start, segment_energy))

            if energies:
                entry = max(energies, key=lambda x: x[1])[0]
                if len(beats) > 0:
                    nearest_beat_idx = np.argmin(np.abs(beats - entry))
                    return int(beats[nearest_beat_idx])
                return entry
        except Exception as e:
            print(f"Structural analysis failed: {e}, falling back to high_energy")
            return find_optimal_entry_point(y, sr, beats, method='high_energy')

    return 0


def detect_song_structure(y, sr):
    """Detect structural segments using Essentia (legacy)."""
    try:
        import essentia.standard as es
        y_essentia = y.astype(np.float32)
        extractor = es.MusicExtractor()
        results = extractor(y_essentia)

        segments = []
        if 'segments' in results:
            for seg in results['segments']:
                start = int(seg['start'] * sr)
                end = int(seg['end'] * sr)
                energy = float(np.mean(librosa.feature.rms(y=y[start:end])))
                segments.append((start, end, seg.get('label', 'unknown'), energy))

        return segments
    except Exception as e:
        print(f"Structure detection failed: {e}")
        return [(0, len(y), 'full_track', 1.0)]


# ---------------------------------------------------------------------------
# Semantic Section Classifier (NEW)
# ---------------------------------------------------------------------------

def classify_sections(y: np.ndarray, sr: int,
                       min_section_duration: float = 8.0) -> List[SongSection]:
    """
    Detect and semantically label the structural sections of a track.

    This uses a multi-feature novelty curve approach:
      1. Compute a timbral/chroma recurrence matrix to find structural boundaries
         (where the music changes character).
      2. For each detected segment, compute energy, brightness, and novelty.
      3. Apply a heuristic rule set to assign semantic labels based on energy
         trajectory, position in the track, and relative brightness:

         - intro       : low energy, track beginning (<15% duration)
         - outro       : low energy, track end (>80% duration)
         - chorus_drop : high energy peak (top 25% RMS), usually 30–70% of track
         - pre_chorus  : rising energy immediately preceding a chorus_drop
         - bridge      : mid song, relatively low energy after a chorus_drop
         - verse       : everything else

    Args:
        y:                    Mono audio array
        sr:                   Sample rate
        min_section_duration: Merge segments shorter than this (seconds)

    Returns:
        List of SongSection objects in chronological order.
    """
    hop_length = 512
    duration = len(y) / sr

    # ---- 1. Feature extraction ----
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]

    # Stack features for the recurrence matrix
    features = np.vstack([
        librosa.util.normalize(mfcc, axis=1),
        librosa.util.normalize(chroma, axis=1),
    ])

    # ---- 2. Segment boundary detection via novelty curve ----
    R = librosa.segment.recurrence_matrix(features, mode='affinity',
                                          sym=True, sparse=False)
    novelty = librosa.segment.recurrence_to_laplacian(R, sym=True)
    # Diagonal novelty score — peaks correspond to structural changes
    novelty_curve = np.mean(novelty, axis=1)
    novelty_curve = np.clip(novelty_curve - novelty_curve.min(), 0, None)
    if novelty_curve.max() > 0:
        novelty_curve /= novelty_curve.max()

    # Pick peaks in the novelty curve as boundaries
    min_frames = int(min_section_duration * sr / hop_length)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(novelty_curve, distance=min_frames, height=0.3)

    # Build boundary list in seconds
    frame_times = librosa.frames_to_time(np.arange(len(novelty_curve)),
                                          sr=sr, hop_length=hop_length)
    boundary_times = [0.0] + list(frame_times[peaks]) + [duration]

    # ---- 3. Compute per-segment descriptors ----
    segments_raw = []
    for i in range(len(boundary_times) - 1):
        t_start = boundary_times[i]
        t_end   = boundary_times[i + 1]
        s_start = int(t_start * sr)
        s_end   = min(int(t_end * sr), len(y))

        if s_end <= s_start:
            continue

        f_start = librosa.time_to_frames(t_start, sr=sr, hop_length=hop_length)
        f_end   = librosa.time_to_frames(t_end,   sr=sr, hop_length=hop_length)
        f_end   = min(f_end, len(rms))

        seg_rms    = float(np.mean(rms[f_start:f_end]))           if f_end > f_start else 0.0
        seg_bright = float(np.mean(spectral_centroid[f_start:f_end])) if f_end > f_start else 0.0
        seg_novelty = float(novelty_curve[f_start]) if f_start < len(novelty_curve) else 0.0

        segments_raw.append({
            'start': t_start, 'end': t_end,
            's_start': s_start, 's_end': s_end,
            'energy': seg_rms, 'brightness': seg_bright,
            'novelty': seg_novelty,
        })

    if not segments_raw:
        # Degenerate case — treat whole track as one section
        return [SongSection('verse', 0, len(y), 0.0, duration,
                             float(np.mean(rms)), float(np.mean(spectral_centroid)), 0.0)]

    # ---- 4. Semantic label assignment ----
    energies  = np.array([s['energy'] for s in segments_raw])
    e_max     = energies.max() if energies.max() > 0 else 1.0
    energies_norm = energies / e_max

    # Energy threshold for a "drop" / chorus
    drop_threshold = 0.75    # top 25% energy considered high-energy
    low_threshold  = 0.40    # bottom 40% energy considered low-energy

    # Positional thresholds (fraction of total duration)
    intro_end_frac  = 0.18
    outro_start_frac = 0.80

    sections: List[SongSection] = []
    prev_label = None

    for i, seg in enumerate(segments_raw):
        pos = (seg['start'] + seg['end']) / 2.0 / duration   # normalised position
        e   = energies_norm[i]

        if pos < intro_end_frac and e < 0.60:
            label = 'intro'
        elif pos > outro_start_frac and e < 0.60:
            label = 'outro'
        elif e >= drop_threshold:
            label = 'chorus_drop'
        elif e < low_threshold and prev_label in ('chorus_drop',):
            label = 'bridge'
        elif prev_label in ('verse', 'intro', 'bridge') and e > 0.55:
            # Rising energy before a potential drop
            label = 'pre_chorus'
        else:
            label = 'verse'

        sections.append(SongSection(
            label=label,
            start_sample=seg['s_start'],
            end_sample=seg['s_end'],
            start_time=seg['start'],
            end_time=seg['end'],
            energy=seg['energy'],
            brightness=seg['brightness'],
            novelty=seg['novelty'],
        ))
        prev_label = label

    return sections


def find_best_outro_section(sections: List[SongSection]) -> Optional[SongSection]:
    """
    From a list of classified sections, return the best outro candidate:
    prefer the last 'chorus_drop' or 'verse' with high energy, not the actual outro.
    This is the most interesting section to mix *out of*.
    """
    candidates = [s for s in sections if s.label in ('chorus_drop', 'verse')]
    if not candidates:
        return sections[-1] if sections else None
    # Prefer the last high-energy section
    return max(candidates[-3:], key=lambda s: s.energy)


def find_best_intro_section(sections: List[SongSection]) -> Optional[SongSection]:
    """
    Return the best section to begin mixing *into* in the incoming track:
    prefer the first 'chorus_drop' or 'verse' (skip the intro).
    """
    for s in sections:
        if s.label in ('verse', 'pre_chorus', 'chorus_drop'):
            return s
    return sections[0] if sections else None
