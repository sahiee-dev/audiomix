import numpy as np
import librosa
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class AudioFingerprint:
    """Musical fingerprint for a section of audio"""
    start_sample: int
    end_sample: int
    bpm: float
    energy_rms: float
    spectral_centroid: float
    chroma_profile: np.ndarray  # 12-dim key profile
    onset_pattern: np.ndarray   # Rhythm signature
    tempo_stability: float

    def __repr__(self):
        return (f"Fingerprint(bpm={self.bpm:.1f}, energy={self.energy_rms:.3f}, "
                f"centroid={self.spectral_centroid:.0f}Hz)")


class PatternMatcher:
    """Match song endings to song beginnings for seamless transitions"""

    # Shared hop length for all STFT-based features so frame indices align
    HOP_LENGTH = 512

    def __init__(self, match_threshold=0.7):
        self.match_threshold = match_threshold  # 0.0-1.0 similarity score

    # ------------------------------------------------------------------
    # Pre-compute all STFT-derived features for an entire audio array once.
    # The scanning loop then merely slices into these matrices.
    # ------------------------------------------------------------------
    def _precompute_features(self, y: np.ndarray, sr: int) -> dict:
        """
        Compute all heavy STFT-based feature matrices for `y` in a single pass.

        Returns a dict of pre-computed matrices keyed by feature name.
        All frame-indexed features share self.HOP_LENGTH so frame arithmetic
        is consistent across them.
        """
        hl = self.HOP_LENGTH
        rms = librosa.feature.rms(y=y, hop_length=hl)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hl)[0]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hl)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hl)
        return {
            'rms': rms,
            'spectral_centroid': spectral_centroid,
            'chroma': chroma,          # shape: [12, T]
            'onset_env': onset_env,    # shape: [T]
            'sr': sr,
        }

    def _fingerprint_from_precomputed(self, precomputed: dict,
                                      start_time: float, end_time: float,
                                      y_full: np.ndarray) -> AudioFingerprint:
        """
        Build an AudioFingerprint from pre-computed feature slices.

        This only does array indexing (O(1)) rather than re-running the STFT.
        BPM and tempo stability still require a short beat-track on the slice,
        but that is unavoidable and cheap relative to the STFT.
        """
        sr = precomputed['sr']
        hl = self.HOP_LENGTH

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        # Convert time bounds to feature frame indices
        start_frame = librosa.time_to_frames(start_time, sr=sr, hop_length=hl)
        end_frame = librosa.time_to_frames(end_time, sr=sr, hop_length=hl)

        # Slice pre-computed matrices — O(1) indexing, no STFT
        rms_slice = precomputed['rms'][start_frame:end_frame]
        centroid_slice = precomputed['spectral_centroid'][start_frame:end_frame]
        chroma_slice = precomputed['chroma'][:, start_frame:end_frame]
        onset_slice = precomputed['onset_env'][start_frame:end_frame]

        energy_rms = float(np.sqrt(np.mean(rms_slice ** 2))) if len(rms_slice) > 0 else 0.0
        spectral_centroid = float(np.mean(centroid_slice)) if len(centroid_slice) > 0 else 0.0
        chroma_profile = np.mean(chroma_slice, axis=1) if chroma_slice.shape[1] > 0 else np.zeros(12)

        # Downsample onset envelope to a fixed 32-point signature
        if len(onset_slice) >= 2:
            idx = np.linspace(0, len(onset_slice) - 1, 33, dtype=int)
            onset_pattern = onset_slice[idx]
        else:
            onset_pattern = np.zeros(33)

        # BPM & tempo stability — run on the audio slice (cheap relative to STFT)
        segment = y_full[start_sample:end_sample]
        tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
        if len(beats) > 2:
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
            tempo_stability = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-9))
        else:
            tempo_stability = 0.5

        tempo = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])

        return AudioFingerprint(
            start_sample=start_sample,
            end_sample=end_sample,
            bpm=tempo,
            energy_rms=energy_rms,
            spectral_centroid=spectral_centroid,
            chroma_profile=chroma_profile,
            onset_pattern=onset_pattern,
            tempo_stability=tempo_stability,
        )

    def create_fingerprint(self, y: np.ndarray, sr: int,
                           start_time: float, end_time: float) -> AudioFingerprint:
        """
        Create a musical fingerprint for a section of audio.

        Public API kept for compatibility. When scanning many windows back-to-back,
        prefer calling _precompute_features() once and passing the result to
        _fingerprint_from_precomputed() to avoid redundant STFT computation.
        """
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        segment = y[start_sample:end_sample]

        tempo, beats = librosa.beat.beat_track(y=segment, sr=sr)
        if len(beats) > 2:
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
            tempo_stability = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-9))
        else:
            tempo_stability = 0.5

        energy_rms = float(np.sqrt(np.mean(segment ** 2)))
        spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        chroma_profile = np.mean(chroma, axis=1)
        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        onset_pattern = librosa.util.sync(
            onset_env[np.newaxis, :],
            np.linspace(0, len(onset_env), 33, dtype=int)
        )[0]

        return AudioFingerprint(
            start_sample=start_sample,
            end_sample=end_sample,
            bpm=float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0]),
            energy_rms=energy_rms,
            spectral_centroid=spectral_centroid,
            chroma_profile=chroma_profile,
            onset_pattern=onset_pattern,
            tempo_stability=tempo_stability,
        )

    def calculate_similarity(self, fp1: AudioFingerprint, fp2: AudioFingerprint,
                             genre='general') -> Tuple[float, List]:
        """
        Calculate similarity score between two fingerprints (0.0-1.0).
        """
        if genre == 'hip-hop':
            weights = {
                'bpm': 0.20, 'energy': 0.30, 'timbre': 0.20,
                'harmony': 0.10, 'rhythm': 0.15, 'stability': 0.05,
            }
            bpm_tolerance = 30.0
        elif genre in ['edm', 'house', 'techno']:
            weights = {
                'bpm': 0.40, 'energy': 0.20, 'timbre': 0.10,
                'harmony': 0.15, 'rhythm': 0.10, 'stability': 0.05,
            }
            bpm_tolerance = 15.0
        else:
            weights = {
                'bpm': 0.35, 'energy': 0.20, 'timbre': 0.15,
                'harmony': 0.15, 'rhythm': 0.10, 'stability': 0.05,
            }
            bpm_tolerance = 20.0

        bpm_diff = abs(fp1.bpm - fp2.bpm)
        bpm_score = max(0, 1.0 - (bpm_diff / bpm_tolerance))

        energy_diff = abs(fp1.energy_rms - fp2.energy_rms)
        energy_score = max(0, 1.0 - (energy_diff / 0.3))

        centroid_diff = abs(fp1.spectral_centroid - fp2.spectral_centroid)
        centroid_score = max(0, 1.0 - (centroid_diff / 2000.0))

        chroma_corr = np.corrcoef(fp1.chroma_profile, fp2.chroma_profile)[0, 1]
        chroma_score = (chroma_corr + 1) / 2

        onset_corr = np.corrcoef(fp1.onset_pattern, fp2.onset_pattern)[0, 1]
        onset_score = (onset_corr + 1) / 2

        tempo_score = (fp1.tempo_stability + fp2.tempo_stability) / 2

        scores = [
            ('BPM',       bpm_score,      weights['bpm']),
            ('Energy',    energy_score,   weights['energy']),
            ('Timbre',    centroid_score, weights['timbre']),
            ('Harmony',   chroma_score,   weights['harmony']),
            ('Rhythm',    onset_score,    weights['rhythm']),
            ('Stability', tempo_score,    weights['stability']),
        ]

        total_score = sum(score * weight for _, score, weight in scores)
        return total_score, scores

    def find_best_match(self, y_outro: np.ndarray, y_song: np.ndarray,
                        sr: int, outro_duration: float = 30.0,
                        genre: str = 'general') -> Tuple[Optional[float], float, List]:
        """
        Find the best matching point in song B for song A's outro.

        Performance improvement: all STFT-based features for the scan region
        of y_song are computed ONCE before the loop. Each iteration only does
        O(1) array slicing instead of running a full STFT.
        """
        print(f"\n🔍 Analyzing outro fingerprint (genre: {genre})...")

        # Fingerprint the outro (one-shot, not in a loop)
        outro_fp = self.create_fingerprint(
            y_outro, sr,
            start_time=max(0, len(y_outro) / sr - outro_duration),
            end_time=len(y_outro) / sr,
        )
        print(f"   Outro: {outro_fp}")

        song_duration = len(y_song) / sr
        scan_until = min(song_duration, song_duration * 0.5)
        window_size = 15.0
        step_size = 5.0

        # --- KEY OPTIMISATION: pre-compute features for the entire scan region ---
        scan_end_sample = int(scan_until * sr)
        print(f"🔍 Pre-computing features for first {scan_until:.0f}s...")
        precomputed = self._precompute_features(y_song[:scan_end_sample], sr)

        best_score = 0.0
        best_time = None
        best_breakdown = []

        n_windows = int(scan_until / step_size)
        print(f"🔍 Scanning {n_windows} positions (STFT computed once)...")

        for t in np.arange(0, scan_until - window_size, step_size):
            # Cheap slice-based fingerprint — no STFT recalculation
            candidate_fp = self._fingerprint_from_precomputed(
                precomputed, t, t + window_size, y_song
            )

            score, breakdown = self.calculate_similarity(outro_fp, candidate_fp, genre=genre)

            print(f"   {t:>5.1f}s: score={score:.3f} "
                  f"(BPM:{breakdown[0][1]:.2f}, Energy:{breakdown[1][1]:.2f}, "
                  f"Rhythm:{breakdown[4][1]:.2f})")

            if score > best_score:
                best_score = score
                best_time = t + (window_size / 2)
                best_breakdown = breakdown

        if best_score >= self.match_threshold:
            print(f"\n✅ MATCH FOUND at {best_time:.1f}s (score: {best_score:.3f})")
            return best_time, best_score, best_breakdown
        else:
            print(f"\n⚠️ NO GOOD MATCH (best: {best_score:.3f} < {self.match_threshold:.3f})")
            print(f"   → Falling back to standard crossfade")
            return None, best_score, best_breakdown
