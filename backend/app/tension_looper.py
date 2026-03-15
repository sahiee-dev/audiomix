"""
tension_looper.py — Beat-grid aware looping engine for transition hype effects.

Core concept: During a transition, isolate a musically clean 1-bar or 2-bar
slice from Track A (ideally a synth stab, drum fill, or acapella phrase).
Loop that slice repeatedly while progressively halving the loop length on
each repetition.  The result is a chaotic, accelerating riser that builds
unbearable tension and naturally resolves when Track B drops.

This is the digital equivalent of the "loop roll" or "loop halving" effect
on a Pioneer CDJ / Rekordbox controller, but executed programmatically and
synced to the actual beat grid.

Public API:
  - TensionLooper.build_riser(y, sr, beats, n_bars, n_halvings) -> np.ndarray
  - TensionLooper.build_stutter(y, sr, beats, start_sample)    -> np.ndarray
"""

import numpy as np
from typing import Optional


class TensionLooper:
    """
    Build tension effects from a segment of audio using beat-aware looping.
    """

    def __init__(self, sr: int = 44100):
        self.sr = sr

    # ------------------------------------------------------------------
    # Core loop-halving riser
    # ------------------------------------------------------------------

    def build_riser(self,
                    y: np.ndarray,
                    beats: np.ndarray,
                    loop_start_sample: int,
                    n_bars: int = 2,
                    n_halvings: int = 4,
                    beats_per_bar: int = 4,
                    gain_ramp: bool = True) -> np.ndarray:
        """
        Create a loop-halving tension riser from a section of audio.

        Procedure:
          1. Cut a `n_bars`-bar loop from `y` starting at `loop_start_sample`.
          2. Play that loop once at full length.
          3. Halve the loop length and play again.
          4. Repeat `n_halvings` times.

        The total output length grows geometrically (2× - 1 of the initial loop)
        but the loop gets tighter and more frenetic on each pass.

        Args:
            y:                  Audio array to source the loop from
            beats:              Beat positions in samples (from beat tracker)
            loop_start_sample:  Sample index of the bar boundary to start the loop
            n_bars:             Initial loop length in bars (default 2)
            n_halvings:         How many times to halve the loop (4 = 1/16th note final)
            beats_per_bar:      Time signature numerator (default 4)
            gain_ramp:          If True, apply a gain ramp (+3dB over the riser)

        Returns:
            Riser audio array (mono).
        """
        sr = self.sr

        # Find the beat interval from the grid around our start point
        near_beats = beats[(beats >= loop_start_sample) &
                           (beats < loop_start_sample + 4 * sr)]
        if len(near_beats) < 2:
            # No clean beat grid found — generate a silent fallback
            print("   ⚠️  TensionLooper: no beats near loop start — skipping riser")
            return np.zeros(0)

        beat_interval = int(np.median(np.diff(near_beats)))
        bar_length    = beat_interval * beats_per_bar
        loop_length   = bar_length * n_bars

        # Ensure we don't overrun the source audio
        loop_end_sample = loop_start_sample + loop_length
        if loop_end_sample > len(y):
            loop_end_sample = len(y)
            loop_length = loop_end_sample - loop_start_sample
            if loop_length < beat_interval:
                print("   ⚠️  TensionLooper: loop source too short — skipping riser")
                return np.zeros(0)

        loop_source = y[loop_start_sample:loop_end_sample].copy()

        # Apply a linear fade-in at the start of the source to avoid a click
        fade_len = min(int(0.01 * sr), len(loop_source) // 4)  # 10 ms max
        if fade_len > 1:
            loop_source[:fade_len] *= np.linspace(0.0, 1.0, fade_len)

        # Build the sequence: full loop → half loop → quarter loop → ...
        segments = []
        current_loop = loop_source
        for i in range(n_halvings + 1):
            seg_len = len(current_loop)
            if seg_len < 64:
                break
            # Repeat the current loop to fill one full bar
            # (keeps the rhythm locked to the beat grid)
            target_len = loop_length
            repeats = int(np.ceil(target_len / seg_len))
            tiled = np.tile(current_loop, repeats)[:target_len]
            segments.append(tiled)
            # Halve for next pass
            half = seg_len // 2
            if half < 64:
                break
            current_loop = loop_source[:half]

        if not segments:
            return np.zeros(0)

        riser = np.concatenate(segments)

        # Optional: gentle gain ramp to build perceived intensity
        if gain_ramp:
            gain_curve = np.linspace(1.0, 1.8, len(riser))   # +~5dB over the riser
            riser = riser * gain_curve
            riser = np.clip(riser, -1.0, 1.0)

        return riser

    # ------------------------------------------------------------------
    # Stutter / glitch effect (one-shot, simpler than a full riser)
    # ------------------------------------------------------------------

    def build_stutter(self,
                      y: np.ndarray,
                      beats: np.ndarray,
                      start_sample: int,
                      n_beats: int = 4,
                      subdivisions: int = 8) -> np.ndarray:
        """
        Create a rapid stutter effect by repeating sub-beat slices.

        Chops `n_beats` worth of audio into `subdivisions` equal slices and
        repeats each slice twice, producing a robotic "glitch" texture.

        Args:
            y:            Audio source array
            beats:        Beat positions in samples
            start_sample: Beat boundary to start from
            n_beats:      How many beats of audio to process
            subdivisions: How many slices per beat (8 = 32nd notes)

        Returns:
            Stutter audio array (same number of samples as the input span).
        """
        near_beats = beats[beats >= start_sample]
        if len(near_beats) < 2:
            return np.zeros(0)

        beat_interval  = int(np.median(np.diff(near_beats[:min(8, len(near_beats) - 1)])))
        total_samples  = beat_interval * n_beats
        end_sample     = min(start_sample + total_samples, len(y))
        source         = y[start_sample:end_sample]

        if len(source) < subdivisions:
            return source

        chunk_len  = len(source) // subdivisions
        stutter_out = []
        for i in range(subdivisions):
            chunk = source[i * chunk_len:(i + 1) * chunk_len]
            # Repeat chunk twice (creates the "rr-rr-rr" stutter texture)
            stutter_out.append(chunk)
            stutter_out.append(chunk)

        result = np.concatenate(stutter_out)
        # Trim/pad to match original length
        if len(result) > len(source):
            result = result[:len(source)]
        elif len(result) < len(source):
            result = np.pad(result, (0, len(source) - len(result)))

        return result
