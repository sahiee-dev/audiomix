"""
dsp_fx.py — Studio-quality DSP effects for professional DJ transitions.

Provides:
  - Three-band EQ kill (low / mid / high isolation)
  - Frequency-targeted bass swap (hard-cut bass on a downbeat)
  - Swept high-pass and low-pass filters for tension build / wash-out effects
  - Utility helpers for applying fade-in/out curves to audio segments
"""

import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Low-level Biquad / IIR filter helpers
# ---------------------------------------------------------------------------

def _butter_filter(y: np.ndarray, cutoff_hz: float, sr: int,
                   btype: str, order: int = 4) -> np.ndarray:
    """Apply a Butterworth IIR filter to a mono audio array."""
    nyq = sr / 2.0
    Wn = np.clip(cutoff_hz / nyq, 1e-4, 0.9999)
    b, a = scipy_signal.butter(order, Wn, btype=btype)
    return scipy_signal.sosfilt(scipy_signal.butter(order, Wn, btype=btype, output='sos'), y)


def lowpass(y: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    """Return the signal with everything above cutoff_hz attenuated."""
    sos = scipy_signal.butter(order, cutoff_hz / (sr / 2), btype='low', output='sos')
    return scipy_signal.sosfilt(sos, y)


def highpass(y: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    """Return the signal with everything below cutoff_hz attenuated."""
    sos = scipy_signal.butter(order, cutoff_hz / (sr / 2), btype='high', output='sos')
    return scipy_signal.sosfilt(sos, y)


def bandpass(y: np.ndarray, lo_hz: float, hi_hz: float, sr: int,
             order: int = 4) -> np.ndarray:
    """Return only the frequency band between lo_hz and hi_hz."""
    nyq = sr / 2.0
    Wn = [np.clip(lo_hz / nyq, 1e-4, 0.9999), np.clip(hi_hz / nyq, 1e-4, 0.9999)]
    sos = scipy_signal.butter(order, Wn, btype='band', output='sos')
    return scipy_signal.sosfilt(sos, y)


# ---------------------------------------------------------------------------
# Three-band EQ Kill
# ---------------------------------------------------------------------------

# Crossover frequencies (in Hz) — industry standard DJ EQ splits
LOW_MID_CROSSOVER = 300      # Hz
MID_HIGH_CROSSOVER = 3000    # Hz


@dataclass
class ThreeBandEQ:
    """
    Split audio into three independent frequency bands.

    Killing a band means multiplying it by 0 before re-summing.  This
    accurately models the "EQ kill" knob on a professional DJ mixer.

    Usage:
        eq = ThreeBandEQ(y, sr)
        # Bass swap — kill bass of Track A
        no_bass_a = eq.apply(kill_low=True)
    """
    y: np.ndarray
    sr: int

    def split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lows, mids, highs) as separate arrays."""
        lows  = lowpass(self.y,  LOW_MID_CROSSOVER, self.sr)
        highs = highpass(self.y, MID_HIGH_CROSSOVER, self.sr)
        mids  = self.y - lows - highs  # mid = full - lows - highs
        return lows, mids, highs

    def apply(self,
              low_gain:  float = 1.0,
              mid_gain:  float = 1.0,
              high_gain: float = 1.0,
              kill_low:  bool  = False,
              kill_mid:  bool  = False,
              kill_high: bool  = False) -> np.ndarray:
        """
        Return a re-combined signal with per-band gain or kill applied.

        Args:
            low_gain / mid_gain / high_gain : Multiplier for each band (0.0–2.0).
            kill_low / kill_mid / kill_high : Hard-zero a band (overrides gain).
        """
        lows, mids, highs = self.split()
        if kill_low:  low_gain  = 0.0
        if kill_mid:  mid_gain  = 0.0
        if kill_high: high_gain = 0.0
        return lows * low_gain + mids * mid_gain + highs * high_gain


# ---------------------------------------------------------------------------
# Bass Swap
# ---------------------------------------------------------------------------

def bass_swap(y_a: np.ndarray, y_b: np.ndarray, swap_sample: int,
              sr: int, fade_samples: int = 256) -> np.ndarray:
    """
    Hard-swap the bass frequency band from Track A to Track B at exactly
    `swap_sample`.  A tiny crossfade of `fade_samples` prevents click
    artifacts while keeping the swap perceptually instant.

    Returns a full-length array the same length as y_a (or longer if y_b
    extends past y_a after the swap point).

    Args:
        y_a:          Track A audio (mono ndarray)
        y_b:          Track B audio (mono ndarray), aligned to y_a time base
        swap_sample:  Exact sample index at which the bass ownership transfers
        sr:           Sample rate
        fade_samples: Samples over which to micro-crossfade to avoid a click

    Returns:
        Combined audio with bass swapped at the downbeat.
    """
    n = max(len(y_a), len(y_b))
    y_a_padded = np.pad(y_a, (0, max(0, n - len(y_a))))
    y_b_padded = np.pad(y_b, (0, max(0, n - len(y_b))))

    # Extract bass from both tracks
    bass_a = lowpass(y_a_padded, LOW_MID_CROSSOVER, sr)
    bass_b = lowpass(y_b_padded, LOW_MID_CROSSOVER, sr)

    # High shelf (everything except bass) — stays as a smooth mix
    high_a = y_a_padded - bass_a
    high_b = y_b_padded - bass_b

    # Build micro-crossfade window around swap_sample
    fade = np.ones(n)
    s = max(0, swap_sample - fade_samples // 2)
    e = min(n, swap_sample + fade_samples // 2)
    ramp_len = e - s
    fade[s:e] = np.linspace(1.0, 0.0, ramp_len)
    fade[e:]  = 0.0
    fade_inv  = 1.0 - fade

    # Bass: Track A before swap, Track B after swap
    bass_combined = bass_a * fade + bass_b * fade_inv

    # Upper bands: maintain a smooth mix with a longer standard crossfade
    # — caller is responsible for this; we just merge bass here.
    upper_combined = high_a  # pass through; caller applies their own crossfade

    return bass_combined + upper_combined


# ---------------------------------------------------------------------------
# Swept High-Pass Filter (Wash-Out Effect)
# ---------------------------------------------------------------------------

def highpass_sweep(y: np.ndarray, sr: int,
                   start_cutoff_hz: float = 20.0,
                   end_cutoff_hz:   float = 4000.0,
                   curve: str = 'exponential') -> np.ndarray:
    """
    Apply a time-varying high-pass filter whose cutoff sweeps from
    `start_cutoff_hz` to `end_cutoff_hz` over the length of the signal.

    This creates the signature DJ "wash-out" / "filter-up" effect as a
    track exits the mix.

    Args:
        y:               Mono audio array
        sr:              Sample rate
        start_cutoff_hz: Initial cutoff frequency (Hz)
        end_cutoff_hz:   Final cutoff frequency (Hz)
        curve:           'linear' or 'exponential' sweep shape

    Returns:
        Filter-swept audio array (same length as y).
    """
    n = len(y)
    # Process in 2048-sample chunks so the cutoff moves smoothly
    chunk = 2048
    out = np.zeros(n)

    if curve == 'exponential':
        cutoffs = np.exp(np.linspace(np.log(max(start_cutoff_hz, 20)),
                                     np.log(max(end_cutoff_hz, 21)), n // chunk + 1))
    else:
        cutoffs = np.linspace(start_cutoff_hz, end_cutoff_hz, n // chunk + 1)

    nyq = sr / 2.0
    for i, cutoff in enumerate(cutoffs):
        s = i * chunk
        e = min(s + chunk, n)
        if s >= n:
            break
        Wn = np.clip(cutoff / nyq, 1e-4, 0.9999)
        sos = scipy_signal.butter(2, Wn, btype='high', output='sos')
        out[s:e] = scipy_signal.sosfilt(sos, y[s:e])

    return out


def lowpass_sweep(y: np.ndarray, sr: int,
                  start_cutoff_hz: float = 20000.0,
                  end_cutoff_hz:   float = 200.0,
                  curve: str = 'exponential') -> np.ndarray:
    """
    Apply a time-varying low-pass filter that sweeps downward.
    Creates a muffling / telephonic effect for incoming tracks or outros.

    Args:
        y:               Mono audio array
        sr:              Sample rate
        start_cutoff_hz: Initial (open) cutoff frequency
        end_cutoff_hz:   Final (closed) cutoff frequency
        curve:           'linear' or 'exponential'

    Returns:
        Filter-swept audio array (same length as y).
    """
    n = len(y)
    chunk = 2048
    out = np.zeros(n)

    if curve == 'exponential':
        cutoffs = np.exp(np.linspace(np.log(max(start_cutoff_hz, 200)),
                                     np.log(max(end_cutoff_hz, 20)), n // chunk + 1))
    else:
        cutoffs = np.linspace(start_cutoff_hz, end_cutoff_hz, n // chunk + 1)

    nyq = sr / 2.0
    for i, cutoff in enumerate(cutoffs):
        s = i * chunk
        e = min(s + chunk, n)
        if s >= n:
            break
        Wn = np.clip(cutoff / nyq, 1e-4, 0.9999)
        sos = scipy_signal.butter(2, Wn, btype='low', output='sos')
        out[s:e] = scipy_signal.sosfilt(sos, y[s:e])

    return out


# ---------------------------------------------------------------------------
# Utility: amplitude fade curves
# ---------------------------------------------------------------------------

def make_fade_out(n_samples: int, shape: str = 'cosine') -> np.ndarray:
    """Return a fade-out curve of length n_samples."""
    if shape == 'linear':
        return np.linspace(1.0, 0.0, n_samples)
    # Cosine (equal-power) — default
    return np.cos(np.linspace(0, np.pi / 2, n_samples)) ** 2


def make_fade_in(n_samples: int, shape: str = 'cosine') -> np.ndarray:
    """Return a fade-in curve of length n_samples."""
    if shape == 'linear':
        return np.linspace(0.0, 1.0, n_samples)
    return np.sin(np.linspace(0, np.pi / 2, n_samples)) ** 2
