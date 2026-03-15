"""
mixer.py — Professional Stem-Level Mashup Engine.

Executes semantic, studio-quality DJ transitions by:
  1. Classifying song sections (Intro / Verse / Drop / Outro) using the segmenter
  2. Choosing a transition blueprint based on section types:
       - chorus_drop → chorus_drop : Bass swap + filter sweep + stem-aware crossfade
       - *           → *           : Filter sweep + standard beat-synced crossfade
  3. Building a tension riser (loop-halving) to hype the transition when applicable
  4. Applying studio-grade harmonic forcing (pitch shift to Camelot-compatible key)
  5. All I/O is in-memory; no intermediate files are written to disk
"""

import io
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from typing import List, Dict, Optional, Tuple

from .align import detect_beats_advanced, align_tracks_by_beats
from .segmenter import (classify_sections, find_best_outro_section,
                         find_best_intro_section, find_optimal_entry_point)
from .pattern_matcher import PatternMatcher
from .source_separator import SourceSeparator
from .audio_normalizer import normalize_tracks_before_mixing
from .dsp_fx import (ThreeBandEQ, bass_swap as dsp_bass_swap,
                      highpass_sweep, make_fade_out, make_fade_in)
from .pitch_engine import force_key, detect_key_simple, time_stretch
from .tension_looper import TensionLooper


# ---------------------------------------------------------------------------
# Low-level crossfade helpers
# ---------------------------------------------------------------------------

def _beat_synced_crossfade(y_a: np.ndarray, y_b: np.ndarray,
                            fade_start: int, fade_samples: int) -> np.ndarray:
    """Equal-power crossfade.  Returns the full mixed array."""
    if fade_start + fade_samples > len(y_a):
        fade_samples = len(y_a) - fade_start

    a_fade = y_a[fade_start:fade_start + fade_samples]
    b_fade = y_b[:fade_samples]
    min_len = min(len(a_fade), len(b_fade))

    curve_out = make_fade_out(min_len)
    curve_in  = make_fade_in(min_len)

    mixed = a_fade[:min_len] * curve_out + b_fade[:min_len] * curve_in
    return np.concatenate([y_a[:fade_start], mixed, y_b[fade_samples:]])


def _stem_aware_crossfade(y_a: np.ndarray, y_b: np.ndarray,
                           fade_start: int, fade_samples: int,
                           stems_a: Optional[Dict], stems_b: Optional[Dict],
                           sr: int) -> np.ndarray:
    """
    Stem-aware crossfade: vocals exit at 30%, instrumentals at 50%.
    Falls back to equal-power if stems not available.
    """
    if stems_a is None or stems_b is None:
        return _beat_synced_crossfade(y_a, y_b, fade_start, fade_samples)

    if fade_start + fade_samples > len(y_a):
        fade_samples = len(y_a) - fade_start

    min_len = min(fade_samples,
                  len(y_a) - fade_start,
                  len(stems_a['vocals']) - fade_start,
                  len(stems_b['vocals']),
                  len(y_b))

    if min_len <= 0:
        return _beat_synced_crossfade(y_a, y_b, fade_start, fade_samples)

    voc_out = make_fade_out(min_len)
    voc_out[int(min_len * 0.3):] = np.cos(
        np.linspace(0, np.pi / 2, min_len - int(min_len * 0.3))) ** 3
    voc_in = np.zeros(min_len)
    voc_in[int(min_len * 0.7):] = np.sin(
        np.linspace(0, np.pi / 2, min_len - int(min_len * 0.7))) ** 3

    inst_out = make_fade_out(min_len)
    inst_in  = make_fade_in(min_len)

    mixed = (stems_a['vocals'][fade_start:fade_start + min_len]     * voc_out  +
             stems_b['vocals'][:min_len]                             * voc_in   +
             stems_a['instrumental'][fade_start:fade_start + min_len] * inst_out +
             stems_b['instrumental'][:min_len]                        * inst_in)

    print("   🎤 Stem-aware crossfade: Vocals out@30%, in@70%")
    return np.concatenate([y_a[:fade_start], mixed, y_b[fade_samples:]])


# ---------------------------------------------------------------------------
# In-memory MP3 export
# ---------------------------------------------------------------------------

def _export_mp3(y: np.ndarray, sr: int, path: str, bitrate: str = "320k"):
    """Encode to MP3 through a BytesIO buffer — no temp WAV on disk."""
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV', subtype='PCM_16')
    buf.seek(0)
    AudioSegment.from_wav(buf).export(path, format="mp3", bitrate=bitrate)


# ---------------------------------------------------------------------------
# Transition blueprint builder
# ---------------------------------------------------------------------------

def _choose_blueprint(section_a_label: str, section_b_label: str) -> Dict:
    """
    Return a transition blueprint dict based on the semantic labels of the
    sections being mixed from and into.

    Blueprint keys:
        crossfade_type  : 'stem_aware' | 'filter_sweep' | 'bass_swap' | 'standard'
        use_tension_riser: bool — build a loop-halving riser before Track B's drop
        filter_sweep_a  : bool — apply HP sweep to Track A during transition
        bass_swap       : bool — hard-swap bass frequencies at the downbeat
        crossfade_bars  : int  — crossfade duration in bars
    """
    drop_into_drop = (section_a_label == 'chorus_drop' and
                      section_b_label == 'chorus_drop')
    drop_into_verse = (section_a_label == 'chorus_drop' and
                       section_b_label in ('verse', 'pre_chorus'))
    verse_into_drop = (section_a_label in ('verse', 'pre_chorus') and
                       section_b_label == 'chorus_drop')

    if drop_into_drop:
        return {
            'crossfade_type': 'stem_aware',
            'use_tension_riser': False,
            'filter_sweep_a': True,
            'bass_swap': True,
            'crossfade_bars': 8,
        }
    elif verse_into_drop:
        return {
            'crossfade_type': 'filter_sweep',
            'use_tension_riser': True,
            'filter_sweep_a': True,
            'bass_swap': False,
            'crossfade_bars': 4,
        }
    elif drop_into_verse:
        return {
            'crossfade_type': 'standard',
            'use_tension_riser': False,
            'filter_sweep_a': False,
            'bass_swap': False,
            'crossfade_bars': 8,
        }
    else:
        return {
            'crossfade_type': 'standard',
            'use_tension_riser': False,
            'filter_sweep_a': False,
            'bass_swap': False,
            'crossfade_bars': 8,
        }


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def mix_tracks_professional(tracks_info: List[Dict], upload_dir: str, session: str,
                             normalize: bool = True,
                             tempo_match: bool = True,
                             harmonic_match: bool = True,
                             crossfade_duration: float = 8.0,
                             entry_method: str = 'high_energy',
                             use_stem_separation: bool = False) -> str:
    """
    Mix a list of tracks into a professional stem-level mashup.

    Returns:
        Absolute path to the final MP3 file.
    """
    if not tracks_info:
        raise ValueError("No tracks to mix")

    sr = 44100
    looper = TensionLooper(sr=sr)
    matcher = PatternMatcher(match_threshold=0.45)
    separator = SourceSeparator() if use_stem_separation else None

    # ------------------------------------------------------------------ #
    # Step 1 — Load every track into memory exactly once
    # ------------------------------------------------------------------ #
    print("\n📂 Loading all tracks into memory...")
    audio_cache = []
    for track in tracks_info:
        path = os.path.join(upload_dir, session, track['filename'])
        y_stereo, _ = librosa.load(path, sr=sr, mono=False)
        y_mono = np.mean(y_stereo, axis=0) if y_stereo.ndim > 1 else y_stereo
        audio_cache.append({'mono': y_mono, 'stereo': y_stereo})

    # ------------------------------------------------------------------ #
    # Step 2 — Normalize all tracks in-memory to -14 LUFS
    # ------------------------------------------------------------------ #
    if normalize:
        print("\n🔊 Normalizing all tracks to -14 LUFS (in-memory)...")
        mono_arrays = [t['mono'] for t in audio_cache]
        normalized = normalize_tracks_before_mixing(mono_arrays, sr)
        for i, y_norm in enumerate(normalized):
            audio_cache[i]['mono'] = y_norm

    # ------------------------------------------------------------------ #
    # Step 3 — Analyse each track: beats, key, semantic sections
    # ------------------------------------------------------------------ #
    print("\n🔬 Analysing track structures...")
    analyses = []
    for idx, (track, cache) in enumerate(zip(tracks_info, audio_cache)):
        y = cache['mono']
        print(f"\n  [{idx + 1}/{len(tracks_info)}] {track['filename']}")

        beats, downbeats, bpm = detect_beats_advanced(y, sr)
        key, mode = detect_key_simple(y, sr)
        sections = classify_sections(y, sr)

        print(f"     BPM: {bpm:.1f}  |  Key: {key} {mode}  |  Sections: {len(sections)}")
        for s in sections:
            print(f"       {s.label:12s}  {s.start_time:5.1f}s – {s.end_time:5.1f}s  "
                  f"energy={s.energy:.3f}")

        analyses.append({
            'y': y, 'stereo': cache['stereo'],
            'beats': beats, 'downbeats': downbeats, 'bpm': bpm,
            'key': key, 'mode': mode, 'sections': sections,
        })

    # ------------------------------------------------------------------ #
    # Step 4 — Mix loop
    # ------------------------------------------------------------------ #
    print(f"\n🎧 Starting Mashup Mix")
    print(f"   Tracks: {len(tracks_info)}")
    print(f"   Stem Separation: {'✅ ON' if use_stem_separation else '❌ OFF'}")
    print(f"   Harmonic Forcing: {'✅ ON' if harmonic_match else '❌ OFF'}")

    mixed_audio = None
    mixed_beats  = None
    mixed_downbeats = None
    mixed_bpm = None
    mixed_key = None
    mixed_mode = None
    prev_stems = None

    for idx, (track, analysis) in enumerate(zip(tracks_info, analyses)):
        print(f"\n{'=' * 70}")
        print(f"Track {idx + 1}/{len(tracks_info)}: {track['filename']}")
        print(f"{'=' * 70}")

        y         = analysis['y']
        beats     = analysis['beats']
        downbeats = analysis['downbeats']
        bpm       = analysis['bpm']
        key       = analysis['key']
        mode      = analysis['mode']
        sections  = analysis['sections']

        # -- Stem separation (optional but powerful) --
        current_stems = None
        if use_stem_separation:
            print("🎵 Separating stems...")
            stems_dir = os.path.join(upload_dir, session, 'stems')
            os.makedirs(stems_dir, exist_ok=True)
            separated = separator.separate(analysis['stereo'], stems_dir, sr_hint=sr)
            out_sr = separated['sr']
            if out_sr != sr:
                from scipy import signal as scipy_signal
                def _resamp(arr):
                    return scipy_signal.resample(arr, int(len(arr) * sr / out_sr))
                current_stems = {
                    'vocals': _resamp(separated['vocals']),
                    'instrumental': _resamp(separated['instrumental']),
                }
            else:
                current_stems = {
                    'vocals': separated['vocals'],
                    'instrumental': separated['instrumental'],
                }

        # -- First track — just set as the base --
        if mixed_audio is None:
            mixed_audio     = y
            mixed_beats     = beats
            mixed_downbeats = downbeats
            mixed_bpm       = bpm
            mixed_key       = key
            mixed_mode      = mode
            prev_stems      = current_stems
            print("✅ Track 1 loaded as base")
            continue

        # ==============================================================
        # Transition logic from mixed_audio → y
        # ==============================================================

        # --- Harmonic forcing ---
        if harmonic_match:
            y, semitones = force_key(y, sr, key, mixed_key,
                                     source_mode=mode, target_mode=mixed_mode)
            if semitones != 0.0:
                # Shift current_stems by the same amount
                if current_stems:
                    from .pitch_engine import pitch_shift as _ps
                    current_stems['vocals'] = _ps(current_stems['vocals'], sr, semitones)
                    current_stems['instrumental'] = _ps(current_stems['instrumental'], sr, semitones)
                # Re-run beat detection on pitch-shifted audio
                beats, downbeats, bpm = detect_beats_advanced(y, sr)

        # --- Pattern match entry point ---
        print("\n🎯 PATTERN MATCHING...")
        outro_samples = int(30 * sr)
        y_outro = mixed_audio[-outro_samples:]
        match_time, match_score, score_breakdown = matcher.find_best_match(
            y_outro, y, sr, outro_duration=30.0, genre=track.get('genre', 'general')
        )

        if match_time is not None:
            print(f"✨ Pattern match at {match_time:.1f}s (score {match_score:.3f})")
            entry_point = int(match_time * sr)
        else:
            best_intro = find_best_intro_section(sections)
            entry_point = best_intro.start_sample if best_intro else 0
            print(f"⚠️ Semantic intro entry at {entry_point / sr:.1f}s")

        y         = y[entry_point:]
        beats     = beats[beats >= entry_point] - entry_point
        downbeats = downbeats[downbeats >= entry_point] - entry_point
        if current_stems:
            current_stems['vocals']       = current_stems['vocals'][entry_point:]
            current_stems['instrumental'] = current_stems['instrumental'][entry_point:]

        # --- Tempo alignment ---
        if tempo_match and abs(bpm - mixed_bpm) > 2:
            stretch_factor = bpm / mixed_bpm
            print(f"⏱️  Time-stretch: {bpm:.1f} → {mixed_bpm:.1f} BPM (×{stretch_factor:.3f})")
            y = time_stretch(y, sr, stretch_factor)
            beats     = (beats     / stretch_factor).astype(int)
            downbeats = (downbeats / stretch_factor).astype(int)
            if current_stems:
                from scipy import signal as scipy_signal
                n_new = int(len(current_stems['vocals']) / stretch_factor)
                current_stems['vocals'] = scipy_signal.resample(
                    current_stems['vocals'], n_new)
                current_stems['instrumental'] = scipy_signal.resample(
                    current_stems['instrumental'], n_new)

        # --- Determine section labels for blueprint selection ---
        outro_section = find_best_outro_section(classify_sections(mixed_audio, sr))
        intro_section = find_best_intro_section(sections)
        label_a = outro_section.label if outro_section else 'verse'
        label_b = intro_section.label if intro_section else 'verse'
        blueprint = _choose_blueprint(label_a, label_b)

        bar_duration = (60.0 / mixed_bpm) * 4
        fade_duration = bar_duration * blueprint['crossfade_bars']
        fade_samples  = int(fade_duration * sr)
        fade_start_a  = max(0, len(mixed_audio) - fade_samples)

        # Snap fade start to nearest downbeat
        if len(mixed_downbeats) > 0:
            valid = mixed_downbeats[mixed_downbeats <= fade_start_a + fade_samples]
            if len(valid) > 0:
                fade_start_a = int(valid[-1])

        print(f"\n🎛️  Blueprint: {label_a!r} → {label_b!r}")
        print(f"   Crossfade type:    {blueprint['crossfade_type']}")
        print(f"   Tension riser:     {blueprint['use_tension_riser']}")
        print(f"   HP filter sweep:   {blueprint['filter_sweep_a']}")
        print(f"   Bass swap:         {blueprint['bass_swap']}")
        print(f"   Fade duration:     {fade_duration:.1f}s ({blueprint['crossfade_bars']} bars)")

        # --- Tension riser (loop-halving) ---
        riser = None
        if blueprint['use_tension_riser'] and len(mixed_beats) > 0:
            riser_start = max(0, fade_start_a - int(2 * bar_duration * sr))
            print(f"🔄 Building tension riser at {riser_start / sr:.1f}s...")
            riser = looper.build_riser(
                mixed_audio, mixed_beats,
                loop_start_sample=riser_start,
                n_bars=2, n_halvings=4
            )
            if len(riser) > 0:
                # Insert riser into mixed_audio before the transition point
                riser_end = riser_start + len(riser)
                riser_end = min(riser_end, fade_start_a)
                mixed_audio = np.concatenate([
                    mixed_audio[:riser_start],
                    riser[:riser_end - riser_start],
                    mixed_audio[riser_end:]
                ])
                # Recalculate fade_start after possible length change
                fade_start_a = max(0, len(mixed_audio) - fade_samples)

        # --- HP filter sweep on Track A's exit ---
        if blueprint['filter_sweep_a']:
            sweep_region = mixed_audio[fade_start_a:]
            swept = highpass_sweep(sweep_region, sr,
                                   start_cutoff_hz=20.0,
                                   end_cutoff_hz=2000.0,
                                   curve='exponential')
            mixed_audio = np.concatenate([mixed_audio[:fade_start_a], swept])

        # --- Bass swap (hard-cut bass at downbeat) ---
        if blueprint['bass_swap']:
            print("🔊 Executing bass swap at downbeat...")
            # Pad y to match length needed for swap
            pad_amount = max(0, len(mixed_audio) - len(y))
            y_padded = np.pad(y, (0, pad_amount))
            mixed_audio = dsp_bass_swap(mixed_audio, y_padded,
                                        swap_sample=fade_start_a, sr=sr)

        # --- Final crossfade ---
        if blueprint['crossfade_type'] == 'stem_aware':
            mixed_audio = _stem_aware_crossfade(
                mixed_audio, y, fade_start_a, fade_samples,
                prev_stems, current_stems, sr
            )
        else:
            mixed_audio = _beat_synced_crossfade(
                mixed_audio, y, fade_start_a, fade_samples
            )

        # Update beat tracking state
        offset = len(mixed_audio) - len(y) + fade_start_a
        mixed_beats     = np.concatenate([mixed_beats[mixed_beats < fade_start_a],
                                           beats + offset])
        mixed_downbeats = np.concatenate([mixed_downbeats[mixed_downbeats < fade_start_a],
                                           downbeats + offset])
        mixed_bpm  = bpm
        mixed_key  = key
        mixed_mode = mode
        prev_stems = current_stems

        print(f"✅ Mixed! Total: {len(mixed_audio) / sr / 60:.2f} min")

    # ------------------------------------------------------------------ #
    # Step 5 — Export MP3 in-memory (no intermediate WAV on disk)
    # ------------------------------------------------------------------ #
    print(f"\n💾 Exporting (in-memory, 320k MP3)...")
    mp3_path = os.path.join(upload_dir, session, "mixed_output.mp3")
    _export_mp3(mixed_audio, sr, mp3_path, bitrate="320k")
    print(f"✅ Complete! {len(mixed_audio) / sr / 60:.2f} minutes → {mp3_path}")
    return mp3_path
