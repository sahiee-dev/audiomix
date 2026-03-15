import numpy as np
import librosa
from typing import Dict, Tuple, Optional

class AutoMixEngine:
    """Apple Music AutoMix-style intelligent transition engine"""
    
    def __init__(self):
        self.min_bpm_diff = 20  # Skip mixing if BPM diff > 20
        self.max_energy_diff = 0.35  # Skip if energy too different
        self.phrase_length = 16  # 16-beat phrases
        
    def should_mix(self, track_a: Dict, track_b: Dict) -> Tuple[bool, str]:
        """
        Determine if tracks should be mixed or hard-cut
        
        Returns:
            (should_mix: bool, reason: str)
        """
        # Check BPM compatibility
        bpm_diff = abs(track_a['bpm'] - track_b['bpm'])
        if bpm_diff > self.min_bpm_diff:
            return False, f"BPM too different ({bpm_diff:.1f})"
        
        # Check energy compatibility
        energy_diff = abs(track_a['energy'] - track_b['energy'])
        if energy_diff > self.max_energy_diff:
            return False, f"Energy mismatch ({energy_diff:.2f})"
        
        # Check duration (skip very short tracks)
        if track_a.get('duration', 0) < 30 or track_b.get('duration', 0) < 30:
            return False, "Track too short"
        
        # All checks passed
        return True, "Compatible"
    
    def get_transition_type(self, track_a: Dict, track_b: Dict) -> str:
        """
        Choose transition type based on track characteristics
        
        Returns:
            'extended' | 'standard' | 'quick' | 'cut'
        """
        bpm_diff = abs(track_a['bpm'] - track_b['bpm'])
        energy_a = track_a.get('energy', 0.5)
        energy_b = track_b.get('energy', 0.5)
        
        # Extended mix: similar BPM, high energy both tracks
        if bpm_diff < 3 and energy_a > 0.6 and energy_b > 0.6:
            return 'extended'  # 16-bar overlap
        
        # Standard: compatible tracks, normal energy
        elif bpm_diff < 8:
            return 'standard'  # 8-bar overlap
        
        # Quick: larger BPM difference but still compatible
        elif bpm_diff < 15:
            return 'quick'  # 4-bar overlap
        
        # Cut: incompatible but user forced it
        else:
            return 'cut'  # No overlap, just gap fill
    
    def find_mix_points(self, y, sr, beats, downbeats, transition_type: str) -> Tuple[int, int]:
        """
        Find optimal entry/exit points for mixing
        
        Returns:
            (exit_point_samples, entry_point_samples)
        """
        if transition_type == 'extended':
            bars_overlap = 16  # 16 bars
        elif transition_type == 'standard':
            bars_overlap = 8
        elif transition_type == 'quick':
            bars_overlap = 4
        else:
            bars_overlap = 0
        
        # Calculate beats needed for overlap
        beats_overlap = bars_overlap * 4  # 4 beats per bar
        
        # Exit point: last 16-bar phrase of track A
        if len(downbeats) > bars_overlap:
            exit_point = downbeats[-bars_overlap]  # Start of last N bars
        else:
            exit_point = max(0, len(y) - int(bars_overlap * 2 * sr))
        
        # Entry point: skip silence and find first strong downbeat
        entry_point = self._find_intro_end(y, sr, downbeats)
        
        return int(exit_point), int(entry_point)
    
    def _find_intro_end(self, y, sr, downbeats) -> int:
        """
        Find where the intro ends (remove silence/quiet parts)
        """
        # Calculate RMS energy in 2-second windows
        hop_length = sr * 2  # 2-second windows
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Find first frame above 25% of max energy
        threshold = np.max(rms) * 0.25
        intro_end_frame = np.argmax(rms > threshold)
        intro_end_sample = intro_end_frame * hop_length
        
        # Snap to nearest downbeat after intro
        if len(downbeats) > 0:
            valid_downbeats = downbeats[downbeats >= intro_end_sample]
            if len(valid_downbeats) > 0:
                return int(valid_downbeats[0])
        
        return int(intro_end_sample)
    
    def calculate_crossfade_duration(self, transition_type: str, bpm: float) -> float:
        """
        Calculate optimal crossfade duration based on BPM and transition type
        
        Returns:
            Duration in seconds
        """
        # Calculate one bar duration
        beat_duration = 60.0 / bpm  # seconds per beat
        bar_duration = beat_duration * 4  # 4 beats per bar
        
        if transition_type == 'extended':
            return bar_duration * 16  # 16 bars
        elif transition_type == 'standard':
            return bar_duration * 8  # 8 bars
        elif transition_type == 'quick':
            return bar_duration * 4  # 4 bars
        else:
            return bar_duration * 2  # 2 bars minimum
