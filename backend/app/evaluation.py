import numpy as np
import librosa
from typing import Dict, List

class MixQualityEvaluator:
    """
    Objective metrics for evaluating mix quality
    
    Research contribution: Novel evaluation framework for automatic DJ mixing
    """
    
    @staticmethod
    def compute_transition_smoothness(y, sr, transition_start, transition_end):
        """
        Measure smoothness of transition using spectral flux
        
        Returns: smoothness score 0-1 (higher = smoother)
        """
        transition_audio = y[transition_start:transition_end]
        
        # Compute spectral flux (sudden changes in spectrum)
        S = np.abs(librosa.stft(transition_audio))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        
        # Normalize and invert (low flux = smooth)
        smoothness = 1 / (1 + np.mean(flux))
        
        return float(smoothness)
    
    @staticmethod
    def compute_beat_alignment_error(beats_a, beats_b, transition_point):
        """
        Measure beat alignment accuracy at transition point
        
        Returns: error in milliseconds
        """
        # Find beats near transition
        beats_a_near = beats_a[beats_a < transition_point]
        beats_b_near = beats_b[beats_b > transition_point]
        
        if len(beats_a_near) == 0 or len(beats_b_near) == 0:
            return 1000.0  # 1 second error (bad)
        
        last_beat_a = beats_a_near[-1]
        first_beat_b = beats_b_near[0]
        
        # Expected beat period
        if len(beats_a_near) > 1:
            beat_period = np.mean(np.diff(beats_a_near[-4:]))
        else:
            return 500.0
        
        # Compute phase error
        expected_beat_b = last_beat_a + beat_period
        error_samples = abs(first_beat_b - expected_beat_b)
        error_ms = (error_samples / 44100) * 1000
        
        return float(error_ms)
    
    @staticmethod
    def compute_loudness_continuity(y, sr, transition_start, transition_end):
        """
        Measure loudness consistency across transition
        
        Returns: continuity score 0-1
        """
        window = 2 * sr  # 2 second windows
        
        # Pre-transition loudness
        pre_audio = y[max(0, transition_start - window):transition_start]
        pre_rms = np.sqrt(np.mean(pre_audio**2))
        
        # Post-transition loudness
        post_audio = y[transition_end:transition_end + window]
        post_rms = np.sqrt(np.mean(post_audio**2))
        
        # Continuity (low difference = high continuity)
        diff = abs(pre_rms - post_rms)
        continuity = np.exp(-diff * 10)  # Exponential decay
        
        return float(continuity)
    
    @staticmethod
    def evaluate_full_mix(y, sr, transition_points: List[int]):
        """
        Comprehensive evaluation of entire mix
        
        Returns: Dict of metrics
        """
        metrics = {
            'smoothness': [],
            'beat_alignment_error_ms': [],
            'loudness_continuity': []
        }
        
        # Evaluate each transition
        for i, trans_start in enumerate(transition_points):
            trans_end = trans_start + int(8 * sr)  # Assume 8s transitions
            
            smoothness = MixQualityEvaluator.compute_transition_smoothness(
                y, sr, trans_start, trans_end
            )
            metrics['smoothness'].append(smoothness)
            
            continuity = MixQualityEvaluator.compute_loudness_continuity(
                y, sr, trans_start, trans_end
            )
            metrics['loudness_continuity'].append(continuity)
        
        # Compute averages
        return {
            'avg_smoothness': np.mean(metrics['smoothness']),
            'avg_loudness_continuity': np.mean(metrics['loudness_continuity']),
            'smoothness_std': np.std(metrics['smoothness']),
            'num_transitions': len(transition_points)
        }
