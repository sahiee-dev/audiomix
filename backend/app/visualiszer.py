import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List
import os

class MixVisualizer:
    """
    Generate publication-quality visualizations for research paper
    """
    
    @staticmethod
    def plot_waveform_with_transitions(y, sr, transition_points, output_path):
        """
        Plot waveform showing transition points
        """
        fig, ax = plt.subplots(figsize=(16, 4))
        
        # Plot waveform
        time = np.arange(len(y)) / sr
        ax.plot(time, y, color='#2E86AB', linewidth=0.5, alpha=0.7)
        
        # Mark transition points
        for trans_time in transition_points:
            ax.axvline(trans_time / sr, color='#A23B72', linestyle='--', 
                      linewidth=2, label='Transition' if trans_time == transition_points[0] else '')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Mix Waveform with Transition Points', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved waveform: {output_path}")
    
    @staticmethod
    def plot_spectrogram_transition(y, sr, transition_start, transition_end, output_path):
        """
        Plot spectrogram focusing on transition region
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract transition region (+/- 5 seconds)
        buffer = 5 * sr
        start = max(0, transition_start - buffer)
        end = min(len(y), transition_end + buffer)
        y_region = y[start:end]
        
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_region)), ref=np.max)
        
        # Plot
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax, cmap='viridis')
        
        # Mark transition boundaries
        trans_start_rel = (transition_start - start) / sr
        trans_end_rel = (transition_end - start) / sr
        ax.axvline(trans_start_rel, color='red', linestyle='--', linewidth=2, label='Transition Start')
        ax.axvline(trans_end_rel, color='yellow', linestyle='--', linewidth=2, label='Transition End')
        
        ax.set_title('Spectrogram During Transition', fontsize=14, fontweight='bold')
        ax.legend()
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved spectrogram: {output_path}")
    
    @staticmethod
    def plot_beat_alignment(beats_a, beats_b, transition_point, sr, output_path):
        """
        Visualize beat alignment across transition
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Convert to time
        beats_a_time = beats_a / sr
        beats_b_time = beats_b / sr
        trans_time = transition_point / sr
        
        # Focus on ±10 seconds around transition
        window = 10
        mask_a = (beats_a_time > trans_time - window) & (beats_a_time < trans_time)
        mask_b = (beats_b_time > trans_time) & (beats_b_time < trans_time + window)
        
        # Plot beats
        ax.scatter(beats_a_time[mask_a], np.ones(mask_a.sum()), 
                  color='#2E86AB', s=100, marker='|', linewidths=3, label='Track A Beats')
        ax.scatter(beats_b_time[mask_b], np.zeros(mask_b.sum()), 
                  color='#A23B72', s=100, marker='|', linewidths=3, label='Track B Beats')
        
        # Mark transition
        ax.axvline(trans_time, color='green', linestyle='--', linewidth=2, label='Transition Point')
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Track B', 'Track A'])
        ax.set_title('Beat Alignment Across Transition', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved beat alignment: {output_path}")
    
    @staticmethod
    def plot_pattern_match_scores(match_scores, match_times, output_path):
        """
        Plot pattern matching scores across scanned positions
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Plot scores
        ax.plot(match_times, match_scores, marker='o', linewidth=2, 
               markersize=8, color='#2E86AB', label='Match Score')
        
        # Mark threshold
        ax.axhline(0.65, color='orange', linestyle='--', linewidth=2, 
                  label='Threshold (0.65)')
        
        # Mark best match
        best_idx = np.argmax(match_scores)
        ax.scatter(match_times[best_idx], match_scores[best_idx], 
                  color='red', s=200, marker='*', zorder=5, label='Best Match')
        
        ax.set_xlabel('Time in Next Track (seconds)', fontsize=12)
        ax.set_ylabel('Pattern Match Score', fontsize=12)
        ax.set_title('Pattern Matching Analysis', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved pattern match scores: {output_path}")
    
    @staticmethod
    def plot_feature_comparison(features_a, features_b, output_path):
        """
        Radar chart comparing audio features of two tracks
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        
        # Metrics to compare
        metrics = ['BPM\nMatch', 'Energy\nMatch', 'Timbre\nMatch', 
                  'Harmony\nMatch', 'Rhythm\nMatch', 'Stability']
        
        # Normalize scores to 0-1 (assuming features_a/b are dicts with scores)
        values_a = [features_a.get(m.replace('\n', ' '), 0.5) for m in metrics]
        values_b = [features_b.get(m.replace('\n', ' '), 0.5) for m in metrics]
        
        # Close the plot by appending first value
        values_a += values_a[:1]
        values_b += values_b[:1]
        
        # Angles
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, values_a, 'o-', linewidth=2, color='#2E86AB', label='Track A Outro')
        ax.fill(angles, values_a, alpha=0.25, color='#2E86AB')
        ax.plot(angles, values_b, 'o-', linewidth=2, color='#A23B72', label='Track B Entry')
        ax.fill(angles, values_b, alpha=0.25, color='#A23B72')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, size=10)
        ax.set_ylim(0, 1)
        ax.set_title('Audio Feature Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Saved feature comparison: {output_path}")
