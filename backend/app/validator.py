import os
import numpy as np
import librosa
import pandas as pd
from typing import Dict, List, Tuple

class DSPValidator:
    """Test BPM, key, and loudness detection accuracy"""
    
    def __init__(self):
        self.bpm_results = []
        self.key_results = []
        self.loudness_results = []
    
    def test_bpm(self, audio_path: str, ground_truth_bpm: float = None, genre: str = None):
        """
        Test BPM detection accuracy
        """
        print(f"\n🎵 Testing: {os.path.basename(audio_path)}")
        
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        
        # Method 1: Librosa (default)
        tempo_librosa, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Ensure it's a scalar
        tempo_librosa = float(tempo_librosa) if not isinstance(tempo_librosa, np.ndarray) else float(tempo_librosa[0])
        
        # Method 2: Madmom (more accurate)
        try:
            import madmom
            proc = madmom.features.beats.RNNBeatProcessor()
            act = proc(audio_path)
            beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(act)
            
            if len(beat_times) > 1:
                tempo_madmom = 60 / np.median(np.diff(beat_times))
            else:
                tempo_madmom = tempo_librosa
        except Exception as e:
            print(f"   ⚠️ Madmom failed: {e}")
            tempo_madmom = tempo_librosa
        
        # Ensure scalar
        tempo_madmom = float(tempo_madmom)
        
        # Double-time / half-time correction
        def correct_tempo(bpm, target_range=(80, 180)):
            if bpm > target_range[1]:
                return bpm / 2
            elif bpm < target_range[0]:
                return bpm * 2
            return bpm
        
        tempo_librosa_corrected = correct_tempo(tempo_librosa)
        tempo_madmom_corrected = correct_tempo(tempo_madmom)
        
        # Calculate errors if ground truth provided
        error_librosa = abs(tempo_librosa_corrected - ground_truth_bpm) if ground_truth_bpm else None
        error_madmom = abs(tempo_madmom_corrected - ground_truth_bpm) if ground_truth_bpm else None
        
        result = {
            'filename': os.path.basename(audio_path),
            'genre': genre,
            'librosa_raw': round(tempo_librosa, 2),
            'librosa_corrected': round(tempo_librosa_corrected, 2),
            'madmom_raw': round(tempo_madmom, 2),
            'madmom_corrected': round(tempo_madmom_corrected, 2),
            'ground_truth': ground_truth_bpm,
            'error_librosa': round(error_librosa, 2) if error_librosa else None,
            'error_madmom': round(error_madmom, 2) if error_madmom else None,
            'num_beats': len(beats)
        }
        
        self.bpm_results.append(result)
        
        print(f"   Librosa: {tempo_librosa:.1f} → {tempo_librosa_corrected:.1f} BPM")
        print(f"   Madmom:  {tempo_madmom:.1f} → {tempo_madmom_corrected:.1f} BPM")
        if ground_truth_bpm:
            print(f"   Truth:   {ground_truth_bpm} BPM")
            print(f"   Error:   Librosa={error_librosa:.1f}, Madmom={error_madmom:.1f}")
        
        return result

    
    def test_key(self, audio_path: str, ground_truth_key: str = None):
        """Test key detection"""
        try:
            import essentia.standard as es
            
            audio = es.MonoLoader(filename=audio_path, sampleRate=44100)()
            key_extractor = es.KeyExtractor()
            key, scale, strength = key_extractor(audio)
            
            detected = f"{key} {scale}"
            
            result = {
                'filename': os.path.basename(audio_path),
                'detected_key': detected,
                'ground_truth': ground_truth_key,
                'correct': detected == ground_truth_key if ground_truth_key else None,
                'confidence': round(strength, 3)
            }
            
            self.key_results.append(result)
            
            print(f"   Key: {detected} (confidence: {strength:.2f})")
            if ground_truth_key:
                print(f"   Truth: {ground_truth_key} {'✓' if detected == ground_truth_key else '✗'}")
            
            return result
            
        except ImportError:
            print("   ⚠️ Essentia not installed. Run: pip install essentia")
            return None
    
    def test_loudness(self, audio_path: str):
        """Test loudness normalization"""
        import pyloudnorm as pyln
        
        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y)
        
        result = {
            'filename': os.path.basename(audio_path),
            'lufs': round(loudness, 2),
            'target': -14.0,
            'deviation': round(abs(loudness - (-14.0)), 2)
        }
        
        self.loudness_results.append(result)
        
        print(f"   LUFS: {loudness:.1f} (target: -14.0, deviation: {result['deviation']:.1f})")
        
        return result
    
    def export_results(self, output_dir='validation_results'):
        """Export all results to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # BPM results
        if self.bpm_results:
            df_bpm = pd.DataFrame(self.bpm_results)
            df_bpm.to_csv(f'{output_dir}/bpm_validation.csv', index=False)
            
            print(f"\n📊 BPM Validation Results:")
            print(f"   Total songs: {len(df_bpm)}")
            if df_bpm['error_librosa'].notna().any():
                print(f"   Mean error (Librosa): {df_bpm['error_librosa'].mean():.2f} BPM")
                print(f"   Mean error (Madmom):  {df_bpm['error_madmom'].mean():.2f} BPM")
        
        # Key results
        if self.key_results:
            df_key = pd.DataFrame(self.key_results)
            df_key.to_csv(f'{output_dir}/key_validation.csv', index=False)
            
            print(f"\n🎹 Key Validation Results:")
            print(f"   Total songs: {len(df_key)}")
            if df_key['correct'].notna().any():
                accuracy = df_key['correct'].sum() / df_key['correct'].notna().sum() * 100
                print(f"   Accuracy: {accuracy:.1f}%")
        
        # Loudness results
        if self.loudness_results:
            df_loud = pd.DataFrame(self.loudness_results)
            df_loud.to_csv(f'{output_dir}/loudness_validation.csv', index=False)
            
            print(f"\n🔊 Loudness Validation Results:")
            print(f"   Mean LUFS: {df_loud['lufs'].mean():.1f}")
            print(f"   Mean deviation from -14 LUFS: {df_loud['deviation'].mean():.1f}")
        
        print(f"\n✅ Results saved to {output_dir}/")
