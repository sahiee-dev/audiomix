import numpy as np
import librosa
from scipy import signal
from typing import Dict

class SpectralAnalyzer:
    """
    Advanced spectral analysis for research-grade audio matching
    """
    
    @staticmethod
    def compute_spectral_features(y, sr):
        """
        Compute comprehensive spectral features
        
        Returns dict with 20+ spectral metrics
        """
        # Short-time Fourier transform
        D = librosa.stft(y)
        S = np.abs(D)
        
        features = {}
        
        # 1. Spectral Centroid (brightness)
        features['centroid_mean'] = np.mean(librosa.feature.spectral_centroid(S=S, sr=sr))
        features['centroid_std'] = np.std(librosa.feature.spectral_centroid(S=S, sr=sr))
        
        # 2. Spectral Rolloff (frequency content)
        features['rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
        
        # 3. Spectral Bandwidth (spread)
        features['bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(S=S, sr=sr))
        
        # 4. Spectral Contrast (texture)
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        features['contrast_mean'] = np.mean(contrast)
        features['contrast_std'] = np.std(contrast)
        
        # 5. Spectral Flatness (noisiness)
        features['flatness_mean'] = np.mean(librosa.feature.spectral_flatness(S=S))
        
        # 6. Zero Crossing Rate
        features['zcr_mean'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 7. MFCCs (timbral texture)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i}_std'] = np.std(mfcc[i])
        
        # 8. Chroma features (harmonic content)
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # 9. Tonnetz (tonal centroid)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz)
        
        return features
    
    @staticmethod
    def compute_spectral_similarity(features1: Dict, features2: Dict):
        """
        Compute similarity between two spectral feature sets
        
        Returns: similarity score 0-1
        """
        # Extract common features
        keys = set(features1.keys()) & set(features2.keys())
        
        # Compute euclidean distance in normalized feature space
        diffs = []
        for key in keys:
            # Normalize by max value
            max_val = max(abs(features1[key]), abs(features2[key]), 1e-6)
            diff = abs(features1[key] - features2[key]) / max_val
            diffs.append(diff)
        
        # Convert distance to similarity (0 = different, 1 = identical)
        avg_diff = np.mean(diffs)
        similarity = np.exp(-avg_diff)  # Exponential decay
        
        return similarity
