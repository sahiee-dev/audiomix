import numpy as np
import librosa
from typing import Dict, Tuple

class SemanticAnalyzer:
    """
    Detect genre, mood, and semantic properties of music
    
    Research contribution: Multi-modal audio understanding
    """
    
    def __init__(self):
        # Genre fingerprints (simplified - use pre-trained models for production)
        self.genre_profiles = {
            'hip-hop': {'bpm_range': (70, 100), 'rhythm_complexity': 'high'},
            'house': {'bpm_range': (120, 130), 'rhythm_complexity': 'low'},
            'techno': {'bpm_range': (125, 135), 'rhythm_complexity': 'medium'},
            'pop': {'bpm_range': (100, 130), 'rhythm_complexity': 'medium'},
            'rock': {'bpm_range': (110, 140), 'rhythm_complexity': 'high'}
        }
    
    def detect_genre(self, y, sr, bpm):
        """
        Detect genre using audio features
        
        Returns: (genre, confidence)
        """
        # Extract features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        onset_density = len(librosa.onset.onset_detect(y=y, sr=sr)) / (len(y) / sr)
        
        # Simple rule-based classification (replace with ML model)
        if 70 <= bpm <= 100 and onset_density > 4:
            return 'hip-hop', 0.75
        elif 120 <= bpm <= 130 and onset_density < 3:
            return 'house', 0.80
        elif 125 <= bpm <= 135 and spectral_centroid > 2000:
            return 'techno', 0.70
        elif spectral_centroid < 1500:
            return 'rock', 0.60
        else:
            return 'pop', 0.50
    
    def detect_mood(self, y, sr):
        """
        Detect mood/emotion using valence and arousal
        
        Returns: (mood, valence, arousal)
        """
        # Arousal: Energy level
        energy = np.mean(librosa.feature.rms(y=y))
        arousal = np.clip(energy * 5, 0, 1)  # Normalize to 0-1
        
        # Valence: Major/minor (simplified)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        # Major chords have more weight on 0, 4, 7 (C, E, G)
        major_weight = np.mean(chroma[[0, 4, 7], :])
        # Minor chords have more weight on 0, 3, 7 (C, Eb, G)
        minor_weight = np.mean(chroma[[0, 3, 7], :])
        valence = major_weight / (major_weight + minor_weight + 1e-6)
        
        # Map to mood categories
        if arousal > 0.6 and valence > 0.6:
            mood = 'energetic'
        elif arousal > 0.6 and valence < 0.4:
            mood = 'aggressive'
        elif arousal < 0.4 and valence > 0.6:
            mood = 'calm'
        elif arousal < 0.4 and valence < 0.4:
            mood = 'sad'
        else:
            mood = 'neutral'
        
        return mood, float(valence), float(arousal)
