import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TransitionQualityPredictor:
    """
    Predict if a transition will sound good using ML
    
    Features: BPM diff, energy diff, spectral similarity, rhythm correlation
    Labels: 0 (bad), 1 (okay), 2 (good), 3 (excellent)
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        
    def extract_features(self, fp1, fp2):
        """Extract features from two fingerprints for ML prediction"""
        features = np.array([
            abs(fp1.bpm - fp2.bpm),                    # BPM difference
            abs(fp1.energy_rms - fp2.energy_rms),      # Energy difference
            abs(fp1.spectral_centroid - fp2.spectral_centroid) / 1000,  # Brightness
            np.corrcoef(fp1.chroma_profile, fp2.chroma_profile)[0, 1],  # Harmonic
            np.corrcoef(fp1.onset_pattern, fp2.onset_pattern)[0, 1],    # Rhythm
            (fp1.tempo_stability + fp2.tempo_stability) / 2,             # Stability
            np.std(fp1.chroma_profile - fp2.chroma_profile),            # Key variance
            abs(np.mean(fp1.onset_pattern) - np.mean(fp2.onset_pattern)) # Onset diff
        ])
        return features
    
    def train(self, X, y):
        """Train the model on labeled transition examples"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.trained = True
        print(f"✅ Model trained on {len(X)} examples")
        print(f"   Feature importances: {self.model.feature_importances_}")
    
    def predict(self, fp1, fp2):
        """
        Predict transition quality
        
        Returns:
            quality_score: 0-3 (bad to excellent)
            confidence: 0-1 probability
        """
        if not self.trained:
            return 1, 0.5  # Default: okay with low confidence
        
        features = self.extract_features(fp1, fp2).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        quality = self.model.predict(features_scaled)[0]
        confidence = np.max(self.model.predict_proba(features_scaled))
        
        return int(quality), float(confidence)
    
    def save(self, path):
        """Save trained model"""
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load(self, path):
        """Load trained model"""
        if os.path.exists(path):
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.trained = True
