import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

class TransitionTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.features = []
        self.labels = []
    
    def load_labeled_data(self, json_path='labeled_transitions.json'):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            self.features.append(np.array(item['features']))
            self.labels.append(item['quality'])
        
        print(f"Loaded {len(self.features)} transition examples")
    
    def train(self):
        X = np.array(self.features)
        y = np.array(self.labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def save(self, out_path='models/transition_predictor.pkl'):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        joblib.dump(self.model, out_path)
        print(f"Model saved to {out_path}")

if __name__ == "__main__":
    trainer = TransitionTrainer()
    trainer.load_labeled_data()
    trainer.train()
    trainer.save()
