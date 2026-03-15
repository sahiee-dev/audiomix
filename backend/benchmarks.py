import pandas as pd
import numpy as np

def create_data_dependent_synthetic_benchmarks(input_csv, output_csv, num_samples=50):
    # Load your real feature dataset
    df = pd.read_csv(input_csv)
    
    # Numerical features to model
    features_num = ['bpm', 'loudness', 'energy']
    
    # Calculate mean and std of each numeric feature (ignore NaNs)
    stats = {}
    for feature in features_num:
        stats[feature] = {
            'mean': df[feature].dropna().mean(),
            'std': df[feature].dropna().std()
        }
    
    # Get categorical feature distribution for 'key'
    key_counts = df['key'].value_counts(normalize=True).to_dict()
    keys = list(key_counts.keys())
    probs = list(key_counts.values())
    
    synthetic_data = []
    for i in range(num_samples):
        # Sample numeric features from normal distributions constrained sensibly
        bpm = max(30, np.random.normal(stats['bpm']['mean'], stats['bpm']['std']))
        loudness = np.random.normal(stats['loudness']['mean'], stats['loudness']['std'])
        energy = min(1.0, max(0.0, np.random.normal(stats['energy']['mean'], stats['energy']['std'])))
        
        # Sample key from empirical distribution
        key = np.random.choice(keys, p=probs)
        
        synthetic_data.append({
            'filename': f'synthetic_track_{i+1}.mp3',
            'bpm': round(bpm, 1),
            'loudness': round(loudness, 1),
            'energy': round(energy, 2),
            'key': key
        })
    
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv(output_csv, index=False)
    print(f"Synthetic benchmark dataset saved to {output_csv}")

if __name__ == "__main__":
    create_data_dependent_synthetic_benchmarks(
        input_csv='validation_results/combined_features.csv',
        output_csv='validation_results/synthetic_benchmarks.csv',
        num_samples=50
    )
