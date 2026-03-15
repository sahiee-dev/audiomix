import pandas as pd

# Load each CSV
bpm_df = pd.read_csv('/Users/lulu/Desktop/audiomix/backend/validation_results/bpm_validation.csv')
key_df = pd.read_csv('/Users/lulu/Desktop/audiomix/backend/validation_results/key_validation.csv')
loudness_df = pd.read_csv('/Users/lulu/Desktop/audiomix/backend/validation_results/loudness_validation.csv')

# Merge on filename, using outer join to include all rows
merged_df = pd.merge(bpm_df, key_df[['filename', 'detected_key']], on='filename', how='outer')
merged_df = pd.merge(merged_df, loudness_df[['filename', 'lufs']], on='filename', how='outer')

# Rename columns if needed to match spotify_benchmark script expectations
merged_df.rename(columns={
    'detected_key': 'key',
    'lufs': 'loudness',
    'librosa_corrected': 'bpm'  # or 'madmom_corrected' if preferred
}, inplace=True)

# For energy, if missing, fill with NaN or estimates
if 'energy' not in merged_df.columns:
    merged_df['energy'] = pd.NA

# Save combined CSV
combined_csv_path = '/Users/lulu/Desktop/audiomix/backend/validation_results/combined_features.csv'
merged_df.to_csv(combined_csv_path, index=False)

print(f"Combined CSV saved to {combined_csv_path}")
print(merged_df.head())
