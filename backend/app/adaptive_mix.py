import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import joblib
from .align import align_tracks_by_beats
from .pattern_matcher import PatternMatcher
from .audio_normalizer import normalize_tracks_before_mixing

def extract_transition_features(track_a, track_b):
    """
    Extract features for ML from two tracks
    Example features:
      - BPM diff
      - Energy diff
      - Spectral centroid diff
      - Chroma similarity
      - Key similarity (optional)
    """
    bpm_diff = abs(track_a['bpm'] - track_b['bpm'])
    energy_diff = abs(track_a.get('energy', 0) - track_b.get('energy', 0))
    spectral_centroid_diff = abs(track_a.get('spectral_centroid', 0) - track_b.get('spectral_centroid', 0))
    chroma_sim = np.corrcoef(track_a.get('chroma', []), track_b.get('chroma', []))[0, 1] if track_a.get('chroma') and track_b.get('chroma') else 0
    key_compatibility = 1.0 if track_a.get('key') == track_b.get('key') else 0.5

    return [
        bpm_diff,
        energy_diff,
        spectral_centroid_diff,
        chroma_sim,
        key_compatibility
    ]

def mix_tracks_adaptive(tracks_info, upload_dir, session,
                        crossfade_base=10,
                        use_stem_separation=False):
    predictor = joblib.load('models/transition_predictor.pkl')
    sr = 44100
    
    print("🔮 Adaptive Mixing with ML")
    
    tracks_audio = []
    for track in tracks_info:
        file_path = os.path.join(upload_dir, session, track['filename'])
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        tracks_audio.append(y)
    
    # Normalize all tracks before mixing
    print("🔊 Normalizing loudness of all tracks...")
    tracks_audio = normalize_tracks_before_mixing(tracks_audio, sr)
    
    mixed_audio = None
    metadata = {'transitions': []}
    prev_track = None
    prev_audio = None
    
    for i, track in enumerate(tracks_info):
        print(f"\n🎵 Mixing track {i+1}/{len(tracks_info)}: {track['filename']}")
        
        current_audio = tracks_audio[i]
        
        if mixed_audio is None:
            mixed_audio = current_audio
            prev_track = track
            prev_audio = current_audio
            continue
        
        # Extract features from prev and current track
        features = extract_transition_features(prev_track, track)
        
        # Predict quality and get crossfade duration
        quality_pred = predictor.predict([features])[0]
        confidence = max(predictor.predict_proba([features])[0])
        
        print(f"   Transition quality prediction: {quality_pred} (confidence: {confidence:.2f})")
        
        # Adaptive crossfade duration
        if quality_pred >= 2:
            crossfade = crossfade_base
        elif quality_pred == 1:
            crossfade = crossfade_base + 2
        else:
            crossfade = crossfade_base + 4
        
        print(f"   Crossfade duration: {crossfade}s")
        
        # Implement simple crossfade, linear for now (extend with stem-aware later)
        fade_samples = int(sr * crossfade)
        
        mixed_len = len(mixed_audio)
        overlap_start = mixed_len - fade_samples
        
        fade_out_region = mixed_audio[overlap_start:]
        fade_in_region = current_audio[:fade_samples]
        
        fade_out_curve = np.linspace(1, 0, fade_samples)
        fade_in_curve = np.linspace(0, 1, fade_samples)
        
        crossfade_mix = fade_out_region * fade_out_curve + fade_in_region * fade_in_curve
        
        mixed_audio = np.concatenate([
            mixed_audio[:overlap_start],
            crossfade_mix,
            current_audio[fade_samples:]
        ])
        
        # Log metadata
        metadata['transitions'].append({
            'track_a': prev_track['filename'],
            'track_b': track['filename'],
            'predicted_quality': int(quality_pred),
            'confidence': float(confidence),
            'crossfade_duration': float(crossfade)
        })
        
        prev_audio = current_audio
        prev_track = track
    
    output_path = os.path.join(upload_dir, session, 'adaptive_mixed_output.mp3')
    sf.write(output_path.replace('.mp3', '.wav'), mixed_audio, sr)
    
    wav_audio = AudioSegment.from_wav(output_path.replace('.mp3', '.wav'))
    wav_audio.export(output_path, format="mp3", bitrate='320k')
    os.remove(output_path.replace('.mp3', '.wav'))
    
    print("✅ Adaptive mixing complete:", output_path)
    
    return {
        'status': 'success',
        'mixed_file': 'adaptive_mixed_output.mp3',
        'download_url': f'/api/download/{session}/adaptive_mixed_output.mp3',
        'metadata': metadata
    }
