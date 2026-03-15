import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import pandas as pd
import time


sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id='962972e977db4cebbb88b1848df305d3',
    client_secret='df1b4f4ddf6b4774a61849a3f55fc802'
))

def fetch_spotify_audio_features(track_ids):
    features = {}
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        try:
            feature_list = sp.audio_features(batch)
            if feature_list is None:
                print(f"⚠️ Received None for IDs: {batch}")
                continue
            for feat in feature_list:
                if feat:
                    features[feat['id']] = {
                        'bpm': feat['tempo'],
                        'energy': feat['energy'],
                        'key': feat['key'],
                        'loudness': feat['loudness'],
                        'mode': feat['mode']
                    }
                else:
                    print(f"⚠️ No feature data for some tracks in {batch}")
        except spotipy.SpotifyException as e:
            print(f"⚠️ Spotify API error on batch {batch}: {e}")
            # Try fetching individually as fallback
            for tid in batch:
                try:
                    single_feat = sp.audio_features([tid])[0]
                    if single_feat:
                        features[tid] = {
                            'bpm': single_feat['tempo'],
                            'energy': single_feat['energy'],
                            'key': single_feat['key'],
                            'loudness': single_feat['loudness'],
                            'mode': single_feat['mode']
                        }
                    else:
                        print(f"⚠️ No data for track {tid}")
                except Exception as ie:
                    print(f"⚠️ Error for track {tid}: {ie}")
            continue
    return features



def key_to_string(key, mode):
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if key == -1:
        return 'Unknown'
    return f"{keys[key]} {'major' if mode == 1 else 'minor'}"

def get_spotify_track_id(artist_name, track_title):
    query = f"track:{track_title} artist:{artist_name}"
    result = sp.search(q=query, type='track', limit=1)
    items = result.get('tracks', {}).get('items', [])
    if not items:
        print(f"⚠️ Not found: {artist_name} - {track_title}")
        return None
    return items[0]['id']



def parse_artist_track(filename):
    base = filename.split('/')[-1].rsplit('.', 1)[0]
    parts = base.split('_')
    if len(parts) < 2:
        return None, None
    artist = parts[0].replace('-', ' ').title()
    track = ' '.join(parts[1:]).replace('-', ' ').title()
    return artist, track


def benchmark_with_spotify_via_search(local_results_csv, output_csv='benchmark_report.csv'):
    df = pd.read_csv(local_results_csv)
    records = []

    for _, row in df.iterrows():
        filename = row['filename']
        artist, track = parse_artist_track(filename)
        if not artist or not track:
            print(f"⚠️ Cannot parse artist/track from {filename}")
            continue

        track_id = get_spotify_track_id(artist, track)
        if not track_id:
            continue

        spotify_feats = fetch_spotify_audio_features([track_id])
        sf = spotify_feats.get(track_id)
        if not sf:
            print(f"⚠️ No features for Spotify ID {track_id}")
            continue

        local_key = row.get('key', 'Unknown')
        spotify_key = key_to_string(sf['key'], sf['mode'])

        record = {
            'filename': filename,
            'local_bpm': row.get('bpm'),
            'spotify_bpm': sf['bpm'],
            'bpm_error': abs(row.get('bpm', 0) - sf['bpm']) if row.get('bpm') else None,
            'local_energy': row.get('energy'),
            'spotify_energy': sf['energy'],
            'energy_diff': abs(row.get('energy', 0) - sf['energy']) if row.get('energy') is not None else None,
            'local_key': local_key,
            'spotify_key': spotify_key,
            'local_loudness': row.get('loudness'),
            'spotify_loudness': sf['loudness'],
            'loudness_diff': abs(row.get('loudness', 0) - sf['loudness']) if row.get('loudness') is not None else None,
        }
        records.append(record)

        # To respect API rate limits
        time.sleep(0.2)

    result_df = pd.DataFrame(records)
    result_df.to_csv(output_csv, index=False)
    print(f"Benchmark report saved to {output_csv}")
    if not result_df.empty:
        print(result_df.describe())
    else:
        print("⚠️ No results to describe — the result dataframe is empty.")



if __name__ == "__main__":
    # Adjust CSV path as needed relative to backend folder
    benchmark_with_spotify_via_search('validation_results/combined_features.csv')
