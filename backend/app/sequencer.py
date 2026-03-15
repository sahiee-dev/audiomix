from typing import List, Dict
import math

def weight_features(track):
    """Calculate composite weight for high/low ordering"""
    return track['bpm'] * 0.6 + track['energy'] * 100.0 * 0.4

def order_high_to_low(tracks):
    """Order from high energy to low energy"""
    return sorted(tracks, key=lambda t: weight_features(t), reverse=True)

def order_low_to_high(tracks):
    """Order from low energy to high energy"""
    return sorted(tracks, key=lambda t: weight_features(t), reverse=False)

def distance(a, b):
    """Calculate multi-dimensional distance between tracks"""
    bpm_diff = abs(a['bpm'] - b['bpm'])
    energy_diff = abs(a['energy'] - b['energy'])
    
    # Penalize key mismatches
    key_penalty = 0
    if a.get('key') and b.get('key') and a['key'] != b['key']:
        key_penalty = 5
    
    # Consider danceability if available
    dance_diff = 0
    if a.get('danceability') and b.get('danceability'):
        dance_diff = abs(a['danceability'] - b['danceability']) * 20
    
    return bpm_diff * 0.5 + energy_diff * 100 * 0.3 + key_penalty + dance_diff * 0.2

def smooth_greedy_order(tracks):
    """Create smooth transitions using greedy nearest-neighbor"""
    if not tracks:
        return []
    
    remaining = tracks.copy()
    
    # Start with a medium-energy track
    bpms = sorted(remaining, key=lambda t: t['bpm'])
    start_idx = len(bpms) // 2
    start = bpms[start_idx]
    
    ordered = [start]
    remaining.remove(start)
    
    while remaining:
        last = ordered[-1]
        # Find the closest track
        nxt = min(remaining, key=lambda t: distance(last, t))
        ordered.append(nxt)
        remaining.remove(nxt)
    
    return ordered

def order_tracks(tracks: List[Dict], mode: str):
    """Main ordering function"""
    if mode == "high_to_low":
        return order_high_to_low(tracks)
    elif mode == "low_to_high":
        return order_low_to_high(tracks)
    else:  # smooth
        return smooth_greedy_order(tracks)
