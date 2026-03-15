import os
from typing import List, Dict

def generate_m3u(tracks: List[Dict], session: str, upload_dir: str) -> str:
    """Generate M3U playlist content"""
    lines = ["#EXTM3U\n"]
    for track in tracks:
        duration = int(track.get('duration', 0))
        filename = track['filename']
        path = os.path.join(upload_dir, session, filename)
        lines.append(f"#EXTINF:{duration},{filename}\n")
        lines.append(f"{path}\n")
    return "".join(lines)
