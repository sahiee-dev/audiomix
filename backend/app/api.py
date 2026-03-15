import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from .audio_utils import extract_features
from .sequencer import order_tracks
from .config import UPLOAD_DIR
from .schemas import TrackFeatures, OrderRequest, OrderResponse
from .export_utils import generate_m3u
from .mixer import mix_tracks_professional
import uuid
import shutil

router = APIRouter()

@router.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    session = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session)
    os.makedirs(session_dir, exist_ok=True)
    for f in files:
        dest = os.path.join(session_dir, f.filename)
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved.append({"filename": f.filename, "path": dest})
    return {"session": session, "files": saved}

@router.post("/analyze")
async def analyze(session: str):
    session_dir = os.path.join(UPLOAD_DIR, session)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    results = []
    for fn in os.listdir(session_dir):
        path = os.path.join(session_dir, fn)
        try:
            feats = extract_features(path)
            results.append(feats)
        except Exception as e:
            results.append({"filename": fn, "error": str(e)})
    return {"session": session, "tracks": results}

@router.post("/order")
async def order(req: OrderRequest):
    ordered = order_tracks([t.dict() if hasattr(t, 'dict') else t for t in req.tracks], req.mode)
    return OrderResponse(ordered=ordered)

@router.get("/preview/{session}/{filename}")
def preview(session: str, filename: str):
    path = os.path.join(UPLOAD_DIR, session, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)

@router.post("/mix")
async def mix(req: OrderRequest, session: str, 
              normalize: bool = True, 
              tempo_match: bool = True,
              harmonic_match: bool = True,
              crossfade_duration: float = 8.0,
              entry_method: str = 'high_energy',
              use_stem_separation: bool = False):
    """
    Mix tracks in the EXACT order provided by frontend
    """
    try:
        from .mixer import mix_tracks_professional
        
        # Convert Pydantic models to dictionaries
        tracks_in_order = [track.dict() if hasattr(track, 'dict') else track for track in req.tracks]
        
        print(f"\n🎧 Mixing {len(tracks_in_order)} tracks in this order:")
        for i, track in enumerate(tracks_in_order):
            print(f"   {i+1}. {track['filename']}")
        
        # Mix in the exact order provided
        mixed_file = mix_tracks_professional(
            tracks_in_order,
            UPLOAD_DIR, 
            session,
            normalize=normalize,
            tempo_match=tempo_match,
            harmonic_match=harmonic_match,
            crossfade_duration=crossfade_duration,
            entry_method=entry_method,
            use_stem_separation=use_stem_separation
        )
        
        return {
            "status": "success",
            "mixed_file": "mixed_output.mp3",
            "download_url": f"/api/download/{session}/mixed_output.mp3"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{session}/{filename}")
def download_file(session: str, filename: str):
    """Download mixed audio file"""
    path = os.path.join(UPLOAD_DIR, session, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
