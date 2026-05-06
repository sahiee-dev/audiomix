"""
api.py — FastAPI router: upload, analyze, order, mix, session management.
"""

import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from .audio_utils import extract_features
from .config import UPLOAD_DIR, ALLOWED_AUDIO_TYPES, MAX_UPLOAD_MB
from .mixer import mix_tracks_professional
from .schemas import (
    HealthResponse,
    MixRequest,
    MixResponse,
    OrderRequest,
    OrderResponse,
    SessionInfo,
    TrackFeatures,
)
from .sequencer import order_tracks
from .export_utils import generate_m3u

router = APIRouter()

_VERSION = "1.0.0"
_MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# ─── Health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe — returns 200 when the API is up."""
    return HealthResponse(status="ok", version=_VERSION)


# ─── Sessions ─────────────────────────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionInfo], tags=["sessions"])
def list_sessions() -> List[SessionInfo]:
    """Return metadata for all active upload sessions."""
    sessions: List[SessionInfo] = []
    if not os.path.isdir(UPLOAD_DIR):
        return sessions
    for session_id in os.listdir(UPLOAD_DIR):
        session_dir = os.path.join(UPLOAD_DIR, session_id)
        if not os.path.isdir(session_dir):
            continue
        files = [
            f for f in os.listdir(session_dir)
            if os.path.isfile(os.path.join(session_dir, f))
        ]
        total_bytes = sum(
            os.path.getsize(os.path.join(session_dir, f)) for f in files
        )
        sessions.append(
            SessionInfo(
                session_id=session_id,
                file_count=len(files),
                total_size_bytes=total_bytes,
            )
        )
    return sessions


@router.delete("/sessions/{session}", tags=["sessions"])
def delete_session(session: str) -> JSONResponse:
    """Delete all files belonging to *session* and free disk space."""
    session_dir = os.path.join(UPLOAD_DIR, session)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")
    shutil.rmtree(session_dir, ignore_errors=True)
    return JSONResponse({"deleted": session})


# ─── Upload ───────────────────────────────────────────────────────────────────

@router.post("/upload", tags=["pipeline"])
async def upload(files: List[UploadFile] = File(...)):
    """Upload one or more audio files and create a session."""
    session = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIR, session)
    os.makedirs(session_dir, exist_ok=True)

    saved = []
    for f in files:
        content = await f.read()
        if len(content) > _MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{f.filename} exceeds the {MAX_UPLOAD_MB} MB upload limit.",
            )
        dest = os.path.join(session_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(content)
        saved.append({"filename": f.filename, "path": dest})

    return {"session": session, "files": saved}


# ─── Analyze ──────────────────────────────────────────────────────────────────

@router.post("/analyze", tags=["pipeline"])
async def analyze(session: str):
    """Extract audio features for every file in *session*."""
    session_dir = os.path.join(UPLOAD_DIR, session)
    if not os.path.isdir(session_dir):
        raise HTTPException(status_code=404, detail="Session not found")

    results = []
    for fn in os.listdir(session_dir):
        path = os.path.join(session_dir, fn)
        if not os.path.isfile(path):
            continue
        try:
            results.append(extract_features(path))
        except Exception as exc:
            results.append({"filename": fn, "error": str(exc)})

    return {"session": session, "tracks": results}


# ─── Order ────────────────────────────────────────────────────────────────────

@router.post("/order", response_model=OrderResponse, tags=["pipeline"])
async def order(req: OrderRequest) -> OrderResponse:
    """Sort tracks according to *mode*."""
    ordered = order_tracks(
        [t.dict() if hasattr(t, "dict") else t for t in req.tracks],
        req.mode,
    )
    return OrderResponse(ordered=ordered)


# ─── Mix ──────────────────────────────────────────────────────────────────────

@router.post("/mix", response_model=MixResponse, tags=["pipeline"])
async def mix(
    req: MixRequest,
    session: str,
    normalize: bool = True,
    tempo_match: bool = True,
    harmonic_match: bool = True,
    crossfade_duration: float = 8.0,
    entry_method: str = "high_energy",
    use_stem_separation: bool = False,
) -> MixResponse:
    """Mix the ordered tracks into a single 320 kbps MP3."""
    try:
        tracks_in_order = [
            t.dict() if hasattr(t, "dict") else t for t in req.tracks
        ]
        mixed_file = mix_tracks_professional(
            tracks_in_order,
            UPLOAD_DIR,
            session,
            normalize=normalize,
            tempo_match=tempo_match,
            harmonic_match=harmonic_match,
            crossfade_duration=crossfade_duration,
            entry_method=entry_method,
            use_stem_separation=use_stem_separation,
        )
        return MixResponse(
            status="success",
            mixed_file="mixed_output.mp3",
            download_url=f"/api/download/{session}/mixed_output.mp3",
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


# ─── File serving ─────────────────────────────────────────────────────────────

@router.get("/preview/{session}/{filename}", tags=["files"])
def preview(session: str, filename: str):
    """Stream an audio file for in-browser playback."""
    path = os.path.join(UPLOAD_DIR, session, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)


@router.get("/download/{session}/{filename}", tags=["files"])
def download_file(session: str, filename: str):
    """Download an audio file (mixed output or original)."""
    path = os.path.join(UPLOAD_DIR, session, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/mpeg", filename=filename)
