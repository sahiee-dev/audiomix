"""
schemas.py — Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


# ─── Track Models ─────────────────────────────────────────────────────────────

class TrackFeatures(BaseModel):
    """Audio features extracted from a single track."""
    filename: str
    bpm: float
    key: Optional[str] = None
    energy: float
    duration: float
    spectral_centroid: Optional[float] = None
    mood: Optional[str] = None
    danceability: Optional[float] = None
    loudness: Optional[float] = None
    genre: Optional[str] = Field(default="general", description="Genre hint used by the pattern matcher.")


# ─── Ordering Models ──────────────────────────────────────────────────────────

class OrderRequest(BaseModel):
    """Request body for the /order endpoint."""
    tracks: List[TrackFeatures]
    mode: str = Field(
        ...,
        description="Ordering strategy: 'high_to_low' | 'low_to_high' | 'smooth' | 'recommended' | 'manual'",
    )


class OrderResponse(BaseModel):
    """Response body for the /order endpoint."""
    ordered: List[TrackFeatures]


# ─── Mix Models ───────────────────────────────────────────────────────────────

class MixRequest(BaseModel):
    """Request body for the /mix endpoint."""
    tracks: List[TrackFeatures]
    mode: str = "recommended"


class MixResponse(BaseModel):
    """Response body for a successful /mix call."""
    status: str
    mixed_file: str
    download_url: str


# ─── Session Models ───────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    """Metadata about an active upload session."""
    session_id: str
    file_count: int
    total_size_bytes: int


# ─── Health Model ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str
    version: str
