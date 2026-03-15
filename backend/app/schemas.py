from pydantic import BaseModel
from typing import List, Optional

class TrackFeatures(BaseModel):
    filename: str
    bpm: float
    key: Optional[str] = None
    energy: float
    duration: float
    spectral_centroid: Optional[float] = None
    mood: Optional[str] = None
    danceability: Optional[float] = None
    loudness: Optional[float] = None

class OrderRequest(BaseModel):
    tracks: List[TrackFeatures]
    mode: str  # "high_to_low" | "low_to_high" | "smooth"

class OrderResponse(BaseModel):
    ordered: List[TrackFeatures]
