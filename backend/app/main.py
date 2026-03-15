from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router
from .config import UPLOAD_DIR
import os

app = FastAPI(title="Smart Mix Orderer - API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def health():
    return {"status": "ok"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
