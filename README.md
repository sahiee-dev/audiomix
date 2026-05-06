# AudioMix 🎧

**AI-powered DJ mixing engine.** Upload your tracks, let the engine analyze BPM, key, energy, and semantic structure, then generate a studio-quality MP3 mix — with harmonic forcing, bass swaps, filter sweeps, and beat-synced crossfades.

---

## Features

- **Semantic section detection** — classifies Intro / Verse / Pre-Chorus / Drop / Outro
- **Pre-computed STFT pattern matching** — finds the best entry point in O(1) per window
- **Harmonic forcing** — pitch-shifts tracks to a compatible Camelot key without changing tempo
- **Bass swaps & HP filter sweeps** — studio-grade DSP transitions
- **Tension riser** — loop-halving engine builds a chaotic riser into every drop
- **Stem-aware crossfade** — vocals exit early, instrumentals carry the transition
- **In-memory export** — no intermediate WAV files written to disk

---

## Quickstart

### Prerequisites

- Python 3.11+
- Node.js 20+
- `ffmpeg` in PATH (required by pydub)

### 1 — Clone

```bash
git clone https://github.com/sahiee-dev/audiomix.git
cd audiomix
cp .env.example .env
```

### 2 — Backend

```bash
cd backend
pip install -r app/requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Docker (optional)

```bash
docker-compose up
```

Both services start automatically. Frontend at `http://localhost:5173`, API at `http://localhost:8000`.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/upload` | Upload audio files |
| `POST` | `/api/analyze?session=<id>` | Extract features |
| `POST` | `/api/order` | Sort tracks by mode |
| `POST` | `/api/mix?session=<id>` | Generate the mix |
| `GET` | `/api/sessions` | List active sessions |
| `DELETE` | `/api/sessions/<id>` | Delete a session |
| `GET` | `/api/preview/<session>/<file>` | Stream audio |
| `GET` | `/api/download/<session>/<file>` | Download file |

Full interactive docs at `http://localhost:8000/docs`.

---

## Developer Commands

```bash
make install        # install backend + frontend deps
make dev-backend    # start FastAPI with hot-reload
make dev-frontend   # start Vite dev server
make test           # run backend unit tests
make lint           # run ruff linter
make clean          # remove uploads and Python caches
```

---

## Project Structure

```
audiomix/
├── backend/
│   └── app/
│       ├── main.py           # FastAPI app entry point
│       ├── api.py            # API router
│       ├── config.py         # Environment-based configuration
│       ├── schemas.py        # Pydantic request/response models
│       ├── mixer.py          # Stem-level mashup engine
│       ├── segmenter.py      # Semantic section classifier
│       ├── pattern_matcher.py# Pre-computed STFT matching
│       ├── pitch_engine.py   # Harmonic forcing via pyrubberband
│       ├── tension_looper.py # Beat-grid loop-halving riser
│       ├── dsp_fx.py         # EQ kills, bass swap, HP sweep
│       └── audio_utils.py    # Feature extraction utilities
└── frontend/
    └── src/
        ├── App.jsx
        ├── api.js            # API client
        └── components/
```

---

## License

MIT
