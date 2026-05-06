# Changelog

All notable changes to AudioMix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- Health check endpoint at `/health` with version metadata
- `CHANGELOG.md` tracking project history
- `.env.example` for environment configuration
- `Makefile` with developer convenience commands
- `docker-compose.yml` for containerized local development
- Type-annotated Pydantic response models for all API endpoints
- `GET /api/sessions` endpoint to list active sessions
- `DELETE /api/sessions/{session}` endpoint to clean up uploads
- Graceful error handling with structured JSON error responses
- Unit tests for core audio utility functions

### Changed
- `config.py` now reads all settings from environment variables with documented defaults
- `audio_utils.py` cleans up temp WAV files in a `finally` block (previously leaked on error)
- `main.py` title updated to `AudioMix API`
- `api.js` now exports a `mix()` function aligning with the `/mix` backend endpoint
- `schemas.py` expanded with `MixRequest`, `MixResponse`, and `SessionInfo` models

### Fixed
- Temp WAV file was not deleted on exception in `extract_features()`
- Missing `genre` field on `TrackFeatures` schema caused silent default `'general'` mismatches

---

## [1.0.0] - 2026-05-06

### Added
- Initial release: upload → analyze → order → mix pipeline
- Professional Stem-Level Mashup Engine (`mixer.py`)
- Semantic section classifier: Intro / Verse / Pre-Chorus / Drop / Outro (`segmenter.py`)
- Beat-grid tension looper with loop-halving riser effect (`tension_looper.py`)
- Studio-grade harmonic forcing via `pyrubberband` (`pitch_engine.py`)
- 3-band EQ kills, bass swap, HP filter sweep DSP chain (`dsp_fx.py`)
- Pre-computed STFT pattern matching — O(1) window scanning (`pattern_matcher.py`)
- In-memory MP3 export — no intermediate WAV written to disk (`mixer.py`)
- React + Vite frontend with drag-and-drop upload and manual track reordering
- Spotify benchmark harness (`spotify_benchmark.py`)
