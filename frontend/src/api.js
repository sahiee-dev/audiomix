import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api';

const client = axios.create({ baseURL: API_BASE });

/** Upload one or more File objects and return { session, files }. */
export async function uploadFiles(files) {
  const fd = new FormData();
  files.forEach((f) => fd.append('files', f));
  const res = await client.post('/upload', fd, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return res.data;
}

/** Analyze all tracks in a session. Returns { session, tracks: TrackFeatures[] }. */
export async function analyze(session) {
  const res = await client.post(`/analyze?session=${session}`);
  return res.data;
}

/** Order tracks by mode. Returns { ordered: TrackFeatures[] }. */
export async function order(tracks, mode) {
  const res = await client.post('/order', { tracks, mode });
  return res.data;
}

/**
 * Mix the ordered tracks and return { status, mixed_file, download_url }.
 *
 * @param {object[]} tracks   Ordered TrackFeatures array.
 * @param {string}   session  Session ID returned by uploadFiles.
 * @param {object}   [opts]   Optional mixing parameters.
 */
export async function mix(tracks, session, opts = {}) {
  const params = new URLSearchParams({ session, ...opts }).toString();
  const res = await client.post(`/mix?${params}`, { tracks, mode: opts.mode || 'recommended' });
  return res.data;
}

/** Return metadata for all active sessions. */
export async function listSessions() {
  const res = await client.get('/sessions');
  return res.data;
}

/** Delete a session and its files from the server. */
export async function deleteSession(session) {
  const res = await client.delete(`/sessions/${session}`);
  return res.data;
}

/** Check whether the API is reachable. Returns { status, version }. */
export async function healthCheck() {
  const res = await client.get('/health');
  return res.data;
}
