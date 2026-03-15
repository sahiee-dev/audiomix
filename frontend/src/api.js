import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000/api";

export async function uploadFiles(files) {
  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  const res = await axios.post(`${API_BASE}/upload`, fd, {
    headers: {'Content-Type': 'multipart/form-data'}
  });
  return res.data;
}

export async function analyze(session) {
  const res = await axios.post(`${API_BASE}/analyze?session=${session}`);
  return res.data;
}

export async function order(tracks, mode) {
  const res = await axios.post(`${API_BASE}/order`, { tracks, mode });
  return res.data;
}
