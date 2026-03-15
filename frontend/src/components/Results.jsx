import React, { useState } from 'react';
import { order } from '../api';
import PlaylistChart from './PlaylistChart';
import MixingOptions from './MixingOptions';

export default function Results({ ordered, session, mode }) {
    const [mixedFile, setMixedFile] = useState(null);

    function handleMixComplete(data) {
        setMixedFile(data);
    }

    async function handleExport() {
        try {
            const res = await order(ordered, mode);
            const m3uRes = await fetch(`http://localhost:8000/api/export/m3u?session=${session}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tracks: res.ordered, mode })
            });
            const data = await m3uRes.json();

            const blob = new Blob([data.m3u], { type: 'audio/x-mpegurl' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = data.filename;
            a.click();
            URL.revokeObjectURL(url);
        } catch (error) {
            alert('Export failed: ' + error.message);
        }
    }

    return (
        <div>
            <h3>Step 3: Ordered Playlist</h3>
            <button
                onClick={handleExport}
                style={{
                    marginBottom: '20px',
                    padding: '10px 20px',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer'
                }}
            >
                📥 Export as M3U
            </button>

            <PlaylistChart ordered={ordered} />

            <MixingOptions
                tracks={ordered}
                session={session}
                mode={mode}
                onMixComplete={handleMixComplete}
            />

            {mixedFile && (
                <div style={{
                    backgroundColor: '#d4edda',
                    padding: '20px',
                    borderRadius: '10px',
                    marginBottom: '20px'
                }}>
                    <h3>✅ Mixed Audio Ready!</h3>
                    <audio
                        controls
                        style={{ width: '100%', marginBottom: '10px' }}
                        src={`http://localhost:8000/api/preview/${session}/${mixedFile.mixed_file}`}
                    />
                    <a
                        href={`http://localhost:8000/api/download/${session}/${mixedFile.mixed_file}`}
                        download
                        style={{
                            padding: '10px 20px',
                            backgroundColor: '#28a745',
                            color: 'white',
                            textDecoration: 'none',
                            borderRadius: '5px',
                            display: 'inline-block'
                        }}
                    >
                        ⬇️ Download Mixed Audio
                    </a>
                </div>
            )}

            <ol style={{ textAlign: 'left', maxWidth: '800px', margin: '0 auto' }}>
                {ordered.map((t, idx) => (
                    <li key={idx} style={{ marginBottom: '15px' }}>
                        <strong>{idx + 1}. {t.filename}</strong>
                        <br />
                        <span style={{ fontSize: '14px', color: '#555' }}>
                            🎵 BPM: {Math.round(t.bpm)} |
                            ⚡ Energy: {t.energy.toFixed(3)} |
                            🎹 Key: {t.key} |
                            ⏱️ Duration: {Math.round(t.duration)}s
                            <br />
                            😊 Mood: {t.mood || 'N/A'} |
                            💃 Danceability: {t.danceability ? t.danceability.toFixed(2) : 'N/A'} |
                            🔊 Loudness: {t.loudness ? t.loudness.toFixed(1) : 'N/A'} dB
                        </span>
                        <br />
                        <audio
                            controls
                            style={{ width: '100%', marginTop: '5px' }}
                            src={`http://localhost:8000/api/preview/${session}/${t.filename}`}
                        />
                    </li>
                ))}
            </ol>
        </div>
    );
}
