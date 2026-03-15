import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function PlaylistChart({ ordered }) {
    const data = ordered.map((track, idx) => ({
        position: idx + 1,
        name: track.filename.substring(0, 20) + '...',
        BPM: Math.round(track.bpm),
        Energy: (track.energy * 100).toFixed(1)
    }));

    return (
        <div style={{ width: '100%', maxWidth: '900px', margin: '30px auto' }}>
            <h3>Playlist Flow Visualization</h3>
            <ResponsiveContainer width="100%" height={300}>
                <LineChart data={data}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="position" label={{ value: 'Track Position', position: 'insideBottom', offset: -5 }} />
                    <YAxis yAxisId="left" label={{ value: 'BPM', angle: -90, position: 'insideLeft' }} />
                    <YAxis yAxisId="right" orientation="right" label={{ value: 'Energy (%)', angle: 90, position: 'insideRight' }} />
                    <Tooltip />
                    <Legend />
                    <Line yAxisId="left" type="monotone" dataKey="BPM" stroke="#8884d8" strokeWidth={2} />
                    <Line yAxisId="right" type="monotone" dataKey="Energy" stroke="#82ca9d" strokeWidth={2} />
                </LineChart>
            </ResponsiveContainer>
        </div>
    );
}
