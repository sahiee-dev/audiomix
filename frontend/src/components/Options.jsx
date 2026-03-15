import React, { useState } from 'react';

export default function Options({ onChoose }) {
    const [mode, setMode] = useState('high_to_low');

    return (
        <div style={{ marginBottom: '20px' }}>
            <h3>Step 2: Choose Ordering Mode</h3>
            <label style={{ display: 'block', margin: '5px 0' }}>
                <input
                    type="radio"
                    name="mode"
                    value="high_to_low"
                    checked={mode === 'high_to_low'}
                    onChange={(e) => setMode(e.target.value)}
                /> High → Low (Party to Chill)
            </label>
            <label style={{ display: 'block', margin: '5px 0' }}>
                <input
                    type="radio"
                    name="mode"
                    value="low_to_high"
                    checked={mode === 'low_to_high'}
                    onChange={(e) => setMode(e.target.value)}
                /> Low → High (Mood Builder)
            </label>
            <label style={{ display: 'block', margin: '5px 0' }}>
                <input
                    type="radio"
                    name="mode"
                    value="smooth"
                    checked={mode === 'smooth'}
                    onChange={(e) => setMode(e.target.value)}
                /> Smooth Mix (Balanced)
            </label>
            <button onClick={() => onChoose(mode)}>
                Order Playlist
            </button>
        </div>
    );
}
