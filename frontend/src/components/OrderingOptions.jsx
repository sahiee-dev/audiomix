import React, { useState } from 'react';

export default function OrderingOptions({ tracks, session, onOrderComplete }) {
    const [mode, setMode] = useState('recommended');
    const [ordering, setOrdering] = useState(false);

    async function handleOrder() {
        setOrdering(true);
        try {
            const res = await fetch('http://localhost:8000/api/order', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tracks, mode })
            });
            const data = await res.json();
            onOrderComplete(data);
        } catch (error) {
            alert('Ordering failed: ' + error.message);
        }
        setOrdering(false);
    }

    return (
        <div style={{
            marginBottom: '20px',
            padding: '20px',
            backgroundColor: '#f5f5f5',
            borderRadius: '8px'
        }}>
            <h3>📊 Auto-Order Tracks</h3>
            <p style={{ fontSize: '14px', color: '#666' }}>
                Let AI suggest the best track order, or manually drag tracks below
            </p>

            <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                style={{
                    padding: '10px',
                    marginRight: '10px',
                    borderRadius: '5px',
                    border: '1px solid #ccc',
                    fontSize: '14px'
                }}
            >
                <option value="recommended">Recommended (BPM + Energy)</option>
                <option value="energy">Energy Flow</option>
                <option value="bpm">BPM Progression</option>
                <option value="shuffle">Random Shuffle</option>
            </select>

            <button
                onClick={handleOrder}
                disabled={ordering}
                style={{
                    padding: '10px 20px',
                    backgroundColor: ordering ? '#ccc' : '#2196F3',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: ordering ? 'not-allowed' : 'pointer',
                    fontSize: '14px',
                    fontWeight: 'bold'
                }}
            >
                {ordering ? 'Analyzing...' : '🔄 Auto-Order Tracks'}
            </button>
        </div>
    );
}
