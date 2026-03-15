import React, { useState } from 'react';

export default function MixingOptions({ tracks, session, mode, onMixComplete }) {
    const [mixing, setMixing] = useState(false);
    const [normalize, setNormalize] = useState(true);
    const [tempoMatch, setTempoMatch] = useState(true);
    const [harmonicMatch, setHarmonicMatch] = useState(false);
    const [crossfadeDuration, setCrossfadeDuration] = useState(10);
    const [entryMethod, setEntryMethod] = useState('skip_intro');
    const [stemSeparation, setStemSeparation] = useState(false);

    async function handleMix() {
        if (!tracks || tracks.length === 0) {
            alert('Please upload tracks first');
            return;
        }

        setMixing(true);

        try {
            // Format tracks to ensure all required fields exist
            const formattedTracks = tracks.map(track => ({
                filename: track.filename,
                bpm: track.bpm || 0,
                duration: track.duration || 0,
                key: track.key || 'C',
                energy: track.energy || 0.5
            }));

            const payload = {
                tracks: formattedTracks,
                mode: mode || 'recommended'
            };

            console.log('🔍 Sending payload:', JSON.stringify(payload, null, 2));
            console.log('🔍 Session:', session);

            const url = `http://localhost:8000/api/mix?session=${session}&normalize=${normalize}&tempo_match=${tempoMatch}&harmonic_match=${harmonicMatch}&crossfade_duration=${crossfadeDuration}&entry_method=${entryMethod}&use_stem_separation=${stemSeparation}`;

            console.log('🔍 Request URL:', url);

            const res = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            console.log('🔍 Response status:', res.status);

            if (!res.ok) {
                const errorText = await res.text();
                console.error('❌ Error response:', errorText);
                throw new Error(`Mix failed: ${res.status} - ${errorText}`);
            }

            const data = await res.json();
            console.log('✅ Mix complete:', data);
            onMixComplete(data);

        } catch (error) {
            console.error('❌ Mix error:', error);
            alert('Mixing failed: ' + error.message);
        } finally {
            setMixing(false);
        }
    }

    return (
        <div style={{
            backgroundColor: '#f0f0f0',
            padding: '25px',
            borderRadius: '10px',
            marginBottom: '20px',
            maxWidth: '900px',
            margin: '0 auto 20px'
        }}>
            <h3>🎛️ Professional DJ Mixing</h3>
            <p style={{ fontSize: '14px', color: '#666', marginBottom: '20px' }}>
                Pattern-matched transitions with AI-powered stem separation
            </p>

            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '15px',
                marginBottom: '20px'
            }}>
                <label style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                        type="checkbox"
                        checked={normalize}
                        onChange={(e) => setNormalize(e.target.checked)}
                        style={{ marginRight: '8px' }}
                    />
                    <span><strong>Normalize Loudness</strong> (LUFS -14)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                        type="checkbox"
                        checked={tempoMatch}
                        onChange={(e) => setTempoMatch(e.target.checked)}
                        style={{ marginRight: '8px' }}
                    />
                    <span><strong>Tempo Matching</strong> (Time-stretch BPM)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                        type="checkbox"
                        checked={harmonicMatch}
                        onChange={(e) => setHarmonicMatch(e.target.checked)}
                        style={{ marginRight: '8px' }}
                    />
                    <span><strong>Harmonic Mixing</strong> (Key compatibility)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center' }}>
                    <input
                        type="checkbox"
                        checked={stemSeparation}
                        onChange={(e) => setStemSeparation(e.target.checked)}
                        style={{ marginRight: '8px' }}
                    />
                    <span><strong>🎤 Stem Separation</strong> (Vocal/Instrumental AI)</span>
                </label>
            </div>

            <div style={{ marginBottom: '15px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>
                    <strong>Entry Point Detection:</strong>
                </label>
                <select
                    value={entryMethod}
                    onChange={(e) => setEntryMethod(e.target.value)}
                    style={{
                        width: '100%',
                        padding: '8px',
                        borderRadius: '5px',
                        border: '1px solid #ccc'
                    }}
                >
                    <option value="skip_intro">Skip Intro (15%)</option>
                    <option value="high_energy">High Energy Section</option>
                    <option value="first_drop">First Drop</option>
                    <option value="structural">Structural Analysis</option>
                </select>
            </div>

            <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', marginBottom: '5px' }}>
                    <strong>Crossfade Duration:</strong> {crossfadeDuration}s
                </label>
                <input
                    type="range"
                    min="4"
                    max="15"
                    step="0.5"
                    value={crossfadeDuration}
                    onChange={(e) => setCrossfadeDuration(parseFloat(e.target.value))}
                    style={{ width: '100%' }}
                />
                <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: '12px',
                    color: '#666'
                }}>
                    <span>Quick (4s)</span>
                    <span>Standard (10s)</span>
                    <span>Extended (15s)</span>
                </div>
            </div>

            {stemSeparation && (
                <div style={{
                    backgroundColor: '#fff3cd',
                    border: '1px solid #ffc107',
                    padding: '10px',
                    borderRadius: '5px',
                    marginBottom: '15px',
                    fontSize: '14px'
                }}>
                    ⚠️ <strong>Note:</strong> Stem separation takes 30-60 seconds per song (AI processing)
                </div>
            )}

            <button
                onClick={handleMix}
                disabled={mixing}
                style={{
                    padding: '15px 30px',
                    backgroundColor: mixing ? '#ccc' : '#FF6B6B',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: mixing ? 'not-allowed' : 'pointer',
                    fontSize: '16px',
                    fontWeight: 'bold',
                    width: '100%',
                    transition: 'all 0.3s'
                }}
            >
                {mixing ? '🎵 Creating Mix... (This may take 2-3 minutes)' : '🎧 Create Professional DJ Mix'}
            </button>

            {mixing && (
                <p style={{
                    textAlign: 'center',
                    marginTop: '15px',
                    fontSize: '14px',
                    color: '#666'
                }}>
                    {stemSeparation
                        ? '🎤 Separating vocals/instrumentals + pattern matching...'
                        : '🎵 Analyzing beats and matching patterns...'}
                </p>
            )}
        </div>
    );
}
