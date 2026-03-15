import React, { useState } from 'react';
import { uploadFiles } from '../api';

export default function UploadZone({ onUploadComplete }) {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);

    async function handleUpload() {
        if (files.length === 0) return;
        setLoading(true);
        try {
            const data = await uploadFiles(files);
            console.log("✅ Upload response:", data);  // DEBUG

            // Pass as object with correct structure
            onUploadComplete({
                session: data.session,
                tracks: data.files || data.tracks  // Handle both 'files' and 'tracks'
            });
        } catch (error) {
            console.error("❌ Upload error:", error);
            alert('Upload failed: ' + error.message);
        }
        setLoading(false);
    }

    return (
        <div style={{ marginBottom: '20px' }}>
            <h3>Step 1: Upload Audio Files</h3>
            <input
                type="file"
                multiple
                accept="audio/*"
                onChange={(e) => setFiles([...e.target.files])}
            />
            <button onClick={handleUpload} disabled={loading}>
                {loading ? 'Uploading...' : 'Upload & Analyze'}
            </button>
            <p>Selected: {files.length} file(s)</p>
        </div>
    );
}
