import React from "react";

export default function TrackList({ tracks }) {
    return (
        <div style={{ marginBottom: 20 }}>
            <h3>🎵 Uploaded Tracks</h3>
            <ul style={{ listStyleType: "none", paddingLeft: 0 }}>
                {tracks.map((track, i) => (
                    <li
                        key={track.filename}
                        style={{
                            padding: 8,
                            borderBottom: "1px solid #eee",
                            fontSize: 14,
                            color: "#333",
                        }}
                    >
                        {i + 1}. {track.filename}{" "}
                        {track.duration ? `- ${Math.round(track.duration)}s` : ""}
                    </li>
                ))}
            </ul>
        </div>
    );
}
