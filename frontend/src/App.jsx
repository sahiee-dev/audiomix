import React, { useState } from "react";
import "./App.css";

import UploadZone from "./components/UploadZone";
import TrackList from "./components/TrackList";
import TrackOrderer from "./components/TrackOrderer";
import OrderingOptions from "./components/OrderingOptions";
import MixingOptions from "./components/MixingOptions";

function App() {
  const [tracks, setTracks] = useState([]);              // Initialize as empty array
  const [orderedTracks, setOrderedTracks] = useState([]);// Initialize as empty array
  const [session, setSession] = useState(null);
  const [mode, setMode] = useState("recommended");
  const [mixedFile, setMixedFile] = useState(null);

  function handleUploadComplete(data) {
    setTracks(data.tracks || []);
    setOrderedTracks(data.tracks || []);
    setSession(data.session);
    setMixedFile(null); // Reset previous result
  }

  function handleOrderComplete(data) {
    setOrderedTracks(data.ordered_tracks || []);
    setMode(data.mode || "recommended");
  }

  function handleManualReorder(reordered) {
    setOrderedTracks(reordered || []);
    setMode("manual");
  }

  function handleMixComplete(data) {
    setMixedFile(data);
  }

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "40px 20px" }}>
      <h1 style={{ textAlign: "center", marginBottom: 40 }}>🎧 AI DJ AutoMix</h1>

      <UploadZone onUploadComplete={handleUploadComplete} />

      {tracks && tracks.length > 0 && (
        <>
          <TrackList tracks={tracks} session={session} />

          <OrderingOptions
            tracks={tracks}
            session={session}
            onOrderComplete={handleOrderComplete}
          />

          {orderedTracks && orderedTracks.length > 0 && (
            <>
              <TrackOrderer
                tracks={orderedTracks}
                onReorder={handleManualReorder}
              />

              <MixingOptions
                tracks={orderedTracks}
                session={session}
                mode={mode}
                onMixComplete={handleMixComplete}
              />
            </>
          )}

          {mixedFile && (
            <div
              style={{
                marginTop: 30,
                padding: 20,
                backgroundColor: "#e8f5e9",
                borderRadius: 8,
                textAlign: "center",
              }}
            >
              <h3>✅ Mix Complete!</h3>
              <audio
                controls
                src={`http://localhost:8000/api/preview/${session}/${mixedFile.mixed_file}`}
                style={{ width: "100%", marginTop: 15 }}
              />
              <div style={{ marginTop: 15 }}>
                <a
                  href={`http://localhost:8000${mixedFile.download_url}`}
                  download
                  style={{
                    padding: "10px 20px",
                    backgroundColor: "#4caf50",
                    color: "white",
                    textDecoration: "none",
                    borderRadius: 5,
                    display: "inline-block",
                  }}
                >
                  💾 Download Mix
                </a>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;
