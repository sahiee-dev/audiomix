import React, { useState } from "react";
import { DragDropContext, Droppable, Draggable } from "@hello-pangea/dnd";

export default function TrackOrderer({ tracks, onReorder }) {
  const [orderedTracks, setOrderedTracks] = useState(tracks);

  React.useEffect(() => {
    setOrderedTracks(tracks);
  }, [tracks]);

  function handleDragEnd(result) {
    if (!result.destination) return;

    const items = Array.from(orderedTracks);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setOrderedTracks(items);
    onReorder(items);
  }

  return (
    <div style={{ marginBottom: 20 }}>
      <h3>🎵 Track Order (Drag to Reorder)</h3>
      <p style={{ fontSize: 14, color: "#666", marginBottom: 15 }}>
        Drag tracks to change the mixing order
      </p>

      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="tracks">
          {(provided) => (
            <div
              {...provided.droppableProps}
              ref={provided.innerRef}
              style={{
                backgroundColor: "#f9f9f9",
                padding: 10,
                borderRadius: 8,
                maxHeight: 350,
                overflowY: "auto",
              }}
            >
              {orderedTracks.map((track, index) => (
                <Draggable
                  key={track.filename}
                  draggableId={track.filename}
                  index={index}
                >
                  {(provided, snapshot) => (
                    <div
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      {...provided.dragHandleProps}
                      style={{
                        userSelect: "none",
                        padding: "12px 16px",
                        marginBottom: 8,
                        backgroundColor: snapshot.isDragging
                          ? "#e3f2fd"
                          : "white",
                        border: snapshot.isDragging
                          ? "2px solid #2196F3"
                          : "1px solid #ddd",
                        borderRadius: 6,
                        display: "flex",
                        alignItems: "center",
                        cursor: "grab",
                        transition: "all 0.2s",
                        ...provided.draggableProps.style,
                      }}
                    >
                      <div
                        style={{
                          fontWeight: "bold",
                          marginRight: 12,
                          color: "#666",
                          fontSize: 18,
                          minWidth: 30,
                        }}
                      >
                        {index + 1}.
                      </div>

                      <div
                        style={{
                          fontSize: 16,
                          marginRight: 12,
                          userSelect: "none",
                        }}
                      >
                        ☰
                      </div>

                      <div style={{ flex: 1 }}>
                        <div
                          style={{
                            fontWeight: 500,
                            color: "#333",
                            marginBottom: 2,
                            overflowWrap: "break-word",
                          }}
                        >
                          {track.filename}
                        </div>
                        {track.bpm && (
                          <div style={{ fontSize: 12, color: "#999" }}>
                            {track.bpm} BPM • {Math.round(track.duration)}s
                          </div>
                        )}
                      </div>

                      <div
                        style={{
                          fontSize: 12,
                          color: "#999",
                          fontStyle: "italic",
                          userSelect: "none",
                        }}
                      >
                        Drag to reorder
                      </div>
                    </div>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </div>
          )}
        </Droppable>
      </DragDropContext>
    </div>
  );
}
