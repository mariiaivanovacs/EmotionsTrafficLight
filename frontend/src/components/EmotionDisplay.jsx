import React, { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const BACKEND_URL = 'http://localhost:5001';

const EmotionDisplay = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [emotions, setEmotions] = useState([]);
  const [fps, setFps] = useState(0);
  const [faceCount, setFaceCount] = useState(0);
  const [socket, setSocket] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(null);

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io(BACKEND_URL);

    newSocket.on('connect', () => {
      console.log('Connected to backend');
    });

    newSocket.on('video_frame', (data) => {
      // Update frame and emotion data
      setCurrentFrame(data.frame);
      setEmotions(data.emotions);
      setFps(data.fps);
      setFaceCount(data.face_count);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend');
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  // Fetch available cameras
  useEffect(() => {
    fetchCameras();
  }, []);

  const fetchCameras = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/cameras`);
      const data = await response.json();
      setCameras(data);
      if (data.length > 0) {
        setSelectedCamera(data[0].id);
      }
    } catch (error) {
      console.error('Error fetching cameras:', error);
    }
  };

  const startCamera = async () => {
    console.log("=== START CAMERA CLICKED ===");
    if (selectedCamera === null) {
      console.log("No camera selected, aborting");
      return;
    }
    console.log("Selected camera:", selectedCamera);

    const url = `${BACKEND_URL}/api/start/${selectedCamera}`;
    console.log("Fetch URL:", url);

    try {
      console.log("Sending fetch request...");
      const response = await fetch(url);
      console.log("Fetch response received:", response.status, response.statusText);

      const data = await response.json();
      console.log("Response data:", data);

      if (data.status === 'success') {
        console.log("✓ Backend started successfully, showing video");
        setIsStreaming(true); // Only show video after backend confirms success
      } else {
        console.error("Backend returned error:", data.message);
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      console.error('Error details:', error.message, error.stack);
    }
  };

const stopCamera = async () => {
  console.log("=== STOP CAMERA CLICKED ===");

  // Immediately stop UI rendering
  setIsStreaming(false);
  setCurrentFrame(null);
  setEmotions([]);
  setFps(0);
  setFaceCount(0);

  try {
    const url = `${BACKEND_URL}/api/stop`;
    console.log("Fetch URL:", url);

    const response = await fetch(url);
    console.log("Fetch completed");

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const data = await response.json();
    console.log("Backend response:", data);
    console.log("Camera stopped successfully");
  } catch (error) {
    console.error("❌ STOP CAMERA FAILED");
    console.error(error);
  }
};


  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>🚦 Emotion Traffic Light</h1>
        <div style={styles.stats}>
          <span style={styles.stat}>FPS: {fps}</span>
          <span style={styles.stat}>Faces: {faceCount}</span>
        </div>
      </div>

      <div style={styles.controls}>
        <select
          value={selectedCamera || ''}
          onChange={(e) => setSelectedCamera(Number(e.target.value))}
          disabled={isStreaming}
          style={styles.select}
        >
          {cameras.map(cam => (
            <option key={cam.id} value={cam.id}>
              {cam.name} ({cam.resolution})
            </option>
          ))}
        </select>

        {!isStreaming ? (
          <button onClick={startCamera} style={styles.button}>
            Start Camera
          </button>
        ) : (
          <button onClick={stopCamera} style={{...styles.button, ...styles.stopButton}}>
            Stop Camera
          </button>
        )}
      </div>

      <div style={styles.content}>
        <div style={styles.videoContainer}>
          {isStreaming && currentFrame ? (
            <img
              src={`data:image/jpeg;base64,${currentFrame}`}
              alt="Emotion Detection Stream"
              style={styles.video}
            />
          ) : (
            <div style={styles.placeholder}>
              <p>Select a camera and click "Start Camera"</p>
            </div>
          )}
        </div>

        <div style={styles.emotionsPanel}>
          <h2 style={styles.panelTitle}>Detected Emotions</h2>
          {emotions.length === 0 ? (
            <p style={styles.noEmotions}>No faces detected</p>
          ) : (
            emotions.map((emotionData) => (
              <div key={emotionData.id} style={styles.emotionCard}>
                <div style={styles.emotionHeader}>
                  <div
                    style={{
                      ...styles.colorIndicator,
                      backgroundColor: emotionData.color
                    }}
                  />
                  <h3 style={styles.emotionTitle}>Face {emotionData.id + 1}</h3>
                </div>

                <div style={styles.dominantEmotion}>
                  <span style={styles.emotionName}>
                    {emotionData.dominant.toUpperCase()}
                  </span>
                  <span style={styles.confidence}>
                    {(emotionData.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                <div style={styles.topEmotions}>
                  <p style={styles.topEmotionsTitle}>Top Emotions:</p>
                  {emotionData.top_emotions.map((em, idx) => (
                    <div key={idx} style={styles.emotionRow}>
                      <span style={styles.emotionLabel}>{em.emotion}</span>
                      <div style={styles.progressBar}>
                        <div
                          style={{
                            ...styles.progressFill,
                            width: `${em.score * 100}%`
                          }}
                        />
                      </div>
                      <span style={styles.emotionScore}>
                        {(em.score * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#1a1a1a',
    color: '#ffffff',
    padding: '20px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
  },
  title: {
    margin: 0,
    fontSize: '2em',
  },
  stats: {
    display: 'flex',
    gap: '20px',
  },
  stat: {
    backgroundColor: '#2a2a2a',
    padding: '10px 20px',
    borderRadius: '8px',
    fontSize: '1em',
    fontWeight: 'bold',
  },
  controls: {
    display: 'flex',
    gap: '10px',
    marginBottom: '20px',
  },
  select: {
    padding: '10px',
    fontSize: '1em',
    borderRadius: '8px',
    border: '2px solid #444',
    backgroundColor: '#2a2a2a',
    color: '#fff',
    cursor: 'pointer',
  },
  button: {
    padding: '10px 20px',
    fontSize: '1em',
    borderRadius: '8px',
    border: 'none',
    backgroundColor: '#00ff00',
    color: '#000',
    fontWeight: 'bold',
    cursor: 'pointer',
  },
  stopButton: {
    backgroundColor: '#ff0000',
    color: '#fff',
  },
  content: {
    display: 'grid',
    gridTemplateColumns: '2fr 1fr',
    gap: '20px',
  },
  videoContainer: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    overflow: 'hidden',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    minHeight: '480px',
  },
  video: {
    width: '100%',
    height: 'auto',
    display: 'block',
  },
  placeholder: {
    textAlign: 'center',
    color: '#666',
    fontSize: '1.2em',
  },
  emotionsPanel: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    padding: '20px',
    maxHeight: '600px',
    overflowY: 'auto',
  },
  panelTitle: {
    marginTop: 0,
    marginBottom: '20px',
    fontSize: '1.5em',
  },
  noEmotions: {
    color: '#666',
    textAlign: 'center',
    padding: '20px',
  },
  emotionCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '15px',
  },
  emotionHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '10px',
  },
  colorIndicator: {
    width: '20px',
    height: '20px',
    borderRadius: '50%',
  },
  emotionTitle: {
    margin: 0,
    fontSize: '1em',
  },
  dominantEmotion: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px',
    padding: '10px',
    backgroundColor: '#2a2a2a',
    borderRadius: '6px',
  },
  emotionName: {
    fontSize: '1.5em',
    fontWeight: 'bold',
  },
  confidence: {
    fontSize: '1.5em',
    fontWeight: 'bold',
    color: '#00ff00',
  },
  topEmotions: {
    marginTop: '15px',
  },
  topEmotionsTitle: {
    margin: '0 0 10px 0',
    fontSize: '0.9em',
    color: '#888',
  },
  emotionRow: {
    display: 'grid',
    gridTemplateColumns: '80px 1fr 50px',
    alignItems: 'center',
    gap: '10px',
    marginBottom: '8px',
  },
  emotionLabel: {
    fontSize: '0.9em',
    textTransform: 'capitalize',
  },
  progressBar: {
    height: '8px',
    backgroundColor: '#444',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#00ff00',
    transition: 'width 0.3s ease',
  },
  emotionScore: {
    fontSize: '0.9em',
    textAlign: 'right',
    color: '#888',
  },
};

export default EmotionDisplay;
