import React, { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';
import FaceMesh3D from './FaceMesh3D';
import FeatureGraphs from './FeatureGraphs';
import ValenceArousalPlot from './ValenceArousalPlot';

const BACKEND_URL = 'http://localhost:5001';

const FaceMeshDisplay = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [facesData, setFacesData] = useState([]);
  const [fps, setFps] = useState(0);
  const [faceCount, setFaceCount] = useState(0);
  const [socket, setSocket] = useState(null);
  const [meshAvailable, setMeshAvailable] = useState(false);
  const [lastNarration, setLastNarration] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(null);

  const audioRef = useRef(new SpeechSynthesisUtterance());

  // Check if face mesh is available
  useEffect(() => {
    fetch(`${BACKEND_URL}/api/face_mesh/available`)
      .then(res => res.json())
      .then(data => setMeshAvailable(data.available))
      .catch(() => setMeshAvailable(false));
  }, []);

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io(BACKEND_URL);

    newSocket.on('connect', () => {
      console.log('Connected to backend for face mesh');
    });

    newSocket.on('face_mesh_frame', (data) => {
      console.log('📊 Face mesh frame received:', {
        face_count: data.face_count,
        fps: data.fps,
        has_frame: !!data.frame
      });

      // Update frame and face data
      setCurrentFrame(data.frame);
      setFacesData(data.faces || []);
      setFps(data.fps || 0);
      setFaceCount(data.face_count || 0);

      // Handle narration for zone changes
      if (data.faces) {
        data.faces.forEach(face => {
          if (face.zone_changed) {
            narrateZoneChange(face.valence_zone);
          }
        });
      }
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
    console.log("=== START FACE MESH CLICKED ===");
    if (selectedCamera === null) {
      console.log("No camera selected, aborting");
      return;
    }
    console.log("Selected camera:", selectedCamera);

    const url = `${BACKEND_URL}/api/face_mesh/start/${selectedCamera}`;
    console.log("Fetch URL:", url);

    try {
      console.log("Sending fetch request...");
      const response = await fetch(url);
      console.log("Fetch response received:", response.status, response.statusText);

      const data = await response.json();
      console.log("Response data:", data);

      if (data.status === 'success') {
        setIsStreaming(true);
        console.log("Face mesh started successfully");
      } else {
        console.error("Backend returned error:", data.message);
        alert(data.message);
      }
    } catch (error) {
      console.error('Error starting camera:', error);
      console.error('Error details:', error.message, error.stack);
    }
  };

  const stopCamera = async () => {
    console.log("=== STOP FACE MESH CLICKED ===");

    // Immediately stop UI rendering
    setIsStreaming(false);
    setCurrentFrame(null);
    setFacesData([]);
    setFps(0);
    setFaceCount(0);

    try {
      const url = `${BACKEND_URL}/api/face_mesh/stop`;
      console.log("Fetch URL:", url);

      const response = await fetch(url);
      console.log("Fetch response received:", response.status, response.statusText);

      const data = await response.json();
      console.log("Response data:", data);
      console.log("Face mesh stopped successfully");
    } catch (error) {
      console.error('Error stopping camera:', error);
      console.error('Error details:', error.message, error.stack);
    }
  };

  const narrateZoneChange = (zone) => {
    // Prevent rapid repeated narrations
    if (lastNarration === zone) return;

    const messages = {
      'positive': 'Entering positive emotional state',
      'neutral': 'Entering neutral emotional state',
      'negative': 'Entering negative emotional state'
    };

    const message = messages[zone] || 'Emotional state changed';

    audioRef.current.text = message;
    audioRef.current.rate = 1.0;
    audioRef.current.pitch = 1.0;

    window.speechSynthesis.cancel(); // Cancel any ongoing speech
    window.speechSynthesis.speak(audioRef.current);

    setLastNarration(zone);

    // Reset after 2 seconds to allow re-narration
    setTimeout(() => setLastNarration(null), 2000);
  };

  if (!meshAvailable) {
    return (
      <div style={styles.container}>
        <div style={styles.errorBox}>
          <h2>⚠️ Face Mesh Not Available</h2>
          <p>MediaPipe is not installed. Please install it to use 3D face analysis:</p>
          <pre style={styles.code}>pip install mediapipe scipy</pre>
          <p>Then restart the backend server.</p>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>🎯 3D Face Mesh Analysis</h1>
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
            Start Face Mesh Analysis
          </button>
        ) : (
          <button onClick={stopCamera} style={{...styles.button, ...styles.stopButton}}>
            Stop Analysis
          </button>
        )}
      </div>

      <div style={styles.mainContent}>
        {/* Left: 2D Camera + 3D Visualization */}
        <div style={styles.visualizations}>
          <div style={styles.videoContainer}>
            <h3 style={styles.sectionTitle}>2D Camera Feed</h3>
            {isStreaming && currentFrame ? (
              <img
                src={`data:image/jpeg;base64,${currentFrame}`}
                alt="Face Mesh Stream"
                style={styles.video}
              />
            ) : (
              <div style={styles.placeholder}>
                <p>Click "Start Face Mesh Analysis" to begin</p>
              </div>
            )}
          </div>

          <div style={styles.visualization3D}>
            <h3 style={styles.sectionTitle}>3D Face Model</h3>
            {facesData.length > 0 ? (
              <FaceMesh3D landmarks={facesData[0].landmarks_3d} />
            ) : (
              <div style={styles.placeholder}>
                <p>3D mesh will appear here when face is detected</p>
              </div>
            )}
          </div>
        </div>

        {/* Right: Analysis Panel */}
        <div style={styles.analysisPanel}>
          <h2 style={styles.panelTitle}>Geometry Analysis</h2>

          {facesData.length === 0 ? (
            <p style={styles.noData}>No faces detected</p>
          ) : (
            facesData.map((face) => (
              <div key={face.face_id} style={styles.faceCard}>
                {/* Large Emotion Label */}
                <div style={styles.emotionHeader}>
                  <span style={styles.emotionEmoji}>{face.emotion_emoji}</span>
                  <span style={styles.emotionLabel}>{face.emotion_label}</span>
                </div>

                {/* Valence/Arousal Plot */}
                <ValenceArousalPlot
                  valence={face.valence}
                  arousal={face.arousal}
                  zone={face.valence_zone}
                  emotionLabel={face.emotion_label}
                />

                {/* Geometry Features */}
                <div style={styles.featuresSection}>
                  <h3 style={styles.subsectionTitle}>Current Features</h3>
                  {face.geometry_features && Object.keys(face.geometry_features).length > 0 ? (
                    Object.entries(face.geometry_features).map(([key, value]) => (
                      <div key={key} style={styles.featureRow}>
                        <span style={styles.featureLabel}>
                          {key.replace(/_/g, ' ')}:
                        </span>
                        <span style={styles.featureValue}>
                          {typeof value === 'number' ? value.toFixed(3) : value}
                        </span>
                      </div>
                    ))
                  ) : (
                    <p style={styles.noData}>No geometry features available</p>
                  )}
                </div>

                {/* Temporal Features with Graphs */}
                <div style={styles.temporalSection}>
                  <h3 style={styles.subsectionTitle}>Temporal Analysis (3s window)</h3>
                  {face.temporal_features && Object.keys(face.temporal_features).length > 0 ? (
                    <FeatureGraphs temporalFeatures={face.temporal_features} />
                  ) : (
                    <p style={styles.noData}>Collecting temporal data...</p>
                  )}
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
  mainContent: {
    display: 'grid',
    gridTemplateColumns: '2fr 1fr',
    gap: '20px',
  },
  visualizations: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  videoContainer: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    overflow: 'hidden',
    padding: '15px',
  },
  visualization3D: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    padding: '15px',
    minHeight: '400px',
  },
  video: {
    width: '100%',
    height: 'auto',
    display: 'block',
    borderRadius: '8px',
  },
  placeholder: {
    textAlign: 'center',
    color: '#666',
    fontSize: '1.2em',
    padding: '60px 20px',
  },
  sectionTitle: {
    marginTop: 0,
    marginBottom: '15px',
    fontSize: '1.2em',
  },
  analysisPanel: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    padding: '20px',
    maxHeight: '90vh',
    overflowY: 'auto',
  },
  panelTitle: {
    marginTop: 0,
    marginBottom: '20px',
    fontSize: '1.5em',
  },
  noData: {
    color: '#666',
    textAlign: 'center',
    padding: '20px',
  },
  faceCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '15px',
  },
  emotionHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '15px',
    padding: '20px',
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
    marginBottom: '20px',
    border: '2px solid #00ff00',
  },
  emotionEmoji: {
    fontSize: '3em',
  },
  emotionLabel: {
    fontSize: '2em',
    fontWeight: 'bold',
    color: '#00ff00',
    textTransform: 'uppercase',
  },
  featuresSection: {
    marginTop: '20px',
  },
  temporalSection: {
    marginTop: '20px',
  },
  subsectionTitle: {
    fontSize: '1.1em',
    marginBottom: '10px',
    color: '#00ff00',
  },
  featureRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '8px',
    borderBottom: '1px solid #333',
  },
  featureLabel: {
    textTransform: 'capitalize',
    color: '#888',
  },
  featureValue: {
    fontFamily: 'monospace',
    color: '#0ff',
  },
  errorBox: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    padding: '40px',
    margin: '40px auto',
    maxWidth: '600px',
    textAlign: 'center',
  },
  code: {
    backgroundColor: '#1a1a1a',
    padding: '10px',
    borderRadius: '4px',
    display: 'inline-block',
    marginTop: '10px',
  },
};

export default FaceMeshDisplay;
