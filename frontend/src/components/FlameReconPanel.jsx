/**
 * FlameReconPanel - Dedicated FLAME 3D Face Reconstruction Panel
 *
 * Displays:
 * - Camera feed with MediaPipe landmark overlay
 * - Real-time FLAME parametric face model reconstruction
 * - Shape, expression, and pose parameters
 */

import { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';
import FlameMeshViewer from './FlameMeshViewer';

const BACKEND_URL = 'http://localhost:5001';

// MediaPipe landmark indices used in FLAME fitting (from flame_fitter.py)
const POSE_LANDMARKS = [10, 152, 234, 454, 1, 168, 33, 263];
const EXPRESSION_LANDMARKS = [13, 14, 61, 291, 78, 308, 95, 324, 159, 145, 386, 374, 70, 300, 105, 334, 1, 172, 397, 152];

const FlameReconPanel = () => {
  const [cameras, setCameras] = useState([]);
  const [selectedCamera, setSelectedCamera] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [, setSocket] = useState(null);
  const [flameMeshData, setFlameMeshData] = useState(null);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [landmarks3d, setLandmarks3d] = useState(null);
  const [faceRotationMatrix, setFaceRotationMatrix] = useState(null);

  const [fps, setFps] = useState(0);
  const [wireframe, setWireframe] = useState(false);
  const [showParams, setShowParams] = useState(true);
  const [showLandmarks, setShowLandmarks] = useState(true);
  const [showLandmarksData, setShowLandmarksData] = useState(false);
  const [error, setError] = useState(null);

  const canvasRef = useRef(null);

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io(BACKEND_URL);

    newSocket.on('connect', () => {
      console.log('[FLAME] Connected to backend');
      setError(null);
    });

    newSocket.on('face_mesh_frame', (data) => {
      // Store camera frame
      if (data.frame) {
        setCurrentFrame(data.frame);
      }

      // Extract FLAME mesh and landmarks from first detected face
      if (data.faces && data.faces.length > 0) {
        const face = data.faces[0];
        if (face.flame_mesh) {
          setFlameMeshData(face.flame_mesh);
        }
        if (face.landmarks_3d) {
          setLandmarks3d(face.landmarks_3d);
          setFaceRotationMatrix(face.rotation_matrix);
        }
      }
      setFps(data.fps || 0);
    });

    newSocket.on('disconnect', () => {
      console.log('[FLAME] Disconnected from backend');
    });

    newSocket.on('connect_error', () => {
      setError('Cannot connect to backend. Is it running?');
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  // Draw landmarks on canvas when frame or landmarks update
  useEffect(() => {
    if (!currentFrame || !canvasRef.current || !showLandmarks) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Create image from base64 frame
    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw landmarks if available
      if (landmarks3d && landmarks3d.length > 0) {
        drawLandmarks(ctx, landmarks3d);
      }
    };
    img.src = `data:image/jpeg;base64,${currentFrame}`;
  }, [currentFrame, landmarks3d, showLandmarks]);

  const drawLandmarks = (ctx, landmarks) => {
    // Draw pose landmarks (used for head tracking) - CYAN
    ctx.fillStyle = '#00ffff';
    ctx.strokeStyle = '#00ffff';
    POSE_LANDMARKS.forEach(idx => {
      if (landmarks[idx]) {
        const [x, y] = landmarks[idx];
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw expression landmarks (used for FLAME fitting) - MAGENTA
    ctx.fillStyle = '#ff00ff';
    ctx.strokeStyle = '#ff00ff';
    EXPRESSION_LANDMARKS.forEach(idx => {
      if (landmarks[idx]) {
        const [x, y] = landmarks[idx];
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw connections for pose landmarks
    ctx.strokeStyle = '#00ffff44';
    ctx.lineWidth = 1;
    ctx.beginPath();
    // Forehead to chin
    if (landmarks[10] && landmarks[152]) {
      ctx.moveTo(landmarks[10][0], landmarks[10][1]);
      ctx.lineTo(landmarks[152][0], landmarks[152][1]);
    }
    // Left cheek to right cheek
    if (landmarks[234] && landmarks[454]) {
      ctx.moveTo(landmarks[234][0], landmarks[234][1]);
      ctx.lineTo(landmarks[454][0], landmarks[454][1]);
    }
    ctx.stroke();

    // Draw mouth outline
    ctx.strokeStyle = '#ff00ff44';
    ctx.lineWidth = 1;
    const mouthPoints = [61, 13, 291, 14, 61];
    ctx.beginPath();
    mouthPoints.forEach((idx, i) => {
      if (landmarks[idx]) {
        if (i === 0) {
          ctx.moveTo(landmarks[idx][0], landmarks[idx][1]);
        } else {
          ctx.lineTo(landmarks[idx][0], landmarks[idx][1]);
        }
      }
    });
    ctx.stroke();

    // Draw legend
    ctx.font = '12px monospace';
    ctx.fillStyle = '#00ffff';
    ctx.fillText('Pose (head tracking)', 10, 20);
    ctx.fillStyle = '#ff00ff';
    ctx.fillText('Expression (FLAME)', 10, 35);
  };

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
    } catch (err) {
      console.error('Error fetching cameras:', err);
      setError('Failed to fetch cameras. Is backend running?');
    }
  };

  const startReconstruction = async () => {
    if (selectedCamera === null) return;
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/api/face_mesh/start/${selectedCamera}`);
      const data = await response.json();

      if (data.status === 'success') {
        setIsStreaming(true);
      } else {
        setError(data.message || 'Failed to start reconstruction');
      }
    } catch (err) {
      console.error('Error starting reconstruction:', err);
      setError('Failed to start reconstruction');
    }
  };

  const stopReconstruction = async () => {
    setIsStreaming(false);
    setFlameMeshData(null);
    setCurrentFrame(null);
    setLandmarks3d(null);
    setFps(0);

    try {
      await fetch(`${BACKEND_URL}/api/face_mesh/stop`);
    } catch (err) {
      console.error('Error stopping reconstruction:', err);
    }
  };

  const saveFaceLandmarks = async () => {
    if (!landmarks3d || landmarks3d.length === 0) {
      alert('No face detected! Please ensure your face is visible in the camera.');
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/face_mesh/save_landmarks`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          landmarks_3d: landmarks3d
        })
      });

      const data = await response.json();

      if (data.status === 'success') {
        alert(`✓ Face landmarks saved!\n${data.message}\nShape: ${data.shape}`);
      } else {
        alert(`Error: ${data.message}`);
      }
    } catch (err) {
      console.error('Error saving landmarks:', err);
      alert('Failed to save landmarks. Check console for details.');
    }
  };

  // Extract parameter visualization data
  const flameParams = flameMeshData?.flame_params;
  const shapeParams = flameParams?.shape?.slice(0, 10) || [];
  const expressionParams = flameParams?.expression?.slice(0, 10) || [];

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <h1 style={styles.title}>FLAME 3D Reconstruction</h1>
        <div style={styles.stats}>
          <span style={styles.stat}>FPS: {fps}</span>
          {flameMeshData && (
            <>
              <span style={styles.stat}>{flameMeshData.vertices?.length || 0} vertices</span>
              <span style={styles.stat}>{flameMeshData.fit_time_ms?.toFixed(1) || 0}ms fit</span>
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      <div style={styles.controls}>
        <select
          value={selectedCamera ?? ''}
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
          <button onClick={startReconstruction} style={styles.button}>
            Start Reconstruction
          </button>
        ) : (
          <button onClick={stopReconstruction} style={{ ...styles.button, ...styles.stopButton }}>
            Stop
          </button>
        )}

        <button
          onClick={() => setShowLandmarks(!showLandmarks)}
          style={{ ...styles.button, ...(showLandmarks ? styles.activeToggle : styles.toggleButton) }}
        >
          Landmarks
        </button>

        <button
          onClick={() => setWireframe(!wireframe)}
          style={{ ...styles.button, ...styles.toggleButton }}
        >
          {wireframe ? 'Solid' : 'Wireframe'}
        </button>

        <button
          onClick={() => setShowParams(!showParams)}
          style={{ ...styles.button, ...styles.toggleButton }}
        >
          {showParams ? 'Hide Params' : 'Show Params'}
        </button>

        <button
          onClick={() => setShowLandmarksData(!showLandmarksData)}
          style={{ ...styles.button, ...(showLandmarksData ? styles.activeToggle : styles.toggleButton) }}
          disabled={!isStreaming}
        >
          📊 Landmarks Data
        </button>

        <button
          onClick={saveFaceLandmarks}
          disabled={!isStreaming || !landmarks3d}
          style={{
            ...styles.button,
            ...styles.saveButton,
            opacity: (!isStreaming || !landmarks3d) ? 0.5 : 1,
            cursor: (!isStreaming || !landmarks3d) ? 'not-allowed' : 'pointer',
            display: 'none'
          }}
          title="Save current face landmarks to mp_neutral_frame.npy"
        >
          💾 Save Face
        </button>
      </div>

      {/* Error display */}
      {error && (
        <div style={styles.error}>
          {error}
        </div>
      )}

      {/* Main content - Split view */}
      <div style={styles.content}>
        {/* Left: Camera + Landmarks */}
        <div style={styles.cameraContainer}>
          <div style={styles.sectionLabel}>Camera + MediaPipe Landmarks</div>
          {isStreaming && currentFrame ? (
            <canvas
              ref={canvasRef}
              style={styles.canvas}
            />
          ) : (
            <div style={styles.placeholder}>
              <div style={styles.placeholderIcon}>
                <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#666" strokeWidth="1">
                  <rect x="2" y="3" width="20" height="14" rx="2" />
                  <circle cx="12" cy="10" r="3" />
                </svg>
              </div>
              <p style={styles.placeholderText}>Camera feed</p>
            </div>
          )}
        </div>

        {/* Center: 3D FLAME Mesh */}
        <div style={styles.viewerContainer}>
          <div style={styles.sectionLabel}>FLAME 3D Mesh</div>
          {isStreaming ? (
            <FlameMeshViewer
              flameMeshData={flameMeshData}
              wireframe={wireframe}
              showStats={false}
              meshColor="#00ff88"
              backgroundColor="#0a0a0a"
            />
          ) : (
            <div style={styles.placeholder}>
              <div style={styles.placeholderIcon}>
                <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#00ff88" strokeWidth="1">
                  <circle cx="12" cy="8" r="4" />
                  <path d="M4 20c0-4 4-6 8-6s8 2 8 6" />
                </svg>
              </div>
              <p style={styles.placeholderText}>3D reconstruction</p>
            </div>
          )}
        </div>

        {/* Right: Parameters Panel */}
        {showParams && (
          <div style={styles.paramsPanel}>
            <h2 style={styles.panelTitle}>FLAME Parameters</h2>

            {/* Shape Parameters */}
            <div style={styles.paramSection}>
              <h3 style={styles.paramSectionTitle}>Shape (Identity)</h3>
              {shapeParams.length > 0 ? (
                shapeParams.map((val, idx) => (
                  <ParamBar key={`shape-${idx}`} label={`b${idx}`} value={val} />
                ))
              ) : (
                <p style={styles.noData}>Waiting...</p>
              )}
            </div>

            {/* Expression Parameters */}
            <div style={styles.paramSection}>
              <h3 style={styles.paramSectionTitle}>Expression</h3>
              {expressionParams.length > 0 ? (
                expressionParams.map((val, idx) => (
                  <ParamBar key={`expr-${idx}`} label={`e${idx}`} value={val} />
                ))
              ) : (
                <p style={styles.noData}>Waiting...</p>
              )}
            </div>

            {/* Live Landmarks Display */}
            <div style={styles.landmarksSection}>
              <h3 style={styles.paramSectionTitle}>Live Landmarks (Sample)</h3>
              <h3 style={styles.paramSectionTitle}>
                    Rotation Matrix: {faceRotationMatrix}
                  </h3>
              {landmarks3d && landmarks3d.length > 0 ? (
                <div style={styles.landmarksContainer}>
                  {/* Show first 5 landmarks as example */}
                  {landmarks3d.slice(0, 5).map((lm, idx) => (
                    <div key={idx} style={styles.landmarkRow}>
                      <span style={styles.landmarkIndex}>#{idx}</span>
                      <div style={styles.landmarkCoords}>
                        <span style={styles.coordLabel}>X:</span>
                        <span style={styles.coordValue}>{lm[0]?.toFixed(1) || '0.0'}</span>
                        <span style={styles.coordLabel}>Y:</span>
                        <span style={styles.coordValue}>{lm[1]?.toFixed(1) || '0.0'}</span>
                        <span style={styles.coordLabel}>Z:</span>
                        <span style={styles.coordValue}>{lm[2]?.toFixed(1) || '0.0'}</span>
                      </div>
                    </div>
                  ))}
                  <div style={styles.landmarkSummary}>
                    Total: {landmarks3d.length} landmarks
                  </div>
                  
                </div>
              ) : (
                <p style={styles.noData}>No landmarks detected</p>
              )}
            </div>

            {/* Landmark Legend */}
            <div style={styles.legendSection}>
              <h3 style={styles.paramSectionTitle}>Landmark Legend</h3>
              <div style={styles.legendItem}>
                <span style={{ ...styles.legendDot, backgroundColor: '#00ffff' }}></span>
                <span>Pose (head tracking)</span>
              </div>
              <div style={styles.legendItem}>
                <span style={{ ...styles.legendDot, backgroundColor: '#ff00ff' }}></span>
                <span>Expression (FLAME fit)</span>
              </div>
            </div>

            {/* Mesh Info */}
            {flameMeshData && (
              <div style={styles.meshInfo}>
                <h3 style={styles.paramSectionTitle}>Mesh Info</h3>
                <div style={styles.infoRow}>
                  <span>Vertices:</span>
                  <span style={styles.infoValue}>{flameMeshData.vertices?.length || 0}</span>
                </div>
                <div style={styles.infoRow}>
                  <span>Faces:</span>
                  <span style={styles.infoValue}>{flameMeshData.faces?.length || 0}</span>
                </div>
                <div style={styles.infoRow}>
                  <span>Fit Time:</span>
                  <span style={styles.infoValue}>{flameMeshData.fit_time_ms?.toFixed(1) || 0}ms</span>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Floating Landmarks Data Panel */}
      {showLandmarksData && landmarks3d && (
        <div style={styles.floatingPanel}>
          <div style={styles.floatingHeader}>
            <h3 style={styles.floatingTitle}>Live Landmarks Data (updating every frame)</h3>
            <button
              onClick={() => setShowLandmarksData(false)}
              style={styles.closeButton}
            >
              ✕
            </button>
          </div>
          <div style={styles.floatingContent}>
            <div style={styles.landmarksGrid}>
              {landmarks3d.slice(0, 20).map((lm, idx) => (
                <div key={idx} style={styles.landmarkCard}>
                  <div style={styles.landmarkCardHeader}>
                    Landmark #{idx}
                  </div>
                  <div style={styles.landmarkCardBody}>
                    <div style={styles.coordRow}>
                      <span style={styles.coordLabelLarge}>X:</span>
                      <span style={styles.coordValueLarge}>{lm[0]?.toFixed(2) || '0.00'}</span>
                    </div>
                    <div style={styles.coordRow}>
                      <span style={styles.coordLabelLarge}>Y:</span>
                      <span style={styles.coordValueLarge}>{lm[1]?.toFixed(2) || '0.00'}</span>
                    </div>
                    <div style={styles.coordRow}>
                      <span style={styles.coordLabelLarge}>Z:</span>
                      <span style={styles.coordValueLarge}>{lm[2]?.toFixed(2) || '0.00'}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div style={styles.floatingFooter}>
              Showing 20 of {landmarks3d.length} landmarks • Updates in real-time
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Parameter visualization bar component
 */
const ParamBar = ({ label, value }) => {
  const normalizedValue = Math.max(-3, Math.min(3, value));
  const isPositive = value >= 0;

  return (
    <div style={styles.paramRow}>
      <span style={styles.paramLabel}>{label}</span>
      <div style={styles.paramBarContainer}>
        <div style={styles.paramBarBackground}>
          <div style={styles.paramBarCenter} />
          <div
            style={{
              ...styles.paramBarFill,
              width: `${Math.abs(normalizedValue) / 3 * 50}%`,
              left: isPositive ? '50%' : `${50 - Math.abs(normalizedValue) / 3 * 50}%`,
              backgroundColor: isPositive ? '#00ff88' : '#ff6b6b',
            }}
          />
        </div>
      </div>
      <span style={styles.paramValue}>{value.toFixed(2)}</span>
    </div>
  );
};

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#1a1a1a',
    color: '#ffffff',
    padding: '15px',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '15px',
  },
  title: {
    margin: 0,
    fontSize: '1.5em',
    color: '#00ff88',
  },
  stats: {
    display: 'flex',
    gap: '10px',
  },
  stat: {
    backgroundColor: '#2a2a2a',
    padding: '6px 12px',
    borderRadius: '6px',
    fontSize: '0.85em',
    fontFamily: 'monospace',
    color: '#00ff88',
  },
  controls: {
    display: 'flex',
    gap: '8px',
    marginBottom: '15px',
    flexWrap: 'wrap',
  },
  select: {
    padding: '8px 12px',
    fontSize: '0.9em',
    borderRadius: '6px',
    border: '2px solid #444',
    backgroundColor: '#2a2a2a',
    color: '#fff',
    cursor: 'pointer',
    minWidth: '180px',
  },
  button: {
    padding: '8px 16px',
    fontSize: '0.9em',
    borderRadius: '6px',
    border: 'none',
    backgroundColor: '#00ff88',
    color: '#000',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'all 0.2s ease',
  },
  stopButton: {
    backgroundColor: '#ff4444',
    color: '#fff',
  },
  toggleButton: {
    backgroundColor: '#444',
    color: '#fff',
  },
  activeToggle: {
    backgroundColor: '#00aaff',
    color: '#fff',
  },
  saveButton: {
    backgroundColor: '#9b59b6',
    color: '#fff',
  },
  error: {
    backgroundColor: '#ff444433',
    border: '1px solid #ff4444',
    color: '#ff6b6b',
    padding: '10px 15px',
    borderRadius: '6px',
    marginBottom: '15px',
  },
  content: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr 280px',
    gap: '15px',
    height: 'calc(100vh - 150px)',
  },
  cameraContainer: {
    backgroundColor: '#0a0a0a',
    borderRadius: '12px',
    overflow: 'hidden',
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
  },
  sectionLabel: {
    position: 'absolute',
    top: '10px',
    left: '10px',
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: '4px 10px',
    borderRadius: '4px',
    fontSize: '0.8em',
    color: '#888',
    zIndex: 10,
  },
  canvas: {
    width: '100%',
    height: '100%',
    objectFit: 'contain',
  },
  viewerContainer: {
    backgroundColor: '#0a0a0a',
    borderRadius: '12px',
    overflow: 'hidden',
    position: 'relative',
  },
  placeholder: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    color: '#666',
  },
  placeholderIcon: {
    marginBottom: '15px',
    opacity: 0.5,
  },
  placeholderText: {
    fontSize: '1em',
    margin: 0,
    color: '#555',
  },
  paramsPanel: {
    backgroundColor: '#2a2a2a',
    borderRadius: '12px',
    padding: '15px',
    overflowY: 'auto',
  },
  panelTitle: {
    margin: '0 0 15px 0',
    fontSize: '1.1em',
    color: '#00ff88',
  },
  paramSection: {
    marginBottom: '20px',
  },
  paramSectionTitle: {
    margin: '0 0 10px 0',
    fontSize: '0.85em',
    color: '#888',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  paramRow: {
    display: 'grid',
    gridTemplateColumns: '30px 1fr 45px',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '4px',
  },
  paramLabel: {
    fontSize: '0.75em',
    color: '#666',
    fontFamily: 'monospace',
  },
  paramBarContainer: {
    height: '6px',
  },
  paramBarBackground: {
    position: 'relative',
    height: '100%',
    backgroundColor: '#1a1a1a',
    borderRadius: '3px',
    overflow: 'hidden',
  },
  paramBarCenter: {
    position: 'absolute',
    left: '50%',
    top: 0,
    bottom: 0,
    width: '1px',
    backgroundColor: '#444',
  },
  paramBarFill: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    borderRadius: '2px',
    transition: 'width 0.1s ease',
  },
  paramValue: {
    fontSize: '0.7em',
    color: '#888',
    fontFamily: 'monospace',
    textAlign: 'right',
  },
  noData: {
    color: '#555',
    fontSize: '0.85em',
    fontStyle: 'italic',
  },
  landmarksSection: {
    marginBottom: '20px',
    paddingTop: '10px',
    borderTop: '1px solid #444',
  },
  landmarksContainer: {
    backgroundColor: '#1a1a1a',
    borderRadius: '6px',
    padding: '8px',
    maxHeight: '200px',
    overflowY: 'auto',
  },
  landmarkRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '6px',
    padding: '4px',
    backgroundColor: '#2a2a2a',
    borderRadius: '4px',
    fontSize: '0.75em',
  },
  landmarkIndex: {
    color: '#00ff88',
    fontFamily: 'monospace',
    fontWeight: 'bold',
    minWidth: '30px',
  },
  landmarkCoords: {
    display: 'flex',
    gap: '6px',
    flex: 1,
    fontFamily: 'monospace',
  },
  coordLabel: {
    color: '#666',
    fontSize: '0.9em',
  },
  coordValue: {
    color: '#00aaff',
    fontWeight: 'bold',
    minWidth: '40px',
  },
  landmarkSummary: {
    marginTop: '8px',
    padding: '4px',
    fontSize: '0.75em',
    color: '#888',
    textAlign: 'center',
    borderTop: '1px solid #333',
    paddingTop: '8px',
  },
  legendSection: {
    marginBottom: '20px',
    paddingTop: '10px',
    borderTop: '1px solid #444',
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '6px',
    fontSize: '0.8em',
    color: '#888',
  },
  legendDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
  },
  meshInfo: {
    borderTop: '1px solid #444',
    paddingTop: '15px',
  },
  infoRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '4px 0',
    fontSize: '0.85em',
    color: '#888',
  },
  infoValue: {
    color: '#00ff88',
    fontFamily: 'monospace',
  },
  floatingPanel: {
    position: 'fixed',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: '80%',
    maxWidth: '900px',
    maxHeight: '80vh',
    backgroundColor: '#1a1a1a',
    border: '2px solid #00ff88',
    borderRadius: '12px',
    boxShadow: '0 10px 40px rgba(0,0,0,0.8)',
    zIndex: 1000,
    display: 'flex',
    flexDirection: 'column',
  },
  floatingHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '15px 20px',
    borderBottom: '1px solid #333',
    backgroundColor: '#2a2a2a',
    borderRadius: '10px 10px 0 0',
  },
  floatingTitle: {
    margin: 0,
    fontSize: '1.1em',
    color: '#00ff88',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: '#888',
    fontSize: '1.5em',
    cursor: 'pointer',
    padding: '0 8px',
    lineHeight: 1,
    transition: 'color 0.2s',
  },
  floatingContent: {
    padding: '20px',
    overflowY: 'auto',
    flex: 1,
  },
  landmarksGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
    gap: '12px',
  },
  landmarkCard: {
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
    padding: '10px',
    border: '1px solid #333',
    transition: 'border-color 0.2s',
  },
  landmarkCardHeader: {
    fontSize: '0.75em',
    color: '#00ff88',
    fontWeight: 'bold',
    marginBottom: '8px',
    fontFamily: 'monospace',
  },
  landmarkCardBody: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  coordRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  coordLabelLarge: {
    color: '#666',
    fontSize: '0.85em',
    fontFamily: 'monospace',
  },
  coordValueLarge: {
    color: '#00aaff',
    fontSize: '0.9em',
    fontWeight: 'bold',
    fontFamily: 'monospace',
  },
  floatingFooter: {
    marginTop: '15px',
    padding: '10px',
    textAlign: 'center',
    fontSize: '0.85em',
    color: '#888',
    borderTop: '1px solid #333',
  },
};

export default FlameReconPanel;
