import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { getFlatTriangleIndices, getContourEdgeIndices } from '../utils/faceMeshTopology';
import FacePoseEstimator from '../utils/facePoseEstimation';
import { FacialAnalyzer } from '../utils/facialAnalysis';

function FaceMeshSurface({ landmarks, renderMode = 'solid', stabilize = false, onPoseUpdate, onAnalysisUpdate }) {
  const meshRef = useRef();
  const contourRef = useRef();
  const groupRef = useRef();
  const [triangleIndices] = useState(() => getFlatTriangleIndices());
  const [contourEdges] = useState(() => getContourEdgeIndices());

  // Initialize analyzers
  const poseEstimator = useMemo(() => new FacePoseEstimator(), []);
  const facialAnalyzer = useMemo(() => new FacialAnalyzer(), []);

  useEffect(() => {
    if (!landmarks || landmarks.length === 0 || !meshRef.current) return;

    // STEP 4: Estimate face pose
    const poseData = poseEstimator.estimatePose(landmarks);

    // STEP 2: Center vertices around face centroid
    // Calculate centroid (geometric center of all landmarks)
    let cx = 0, cy = 0, cz = 0;
    landmarks.forEach(landmark => {
      cx += landmark[0];
      cy += landmark[1];
      cz += landmark[2];
    });
    cx /= landmarks.length;
    cy /= landmarks.length;
    cz /= landmarks.length;

    // Create vertex positions array
    const positions = new Float32Array(landmarks.length * 3);

    // Define face coordinate system:
    // X-axis: left (-) to right (+)
    // Y-axis: down (-) to up (+)  [inverted for correct orientation]
    // Z-axis: back (-) to front (+) [inverted for correct depth]
    const scale = 160; // Normalization scale factor

    // Store centered landmarks for analysis
    const centeredLandmarks = [];

    landmarks.forEach((landmark, i) => {
      // Center around origin (face centroid)
      let x = (landmark[0] - cx) / scale;
      let y = -(landmark[1] - cy) / scale;  // Invert Y for correct up/down
      let z = -(landmark[2] - cz) / scale;  // Invert Z for correct depth

      // STEP 4: Apply stabilization if enabled
      if (stabilize && poseData) {
        const stabilizationMatrix = poseEstimator.getStabilizationMatrix(poseData.rotationMatrix);
        const stabilized = poseEstimator.applyRotation([x, y, z], stabilizationMatrix);
        x = stabilized[0];
        y = stabilized[1];
        z = stabilized[2];
      }

      positions[i * 3]     = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;

      centeredLandmarks.push([x, y, z]);
    });

    // STEP 5: Perform facial analysis
    const analysis = facialAnalyzer.analyze(landmarks, centeredLandmarks);

    // Update mesh geometry vertices
    const geometry = meshRef.current.geometry;
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // STEP 3: Triangle indices are already set in geometry initialization
    // Topology is locked - only vertex positions update

    // STEP 2 & 3: Compute surface normals for proper lighting
    // This ensures faces are lit correctly based on their orientation
    geometry.computeVertexNormals();

    // Mark for update
    geometry.attributes.position.needsUpdate = true;
    geometry.computeBoundingSphere();

    // Send pose and analysis data to parent
    if (onPoseUpdate && poseData) {
      onPoseUpdate(poseData);
    }
    if (onAnalysisUpdate && analysis) {
      onAnalysisUpdate(analysis);
    }

    // Update contour lines geometry if in contour mode
    if (contourRef.current && renderMode === 'contours') {
      const contourGeometry = contourRef.current.geometry;
      contourGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      contourGeometry.attributes.position.needsUpdate = true;
    }

  }, [landmarks, stabilize, poseEstimator, facialAnalyzer, onPoseUpdate, onAnalysisUpdate, renderMode]);

  return (
    <group ref={groupRef}>
      {/* Main mesh surface */}
      <mesh ref={meshRef} position={[0, 0, -2]}>
        {/* BufferGeometry with locked topology from MediaPipe */}
        <bufferGeometry>
          {/* Vertex positions - updated dynamically */}
          <bufferAttribute
            attach="attributes-position"
            count={landmarks?.length || 468}
            array={new Float32Array((landmarks?.length || 468) * 3)}
            itemSize={3}
          />
          {/* Triangle indices - locked topology, CCW winding */}
          <bufferAttribute
            attach="index"
            array={new Uint16Array(triangleIndices)}
            count={triangleIndices.length}
            itemSize={1}
          />
        </bufferGeometry>

        {/* Material based on render mode */}
        {renderMode === 'wireframe' ? (
          <meshBasicMaterial
            color="0x00ff00"
            wireframe={true}
            side={THREE.DoubleSide}
          />
        ) : (
          <meshStandardMaterial
            color="#00ff00"
            wireframe={true}
            side={THREE.FrontSide}
            flatShading={false}
            metalness={0.2}
            roughness={0.8}
            emissive="#003300"
            emissiveIntensity={0.1}
          />
        )}
      </mesh>

      {/* Selective contour edges (cleaner than full wireframe) */}
      {renderMode === 'contours' && (
        <lineSegments ref={contourRef} position={[0, 0, -2]}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={landmarks?.length || 468}
              array={new Float32Array((landmarks?.length || 468) * 3)}
              itemSize={3}
            />
            <bufferAttribute
              attach="index"
              array={new Uint16Array(contourEdges)}
              count={contourEdges.length}
              itemSize={1}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#00ff00" linewidth={2} />
        </lineSegments>
      )}
    </group>
  );
}

function AxisHelper() {
  return (
    <group position={[0, 0, -2]}>
      {/* X-axis: Red (left to right) */}
      <arrowHelper args={[new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0), 1, 0xff0000]} />
      {/* Y-axis: Green (down to up) */}
      <arrowHelper args={[new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0), 1, 0x00ff00]} />
      {/* Z-axis: Blue (back to front) */}
      <arrowHelper args={[new THREE.Vector3(0, 0, 1), new THREE.Vector3(0, 0, -1), 1, 0x0000ff]} />
    </group>
  );
}

const FaceMesh3D = ({ landmarks }) => {
  const [renderMode, setRenderMode] = useState('solid'); // 'solid', 'wireframe', 'contours'
  const [showAxes, setShowAxes] = useState(false);
  const [stabilize, setStabilize] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [poseData, setPoseData] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);

  const handlePoseUpdate = (data) => {
    setPoseData(data);
  };

  const handleAnalysisUpdate = (data) => {
    setAnalysisData(data);
  };

  const cycleRenderMode = () => {
    const modes = ['solid', 'wireframe', 'contours'];
    const currentIndex = modes.indexOf(renderMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    setRenderMode(modes[nextIndex]);
  };

  const getRenderModeIcon = () => {
    switch (renderMode) {
      case 'wireframe': return '🌐';
      case 'contours': return '📐';
      default: return '🎭';
    }
  };

  const getRenderModeName = () => {
    switch (renderMode) {
      case 'wireframe': return 'Wireframe';
      case 'contours': return 'Contours';
      default: return 'Solid';
    }
  };

  if (!landmarks || landmarks.length === 0) {
    return (
      <div style={styles.empty}>
        <p>Waiting for face detection...</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <Canvas camera={{ position: [0, 0, 3], fov: 50 }}>
        {/* Lighting setup for proper mesh rendering */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <directionalLight position={[-5, -5, -5]} intensity={0.3} />
        <pointLight position={[0, 0, 5]} intensity={0.5} />

        {/* Face mesh surface with pose estimation and analysis */}
        <FaceMeshSurface
          landmarks={landmarks}
          renderMode={renderMode}
          stabilize={stabilize}
          onPoseUpdate={handlePoseUpdate}
          onAnalysisUpdate={handleAnalysisUpdate}
        />

        {/* Optional coordinate axes */}
        {showAxes && <AxisHelper />}

        {/* Camera controls */}
        <OrbitControls
          target={[0, 0, -2]}
          enableZoom
          enableRotate
          enablePan
        />

        {/* Grid helper */}
        <gridHelper args={[10, 10, '#333', '#111']} />
      </Canvas>

      {/* Control panel */}
      <div style={styles.controls}>
        <button
          onClick={cycleRenderMode}
          style={{
            ...styles.button,
            backgroundColor: renderMode !== 'solid' ? '#00ff00' : '#333'
          }}
        >
          {getRenderModeIcon()} {getRenderModeName()}
        </button>
        <button
          onClick={() => setShowAxes(!showAxes)}
          style={{
            ...styles.button,
            backgroundColor: showAxes ? '#00ff00' : '#333'
          }}
        >
          {showAxes ? '📍 Axes' : '📍 Axes'}
        </button>
        <button
          onClick={() => setStabilize(!stabilize)}
          style={{
            ...styles.button,
            backgroundColor: stabilize ? '#00ff00' : '#333'
          }}
        >
          {stabilize ? '🎯 Stable' : '🎯 Track'}
        </button>
        <button
          onClick={() => setShowAnalysis(!showAnalysis)}
          style={{
            ...styles.button,
            backgroundColor: showAnalysis ? '#00ff00' : '#333'
          }}
        >
          {showAnalysis ? '📊 Data' : '📊 Data'}
        </button>
        <span style={styles.hint}>🖱️ Drag | Scroll zoom</span>
      </div>

      {/* Analysis panel */}
      {showAnalysis && (poseData || analysisData) && (
        <div style={styles.analysisPanel}>
          {poseData && (
            <div style={styles.poseSection}>
              <h4 style={styles.sectionTitle}>Head Pose</h4>
              <div style={styles.poseGrid}>
                <div style={styles.poseItem}>
                  <span style={styles.poseLabel}>Yaw (L/R):</span>
                  <span style={styles.poseValue}>{poseData.smoothedAngles.yaw.toFixed(1)}°</span>
                  <div style={styles.stabilityBar}>
                    <div style={{...styles.stabilityFill, width: `${(1 - poseData.stability.yaw) * 100}%`}} />
                  </div>
                </div>
                <div style={styles.poseItem}>
                  <span style={styles.poseLabel}>Pitch (U/D):</span>
                  <span style={styles.poseValue}>{poseData.smoothedAngles.pitch.toFixed(1)}°</span>
                  <div style={styles.stabilityBar}>
                    <div style={{...styles.stabilityFill, width: `${(1 - poseData.stability.pitch) * 100}%`}} />
                  </div>
                </div>
                <div style={styles.poseItem}>
                  <span style={styles.poseLabel}>Roll (Tilt):</span>
                  <span style={styles.poseValue}>{poseData.smoothedAngles.roll.toFixed(1)}°</span>
                  <div style={styles.stabilityBar}>
                    <div style={{...styles.stabilityFill, width: `${(1 - poseData.stability.roll) * 100}%`}} />
                  </div>
                </div>
                <div style={styles.poseItem}>
                  <span style={styles.poseLabel}>Stability:</span>
                  <span style={{
                    ...styles.poseValue,
                    color: poseData.stability.overall < 0.3 ? '#00ff00' : poseData.stability.overall < 0.6 ? '#ffff00' : '#ff0000'
                  }}>
                    {((1 - poseData.stability.overall) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {analysisData && (
            <div style={styles.measurementSection}>
              <h4 style={styles.sectionTitle}>Facial Measurements</h4>
              <div style={styles.measurementGrid}>
                <div style={styles.measurementItem}>
                  <span style={styles.measurementLabel}>Eye Distance:</span>
                  <span style={styles.measurementValue}>
                    {analysisData.distances.EYE_TO_EYE?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div style={styles.measurementItem}>
                  <span style={styles.measurementLabel}>Mouth Width:</span>
                  <span style={styles.measurementValue}>
                    {analysisData.distances.MOUTH_WIDTH?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div style={styles.measurementItem}>
                  <span style={styles.measurementLabel}>Mouth Open:</span>
                  <span style={styles.measurementValue}>
                    {analysisData.distances.MOUTH_OPENNESS?.toFixed(2) || 'N/A'}
                  </span>
                </div>
                <div style={styles.measurementItem}>
                  <span style={styles.measurementLabel}>Symmetry:</span>
                  <span style={{
                    ...styles.measurementValue,
                    color: analysisData.symmetry.score > 85 ? '#00ff00' : analysisData.symmetry.score > 70 ? '#ffff00' : '#ff8800'
                  }}>
                    {analysisData.symmetry.score.toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    width: '100%',
    height: '400px',
    position: 'relative',
    backgroundColor: '#000',
    borderRadius: '8px',
    overflow: 'hidden',
  },
  empty: {
    width: '100%',
    height: '400px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#000',
    borderRadius: '8px',
    color: '#666',
  },
  controls: {
    position: 'absolute',
    bottom: '10px',
    left: '50%',
    transform: 'translateX(-50%)',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: '8px 15px',
    borderRadius: '20px',
    display: 'flex',
    gap: '8px',
    alignItems: 'center',
    zIndex: 10,
  },
  button: {
    padding: '5px 12px',
    fontSize: '0.7em',
    border: '1px solid #444',
    borderRadius: '12px',
    cursor: 'pointer',
    color: '#fff',
    transition: 'all 0.2s',
  },
  hint: {
    margin: 0,
    fontSize: '0.7em',
    color: '#888',
    marginLeft: '5px',
  },
  analysisPanel: {
    position: 'absolute',
    top: '10px',
    right: '10px',
    backgroundColor: 'rgba(0, 0, 0, 0.85)',
    padding: '12px',
    borderRadius: '8px',
    maxWidth: '280px',
    maxHeight: '380px',
    overflowY: 'auto',
    fontSize: '0.8em',
    zIndex: 10,
  },
  poseSection: {
    marginBottom: '15px',
  },
  measurementSection: {
    marginTop: '15px',
    paddingTop: '15px',
    borderTop: '1px solid #333',
  },
  sectionTitle: {
    margin: '0 0 10px 0',
    fontSize: '0.9em',
    color: '#00ff00',
    fontWeight: 'bold',
  },
  poseGrid: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  poseItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
  },
  poseLabel: {
    fontSize: '0.85em',
    color: '#888',
  },
  poseValue: {
    fontSize: '1.1em',
    color: '#0ff',
    fontFamily: 'monospace',
    fontWeight: 'bold',
  },
  stabilityBar: {
    width: '100%',
    height: '4px',
    backgroundColor: '#333',
    borderRadius: '2px',
    overflow: 'hidden',
  },
  stabilityFill: {
    height: '100%',
    backgroundColor: '#00ff00',
    transition: 'width 0.3s ease',
  },
  measurementGrid: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '8px',
  },
  measurementItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  measurementLabel: {
    fontSize: '0.75em',
    color: '#888',
  },
  measurementValue: {
    fontSize: '0.95em',
    color: '#0ff',
    fontFamily: 'monospace',
  },
};

export default FaceMesh3D;
