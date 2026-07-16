/**
 * FlameMeshViewer - Real-time FLAME 3D Face Mesh Visualization
 *
 * Displays parametric FLAME face model with:
 * - 5023 vertices
 * - 9976 triangular faces
 * - Vertex normals for proper lighting
 * - Real-time updates from webcam
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

/**
 * FLAME Mesh Component - Renders the 3D face mesh
 */
function FlameMesh({ vertices, faces, normals, color = '#00ff88', wireframe = false }) {
  const meshRef = useRef();
  const geometryRef = useRef();

  useEffect(() => {
    if (!vertices || !faces || vertices.length === 0 || faces.length === 0) {
      return;
    }

    // Create or update geometry
    if (!geometryRef.current) {
      geometryRef.current = new THREE.BufferGeometry();
    }

    const geometry = geometryRef.current;

    // ========================================
    // COORDINATE TRANSFORMATION
    // ========================================
    // MediaPipe coordinates: (0,0) = top-left, Y-down, pixel units
    // Three.js coordinates: (0,0,0) = center, Y-up, normalized units
    //
    // We need to:
    // 1. Center the mesh (subtract centroid)
    // 2. Flip Y-axis (MediaPipe Y-down → Three.js Y-up)
    // 3. Normalize scale (pixels → normalized units)
    // ========================================

    // Calculate centroid for centering
    let cx = 0, cy = 0, cz = 0;
    vertices.forEach(v => {
      cx += v[0];
      cy += v[1];
      cz += v[2];
    });
    cx /= vertices.length;
    cy /= vertices.length;
    cz /= vertices.length;

    // Normalization scale (adjust based on typical face size in pixels)
    const scale = 200; // Typical face width in pixels

    // Convert vertices to Float32Array with proper coordinate transformation
    const positionArray = new Float32Array(vertices.length * 3);
    vertices.forEach((vertex, i) => {
      // Center and normalize
      const x = (vertex[0] - cx) / scale;
      const y = -(vertex[1] - cy) / scale;  // ✅ Flip Y-axis
      const z = (vertex[2] - cz) / scale;

      positionArray[i * 3] = x;
      positionArray[i * 3 + 1] = y;
      positionArray[i * 3 + 2] = z;
    });

    // Convert faces to Uint32Array
    // faces = [[i, j, k], ...] -> [i, j, k, i, j, k, ...]
    const indexArray = new Uint32Array(faces.length * 3);
    faces.forEach((face, i) => {
      indexArray[i * 3] = face[0];
      indexArray[i * 3 + 1] = face[1];
      indexArray[i * 3 + 2] = face[2];
    });

    // Set positions and indices
    geometry.setAttribute('position', new THREE.BufferAttribute(positionArray, 3));
    geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));

    // Set normals if provided, otherwise compute automatically
    if (normals && normals.length === vertices.length) {
      const normalArray = new Float32Array(normals.length * 3);
      normals.forEach((normal, i) => {
        normalArray[i * 3] = normal[0];
        normalArray[i * 3 + 1] = normal[1];
        normalArray[i * 3 + 2] = normal[2];
      });
      geometry.setAttribute('normal', new THREE.BufferAttribute(normalArray, 3));
    } else {
      // Compute normals automatically
      geometry.computeVertexNormals();
    }

    // Update geometry
    geometry.attributes.position.needsUpdate = true;
    if (geometry.attributes.normal) {
      geometry.attributes.normal.needsUpdate = true;
    }
    geometry.computeBoundingSphere();

    // Update mesh
    if (meshRef.current) {
      meshRef.current.geometry = geometry;
    }
  }, [vertices, faces, normals]);

  // ⚠️ Animation disabled - mesh updates come from real-time face tracking
  // Breathing effect was causing unwanted zooming
  // useFrame((state) => {
  //   if (meshRef.current) {
  //     const breathScale = 1 + Math.sin(state.clock.elapsedTime * 0.5) * 0.01;
  //     meshRef.current.scale.set(breathScale, breathScale, breathScale);
  //   }
  // });

  if (!vertices || vertices.length === 0) {
    return null;
  }

  return (
    <mesh ref={meshRef}>
      <bufferGeometry ref={geometryRef} />
      <meshStandardMaterial
        color={color}
        wireframe={wireframe}
        side={THREE.DoubleSide}
        flatShading={false}
        metalness={0.2}
        roughness={0.7}
      />
    </mesh>
  );
}

/**
 * Scene Setup - Lights, Camera, Controls
 */
function SceneSetup() {
  return (
    <>
      {/* Camera - positioned to view face from front */}
      <PerspectiveCamera makeDefault position={[0, 0, 3]} fov={50} />

      {/* Lights - optimized for face viewing */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[2, 2, 3]} intensity={0.8} castShadow />
      <directionalLight position={[-2, -1, 2]} intensity={0.3} />
      <pointLight position={[0, 0, 2]} intensity={0.5} />

      {/* Controls - optimized for face interaction */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={1.5}
        maxDistance={10}
        target={[0, 0, 0]}
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
      />

      {/* Grid Helper (optional - for debugging) */}
      {/* <gridHelper args={[5, 10]} /> */}
    </>
  );
}

/**
 * Performance Stats Display
 */
function PerformanceStats({ flameMeshData }) {
  if (!flameMeshData) return null;

  return (
    <div style={{
      position: 'absolute',
      top: '10px',
      right: '10px',
      background: 'rgba(0, 0, 0, 0.7)',
      color: '#00ff88',
      padding: '10px',
      borderRadius: '5px',
      fontFamily: 'monospace',
      fontSize: '12px',
      zIndex: 10
    }}>
      <div><strong>FLAME Mesh Stats:</strong></div>
      <div>Vertices: {flameMeshData.vertices?.length || 0}</div>
      <div>Faces: {flameMeshData.faces?.length || 0}</div>
      <div>Fit Time: {flameMeshData.fit_time_ms?.toFixed(1) || 0} ms</div>
      <div>Landmarks Used: {flameMeshData.num_landmarks_used || 0}</div>
    </div>
  );
}

/**
 * Main FLAME Mesh Viewer Component
 */
export default function FlameMeshViewer({
  flameMeshData,
  showStats = true,
  wireframe = false,
  meshColor = '#00ff88',
  backgroundColor = '#0a0a0a'
}) {
  const [displayMode, setDisplayMode] = useState('solid'); // 'solid' or 'wireframe'

  // Extract mesh data
  const vertices = flameMeshData?.vertices || [];
  const faces = flameMeshData?.faces || [];
  const normals = flameMeshData?.normals || [];

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {/* Performance Stats */}
      {showStats && <PerformanceStats flameMeshData={flameMeshData} />}

      {/* Controls */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        zIndex: 10,
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '10px',
        borderRadius: '5px'
      }}>
        <button
          onClick={() => setDisplayMode(displayMode === 'solid' ? 'wireframe' : 'solid')}
          style={{
            padding: '5px 10px',
            background: '#00ff88',
            border: 'none',
            borderRadius: '3px',
            cursor: 'pointer',
            fontFamily: 'monospace',
            fontSize: '12px'
          }}
        >
          {displayMode === 'solid' ? '📐 Wireframe' : '🎨 Solid'}
        </button>
      </div>

      {/* Status Message */}
      {vertices.length === 0 && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          color: '#00ff88',
          fontFamily: 'monospace',
          fontSize: '16px',
          textAlign: 'center',
          zIndex: 10
        }}>
          <div>⏳ Waiting for FLAME mesh data...</div>
          <div style={{ fontSize: '12px', marginTop: '10px', opacity: 0.7 }}>
            Start face detection to see 3D reconstruction
          </div>
        </div>
      )}

      {/* 3D Canvas */}
      <Canvas
        style={{ background: backgroundColor }}
        shadows
        gl={{ antialias: true, alpha: true }}
      >
        <SceneSetup />
        <FlameMesh
          vertices={vertices}
          faces={faces}
          normals={normals}
          color={meshColor}
          wireframe={wireframe || displayMode === 'wireframe'}
        />
      </Canvas>
    </div>
  );
}

/**
 * Variant: Side-by-side comparison view
 */
export function FlameMeshComparison({ flameMeshData }) {
  return (
    <div style={{ display: 'flex', width: '100%', height: '100%' }}>
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          color: 'white',
          fontFamily: 'monospace',
          zIndex: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          padding: '5px 10px',
          borderRadius: '3px'
        }}>
          Solid Mesh
        </div>
        <FlameMeshViewer flameMeshData={flameMeshData} wireframe={false} showStats={false} />
      </div>
      <div style={{ flex: 1, position: 'relative' }}>
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '10px',
          color: 'white',
          fontFamily: 'monospace',
          zIndex: 10,
          background: 'rgba(0, 0, 0, 0.7)',
          padding: '5px 10px',
          borderRadius: '3px'
        }}>
          Wireframe
        </div>
        <FlameMeshViewer flameMeshData={flameMeshData} wireframe={true} showStats={false} />
      </div>
    </div>
  );
}
