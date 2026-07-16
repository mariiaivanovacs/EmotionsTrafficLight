import React, { useState } from 'react';
import EmotionDisplay from './components/EmotionDisplay';
import FaceMeshDisplay from './components/FaceMeshDisplay';
import FlameReconPanel from './components/FlameReconPanel';

function App() {
  const [activeTab, setActiveTab] = useState('emotion');

  return (
    <div style={styles.container}>
      <div style={styles.tabBar}>
        <button
          onClick={() => setActiveTab('emotion')}
          style={{
            ...styles.tab,
            ...(activeTab === 'emotion' ? styles.activeTab : {})
          }}
        >
          🎭 Emotion Detection (FER)
        </button>
        <button
          onClick={() => setActiveTab('facemesh')}
          style={{
            ...styles.tab,
            ...(activeTab === 'facemesh' ? styles.activeTab : {})
          }}
        >
          🎯 Face Mesh Analysis (3D)
        </button>
        <button
          onClick={() => setActiveTab('flame3d')}
          style={{
            ...styles.tab,
            ...(activeTab === 'flame3d' ? styles.activeTab : {})
          }}
        >
          🔥 FLAME 3D Reconstruction
        </button>
      </div>

      <div style={styles.content}>
        {activeTab === 'emotion' && <EmotionDisplay />}
        {activeTab === 'facemesh' && <FaceMeshDisplay />}
        {activeTab === 'flame3d' && <FlameReconPanel />}
      </div>
    </div>
  );
}

const styles = {
  container: {
    minHeight: '100vh',
    backgroundColor: '#1a1a1a',
  },
  tabBar: {
    display: 'flex',
    backgroundColor: '#2a2a2a',
    borderBottom: '2px solid #444',
  },
  tab: {
    flex: 1,
    padding: '15px 30px',
    border: 'none',
    background: 'none',
    color: '#888',
    fontSize: '1.1em',
    cursor: 'pointer',
    transition: 'all 0.3s ease',
  },
  activeTab: {
    color: '#fff',
    backgroundColor: '#1a1a1a',
    borderBottom: '3px solid #00ff00',
  },
  content: {
    minHeight: 'calc(100vh - 60px)',
  },
};

export default App;
