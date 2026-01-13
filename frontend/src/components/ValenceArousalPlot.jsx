import React from 'react';

const ValenceArousalPlot = ({ valence, arousal, zone }) => {
  // Convert valence from -1 to 1 to 0 to 200 (plot coordinates)
  const x = ((valence + 1) / 2) * 200;
  // Convert arousal from 0 to 1 to 0 to 200
  const y = 200 - (arousal * 200); // Invert Y for SVG coordinates

  const getZoneColor = (zone) => {
    switch (zone) {
      case 'positive': return '#00ff00';
      case 'neutral': return '#ffff00';
      case 'negative': return '#ff0000';
      default: return '#888';
    }
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.title}>Valence-Arousal Space</h3>

      <svg width="220" height="220" style={styles.svg}>
        {/* Background quadrants */}
        <rect x="0" y="0" width="100" height="100" fill="rgba(255,0,0,0.1)" />
        <rect x="100" y="0" width="100" height="100" fill="rgba(0,255,0,0.1)" />
        <rect x="0" y="100" width="100" height="100" fill="rgba(255,100,0,0.1)" />
        <rect x="100" y="100" width="100" height="100" fill="rgba(0,100,255,0.1)" />

        {/* Grid lines */}
        <line x1="100" y1="0" x2="100" y2="200" stroke="#444" strokeWidth="1" />
        <line x1="0" y1="100" x2="200" y2="100" stroke="#444" strokeWidth="1" />

        {/* Axes */}
        <line x1="100" y1="0" x2="100" y2="200" stroke="#666" strokeWidth="2" />
        <line x1="0" y1="100" x2="200" y2="100" stroke="#666" strokeWidth="2" />

        {/* Labels */}
        <text x="105" y="15" fill="#888" fontSize="10">High Arousal</text>
        <text x="105" y="195" fill="#888" fontSize="10">Low Arousal</text>
        <text x="5" y="105" fill="#888" fontSize="10">Negative</text>
        <text x="150" y="105" fill="#888" fontSize="10">Positive</text>

        {/* Quadrant labels */}
        <text x="20" y="30" fill="#888" fontSize="9" opacity="0.5">Tense</text>
        <text x="150" y="30" fill="#888" fontSize="9" opacity="0.5">Excited</text>
        <text x="20" y="180" fill="#888" fontSize="9" opacity="0.5">Sad</text>
        <text x="145" y="180" fill="#888" fontSize="9" opacity="0.5">Calm</text>

        {/* Current position */}
        <circle
          cx={x}
          cy={y}
          r="8"
          fill={getZoneColor(zone)}
          stroke="#fff"
          strokeWidth="2"
        />

        {/* Trail effect */}
        <circle
          cx={x}
          cy={y}
          r="12"
          fill="none"
          stroke={getZoneColor(zone)}
          strokeWidth="1"
          opacity="0.5"
        />
      </svg>

      <div style={styles.values}>
        <div style={styles.valueRow}>
          <span style={styles.valueLabel}>Valence:</span>
          <span style={styles.valueNumber}>{valence.toFixed(2)}</span>
        </div>
        <div style={styles.valueRow}>
          <span style={styles.valueLabel}>Arousal:</span>
          <span style={styles.valueNumber}>{arousal.toFixed(2)}</span>
        </div>
        <div style={styles.valueRow}>
          <span style={styles.valueLabel}>Zone:</span>
          <span style={{
            ...styles.valueNumber,
            color: getZoneColor(zone)
          }}>
            {zone.toUpperCase()}
          </span>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: '#1a1a1a',
    borderRadius: '8px',
    padding: '15px',
    marginBottom: '15px',
  },
  title: {
    margin: '0 0 15px 0',
    fontSize: '1.1em',
    color: '#0ff',
  },
  svg: {
    display: 'block',
    margin: '0 auto',
    backgroundColor: '#000',
    borderRadius: '4px',
  },
  values: {
    marginTop: '15px',
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  valueRow: {
    display: 'flex',
    justifyContent: 'space-between',
    padding: '5px',
    backgroundColor: '#2a2a2a',
    borderRadius: '4px',
  },
  valueLabel: {
    color: '#888',
  },
  valueNumber: {
    fontFamily: 'monospace',
    color: '#fff',
    fontWeight: 'bold',
  },
};

export default ValenceArousalPlot;
