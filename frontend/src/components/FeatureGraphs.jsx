import React from 'react';

const FeatureGraphs = ({ temporalFeatures }) => {
  if (!temporalFeatures) {
    return <p style={styles.noData}>No temporal data yet</p>;
  }

  const features = ['mouth_openness', 'smile_amplitude', 'eye_openness', 'eyebrow_raise'];

  return (
    <div style={styles.container}>
      {features.map(feature => {
        const data = temporalFeatures[feature];
        if (!data) return null;

        return (
          <div key={feature} style={styles.featureBox}>
            <h4 style={styles.featureName}>
              {feature.replace(/_/g, ' ').toUpperCase()}
            </h4>
            <div style={styles.metrics}>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Mean:</span>
                <span style={styles.metricValue}>{data.mean.toFixed(3)}</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Std:</span>
                <span style={styles.metricValue}>{data.std.toFixed(3)}</span>
              </div>
              <div style={styles.metric}>
                <span style={styles.metricLabel}>Velocity:</span>
                <span style={styles.metricValue}>{data.velocity.toFixed(3)}</span>
              </div>
            </div>

            {/* Simple progress bar for mean value */}
            <div style={styles.progressBar}>
              <div
                style={{
                  ...styles.progressFill,
                  width: `${Math.min(data.mean * 100, 100)}%`,
                  backgroundColor: getColorForValue(data.mean)
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
};

function getColorForValue(value) {
  if (value < 0.3) return '#ff4444';
  if (value < 0.6) return '#ffaa00';
  return '#00ff00';
}

const styles = {
  container: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '15px',
  },
  featureBox: {
    backgroundColor: '#2a2a2a',
    borderRadius: '8px',
    padding: '15px',
  },
  featureName: {
    margin: '0 0 10px 0',
    fontSize: '0.9em',
    color: '#0ff',
  },
  metrics: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
    marginBottom: '10px',
  },
  metric: {
    display: 'flex',
    justifyContent: 'space-between',
    fontSize: '0.85em',
  },
  metricLabel: {
    color: '#888',
  },
  metricValue: {
    fontFamily: 'monospace',
    color: '#fff',
  },
  progressBar: {
    height: '8px',
    backgroundColor: '#1a1a1a',
    borderRadius: '4px',
    overflow: 'hidden',
    marginTop: '10px',
  },
  progressFill: {
    height: '100%',
    transition: 'width 0.3s ease',
  },
  noData: {
    color: '#666',
    textAlign: 'center',
    padding: '20px',
  },
};

export default FeatureGraphs;
