/**
 * Facial Analysis Utility
 * Measurements, regions, and interpretation of facial geometry
 */

/**
 * STEP 5: Define facial regions using MediaPipe landmark indices
 */
export const FACIAL_REGIONS = {
  // Eyes
  LEFT_EYE: {
    name: 'Left Eye',
    outer: [33, 246, 161, 160, 159, 158, 157, 173, 133],
    upper: [246, 161, 160, 159, 158, 157, 173],
    lower: [33, 7, 163, 144, 145, 153, 154, 155, 133],
    center: 468  // Left iris center (if available)
  },
  RIGHT_EYE: {
    name: 'Right Eye',
    outer: [263, 466, 388, 387, 386, 385, 384, 398, 362],
    upper: [466, 388, 387, 386, 385, 384, 398],
    lower: [263, 249, 390, 373, 374, 380, 381, 382, 362],
    center: 473  // Right iris center (if available)
  },

  // Eyebrows
  LEFT_EYEBROW: [70, 63, 105, 66, 107, 55, 189],
  RIGHT_EYEBROW: [300, 293, 334, 296, 336, 285, 417],

  // Nose
  NOSE: {
    bridge: [6, 197, 195, 5],
    tip: [1, 2],
    left_wing: [220, 237, 44, 1],
    right_wing: [440, 457, 274, 1],
    base: [19, 94, 2, 165, 328]
  },

  // Mouth
  MOUTH: {
    outer_upper: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    outer_lower: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    inner_upper: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    inner_lower: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    left_corner: 61,
    right_corner: 291,
    center_top: 13,
    center_bottom: 14
  },

  // Jaw and chin
  JAW: {
    left: [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
    right: [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
    chin: [152, 378, 379, 365, 397, 288, 361, 323, 454]
  },

  // Forehead (estimated from available points)
  FOREHEAD: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288],

  // Cheeks
  LEFT_CHEEK: [116, 117, 118, 119, 120, 121, 128, 245],
  RIGHT_CHEEK: [345, 346, 347, 348, 349, 350, 357, 465],

  // Face oval (contour)
  FACE_OVAL: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109, 10
  ]
};

/**
 * Key measurements between landmarks
 */
export const MEASUREMENT_PAIRS = {
  // Inter-eye distance
  EYE_TO_EYE: [33, 263],

  // Eye dimensions
  LEFT_EYE_WIDTH: [33, 133],
  RIGHT_EYE_WIDTH: [263, 362],

  // Nose dimensions
  NOSE_HEIGHT: [6, 2],
  NOSE_WIDTH: [220, 440],

  // Mouth dimensions
  MOUTH_WIDTH: [61, 291],
  MOUTH_HEIGHT: [13, 14],

  // Vertical face measurements
  FOREHEAD_TO_NOSE: [10, 1],
  NOSE_TO_CHIN: [1, 152],
  FACE_HEIGHT: [10, 152],

  // Facial thirds
  UPPER_THIRD: [10, 6],    // Forehead to nose bridge
  MIDDLE_THIRD: [6, 2],    // Nose bridge to nose base
  LOWER_THIRD: [2, 152]    // Nose base to chin
};

/**
 * Facial Analysis Engine
 */
export class FacialAnalyzer {
  constructor() {
    this.measurementHistory = [];
    this.maxHistoryLength = 30;
    this.baselineMeasurements = null;
  }

  /**
   * STEP 5: Analyze facial geometry from landmarks
   * @param {Array} landmarks - Raw 3D landmark positions
   * @param {Array} centeredLandmarks - Centered/normalized landmarks
   * @returns {Object} Analysis results
   */
  analyze(landmarks, centeredLandmarks = null) {
    if (!landmarks || landmarks.length < 468) {
      return null;
    }

    // Use centered landmarks if available, otherwise use raw
    const points = centeredLandmarks || landmarks;

    // Compute key distances
    const distances = this.computeDistances(points);

    // Compute facial angles and symmetry
    const symmetry = this.computeSymmetry(points);

    // Compute regional curvature
    const curvature = this.computeRegionalCurvature(points);

    // Detect expression changes
    const expression = this.detectExpressionChanges(distances);

    // Compute normalized measurements (relative to face size)
    const normalized = this.normalizeToFaceSize(distances);

    // Validate anatomical constraints
    const validation = this.validateAnatomy(normalized);

    // Store in history
    this.measurementHistory.push({
      timestamp: Date.now(),
      distances,
      normalized
    });
    if (this.measurementHistory.length > this.maxHistoryLength) {
      this.measurementHistory.shift();
    }

    return {
      distances,
      symmetry,
      curvature,
      expression,
      normalized,
      validation,
      regions: this.getRegionCenters(points)
    };
  }

  /**
   * Compute distances between key landmark pairs
   */
  computeDistances(landmarks) {
    const distance = (p1, p2) => {
      const dx = p2[0] - p1[0];
      const dy = p2[1] - p1[1];
      const dz = p2[2] - p1[2];
      return Math.sqrt(dx * dx + dy * dy + dz * dz);
    };

    const distances = {};
    for (const [name, [idx1, idx2]] of Object.entries(MEASUREMENT_PAIRS)) {
      if (landmarks[idx1] && landmarks[idx2]) {
        distances[name] = distance(landmarks[idx1], landmarks[idx2]);
      }
    }

    // Additional useful measurements
    distances.LEFT_EYE_HEIGHT = this.computeEyeHeight(landmarks, 'left');
    distances.RIGHT_EYE_HEIGHT = this.computeEyeHeight(landmarks, 'right');
    distances.MOUTH_OPENNESS = this.computeMouthOpenness(landmarks);

    return distances;
  }

  /**
   * Compute eye height (vertical eye opening)
   */
  computeEyeHeight(landmarks, side) {
    const region = side === 'left' ? FACIAL_REGIONS.LEFT_EYE : FACIAL_REGIONS.RIGHT_EYE;
    const upperPoints = region.upper.map(idx => landmarks[idx]);
    const lowerPoints = region.lower.map(idx => landmarks[idx]);

    // Average vertical distance between upper and lower eyelids
    let totalDist = 0;
    let count = 0;

    upperPoints.forEach(upper => {
      lowerPoints.forEach(lower => {
        if (upper && lower) {
          totalDist += Math.abs(upper[1] - lower[1]);
          count++;
        }
      });
    });

    return count > 0 ? totalDist / count : 0;
  }

  /**
   * Compute mouth openness
   */
  computeMouthOpenness(landmarks) {
    const top = landmarks[FACIAL_REGIONS.MOUTH.center_top];
    const bottom = landmarks[FACIAL_REGIONS.MOUTH.center_bottom];

    if (!top || !bottom) return 0;

    return Math.sqrt(
      Math.pow(bottom[0] - top[0], 2) +
      Math.pow(bottom[1] - top[1], 2) +
      Math.pow(bottom[2] - top[2], 2)
    );
  }

  /**
   * Compute facial symmetry metrics
   */
  computeSymmetry(landmarks) {
    // Compare left and right eye dimensions
    const leftEyeWidth = landmarks[33] && landmarks[133] ?
      this.distance3D(landmarks[33], landmarks[133]) : 0;
    const rightEyeWidth = landmarks[263] && landmarks[362] ?
      this.distance3D(landmarks[263], landmarks[362]) : 0;

    const eyeSymmetry = leftEyeWidth > 0 && rightEyeWidth > 0 ?
      1 - Math.abs(leftEyeWidth - rightEyeWidth) / Math.max(leftEyeWidth, rightEyeWidth) : 1;

    // Compare left and right eyebrow heights
    const leftBrowHeight = this.getRegionHeight(landmarks, FACIAL_REGIONS.LEFT_EYEBROW);
    const rightBrowHeight = this.getRegionHeight(landmarks, FACIAL_REGIONS.RIGHT_EYEBROW);

    const browSymmetry = leftBrowHeight > 0 && rightBrowHeight > 0 ?
      1 - Math.abs(leftBrowHeight - rightBrowHeight) / Math.max(leftBrowHeight, rightBrowHeight) : 1;

    // Overall symmetry score (0 = asymmetric, 1 = perfect symmetry)
    const overall = (eyeSymmetry + browSymmetry) / 2;

    return {
      overall,
      eyes: eyeSymmetry,
      eyebrows: browSymmetry,
      score: overall * 100  // 0-100 percentage
    };
  }

  /**
   * Compute surface curvature in key regions
   */
  computeRegionalCurvature(landmarks) {
    const computeCurvature = (indices) => {
      if (indices.length < 3) return 0;

      const points = indices.map(idx => landmarks[idx]).filter(p => p);
      if (points.length < 3) return 0;

      // Simplified curvature: deviation from plane
      // Higher values = more curved
      let totalDeviation = 0;
      for (let i = 1; i < points.length - 1; i++) {
        const prev = points[i - 1];
        const curr = points[i];
        const next = points[i + 1];

        // Angle between vectors
        const v1 = [curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2]];
        const v2 = [next[0] - curr[0], next[1] - curr[1], next[2] - curr[2]];

        const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
        const len1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2);
        const len2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2);

        if (len1 > 0 && len2 > 0) {
          const cosAngle = dot / (len1 * len2);
          const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
          totalDeviation += angle;
        }
      }

      return totalDeviation / (points.length - 2);
    };

    return {
      noseBridge: computeCurvature(FACIAL_REGIONS.NOSE.bridge),
      leftCheek: computeCurvature(FACIAL_REGIONS.LEFT_CHEEK),
      rightCheek: computeCurvature(FACIAL_REGIONS.RIGHT_CHEEK),
      jawLeft: computeCurvature(FACIAL_REGIONS.JAW.left),
      jawRight: computeCurvature(FACIAL_REGIONS.JAW.right),
      chin: computeCurvature(FACIAL_REGIONS.JAW.chin)
    };
  }

  /**
   * Detect expression changes from baseline
   */
  detectExpressionChanges(currentDistances) {
    if (this.measurementHistory.length < 5) {
      return { hasChange: false, changes: {} };
    }

    // Use first stable measurements as baseline
    if (!this.baselineMeasurements) {
      if (this.measurementHistory.length >= 10) {
        this.baselineMeasurements = this.measurementHistory[5].distances;
      } else {
        return { hasChange: false, changes: {} };
      }
    }

    const changes = {};
    let significantChange = false;

    // Compare current to baseline
    for (const [key, value] of Object.entries(currentDistances)) {
      if (this.baselineMeasurements[key]) {
        const baseline = this.baselineMeasurements[key];
        const percentChange = ((value - baseline) / baseline) * 100;

        // Detect significant changes (>10%)
        if (Math.abs(percentChange) > 10) {
          changes[key] = {
            current: value,
            baseline: baseline,
            percentChange,
            direction: percentChange > 0 ? 'increase' : 'decrease'
          };
          significantChange = true;
        }
      }
    }

    return {
      hasChange: significantChange,
      changes
    };
  }

  /**
   * Normalize measurements to face size for consistency
   */
  normalizeToFaceSize(distances) {
    // Use inter-eye distance as reference scale
    const scale = distances.EYE_TO_EYE || 1;

    const normalized = {};
    for (const [key, value] of Object.entries(distances)) {
      normalized[key] = value / scale;
    }

    return normalized;
  }

  /**
   * Validate measurements against anatomical constraints
   */
  validateAnatomy(normalizedDistances) {
    const warnings = [];

    // Facial proportions (based on normalized inter-eye distance = 1.0)
    const checks = {
      mouthWidth: { value: normalizedDistances.MOUTH_WIDTH, min: 1.2, max: 2.0 },
      noseWidth: { value: normalizedDistances.NOSE_WIDTH, min: 0.6, max: 1.2 },
      faceHeight: { value: normalizedDistances.FACE_HEIGHT, min: 3.0, max: 5.0 }
    };

    for (const [name, check] of Object.entries(checks)) {
      if (check.value < check.min) {
        warnings.push(`${name} too small (${check.value.toFixed(2)} < ${check.min})`);
      } else if (check.value > check.max) {
        warnings.push(`${name} too large (${check.value.toFixed(2)} > ${check.max})`);
      }
    }

    return {
      valid: warnings.length === 0,
      warnings
    };
  }

  /**
   * Helper: Get region height (average Y coordinate)
   */
  getRegionHeight(landmarks, indices) {
    const points = indices.map(idx => landmarks[idx]).filter(p => p);
    if (points.length === 0) return 0;

    const avgY = points.reduce((sum, p) => sum + p[1], 0) / points.length;
    return avgY;
  }

  /**
   * Helper: 3D distance between two points
   */
  distance3D(p1, p2) {
    return Math.sqrt(
      Math.pow(p2[0] - p1[0], 2) +
      Math.pow(p2[1] - p1[1], 2) +
      Math.pow(p2[2] - p1[2], 2)
    );
  }

  /**
   * Get geometric centers of facial regions
   */
  getRegionCenters(landmarks) {
    const computeCenter = (indices) => {
      const points = indices.map(idx => landmarks[idx]).filter(p => p);
      if (points.length === 0) return null;

      const center = [0, 0, 0];
      points.forEach(p => {
        center[0] += p[0];
        center[1] += p[1];
        center[2] += p[2];
      });

      return [
        center[0] / points.length,
        center[1] / points.length,
        center[2] / points.length
      ];
    };

    return {
      leftEye: computeCenter(FACIAL_REGIONS.LEFT_EYE.outer),
      rightEye: computeCenter(FACIAL_REGIONS.RIGHT_EYE.outer),
      nose: computeCenter(FACIAL_REGIONS.NOSE.bridge),
      mouth: computeCenter([
        ...FACIAL_REGIONS.MOUTH.outer_upper,
        ...FACIAL_REGIONS.MOUTH.outer_lower
      ]),
      leftCheek: computeCenter(FACIAL_REGIONS.LEFT_CHEEK),
      rightCheek: computeCenter(FACIAL_REGIONS.RIGHT_CHEEK)
    };
  }

  /**
   * Reset baseline for expression detection
   */
  resetBaseline() {
    this.baselineMeasurements = null;
    this.measurementHistory = [];
  }
}

export default FacialAnalyzer;
