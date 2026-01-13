/**
 * Face Pose Estimation Utility
 * Computes 3D face orientation (yaw, pitch, roll) and stabilization
 */

import { KEY_LANDMARKS } from './faceMeshTopology';

/**
 * Stable facial anchor points for pose estimation
 */
export const POSE_ANCHORS = {
  LEFT_EYE: 33,      // Left eye outer corner
  RIGHT_EYE: 263,    // Right eye outer corner
  NOSE_TIP: 1,       // Nose tip
  CHIN: 152,         // Chin bottom
  LEFT_EYE_INNER: 133,
  RIGHT_EYE_INNER: 362,
  FOREHEAD: 10
};

/**
 * Vector math utilities
 */
class Vector3 {
  constructor(x, y, z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  static fromArray(arr) {
    return new Vector3(arr[0], arr[1], arr[2]);
  }

  subtract(v) {
    return new Vector3(this.x - v.x, this.y - v.y, this.z - v.z);
  }

  add(v) {
    return new Vector3(this.x + v.x, this.y + v.y, this.z + v.z);
  }

  multiply(scalar) {
    return new Vector3(this.x * scalar, this.y * scalar, this.z * scalar);
  }

  length() {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }

  normalize() {
    const len = this.length();
    if (len === 0) return new Vector3(0, 0, 0);
    return new Vector3(this.x / len, this.y / len, this.z / len);
  }

  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  cross(v) {
    return new Vector3(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }

  toArray() {
    return [this.x, this.y, this.z];
  }
}

/**
 * Exponential moving average for smoothing
 */
class ExponentialSmoother {
  constructor(alpha = 0.3) {
    this.alpha = alpha;
    this.value = null;
  }

  smooth(newValue) {
    if (this.value === null) {
      this.value = newValue;
      return newValue;
    }
    this.value = this.alpha * newValue + (1 - this.alpha) * this.value;
    return this.value;
  }

  reset() {
    this.value = null;
  }
}

/**
 * Face Pose Estimator
 */
export class FacePoseEstimator {
  constructor() {
    // Smoothers for pose angles
    this.yawSmoother = new ExponentialSmoother(0.3);
    this.pitchSmoother = new ExponentialSmoother(0.3);
    this.rollSmoother = new ExponentialSmoother(0.3);

    // History for stability tracking
    this.poseHistory = [];
    this.maxHistoryLength = 30;
  }

  /**
   * STEP 4: Compute face pose from landmarks
   * @param {Array} landmarks - Raw landmark positions [[x,y,z], ...]
   * @returns {Object} Pose data with rotation matrix, angles, and stability
   */
  estimatePose(landmarks) {
    if (!landmarks || landmarks.length < 468) {
      return null;
    }

    // Get anchor points
    const leftEye = Vector3.fromArray(landmarks[POSE_ANCHORS.LEFT_EYE]);
    const rightEye = Vector3.fromArray(landmarks[POSE_ANCHORS.RIGHT_EYE]);
    const noseTip = Vector3.fromArray(landmarks[POSE_ANCHORS.NOSE_TIP]);
    const chin = Vector3.fromArray(landmarks[POSE_ANCHORS.CHIN]);
    const leftEyeInner = Vector3.fromArray(landmarks[POSE_ANCHORS.LEFT_EYE_INNER]);
    const rightEyeInner = Vector3.fromArray(landmarks[POSE_ANCHORS.RIGHT_EYE_INNER]);

    // Compute face coordinate axes
    const axes = this.computeFaceAxes(leftEye, rightEye, noseTip, chin);

    // Build rotation matrix
    const rotationMatrix = this.buildRotationMatrix(axes);

    // Extract Euler angles (yaw, pitch, roll)
    const rawAngles = this.extractEulerAngles(rotationMatrix);

    // Smooth angles to reduce jitter
    const smoothedAngles = {
      yaw: this.yawSmoother.smooth(rawAngles.yaw),
      pitch: this.pitchSmoother.smooth(rawAngles.pitch),
      roll: this.rollSmoother.smooth(rawAngles.roll)
    };

    // Compute stability metrics
    this.poseHistory.push(smoothedAngles);
    if (this.poseHistory.length > this.maxHistoryLength) {
      this.poseHistory.shift();
    }
    const stability = this.computeStability();

    // Compute eye center for reference
    const eyeCenter = leftEyeInner.add(rightEyeInner).multiply(0.5);

    return {
      axes,
      rotationMatrix,
      rawAngles,
      smoothedAngles,
      stability,
      anchorPoints: {
        leftEye: leftEye.toArray(),
        rightEye: rightEye.toArray(),
        noseTip: noseTip.toArray(),
        chin: chin.toArray(),
        eyeCenter: eyeCenter.toArray()
      }
    };
  }

  /**
   * Compute the face's coordinate axes from anchor points
   */
  computeFaceAxes(leftEye, rightEye, noseTip, chin) {
    // Horizontal axis (left to right) - from eye positions
    const horizontal = rightEye.subtract(leftEye).normalize();

    // Vertical axis (up to down) - from nose to chin direction
    const verticalRaw = chin.subtract(noseTip).normalize();

    // Forward direction - perpendicular to horizontal and vertical
    const forward = horizontal.cross(verticalRaw).normalize();

    // Recompute vertical to ensure orthogonality
    const vertical = forward.cross(horizontal).normalize();

    return {
      right: horizontal,      // X-axis: left to right
      up: vertical,          // Y-axis: down to up (inverted from nose-chin)
      forward: forward       // Z-axis: back to front
    };
  }

  /**
   * Build 3x3 rotation matrix from face axes
   */
  buildRotationMatrix(axes) {
    // Rotation matrix columns are the face axes
    return [
      [axes.right.x, axes.up.x, axes.forward.x],
      [axes.right.y, axes.up.y, axes.forward.y],
      [axes.right.z, axes.up.z, axes.forward.z]
    ];
  }

  /**
   * Extract yaw, pitch, roll from rotation matrix
   * Yaw: rotation around Y-axis (left/right)
   * Pitch: rotation around X-axis (up/down)
   * Roll: rotation around Z-axis (tilt)
   */
  extractEulerAngles(R) {
    // Extract angles from rotation matrix using ZYX convention
    const sy = Math.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0]);

    const singular = sy < 1e-6;

    let yaw, pitch, roll;

    if (!singular) {
      yaw = Math.atan2(R[1][0], R[0][0]);
      pitch = Math.atan2(-R[2][0], sy);
      roll = Math.atan2(R[2][1], R[2][2]);
    } else {
      yaw = Math.atan2(-R[0][1], R[1][1]);
      pitch = Math.atan2(-R[2][0], sy);
      roll = 0;
    }

    // Convert to degrees
    return {
      yaw: yaw * 180 / Math.PI,
      pitch: pitch * 180 / Math.PI,
      roll: roll * 180 / Math.PI
    };
  }

  /**
   * Compute stability metric from pose history
   * Lower values = more stable
   */
  computeStability() {
    if (this.poseHistory.length < 5) {
      return { overall: 1.0, yaw: 1.0, pitch: 1.0, roll: 1.0 };
    }

    const computeVariance = (values) => {
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
      return Math.sqrt(variance);
    };

    const yawValues = this.poseHistory.map(p => p.yaw);
    const pitchValues = this.poseHistory.map(p => p.pitch);
    const rollValues = this.poseHistory.map(p => p.roll);

    const yawStd = computeVariance(yawValues);
    const pitchStd = computeVariance(pitchValues);
    const rollStd = computeVariance(rollValues);

    // Normalize to 0-1 range (0 = very stable, 1 = very unstable)
    // Typical head motion is within 30 degrees
    const normalize = (std) => Math.min(std / 10, 1.0);

    return {
      overall: normalize((yawStd + pitchStd + rollStd) / 3),
      yaw: normalize(yawStd),
      pitch: normalize(pitchStd),
      roll: normalize(rollStd)
    };
  }

  /**
   * Get inverse rotation for stabilization
   * Apply this to face vertices to keep face forward-facing
   */
  getStabilizationMatrix(rotationMatrix) {
    // Transpose of rotation matrix = inverse for orthogonal matrices
    return [
      [rotationMatrix[0][0], rotationMatrix[1][0], rotationMatrix[2][0]],
      [rotationMatrix[0][1], rotationMatrix[1][1], rotationMatrix[2][1]],
      [rotationMatrix[0][2], rotationMatrix[1][2], rotationMatrix[2][2]]
    ];
  }

  /**
   * Apply rotation matrix to a 3D point
   */
  applyRotation(point, matrix) {
    const [x, y, z] = point;
    return [
      matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
      matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
      matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z
    ];
  }

  /**
   * Reset smoothing filters
   */
  reset() {
    this.yawSmoother.reset();
    this.pitchSmoother.reset();
    this.rollSmoother.reset();
    this.poseHistory = [];
  }
}

/**
 * Get pose description from angles
 */
export function getPoseDescription(angles) {
  const { yaw, pitch, roll } = angles;

  const yawDesc = Math.abs(yaw) < 10 ? 'Center' :
                  yaw > 10 ? 'Right' : 'Left';

  const pitchDesc = Math.abs(pitch) < 10 ? 'Level' :
                    pitch > 10 ? 'Down' : 'Up';

  const rollDesc = Math.abs(roll) < 5 ? 'Straight' :
                   roll > 5 ? 'Tilted Right' : 'Tilted Left';

  return {
    yaw: yawDesc,
    pitch: pitchDesc,
    roll: rollDesc,
    overall: `${yawDesc}, ${pitchDesc}, ${rollDesc}`
  };
}

export default FacePoseEstimator;
