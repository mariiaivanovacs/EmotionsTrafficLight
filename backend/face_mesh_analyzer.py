"""
Face Mesh Analyzer - Geometry-based emotion analysis
Uses MediaPipe Face Mesh for 3D facial landmark detection
"""

import numpy as np
import cv2
import mediapipe as mp
from collections import deque, defaultdict
from scipy.spatial import distance
import time
import os

# MediaPipe Face Landmarker (new API for MediaPipe 0.10+)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download the model file if needed
# Model: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'face_landmarker.task')

# Initialize face landmarker options
face_landmarker_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# Key landmark indices for MediaPipe Face Mesh (468 points)
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py

LANDMARKS = {
    # Eyes
    'left_eye_top': 159,
    'left_eye_bottom': 145,
    'left_eye_left': 33,
    'left_eye_right': 133,
    'right_eye_top': 386,
    'right_eye_bottom': 374,
    'right_eye_left': 362,
    'right_eye_right': 263,

    # Eyebrows
    'left_eyebrow_inner': 70,
    'left_eyebrow_outer': 46,
    'left_eyebrow_top': 105,
    'right_eyebrow_inner': 300,
    'right_eyebrow_outer': 276,
    'right_eyebrow_top': 334,

    # Mouth
    'mouth_left': 61,
    'mouth_right': 291,
    'mouth_top': 13,
    'mouth_bottom': 14,
    'upper_lip_top': 0,
    'lower_lip_bottom': 17,

    # Mouth corners (smile)
    'mouth_corner_left': 61,
    'mouth_corner_right': 291,

    # Face outline
    'face_left': 234,
    'face_right': 454,
    'face_top': 10,
    'face_bottom': 152,

    # Nose
    'nose_tip': 1,
    'nose_bridge': 168,
}

# Temporal feature window (seconds)
TEMPORAL_WINDOW = 3.0  # 3 second window
MAX_HISTORY_POINTS = 90  # At 30 FPS = 3 seconds


class FaceMeshAnalyzer:
    """Analyzes facial geometry features from MediaPipe Face Mesh"""

    def __init__(self):
        try:
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Face Landmarker: {e}")
        self.feature_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_POINTS))
        self.timestamps = defaultdict(lambda: deque(maxlen=MAX_HISTORY_POINTS))
        self.previous_valence_zone = {}

    def process_frame(self, frame):
        """Process a single frame and extract face mesh data"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process with MediaPipe Face Landmarker
        results = self.face_landmarker.detect(mp_image)

        if not results.face_landmarks:
            return None

        faces_data = []
        current_time = time.time()

        for face_idx, face_landmarks in enumerate(results.face_landmarks):
            # Extract 3D landmark coordinates
            landmarks_3d = self._extract_landmarks_3d_new(face_landmarks, frame.shape)

            # Compute geometry features
            features = self._compute_geometry_features(landmarks_3d, face_idx)

            # Store in temporal history
            for feature_name, value in features.items():
                self.feature_history[f"face_{face_idx}_{feature_name}"].append(value)
                self.timestamps[f"face_{face_idx}_{feature_name}"].append(current_time)

            # Compute temporal features
            temporal_features = self._compute_temporal_features(face_idx)

            # Compute valence/arousal
            valence, arousal = self._compute_valence_arousal(features)

            # Get emotion label from valence/arousal
            emotion_label, emotion_emoji = self._valence_arousal_to_emotion(valence, arousal)

            # Get traffic light color
            color, zone = self._valence_to_color(valence)

            # Check for zone change
            zone_changed = self._check_zone_change(face_idx, zone)

            faces_data.append({
                'face_id': face_idx,
                'landmarks_3d': landmarks_3d.tolist(),
                'geometry_features': features,
                'temporal_features': temporal_features,
                'valence': float(valence),
                'arousal': float(arousal),
                'valence_zone': zone,
                'emotion_label': emotion_label,
                'emotion_emoji': emotion_emoji,
                'color': color,
                'zone_changed': zone_changed,
            })

        return faces_data

    def _extract_landmarks_3d(self, face_landmarks, frame_shape):
        """Extract 3D coordinates of all landmarks (old API - kept for compatibility)"""
        height, width = frame_shape[:2]
        landmarks = []

        for landmark in face_landmarks.landmark:
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Normalized depth
            landmarks.append([x, y, z])

        return np.array(landmarks)

    def _extract_landmarks_3d_new(self, face_landmarks, frame_shape):
        """Extract 3D coordinates of all landmarks (new API)"""
        height, width = frame_shape[:2]
        landmarks = []

        for landmark in face_landmarks:
            x = landmark.x * width
            y = landmark.y * height
            z = landmark.z * width  # Normalized depth
            landmarks.append([x, y, z])

        return np.array(landmarks)

    def _get_landmark(self, landmarks, key):
        """Get specific landmark by key"""
        idx = LANDMARKS[key]
        return landmarks[idx]

    def _compute_distance(self, p1, p2):
        """Compute Euclidean distance between two points"""
        return distance.euclidean(p1[:2], p2[:2])  # Use only x, y

    def _compute_geometry_features(self, landmarks, face_idx):
        """Compute Phase 1 geometry features"""

        # Get face size for normalization
        face_left = self._get_landmark(landmarks, 'face_left')
        face_right = self._get_landmark(landmarks, 'face_right')
        face_width = self._compute_distance(face_left, face_right)

        # Avoid division by zero
        if face_width < 1:
            face_width = 1

        # MOUTH OPENNESS
        mouth_top = self._get_landmark(landmarks, 'mouth_top')
        mouth_bottom = self._get_landmark(landmarks, 'mouth_bottom')
        mouth_height = self._compute_distance(mouth_top, mouth_bottom)
        mouth_openness = mouth_height / face_width

        # SMILE AMPLITUDE
        mouth_left = self._get_landmark(landmarks, 'mouth_corner_left')
        mouth_right = self._get_landmark(landmarks, 'mouth_corner_right')
        mouth_width = self._compute_distance(mouth_left, mouth_right)

        # Smile is measured by mouth width relative to face width
        # and upward movement of corners
        smile_width_ratio = mouth_width / face_width

        # Calculate vertical position of mouth corners relative to mouth center
        mouth_center_y = (mouth_top[1] + mouth_bottom[1]) / 2
        corner_lift = mouth_center_y - ((mouth_left[1] + mouth_right[1]) / 2)
        corner_lift_normalized = corner_lift / face_width

        smile_amplitude = smile_width_ratio + corner_lift_normalized

        # EYE OPENNESS (both eyes)
        left_eye_top = self._get_landmark(landmarks, 'left_eye_top')
        left_eye_bottom = self._get_landmark(landmarks, 'left_eye_bottom')
        left_eye_height = self._compute_distance(left_eye_top, left_eye_bottom)

        right_eye_top = self._get_landmark(landmarks, 'right_eye_top')
        right_eye_bottom = self._get_landmark(landmarks, 'right_eye_bottom')
        right_eye_height = self._compute_distance(right_eye_top, right_eye_bottom)

        avg_eye_openness = (left_eye_height + right_eye_height) / (2 * face_width)

        # EYEBROW RAISE
        left_eyebrow = self._get_landmark(landmarks, 'left_eyebrow_top')
        left_eye_top_pt = self._get_landmark(landmarks, 'left_eye_top')
        left_eyebrow_raise = (left_eye_top_pt[1] - left_eyebrow[1]) / face_width

        right_eyebrow = self._get_landmark(landmarks, 'right_eyebrow_top')
        right_eye_top_pt = self._get_landmark(landmarks, 'right_eye_top')
        right_eyebrow_raise = (right_eye_top_pt[1] - right_eyebrow[1]) / face_width

        avg_eyebrow_raise = (left_eyebrow_raise + right_eyebrow_raise) / 2

        # HEAD TILT (using nose and face points)
        nose_tip = self._get_landmark(landmarks, 'nose_tip')
        nose_bridge = self._get_landmark(landmarks, 'nose_bridge')

        # Pitch (up/down rotation)
        pitch = np.arctan2(nose_tip[1] - nose_bridge[1], nose_tip[2] - nose_bridge[2])

        # Yaw (left/right rotation)
        yaw = np.arctan2(nose_tip[0] - nose_bridge[0], nose_tip[2] - nose_bridge[2])

        # Roll (tilt)
        roll = np.arctan2(face_right[1] - face_left[1], face_right[0] - face_left[0])

        # Z-MOTION (depth variance)
        # Use average Z coordinate of face outline points
        face_outline_z = np.mean([
            face_left[2], face_right[2],
            self._get_landmark(landmarks, 'face_top')[2],
            self._get_landmark(landmarks, 'face_bottom')[2]
        ])

        return {
            'mouth_openness': float(mouth_openness),
            'smile_amplitude': float(smile_amplitude),
            'eye_openness': float(avg_eye_openness),
            'eyebrow_raise': float(avg_eyebrow_raise),
            'head_pitch': float(pitch),
            'head_yaw': float(yaw),
            'head_roll': float(roll),
            'face_depth': float(face_outline_z),
        }

    def _compute_temporal_features(self, face_idx):
        """Compute temporal features from history window"""
        temporal = {}

        for feature_name in ['mouth_openness', 'smile_amplitude', 'eye_openness', 'eyebrow_raise']:
            key = f"face_{face_idx}_{feature_name}"
            history = list(self.feature_history[key])
            timestamps = list(self.timestamps[key])

            if len(history) < 2:
                temporal[feature_name] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'velocity': 0.0,
                }
                continue

            # Filter to temporal window
            current_time = time.time()
            windowed_values = []
            windowed_times = []

            for val, ts in zip(history, timestamps):
                if current_time - ts <= TEMPORAL_WINDOW:
                    windowed_values.append(val)
                    windowed_times.append(ts)

            if len(windowed_values) < 2:
                temporal[feature_name] = {
                    'mean': windowed_values[0] if windowed_values else 0.0,
                    'std': 0.0,
                    'velocity': 0.0,
                }
                continue

            values_array = np.array(windowed_values)
            times_array = np.array(windowed_times)

            # Mean and std
            mean_val = float(np.mean(values_array))
            std_val = float(np.std(values_array))

            # Velocity (derivative)
            velocity = float((values_array[-1] - values_array[0]) / (times_array[-1] - times_array[0]))

            temporal[feature_name] = {
                'mean': mean_val,
                'std': std_val,
                'velocity': velocity,
            }

        return temporal

    def _compute_valence_arousal(self, features):
        """
        Compute valence and arousal from geometry features

        Valence: pleasure/displeasure (-1 to 1)
        Arousal: activation/deactivation (0 to 1)
        """
        # VALENCE (positive/negative emotion)
        # Positive indicators: smile, eyebrow raise
        # Negative indicators: frown (negative smile), eyebrow furrow

        smile_contrib = features['smile_amplitude'] * 2.0  # Weight smile heavily
        eyebrow_contrib = features['eyebrow_raise'] * 0.5

        valence = smile_contrib + eyebrow_contrib - 0.5  # Center around 0
        valence = np.clip(valence, -1, 1)

        # AROUSAL (intensity/activation)
        # High arousal: wide eyes, open mouth, fast movements
        # Low arousal: relaxed features

        eye_contrib = features['eye_openness'] * 2.0
        mouth_contrib = features['mouth_openness'] * 1.5

        arousal = eye_contrib + mouth_contrib
        arousal = np.clip(arousal, 0, 1)

        return valence, arousal

    def _valence_arousal_to_emotion(self, valence, arousal):
        """
        Map valence and arousal to emotion label (4-quadrant model)

        High Arousal + Negative Valence = Angry/Tense
        High Arousal + Positive Valence = Excited/Happy
        Low Arousal + Negative Valence = Sad/Depressed
        Low Arousal + Positive Valence = Calm/Relaxed
        """
        arousal_threshold = 0.5
        valence_threshold = 0.0

        if arousal > arousal_threshold:
            # High arousal
            if valence > valence_threshold:
                return "Excited", "😄"
            else:
                return "Tense", "😠"
        else:
            # Low arousal
            if valence > valence_threshold:
                return "Calm", "😌"
            else:
                return "Sad", "😢"

    def _valence_to_color(self, valence):
        """Map valence to traffic light color"""
        if valence > 0.2:
            return (0, 255, 0), 'positive'  # Green
        elif valence < -0.2:
            return (0, 0, 255), 'negative'  # Red
        else:
            return (0, 255, 255), 'neutral'  # Yellow

    def _check_zone_change(self, face_idx, new_zone):
        """Check if valence zone has changed"""
        old_zone = self.previous_valence_zone.get(face_idx)
        changed = old_zone is not None and old_zone != new_zone
        self.previous_valence_zone[face_idx] = new_zone
        return changed

    def get_landmark_indices_for_visualization(self):
        """Return landmark indices for 3D visualization"""
        return {
            'face_oval': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
            'left_eye': [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
            'right_eye': [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382],
            'lips': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
            'left_eyebrow': [46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
            'right_eyebrow': [276, 283, 282, 295, 285, 300, 293, 334, 296, 336],
        }


# Global analyzer instance
analyzer = FaceMeshAnalyzer()
