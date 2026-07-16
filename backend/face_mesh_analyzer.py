"""
Face Mesh Analyzer - Geometry-based emotion analysis
Uses MediaPipe Face Mesh for 3D facial landmark detection
Integrates FLAME parametric 3D face model for dense mesh reconstruction
"""

import numpy as np
import cv2
import mediapipe as mp
from collections import deque, defaultdict
from scipy.spatial import distance
import time
import os
import logging
logger = logging.getLogger(__name__)

# Try to import FLAME model (optional - falls back to landmark-only if not available)
FLAME_AVAILABLE = False
flame_model = None
flame_fitter = None

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


try:
    from flame_model import get_flame_model
    from flame_fitter import FLAMEFitter
    FLAME_AVAILABLE = True
    print("✓ FLAME model module available")
except ImportError as e:
    print(f"⚠️  FLAME not available (will use landmarks only): {e}")

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
    """Analyzes facial geometry features from MediaPipe Face Mesh and FLAME 3D model"""

    def __init__(self, use_flame=True):
        """
        Initialize Face Mesh Analyzer

        Args:
            use_flame: If True, use FLAME parametric model for dense mesh reconstruction
        """
        # Initialize MediaPipe Face Landmarker
        try:
            self.face_landmarker = FaceLandmarker.create_from_options(face_landmarker_options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Face Landmarker: {e}")

        self.feature_history = defaultdict(lambda: deque(maxlen=MAX_HISTORY_POINTS))
        self.timestamps = defaultdict(lambda: deque(maxlen=MAX_HISTORY_POINTS))
        self.previous_valence_zone = {}

        # Initialize FLAME model if requested and available
        self.use_flame = use_flame and FLAME_AVAILABLE
        self.flame_model = None
        self.flame_fitter = None

        if self.use_flame:
            try:
                global flame_model, flame_fitter
                if flame_model is None:
                    print("Initializing FLAME model...")
                    flame_model = get_flame_model(use_gpu=False)
                    # Initialize with PnP solver (uses only MediaPipe X,Y)
                    flame_fitter = FLAMEFitter(
                        flame_model,
                        image_size=(640, 480),  # Camera resolution
                        focal_length=640,        # Focal length in pixels (default: image_width)
                        use_pnp=True            # Use PnP solver for metric-correct pose
                    )
                    print("✓ FLAME model initialized successfully (PnP solver enabled)")

                self.flame_model = flame_model
                self.flame_fitter = flame_fitter
            except Exception as e:
                print(f"⚠️  Failed to initialize FLAME: {e}")
                print("   Falling back to landmark-only mode")
                self.use_flame = False
                
                
                

    def _umeyama_similarity(src, dst, with_scale=True):
        """
        Compute similarity transform (s, R, t) that maps src -> dst.
        src, dst: (N,3) arrays.
        Returns scale s, rotation R (3x3), translation t (3,)
        """
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3

        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_centered = src - mu_src
        dst_centered = dst - mu_dst

        cov = (dst_centered.T @ src_centered) / src.shape[0]
        U, D, Vt = np.linalg.svd(cov)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1
        R = U @ S @ Vt

        if with_scale:
            var_src = (src_centered ** 2).sum() / src.shape[0]
            s = np.trace(np.diag(D) @ S) / var_src
        else:
            s = 1.0

        t = mu_dst - s * (R @ mu_src)
        return s, R, t

    def process_face_frame(self, face_landmarks, frame, face_idx):
        """
        Full preprocessing pipeline (steps 1-6). Returns dict with:
        landmarks_3d, features, temporal_features, valence, arousal,
        emotion_label, color, zone, zone_changed,
        s, R, t, expr_landmarks_canonical, pose_landmarks_xy
        """
        # 1) Extract landmarks
        landmarks_3d = self._extract_landmarks_3d_new(face_landmarks, frame.shape)  # (N,3)
        logger.info(f"Landmarks 3D: {landmarks_3d.shape}")

        # 2) Compute geometry features + temporal bookkeeping
        features = self._compute_geometry_features(landmarks_3d, face_idx)
        for feature_name, value in features.items():
            self.feature_history[f"face_{face_idx}_{feature_name}"].append(value)
            self.timestamps[f"face_{face_idx}_{feature_name}"].append(time.time())
        temporal_features = self._compute_temporal_features(face_idx)

        # 3) Valence / arousal / UI outputs
        valence, arousal = self._compute_valence_arousal(features)
        emotion_label, emotion_emoji = self._valence_arousal_to_emotion(valence, arousal)
        color, zone = self._valence_to_color(valence)
        zone_changed = self._check_zone_change(face_idx, zone)

        # --- FLAME alignment pipeline ---
        # Use the FLAME fitter reference indices (must exist)
        pose_mp_idx = getattr(self, "pose_mp_indices", None)
        pose_flame_pts = getattr(self.flame_fitter, "pose_reference_3d", None)  # (M,3) canonical FLAME points

        if pose_mp_idx is None or pose_flame_pts is None or len(pose_flame_pts) < 4:
            # fallback: return minimal info
            
            logger.info(f"pose_mp_idx: {pose_mp_idx}")
            logger.info(f"pose_flame_pts: {pose_flame_pts}")
            logger.info(f"len(pose_flame_pts): {len(pose_flame_pts)}")
            logger.warning("[PIPE] insufficient pose reference for rigid alignment")
            return {
                "landmarks_3d": landmarks_3d,
                "features": features,
                "temporal_features": temporal_features,
                "valence": valence, "arousal": arousal,
                "emotion_label": emotion_label, "emotion_emoji": emotion_emoji,
                "color": color, "zone": zone, "zone_changed": zone_changed,
                "s": 1.0, "R": np.eye(3), "t": np.zeros(3),
                "expr_landmarks_canonical": None,
                "pose_landmarks_xy": landmarks_3d[:, :2]
            }

        # 4) Prepare corresponding sets for similarity fit
        # MediaPipe pose landmarks (use X,Y,Z provided by _extract_landmarks_3d_new)
        pose_landmarks_mp = landmarks_3d[pose_mp_idx].astype(np.float64)  # (M,3)
        object_points_flame = np.asarray(pose_flame_pts, dtype=np.float64)  # (M,3)

        # Optional: flip Y/Z to match coordinate conventions if needed (depends how you initialized pose_reference_3d)
        # (Assume pose_reference_3d already adjusted in init; adjust mp coords if needed)
        # e.g., if MP uses Y-down and your FLAME uses Y-up, flip MP Y: pose_landmarks_mp[:,1] *= -1

        # 5) Compute similarity transform src=object_points_flame -> dst=pose_landmarks_mp
        try:
            s, R, t = self._umeyama_similarity(object_points_flame, pose_landmarks_mp, with_scale=True)
        except Exception as e:
            logger.exception("[PIPE] Umeyama failed: %s", e)
            s, R, t = 1.0, np.eye(3), np.zeros(3)

        # 6) Canonicalize expression landmarks: bring MP expression landmarks into FLAME canonical space
        mp_expr_idx = getattr(self, "mp_expr_indices", None)
        expr_landmarks_canonical = None
        if mp_expr_idx is not None and len(mp_expr_idx) > 0:
            expr_landmarks_mp = landmarks_3d[mp_expr_idx].astype(np.float64)  # (K,3) in MP camera space
            # invert similarity: x_canonical = (1/s) * R.T @ (x_mp - t)
            expr_landmarks_canonical = ((R.T @ (expr_landmarks_mp - t).T) / s).T
        else:
            logger.warning("[PIPE] no expression indices available")

        # return everything needed for downstream (expression optimization + mesh generation)
        return {
            "landmarks_3d": landmarks_3d,
            "features": features,
            "temporal_features": temporal_features,
            "valence": valence,
            "arousal": arousal,
            "emotion_label": emotion_label,
            "emotion_emoji": emotion_emoji,
            "color": color,
            "zone": zone,
            "zone_changed": zone_changed,
            "s": float(s),
            "R": R,
            "t": t,
            "expr_landmarks_canonical": expr_landmarks_canonical,
            "pose_landmarks_xy": pose_landmarks_mp[:, :2]
        }


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
        # current_time = time.time()

        for face_idx, face_landmarks in enumerate(results.face_landmarks):
            flame_mesh_data = None

            # ----- Step 1: preprocess (process_face_frame does steps 1-6) -----
            try:
                out = self.process_face_frame(face_landmarks, frame, face_idx)
            except Exception as e:
                logger.exception("process_face_frame failed: %s", e)
                # fallback: minimal processing so we still return useful data
                landmarks_3d = self._extract_landmarks_3d_new(face_landmarks, frame.shape)
                features = self._compute_geometry_features(landmarks_3d, face_idx)
                temporal_features = self._compute_temporal_features(face_idx)
                valence, arousal = self._compute_valence_arousal(features)
                emotion_label, emotion_emoji = self._valence_arousal_to_emotion(valence, arousal)
                color, zone = self._valence_to_color(valence)
                zone_changed = self._check_zone_change(face_idx, zone)

                out = {
                    "landmarks_3d": landmarks_3d,
                    "features": features,
                    "temporal_features": temporal_features,
                    "valence": valence,
                    "arousal": arousal,
                    "emotion_label": emotion_label,
                    "emotion_emoji": emotion_emoji,
                    "color": color,
                    "zone": zone,
                    "zone_changed": zone_changed,
                    "s": 1.0,
                    "R": np.eye(3),
                    "t": np.zeros(3),
                    "expr_landmarks_canonical": None
                }

            # ----- Step 2: FLAME fitting (use existing fit API) -----
            if self.use_flame and self.flame_fitter is not None:
                try:
                    # Use the processed landmarks (we pass full mediapipe landmarks so existing fit() works)
                    flame_mesh_data = self.flame_fitter.fit(
                        mediapipe_landmarks_3d=out["landmarks_3d"],
                        optimize_shape=False,
                        # Change after 
                        optimize_expression=False,
                    )
                except Exception as e:
                    logger.exception("FLAME fitting error: %s", e)
                    flame_mesh_data = None

         # ----- Step 3: assemble face data -----
            face_data = {
                "face_id": int(face_idx),

                # Landmarks
                "landmarks_3d": np.asarray(out["landmarks_3d"]).tolist(),

                # Features (may be numpy)
                "geometry_features": np.asarray(out["features"]).tolist() if out.get("features") is not None else None,
                "temporal_features": np.asarray(out["temporal_features"]).tolist() if out.get("temporal_features") is not None else None,

                # Emotion values (force native Python types)
                "valence": float(out["valence"]),
                "arousal": float(out["arousal"]),
                "valence_zone": str(out["zone"]),
                "emotion_label": str(out["emotion_label"]),
                "emotion_emoji": str(out["emotion_emoji"]),

                # Color (ensure ints)
                "color": [int(c) for c in out["color"]],

                "zone_changed": bool(out["zone_changed"]),

                # FLAME mesh (deep convert safely)
                "flame_mesh": to_serializable(flame_mesh_data),

                # Head pose transform (JSON safe)
                "pose_transform": {
                    "scale": float(out.get("s", 1.0)),
                    "R": np.asarray(out.get("R", np.eye(3))).tolist(),
                    "t": np.asarray(out.get("t", np.zeros(3))).tolist()
                }
            }


            faces_data.append(face_data)

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

    def _generate_flame_mesh(self, landmarks_3d):
        """
        Generate FLAME 3D mesh from MediaPipe landmarks

        Args:
            landmarks_3d: (468, 3) numpy array of MediaPipe landmarks

        Returns:
            dict with FLAME mesh data:
                - vertices: (5023, 3) mesh vertices
                - faces: (N, 3) triangle indices
                - normals: (5023, 3) vertex normals
                - centroid: (3,) mesh center
                - flame_params: dict of FLAME parameters
                - fit_time_ms: fitting time in milliseconds
        """
        if not self.use_flame or self.flame_fitter is None:
            return None

        # Fit FLAME model to landmarks
        result = self.flame_fitter.fit(
            mediapipe_landmarks_3d=landmarks_3d,
            optimize_shape=False,  # Keep identity stable (too slow for real-time)
            optimize_expression=False   # Optimize expressions each frame
            # Change after
        )

        # Convert numpy arrays to lists for JSON serialization
        flame_mesh_data = {
            'vertices': result['vertices'].tolist(),
            'faces': result['faces'].tolist(),
            'normals': result['normals'].tolist(),
            'centroid': result['centroid'].tolist(),
            'flame_params': {
                'shape': result['shape_params'].tolist(),
                'expression': result['expression_params'].tolist(),
                'pose': result['pose_params'].tolist(),
            },
            'fit_time_ms': result['fit_time_ms'],
            'num_landmarks_used': result['num_landmarks_used'],
        }

        return flame_mesh_data

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
