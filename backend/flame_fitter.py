"""
FLAME Fitter - Two-Stage Production Approach

Stage 1: Head Pose Estimation (NO FLAME)
- Uses MediaPipe landmarks directly
- Computes rigid transformation: R_head, t_head, scale
- Rock-solid head tracking

Stage 2: Expression Fitting (FLAME)
- Removes rigid head motion from landmarks
- Optimizes only: expression params + jaw pose
- Shape stays fixed (calibrated once per session)
"""
from pathlib import Path
import json
import numpy as np
import torch
import time
import cv2

import logging

logger = logging.getLogger(__name__)



def transform_flame_vertices(vertices, R, t, s=1.0):
    """
    Apply rotation, translation, and optional scale to FLAME vertices.

    Args:
        vertices (np.ndarray): FLAME mesh vertices, shape (N, 3)
        R (np.ndarray): Rotation matrix, shape (3, 3)
        t (np.ndarray): Translation vector, shape (3,)
        s (float): Optional scale factor

    Returns:
        np.ndarray: Transformed vertices, shape (N, 3)
    """
    return (s * (R @ vertices.T)).T + t



class FLAMEFitter:
    """
    Two-stage FLAME fitter:
    1. Head pose from MediaPipe (no FLAME)
    2. Expression-only optimization in FLAME
    """

    def __init__(self, flame_model, image_size=(640, 480), focal_length=None, use_pnp=True):
        """
        Initialize FLAME fitter

        Args:
            flame_model: FLAMEModel instance
            image_size: (width, height) of camera image for PnP solver
            focal_length: camera focal length in pixels (default: image_width)
            use_pnp: if True, use PnP solver for head pose; if False, use Procrustes
        """
        self.flame = flame_model
        self.device = flame_model.device

        # Camera parameters for PnP solver
        self.image_size = image_size
        self.focal_length = focal_length if focal_length is not None else image_size[0]
        self.use_pnp = use_pnp

        # Create landmark mappings
        self._create_pose_landmarks()      # For head pose estimation
        self._create_expression_landmarks() # For expression fitting

        # Initialize pose reference from ACTUAL FLAME template vertices
        flame_template = self.flame.v_template.cpu().numpy()
        self.pose_reference_3d = flame_template[self.pose_flame_indices].copy()
        logger.info(f"[FLAME] INITIAL pose_reference_3d: {self.pose_reference_3d.shape}")

        # FLAME model has Y-up, but we need to flip Y to match MediaPipe (Y-down)
        # Also flip Z so face points towards camera
        self.pose_reference_3d[:, 1] *= -1  # Flip Y
        self.pose_reference_3d[:, 2] *= -1  # Flip Z (face now points towards +Z)

        logger.info(f"[FLAME] Pose reference initialized from template: {self.pose_reference_3d.shape}")
        logger.info(f"[FLAME] Pose reference (first 5): {self.pose_reference_3d[:5]}")

        # Persistent parameters
        self.current_shape = np.zeros(self.flame.num_betas)
        self.current_expression = np.zeros(self.flame.num_expressions)
        self.current_jaw_pose = np.zeros(3)  # Jaw rotation only

        # Head pose (computed each frame, not optimized with FLAME)
        self.head_rotation = np.eye(3)
        self.head_translation = np.zeros(3)
        # self.head_scale = 1.0
        self.head_scale = 1500  # example value, adjust so the FLAME head appears natural


        # Config
        self.expression_iterations = 8  # Fast real-time
        self.use_temporal_smoothing = True
        self.expression_smoothing = 0.7  # Blend with previous frame

        logger.info(f"[FLAME] FLAMEFitter initialized (PnP={'ON' if use_pnp else 'OFF'}, focal_length={self.focal_length:.1f}px)")

    # def _create_pose_landmarks(self):
    #     """
    #     Select stable MediaPipe landmarks for head pose estimation.
    #     These should be on rigid parts of the face (not affected by expression).

    #     We use ACTUAL FLAME template vertices as reference to ensure proper alignment.
    #     """
    #     # MediaPipe landmarks for pose (stable, rigid parts)
    #     self.pose_mp_indices = np.array([
    #         # Face contour (stable, not affected by expressions)
    #         10,   # Forehead center
    #         152,  # Chin
    #         234,  # Left cheek outer
    #         454,  # Right cheek outer
    #         # Nose (very stable)
    #         1,    # Nose tip
    #         168,  # Nose bridge
    #         # Eye corners (reasonably stable)
    #         33,   # Left eye outer
    #         263,  # Right eye outer
    #     ])

    #     # Corresponding FLAME vertex indices
    #     # These MUST match the MediaPipe landmarks above
    #     self.pose_flame_indices = np.array([
    #         # Face contour
    #         335,   # Forehead center (approx)
    #         2848,  # Chin
    #         2199,  # Left cheek outer
    #         3872,  # Right cheek outer
    #         # Nose
    #         3509,  # Nose tip
    #         3508,  # Nose bridge
    #         # Eye corners
    #         2536,  # Left eye outer
    #         4207,  # Right eye outer
    #     ])

    #     # Get actual FLAME template positions (set later after model loaded)
    #     self.pose_reference_3d = None  # Will be populated from FLAME template

    #     logger.info(f"[FLAME] Pose landmarks: {len(self.pose_mp_indices)} points")

    # def _create_expression_landmarks(self):
    #     """
    #     Select MediaPipe landmarks that correspond to FLAME vertices.
    #     These are for expression fitting after removing head pose.
    #     """
    #     # MediaPipe landmark -> FLAME vertex correspondence
    #     # Focus on expression-relevant areas: mouth, eyes, eyebrows
    #     self.expression_correspondence = {
    #         # Mouth (most important for expression)
    #         13: 1710,    # Upper lip center
    #         14: 1880,    # Lower lip center
    #         61: 2760,    # Mouth left corner
    #         291: 3433,   # Mouth right corner
    #         78: 2850,    # Upper lip left
    #         308: 3522,   # Upper lip right
    #         95: 2865,    # Lower lip left
    #         324: 3537,   # Lower lip right

    #         # Eyes
    #         159: 1308,   # Left eye top
    #         145: 1296,   # Left eye bottom
    #         386: 4120,   # Right eye top
    #         374: 4108,   # Right eye bottom

    #         # Eyebrows
    #         70: 1067,    # Left eyebrow inner
    #         300: 3879,   # Right eyebrow inner
    #         105: 1045,   # Left eyebrow outer
    #         334: 3857,   # Right eyebrow outer

    #         # Nose tip (for reference)
    #         1: 3509,     # Nose tip

    #         # Jaw line (for jaw pose)
    #         172: 2863,   # Left jaw
    #         397: 3536,   # Right jaw
    #         152: 2848,   # Chin center
    #     }

    #     self.mp_expr_indices = np.array(list(self.expression_correspondence.keys()))
    #     self.flame_expr_indices = np.array(list(self.expression_correspondence.values()))

    #     logger.info(f"[FLAME] Expression landmarks: {len(self.expression_correspondence)} points")

    def _create_pose_landmarks(self, mapping_path="mapping.json", min_points=20):
        """
        Populate:
        - self.pose_mp_indices  (MediaPipe indices used for pose)
        - self.pose_flame_indices (corresponding FLAME vertex indices)
        - self.pose_flame_idx_torch (torch.LongTensor on device)

        Behavior:
        - If a mapping file exists, use it and the default semantic list.
        - If not, fall back to an expanded default MP index list and
            pick `len(default_pose_mp)` FLAME vertices by evenly sampling the
            FLAME template so points are well-distributed (avoids coplanarity).
        """
        mapping_file = Path(mapping_path)

        # Expanded default MediaPipe indices for rigid head pose (widely distributed)
        # default_pose_mp = [
        #     10, 152, 234, 454, 1, 168, 33, 263, 127, 356,
        #     199, 5, 200, 425, 50, 280, 10, 109, 338, 337, 351, 389
        # ]
        
        default_pose_mp = [
            10, 152, 234, 454, 1, 168, 33, 263, 127, 356,
            199, 5, 200, 425, 50, 280, 16, 17, 18, 19, 20
        ]

        # make unique and limit to min_points or more
        default_pose_mp = list(dict.fromkeys(default_pose_mp))[:max(min_points, len(default_pose_mp))]

        # if mapping_file.exists():
        #     with open(mapping_file, "r") as f:
        #         mp_to_flame = json.load(f)
        #     # keep only the default ones present in the mapping
        #     selected_mp = [i for i in default_pose_mp if str(i) in mp_to_flame]
        #     if not selected_mp:
        #         raise RuntimeError("[FLAME] No pose landmarks found in mapping file; aborting.")
        #     selected_flame = [int(mp_to_flame[str(i)]) for i in selected_mp]
        #     self.pose_mp_indices = np.array(selected_mp, dtype=np.int64)
        #     self.pose_flame_indices = np.array(selected_flame, dtype=np.int64)
        #     logger.info(f"[FLAME] Loaded pose mapping from file: {len(self.pose_mp_indices)} points")
        # else:
        # logger.warning(f"[FLAME] Mapping file {mapping_file} not found — using expanded defaults and sampling FLAME vertices.")
        self.pose_mp_indices = np.array(default_pose_mp, dtype=np.int64)

        # If FLAME template available, evenly sample vertices to get well-spread 3D points.
        flame_template = getattr(self.flame, "v_template", None)
        if flame_template is None:
            # fallback to a conservative small set (shouldn't happen if flame is loaded)
            self.pose_flame_indices = np.array([335, 2848, 2199, 3872, 3509, 3508, 2536, 4207], dtype=np.int64)
        else:
            # convert to numpy if torch tensor
            if hasattr(flame_template, "cpu"):
                flame_template_np = flame_template.cpu().numpy()
            else:
                flame_template_np = np.asarray(flame_template)
            n_vertices = flame_template_np.shape[0]
            k = len(self.pose_mp_indices)
            # even sampling across the vertex index range gives a distributed set
            sampled = np.linspace(0, n_vertices - 1, k, dtype=int)
            self.pose_flame_indices = sampled.astype(np.int64)

        logger.info(f"[FLAME] Fallback pose_flame_indices: {len(self.pose_flame_indices)} points")

        # Torch index tensor (defer device conversion if necessary)
        try:
            self.pose_flame_idx_torch = torch.as_tensor(self.pose_flame_indices, dtype=torch.long, device=self.device)
        except Exception:
            self.pose_flame_idx_torch = torch.as_tensor(self.pose_flame_indices, dtype=torch.long)

        logger.info(f"[FLAME] Pose landmarks configured: {len(self.pose_mp_indices)} points")


    def _create_expression_landmarks(self, mapping_path="mapping.json", min_points=40):
        """
        Populate:
        - self.mp_expr_indices
        - self.flame_expr_indices
        - self.flame_expr_idx_torch

        Behavior:
        - If mapping file exists, use it for the sensible defaults.
        - If not, use an expanded default MP list and sample FLAME vertices evenly
            to produce a larger, well-distributed set for stable expression fitting.
        """
        mapping_file = Path(mapping_path)

        # Expanded default MediaPipe indices focusing on mouth, eyes, brows, jaw and cheeks
        default_expr_mp = [
            13, 14, 61, 291, 78, 308, 95, 324, 159, 145, 386, 374,
            70, 300, 105, 334, 1, 172, 397, 152, 10, 109, 338, 337,
            351, 389, 199, 5, 200, 425, 50, 280, 129, 359, 234, 454,
            93, 323, 267, 287  # add more around lips/cheeks
        ]
        default_expr_mp = list(dict.fromkeys(default_expr_mp))[:max(min_points, len(default_expr_mp))]

        # if mapping_file.exists():
        #     with open(mapping_file, "r") as f:
        #         mp_to_flame = json.load(f)
        #     selected_mp = [i for i in default_expr_mp if str(i) in mp_to_flame]
        #     if not selected_mp:
        #         raise RuntimeError("[FLAME] No expression landmarks found in mapping file; aborting.")
        #     selected_flame = [int(mp_to_flame[str(i)]) for i in selected_mp]
        #     self.mp_expr_indices = np.array(selected_mp, dtype=np.int64)
        #     self.flame_expr_indices = np.array(selected_flame, dtype=np.int64)
        #     self.expression_correspondence = {int(k): int(v) for k, v in zip(selected_mp, selected_flame)}
        #     logger.info(f"[FLAME] Loaded expression mapping from file: {len(self.mp_expr_indices)} points")
        # else:
        # logger.warning(f"[FLAME] Mapping file {mapping_file} not found — using expanded defaults and sampling FLAME vertices.")
        self.mp_expr_indices = np.array(default_expr_mp, dtype=np.int64)

        flame_template = getattr(self.flame, "v_template", None)
        if flame_template is None:
            # conservative fallback mapping
            fallback = list(self.expression_correspondence.keys()) if hasattr(self, "expression_correspondence") else list(self.mp_expr_indices[:min(20, len(self.mp_expr_indices))])
            self.flame_expr_indices = np.array([int(x) for x in fallback], dtype=np.int64)
        else:
            if hasattr(flame_template, "cpu"):
                flame_template_np = flame_template.cpu().numpy()
            else:
                flame_template_np = np.asarray(flame_template)
            n_vertices = flame_template_np.shape[0]
            k = len(self.mp_expr_indices)
            sampled = np.linspace(0, n_vertices - 1, k, dtype=int)
            self.flame_expr_indices = sampled.astype(np.int64)

        # Build a simple correspondence dict (mp -> flame) for debugging
        self.expression_correspondence = {int(mp): int(flame) for mp, flame in zip(self.mp_expr_indices, self.flame_expr_indices)}
        logger.info(f"[FLAME] Fallback flame_expr_indices: {len(self.flame_expr_indices)} points")

        # Torch index tensor
        try:
            self.flame_expr_idx_torch = torch.as_tensor(self.flame_expr_indices, dtype=torch.long, device=self.device)
        except Exception:
            self.flame_expr_idx_torch = torch.as_tensor(self.flame_expr_indices, dtype=torch.long)

        logger.info(f"[FLAME] Expression landmarks configured: {len(self.mp_expr_indices)} points")

    # def fit(self, mediapipe_landmarks_3d, optimize_shape=False, optimize_expression=True):
    #     """
        
        
    #     Two-stage fitting:
    #     1. Estimate head pose from MediaPipe landmarks (no FLAME)
    #     2. Fit expression parameters in FLAME (after removing head pose)
    #     """
        
    #     logger.info(f"mediapipe_landmarks_3d from fit: {mediapipe_landmarks_3d}")
    #     start_time = time.time()

    #     # ========================================
    #     # STAGE 1: HEAD POSE (No FLAME)
    #     # ========================================
    #     pose_landmarks = mediapipe_landmarks_3d[self.pose_mp_indices]
    #     scale, R_head, t_head = self._estimate_head_pose(
    #         pose_landmarks,
    #         image_size=self.image_size,
    #         use_pnp=self.use_pnp
    #     )

    #     self.head_scale = scale
    #     self.head_rotation = R_head
    #     self.head_translation = t_head
        
    #     logger.info(f"scale: {scale}, R_head: {R_head}, t_head: {t_head}")
        

    #     # ========================================
    #     # STAGE 2: EXPRESSION FITTING
    #     # ========================================
    #     if optimize_expression:
    #         # Get expression-relevant landmarks
    #         expr_landmarks_mp = mediapipe_landmarks_3d[self.mp_expr_indices]

    #         # Remove rigid head motion to get pure expression deformation
    #         # Note: _remove_head_pose now handles both PnP (X,Y only) and Procrustes (X,Y,Z)
    #         expr_landmarks_canonical = self._remove_head_pose(expr_landmarks_mp, R_head, t_head)

    #         # Optimize expression in canonical space
    #         expression_params, jaw_pose = self._optimize_expression_only(expr_landmarks_canonical)

    #         # Apply temporal smoothing
    #         if self.use_temporal_smoothing:
    #             alpha = self.expression_smoothing
    #             expression_params = alpha * self.current_expression + (1 - alpha) * expression_params
    #             jaw_pose = alpha * self.current_jaw_pose + (1 - alpha) * jaw_pose

    #         self.current_expression = expression_params
    #         self.current_jaw_pose = jaw_pose

    #     # Optional: calibrate shape (usually done once, not per-frame)
    #     if optimize_shape:
    #         self.calibrate_shape(mediapipe_landmarks_3d, num_iterations=20)

    #     # ========================================
    #     # STAGE 3: GENERATE FINAL MESH
    #     # ========================================
    #     # Build full pose vector: [global_rot(3), neck(3), jaw(3), eyes(6)]
    #     # We only animate jaw, others stay at zero
    #     full_pose = np.zeros(15)
    #     full_pose[6:9] = self.current_jaw_pose  # Jaw rotation

    #     # Generate FLAME mesh in canonical pose
    #     vertices_canonical = self.flame.forward(
    #         shape_params=self.current_shape,
    #         expression_params=self.current_expression,
    #         pose_params=full_pose[:12]  # FLAME expects 12 pose params
    #     )

    #     if isinstance(vertices_canonical, torch.Tensor):
    #         vertices_canonical = vertices_canonical.cpu().numpy()

    #     # Apply head pose to transform mesh to camera space
    #     vertices = self._apply_head_pose(vertices_canonical, scale, R_head, t_head)

    #     # Compute normals
    #     normals = self.flame.compute_normals(vertices)
    #     centroid = self.flame.compute_centroid(vertices)

    #     fit_time = time.time() - start_time

    #     return {
    #         'vertices': vertices,
    #         'faces': self.flame.get_faces_numpy(),
    #         'normals': normals,
    #         'centroid': centroid,
    #         'shape_params': self.current_shape,
    #         'expression_params': self.current_expression,
    #         'pose_params': full_pose[:12],
    #         'head_pose': {
    #             'rotation': R_head.tolist(),
    #             'translation': t_head.tolist(),
    #             'scale': float(scale),
    #         },
    #         'jaw_pose': self.current_jaw_pose.tolist(),
    #         'landmarks_3d': mediapipe_landmarks_3d,
    #         'fit_time_ms': fit_time * 1000,
    #         'num_landmarks_used': len(self.mp_expr_indices),
    #     }
    
    
    
    
    def fit(self, mediapipe_landmarks_3d, optimize_shape=False, optimize_expression=True):
        """
        Robust dynamic-fit pipeline (no static FLAME vertex indices required).

        Strategy summary:
        - Build a stable head frame from MediaPipe rigid landmarks (PCA/SVD).
        - Build a canonical face frame from FLAME neutral face vertices (auto-selected).
        - Compute R by aligning the two frames, compute scale from IPD (MP pixels / FLAME meters),
        compute t from centroids so s * R * V_flame + t = V_mp.
        - Convert MP expression landmarks into canonical FLAME space and optimize expressions.
        - Generate FLAME vertices in canonical space and re-apply s,R,t to get camera-space vertices.

        Notes:
        - This function prefers `self.mp_eye_indices` and `self.pose_mp_indices` to be set (MediaPipe indices).
        - If `self.flame_face_indices` exists it will use it to get FLAME face region; otherwise it heuristically picks a frontal face subset.
        """
        start_time = time.time()

        # ---------- helpers ----------
        def svd_pca_axes(points):
            # points: (N,3)
            # returns axes matrix R where columns are principal axes [pc1, pc2, pc3]
            C = points - points.mean(axis=0)
            U, S, Vt = np.linalg.svd(C.T @ C)
            axes = Vt.T  # columns are principal directions sorted by variance desc
            # Ensure right-handed basis
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1
            return axes

        def kabsch_rotation(A, B):
            # A,B centered (N,3) -> rotation mapping A -> B
            H = A.T @ B
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            return R

        def orthonormalize(M):
            U, _, Vt = np.linalg.svd(M)
            return U @ Vt

        # ---------- sanity ----------
        mp_all = np.asarray(mediapipe_landmarks_3d)
        assert mp_all.ndim == 2 and mp_all.shape[1] == 3

        # ---------- select stable MediaPipe rigid landmarks ----------
        # Use the class-provided pose indices if available; fallback to a recommended set
        pose_mp_idx = getattr(self, "pose_mp_indices", None)
        if pose_mp_idx is None:
            # recommended stable indices (MediaPipe face mesh): left eye outer, right eye outer, nose bridge, temples
            pose_mp_idx = [33, 263, 6, 234, 454]
            logger.warning("pose_mp_indices not set; using default stable MP indices: %s", pose_mp_idx)
        mp_rigid = mp_all[pose_mp_idx]  # (K,3)

        # Quick validity check
        if len(mp_rigid) < 3:
            logger.error("[fit] insufficient MP rigid points for alignment")
            return None

        # ---------- compute MediaPipe head frame (PCA) ----------
        mp_centroid = mp_rigid.mean(axis=0)
        R_mp_axes = svd_pca_axes(mp_rigid)   # columns: principal axes (pc1, pc2, pc3)
        # Build R_mp such that it maps FLAME canonical axes to MP world axes later; keep as a 3x3
        R_mp = R_mp_axes

        # ---------- prepare FLAME canonical (neutral) face points ----------
        V_neutral = np.asarray(self.flame.get_template_vertices())  # (V,3) in meters
        
        logger.info(f"First rows of V_neutral: {V_neutral[:5]}")
        logger.info(f"V_neutral shape: {V_neutral.shape}")
        # Use provided face indices if available, otherwise heuristically pick frontal face region
        flame_face_idx = getattr(self, "flame_face_indices", None)
        if flame_face_idx is not None and len(flame_face_idx) > 50:
            flame_face_vertices = V_neutral[flame_face_idx]
        else:
            # Heuristic: choose vertices whose Z is near the nose region and Y is above the lower head (a frontal face band)
            vy = V_neutral[:, 1]
            vz = V_neutral[:, 2]
            mean_y = vy.mean()
            mean_z = vz.mean()
            # heuristics thresholds (adjust if your FLAME coordinate system differs)
            face_mask = (vy > mean_y - 0.06) & (vz > mean_z - 0.06)
            idxs = np.where(face_mask)[0]
            if len(idxs) < 100:
                # fallback to central band if too few
                idxs = np.where((vy > mean_y - 0.12) & (vy < mean_y + 0.06))[0]
            flame_face_vertices = V_neutral[idxs]
            logger.info("[fit] using heuristic FLAME face subset of %d vertices", len(idxs))

        # ---------- compute FLAME face frame (PCA) ----------
        flame_centroid = flame_face_vertices.mean(axis=0)
        R_flame_axes = svd_pca_axes(flame_face_vertices)  # canonical axes for FLAME face region

        # ---------- compute rotation mapping FLAME -> MP ----------
        # We want R such that: roughly B = R @ A  (A is flame canonical axes, B is mp axes)
        # So R = R_mp @ R_flame.T
        R_est = R_mp @ R_flame_axes.T
        # re-orthonormalize in case of numerical drift
        R_est = orthonormalize(R_est)

        # ---------- compute scale s ----------
        # Use interpupillary distance if possible (robust). Fallback to ratio of bounding-box widths.
        mp_left_eye_idx, mp_right_eye_idx = getattr(self, "mp_eye_indices", (33, 263))
        flame_left_eye_idx, flame_right_eye_idx = getattr(self, "flame_eye_indices", (None, None))

        # MP IPD (in input units — pixels)
        try:
            mp_ipd = float(np.linalg.norm(mp_all[mp_left_eye_idx] - mp_all[mp_right_eye_idx]))
        except Exception:
            mp_ipd = None

        if flame_left_eye_idx is not None and flame_right_eye_idx is not None:
            flame_ipd = float(np.linalg.norm(V_neutral[flame_left_eye_idx] - V_neutral[flame_right_eye_idx]))
        else:
            # fallback: estimate FLAME "face width" along first principal axis (pc1)
            proj = (flame_face_vertices - flame_centroid) @ R_flame_axes[:, 0]  # projection onto pc1
            flame_ipd = float(proj.max() - proj.min()) * 0.5  # scale heuristic (half-width ~ eye distance)
            # If this seems too small, the subsequent s will be adjusted but it's a heuristic.

        # final scale: map FLAME meters -> MP units (pixels)
        if mp_ipd is None or flame_ipd <= 1e-8:
            s = getattr(self, "head_scale", 1.0)  # fallback
            logger.warning("[fit] IPD fallback used for scale (mp_ipd=%s, flame_ipd=%s)", mp_ipd, flame_ipd)
        else:
            s = float(mp_ipd / flame_ipd)

        # ---------- compute translation t so that s * R * flame_centroid + t = mp_centroid ----------
        t_est = mp_centroid - s * (R_est @ flame_centroid)

        # ---------- optional temporal smoothing ----------
        if getattr(self, "use_temporal_smoothing", False):
            alpha = getattr(self, "pose_smoothing", 0.85)
            # scale
            prev_s = getattr(self, "head_scale", s)
            s = alpha * prev_s + (1 - alpha) * s
            # translation
            prev_t = getattr(self, "head_translation", t_est)
            t_est = alpha * prev_t + (1 - alpha) * t_est
            # rotation: blend matrices then re-orthonormalize
            prev_R = getattr(self, "head_rotation", R_est)
            R_blend = alpha * prev_R + (1 - alpha) * R_est
            R_est = orthonormalize(R_blend)

        # save pose
        self.head_scale = s
        self.head_rotation = R_est
        self.head_translation = t_est

        logger.info(f"[fit] head_scale={s:.4f}, head_translation={t_est}, R_est det={np.linalg.det(R_est):.6f}")

        # ---------- expression optimization ----------
        # expression_params = getattr(self, "current_expression", np.zeros(self.flame.n_expression_params))
        
        expression_params = getattr(self, "current_expression", np.zeros(self.flame.num_expressions))

        jaw_pose = getattr(self, "current_jaw_pose", np.zeros(3))

        if optimize_expression:
            expr_mp_idx = getattr(self, "mp_expr_indices", None)
            if expr_mp_idx is None:
                logger.warning("[fit] mp_expr_indices not set; skipping expression optimization")
            else:
                mp_expr_points = mp_all[expr_mp_idx]  # (M,3) in mp units

                # Remove rigid head pose: X_canon = R.T * ((X_mp - t) / s)
                expr_canonical = (mp_expr_points - t_est) / s
                expr_canonical = (R_est.T @ expr_canonical.T).T  # now in FLAME canonical meters (approx)

                # call expression optimizer (expects FLAME-canonical landmarks in meters)
                try:
                    expression_params, jaw_pose = self._optimize_expression_only(expr_canonical)
                except Exception as e:
                    logger.error("[fit] _optimize_expression_only failed: %s", e)
                    expression_params = getattr(self, "current_expression", expression_params)
                    jaw_pose = getattr(self, "current_jaw_pose", jaw_pose)

                # smoothing
                if getattr(self, "use_temporal_smoothing", False):
                    alpha_e = getattr(self, "expression_smoothing", 0.85)
                    expression_params = alpha_e * np.asarray(self.current_expression) + (1 - alpha_e) * np.asarray(expression_params)
                    jaw_pose = alpha_e * np.asarray(self.current_jaw_pose) + (1 - alpha_e) * np.asarray(jaw_pose)

                self.current_expression = np.asarray(expression_params)
                self.current_jaw_pose = np.asarray(jaw_pose)

        # ---------- optional: calibrate / lock shape ----------
        if optimize_shape and not getattr(self, "shape_locked", False):
            self.calibrate_shape(mediapipe_landmarks_3d, num_iterations=20)
            self.shape_locked = True

        # ---------- generate canonical FLAME mesh and reapply pose ----------
        full_pose = np.zeros(15)
        full_pose[6:9] = np.asarray(self.current_jaw_pose).reshape(3,)

        vertices_canonical = self.flame.forward(
            shape_params=np.asarray(self.current_shape),
            expression_params=np.asarray(self.current_expression),
            pose_params=full_pose[:12]
        )
        if isinstance(vertices_canonical, torch.Tensor):
            vertices_canonical = vertices_canonical.detach().cpu().numpy()
        vertices_canonical = np.asarray(vertices_canonical)

        # Apply rigid pose: V_world = s * (R_est @ V_canonical) + t_est
        # vertices_world = (s * (R_est @ vertices_canonical.T)).T + t_est
        
        vertices_world = transform_flame_vertices(vertices_canonical, R_est, t_est, s)


        normals = self.flame.compute_normals(vertices_world)
        centroid = self.flame.compute_centroid(vertices_world)

        fit_time = (time.time() - start_time)

        # ---------- prepare result ----------
        result = {
            'vertices': vertices_world,
            'faces': self.flame.get_faces_numpy(),
            'normals': normals,
            'centroid': centroid,
            'shape_params': np.asarray(self.current_shape),
            'expression_params': np.asarray(self.current_expression),
            'pose_params': full_pose[:12],
            'head_pose': {
                'rotation': R_est.tolist(),
                'translation': t_est.tolist(),
                'scale': float(s),
            },
            'jaw_pose': np.asarray(self.current_jaw_pose).tolist(),
            'landmarks_3d': mediapipe_landmarks_3d,
            'fit_time_ms': fit_time * 1000.0,
            'num_landmarks_used': int(len(getattr(self, "mp_expr_indices", []))),
        }

        return result


    # def _estimate_head_pose(self, pose_landmarks):
    #     """
    #     STAGE 1: Estimate head pose using Kabsch algorithm.
    #     No FLAME involved - pure geometry from MediaPipe landmarks.

    #     Returns:
    #         scale: uniform scale factor
    #         R: (3,3) rotation matrix
    #         t: (3,) translation vector
    #     """
    #     # Reference points in canonical space
    #     src = self.pose_reference_3d.copy()

    #     # Target points from MediaPipe (in pixel coordinates)
    #     tgt = pose_landmarks.copy()

    #     # Compute centroids
    #     src_centroid = np.mean(src, axis=0)
    #     tgt_centroid = np.mean(tgt, axis=0)

    #     # Center both point sets
    #     src_centered = src - src_centroid
    #     tgt_centered = tgt - tgt_centroid

    #     # Compute scale
    #     src_scale = np.sqrt(np.sum(src_centered ** 2))
    #     tgt_scale = np.sqrt(np.sum(tgt_centered ** 2))
    #     scale = tgt_scale / src_scale if src_scale > 1e-6 else 1.0

    #     # Scale source to match target
    #     src_scaled = src_centered * scale

    #     # Kabsch algorithm: find optimal rotation
    #     # Covariance matrix: H = src^T @ tgt
    #     H = src_scaled.T @ tgt_centered

    #     # SVD
    #     U, S, Vt = np.linalg.svd(H)

    #     # Rotation: R = V @ U^T
    #     R = Vt.T @ U.T
        
        

    #     # Handle reflection (ensure det(R) = +1)
    #     if np.linalg.det(R) < 0:
    #         Vt[-1, :] *= -1
    #         R = Vt.T @ U.T

    #     # Translation: t = tgt_centroid - scale * R @ src_centroid
    #     t = tgt_centroid - scale * (src_centroid @ R)
        
        
    #     if not np.isfinite(R).all() or not np.isfinite(t).all() or not np.isfinite(scale):
    #         logger.info("[DEBUG] Invalid head pose detected!")
    #         logger.info("R:\n", R)
    #         logger.info("t:\n", t)
    #         logger.info("scale:", scale)
    #         # Safe fallback
    #         R = np.eye(3)
    #         t = np.zeros(3)
    #         scale = 1.0
    #     logger.info(f"[FLAME] Head pose estimated: scale={scale:.3f}, R={R}, t={t}")

    #     return scale, R, t
    
    # def _estimate_head_pose_pnp(self,
    #                             pose_landmarks,
    #                             image_size=(640, 480),
    #                             focal_length=None):
    #     """
    #     STAGE 1: Estimate head pose using PnP (Perspective-n-Point) solver.

    #     This uses proper camera projection model instead of Procrustes alignment,
    #     giving metric-correct head pose independent of MediaPipe's relative depth.

    #     Args:
    #         pose_landmarks: (N,3) array of MediaPipe landmarks in pixel coords
    #                        Only X,Y are used (2D image positions)
    #         image_size: tuple (img_w, img_h) for camera intrinsics
    #         focal_length: camera focal length in pixels (default: 1.0 * img_w)

    #     Returns:
    #         scale: uniform scale factor (always 1.0 for PnP - scale is in translation)
    #         R: (3,3) rotation matrix (camera to world)
    #         t: (3,) translation vector (world origin in camera coords)
    #     """
    #     # Defensive checks
    #     pose_landmarks = np.asarray(pose_landmarks, dtype=np.float64)
    #     if pose_landmarks.size == 0 or pose_landmarks.shape[1] != 3:
    #         logger.info("[FLAME] _estimate_head_pose_pnp: invalid input landmarks shape %s", pose_landmarks.shape)
    #         return 1.0, np.eye(3), np.zeros(3)
        
        

    #     img_w, img_h = image_size

    #     # Build camera intrinsic matrix
    #     if focal_length is None:
    #         focal_length = img_w  # Default: focal length = image width

    #     cx = img_w / 2.0  # Principal point X
    #     cy = img_h / 2.0  # Principal point Y

    #     camera_matrix = np.array([
    #         [focal_length, 0, cx],
    #         [0, focal_length, cy],
    #         [0, 0, 1]
    #     ], dtype=np.float64)

    #     # Distortion coefficients (assume no distortion)
    #     dist_coeffs = np.zeros(4, dtype=np.float64)

    #     # 3D points: FLAME template vertices at pose landmark indices
    #     # These are in FLAME canonical space (millimeters, centered at face)
    #     object_points = self.pose_reference_3d.copy()  # (N, 3)
        
        
    #     logger.info(f"object_points: {pose_landmarks.shape[0] == object_points.shape[0]}")
    #     # check correspondence
    #     logger.info(f"pose_landmarks: {pose_landmarks}")
    #     logger.info(f"object_points: {object_points}")

    #     # 2D points: MediaPipe landmarks in image space (pixels)
    #     # Only use X, Y (ignore MediaPipe's Z which is relative depth)
    #     image_points = pose_landmarks[:, :2].copy()  # (N, 2)
        
    #     logger.info(f"image_points: {image_points}")

    #     # Validate we have enough points
    #     if len(object_points) < 4:
    #         logger.warning("[FLAME] _estimate_head_pose_pnp: need at least 4 points, got %d", len(object_points))
    #         return 1.0, np.eye(3), np.zeros(3)
        
        

    #     pts = object_points  # (N,3) FLAME points
    #     # Compute the rank of the matrix after centering
    #     rank = np.linalg.matrix_rank(pts - pts.mean(axis=0))
    #     logger.info(f"3D rank: {rank}")
        
    #     focal_length = 1.0  # use normalized units matching FLAME metric
    #     camera_matrix = np.array([
    #         [focal_length, 0, img_w/2],
    #         [0, focal_length, img_h/2],
    #         [0, 0, 1]
    #     ], dtype=np.float64)
        
    #     logger.info(f"camera_matrix: {camera_matrix}")



    #     # Solve PnP using RANSAC for robustness
    #     try:
    #         success, rvec, tvec, inliers = cv2.solvePnPRansac(
    #             objectPoints=object_points,
    #             imagePoints=image_points,
    #             cameraMatrix=camera_matrix,
    #             distCoeffs=dist_coeffs,
    #             flags=cv2.SOLVEPNP_ITERATIVE,
    #             reprojectionError=8.0,  # Pixel threshold for inliers
    #             confidence=0.99,
    #             iterationsCount=100
    #         )
            
    #         logger.info(f"rvec: {rvec}, tvec: {tvec}, inliers: {inliers}")
    #         logger.info(f"success: {success}")

    #         if not success or rvec is None or tvec is None:
    #             logger.warning("[FLAME] _estimate_head_pose_pnp: PnP solver failed")
    #             return 1.0, np.eye(3), np.zeros(3)

    #         # Convert rotation vector to rotation matrix
    #         R, _ = cv2.Rodrigues(rvec)
    #         t = tvec.flatten()

    #         # Validate results
    #         if not (np.isfinite(R).all() and np.isfinite(t).all()):
    #             logger.warning("[FLAME] _estimate_head_pose_pnp: Non-finite pose computed")
    #             return 1.0, np.eye(3), np.zeros(3)

    #         # Check rotation matrix is valid (det(R) ≈ 1)
    #         det_R = np.linalg.det(R)
    #         if abs(det_R - 1.0) > 0.1:
    #             logger.warning("[FLAME] _estimate_head_pose_pnp: Invalid rotation matrix det(R)=%.3f", det_R)
    #             return 1.0, np.eye(3), np.zeros(3)

    #         # Clamp translation to reasonable range (prevent extreme values)
    #         t = np.clip(t, -1000.0, 1000.0)

    #         # Log results
    #         inlier_ratio = len(inliers) / len(object_points) if inliers is not None else 0.0
    #         logger.info(f"[FLAME] PnP head pose: det(R)={det_R:.6f}, t={t.tolist()}, inliers={inlier_ratio:.1%}")

    #         # Return scale=1.0 (PnP gives metric pose, scale is implicit in translation)
    #         return 1.0, R, t

    #     except Exception as e:
    #         logger.error(f"[FLAME] _estimate_head_pose_pnp: Exception during PnP: {e}")
    #         return 1.0, np.eye(3), np.zeros(3)
    
    def _estimate_head_pose_pnp(self,
                           pose_landmarks,
                           image_size=(640, 480),
                           focal_length=None):
        """
        Estimate head pose using PnP (with RANSAC). Returns (scale, R, t).
        Robust behavior:
        - verifies input shapes & correspondence with self.pose_reference_3d
        - tries pixel-based PnP first (camera intrinsics in pixels)
        - if that fails, retries using normalized image coords (focal=1.0)
        - logs details and returns identity pose on failure

        Args:
            pose_landmarks: (N,3) or (N,2) array of MediaPipe landmarks (pixels).
                        Only X,Y are used for PnP.
            image_size: (img_w, img_h)
            focal_length: focal length in pixels. If None, defaults to img_w.

        Returns:
            scale (float), R (3x3), t (3,)
        """
        pose_landmarks = np.asarray(pose_landmarks, dtype=np.float64)
        if pose_landmarks.size == 0:
            logger.warning("[FLAME] _estimate_head_pose_pnp: empty input")
            return 1.0, np.eye(3), np.zeros(3)

        # Accept (N,3) or (N,2)
        if pose_landmarks.ndim != 2 or pose_landmarks.shape[1] not in (2, 3):
            logger.warning("[FLAME] _estimate_head_pose_pnp: invalid landmarks shape %s", pose_landmarks.shape)
            return 1.0, np.eye(3), np.zeros(3)

        img_w, img_h = image_size
        if focal_length is None:
            focal_length = float(img_w)
        
        
        # logger.info(f"pose_landmarks: {pose_landmarks}")
        # logger.info(f" self.pose_reference_3d: {self.pose_reference_3d}")

        # Prepare object points (FLAME canonical 3D)
        object_points = np.asarray(self.pose_reference_3d, dtype=np.float64).copy()  # (N,3)
        if object_points.ndim != 2 or object_points.shape[1] != 3:
            logger.warning("[FLAME] _estimate_head_pose_pnp: invalid pose_reference_3d shape %s", object_points.shape)
            return 1.0, np.eye(3), np.zeros(3)

        # Ensure correspondence length matches
        image_points_xy = pose_landmarks[:, :2].copy()  # drop MediaPipe z if present
        if image_points_xy.shape[0] != object_points.shape[0]:
            logger.warning("[FLAME] _estimate_head_pose_pnp: landmark count mismatch image(%d) vs object(%d)",
                        image_points_xy.shape[0], object_points.shape[0])
            return 1.0, np.eye(3), np.zeros(3)

        logger.debug(f"[FLAME] object_points: {object_points.shape}, image_points: {image_points_xy.shape}")

        # Check 3D rank to avoid coplanar degenerate sets
        rank = np.linalg.matrix_rank(object_points - object_points.mean(axis=0))
        logger.debug(f"[FLAME] 3D rank: {rank}")
        if rank < 3:
            logger.warning("[FLAME] _estimate_head_pose_pnp: object_points are (nearly) coplanar (rank=%d)", rank)
            return 1.0, np.eye(3), np.zeros(3)

        # Convert shapes to OpenCV expected formats
        obj_pts_cv = object_points.reshape(-1, 1, 3).astype(np.float64)
        img_pts_cv = image_points_xy.reshape(-1, 1, 2).astype(np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Primary camera matrix (pixel units)
        camera_matrix_px = np.array([
            [focal_length, 0.0, img_w / 2.0],
            [0.0, focal_length, img_h / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # RANSAC parameters (more robust defaults)
        reproj_px = 12.0
        iterations = 1000
        confidence = 0.99

        def try_pnp(cam_mat, img_pts, obj_pts, reproj_thresh, iters):
            try:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    cameraMatrix=cam_mat,
                    distCoeffs=dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=reproj_thresh,
                    confidence=confidence,
                    iterationsCount=iters
                )
                logger.debug(f"[FLAME] solvePnP retval={retval}, inliers={None if inliers is None else len(inliers)}")
                if not retval or rvec is None or tvec is None:
                    return False, None, None, None
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()
                if not (np.isfinite(R).all() and np.isfinite(t).all()):
                    return False, None, None, None
                if abs(np.linalg.det(R) - 1.0) > 0.1:
                    logger.debug("[FLAME] solvePnP produced invalid rotation det=%f", np.linalg.det(R))
                    return False, None, None, None
                return True, R, t, inliers
            except Exception as e:
                logger.exception("[FLAME] solvePnPRansac exception: %s", e)
                return False, None, None, None

        # Try pixel-based PnP first
        success, R, t, inliers = try_pnp(camera_matrix_px, img_pts_cv, obj_pts_cv, reproj_px, iterations)
        logger.info(f"success: {success}, R: {R}, t: {t}, inliers: {inliers}")

        # Fallback: try normalized coordinates (camera focal=1, principal=(0.5,0.5))
        if not success:
            logger.info("[FLAME] pixel PnP failed — retrying with normalized coordinates")
            # normalize image points to [0,1] range and set camera focal=1
            img_pts_norm = np.empty_like(img_pts_cv)
            img_pts_norm[:, 0, 0] = image_points_xy[:, 0] / float(img_w)
            img_pts_norm[:, 0, 1] = image_points_xy[:, 1] / float(img_h)
            cam_norm = np.array([
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            # Use looser reprojection in normalized coords
            success, R, t, inliers = try_pnp(cam_norm, img_pts_norm, obj_pts_cv, reproj_thresh=0.02, iters=iterations)

        if not success:
            logger.warning("[FLAME] _estimate_head_pose_pnp: PnP failed (returning identity)")
            return 1.0, np.eye(3), np.zeros(3)

        # Clip translation to avoid wild values (in object point units)
        t = np.clip(t, -1e3, 1e3)

        inlier_ratio = (0.0 if inliers is None else float(len(inliers)) / float(len(object_points)))
        # logger.info(f"[FLAME] PnP head pose: det(R)={np.linalg.det(R):.6f}, t={t.tolist()}, inliers={inlier_ratio:.1%}")

        # PnP returns metric pose (units same as object_points), scale is implicit in t
        return 1.0, R, t



    def _estimate_head_pose(self,
                        pose_landmarks,
                        image_size=(640, 480),
                        z_scale=100.0,
                        normalize=True,
                        use_pnp=True):
        """
        STAGE 1: Estimate head pose using either PnP solver or Procrustes alignment.

        Args:
            pose_landmarks: (N,3) array of MediaPipe landmarks (pixel coords by default)
            image_size: tuple (img_w, img_h) used for camera intrinsics or normalization
            z_scale: divisor for MediaPipe Z (only used if use_pnp=False)
            normalize: if True, normalize coords (only used if use_pnp=False)
            use_pnp: if True, use PnP solver (recommended); if False, use Procrustes

        Returns:
            scale: uniform scale factor (1.0 for PnP, computed for Procrustes)
            R: (3,3) rotation matrix
            t: (3,) translation vector
        """
        if use_pnp:
            # Use PnP solver (metric-correct, camera-aware)
            return self._estimate_head_pose_pnp(pose_landmarks, image_size)
        else:
            # Fallback to Procrustes alignment (original method)
            return self._estimate_head_pose_procrustes(pose_landmarks, image_size, z_scale, normalize)


    def _estimate_head_pose_procrustes(self,
                                      pose_landmarks,
                                      image_size=(640, 480),
                                      z_scale=100.0,
                                      normalize=True):
        """
        STAGE 1: Estimate head pose using Kabsch/Procrustes algorithm (LEGACY).

        NOTE: This is the old method. Use _estimate_head_pose_pnp() instead for
        metric-correct pose estimation.

        Args:
            pose_landmarks: (N,3) array of MediaPipe landmarks (pixel coords by default)
            image_size: tuple (img_w, img_h) used to normalize XY to [-1, 1]
            z_scale: divisor to bring MediaPipe Z into similar canonical range (tune per camera)
            normalize: if True, convert pixel coords -> normalized coords before fitting

        Returns:
            scale: uniform scale factor (clamped)
            R: (3,3) rotation matrix
            t: (3,) translation vector
        """
        # Defensive checks
        pose_landmarks = np.asarray(pose_landmarks, dtype=np.float64)
        if pose_landmarks.size == 0 or pose_landmarks.shape[1] != 3:
            logger.info("[FLAME] _estimate_head_pose_procrustes: invalid input landmarks shape %s", pose_landmarks.shape)
            return 1.0, np.eye(3), np.zeros(3)

        # Optionally normalize MediaPipe pixel coords to ~[-1,1] range so units match FLAME
        tgt = pose_landmarks.copy()
        if normalize:
            img_w, img_h = image_size
            # center and normalize X,Y to [-1,1]
            tgt[:, 0] = (tgt[:, 0] - (img_w / 2.0)) / (img_w / 2.0)
            tgt[:, 1] = (tgt[:, 1] - (img_h / 2.0)) / (img_h / 2.0)
            # scale Z to roughly similar magnitude (tune z_scale for your camera)
            tgt[:, 2] = tgt[:, 2] / float(z_scale)

        # Reference points in canonical (FLAME) space
        src = self.pose_reference_3d.copy()

        # Compute centroids
        src_centroid = np.mean(src, axis=0)
        tgt_centroid = np.mean(tgt, axis=0)

        # Center both point sets
        src_centered = src - src_centroid
        tgt_centered = tgt - tgt_centroid

        # Compute Frobenius norms (scale measures)
        src_scale = np.linalg.norm(src_centered)
        tgt_scale = np.linalg.norm(tgt_centered)

        # Guard against degenerate configurations
        eps = 1e-6
        if not np.isfinite(src_scale) or src_scale < eps or not np.isfinite(tgt_scale) or tgt_scale < eps:
            logger.info("[FLAME] _estimate_head_pose_procrustes: degenerate scales src_scale=%s tgt_scale=%s - fallback to identity",
                        src_scale, tgt_scale)
            return 1.0, np.eye(3), np.zeros(3)

        # Compute scale (clamped)
        raw_scale = tgt_scale / src_scale
        scale = float(np.clip(raw_scale, 0.01, 10.0))

        # Scale source to match target for covariance computation
        src_scaled = src_centered * scale

        # Compute covariance matrix and run SVD (Kabsch)
        H = src_scaled.T @ tgt_centered
        try:
            U, S, Vt = np.linalg.svd(H)
        except Exception as e:
            logger.info("[FLAME] _estimate_head_pose_procrustes: SVD failed: %s", e)
            return 1.0, np.eye(3), np.zeros(3)

        # Rotation: R = V * U^T  (V = Vt.T)
        R = Vt.T @ U.T

        # Correct reflection if any (ensure proper rotation)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Translation: t = tgt_centroid - scale * (src_centroid @ R)
        # (we use row-vector multiply to be consistent with vertices @ R convention)
        t = tgt_centroid - (scale * (src_centroid @ R))

        # Final sanity checks & clamping to avoid numerical explosion
        if not (np.isfinite(R).all() and np.isfinite(t).all() and np.isfinite(scale)):
            logger.info("[FLAME] _estimate_head_pose_procrustes: Non-finite pose computed - falling back to identity")
            return 1.0, np.eye(3), np.zeros(3)

        # Optionally clamp translation magnitude (tunable)
        t = np.clip(t, -10.0, 10.0)

        logger.info(f"[FLAME] Head pose (Procrustes): scale={scale:.3f}, det(R)={np.linalg.det(R):.6f}, t={t.tolist()}")
        return scale, R, t


    def _project_landmarks_to_3d(self, landmarks_2d, R, t):
        """
        Project 2D MediaPipe landmarks to 3D using PnP-estimated head pose.

        Since MediaPipe only gives reliable X,Y (not Z), we reconstruct 3D positions
        by back-projecting through the camera using the known head pose.

        Args:
            landmarks_2d: (N, 2) array of MediaPipe X,Y positions in pixels
            R: (3, 3) rotation matrix from PnP
            t: (3,) translation vector from PnP

        Returns:
            landmarks_3d: (N, 3) array in camera coordinate system
        """
        # Build camera matrix
        img_w, img_h = self.image_size
        fx = fy = self.focal_length
        cx = img_w / 2.0
        cy = img_h / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # For each 2D point, we need to find its 3D position
        # We know: 2D_point = K @ (R @ 3D_point_canonical + t)
        # We assume the landmark lies on the FLAME surface (depth from template)

        landmarks_3d = []
        K_inv = np.linalg.inv(K)

        for i, pt_2d in enumerate(landmarks_2d):
            # Normalize 2D point
            pt_homogeneous = np.array([pt_2d[0], pt_2d[1], 1.0])
            ray_camera = K_inv @ pt_homogeneous  # Ray direction in camera space

            # We need depth. Use a reasonable default (average face depth)
            # Or better: use the corresponding FLAME template vertex depth
            # For now, use average depth from translation vector
            depth = np.linalg.norm(t)  # Approximate depth

            # 3D point in camera space
            pt_3d_camera = ray_camera * depth
            landmarks_3d.append(pt_3d_camera)

        return np.array(landmarks_3d)


    def _remove_head_pose(self, landmarks_2d, R, t, use_pnp_projection=None):
        """
        Transform MediaPipe landmarks to FLAME canonical space.

        Args:
            landmarks_2d: (N, 2) or (N, 3) array
                         If use_pnp=True: only X,Y used (ignore Z)
                         If use_pnp=False: all X,Y,Z used (legacy)
            R: (3, 3) rotation matrix from pose estimation
            t: (3,) translation vector from pose estimation
            use_pnp_projection: Override self.use_pnp (for testing)

        Returns:
            landmarks_canonical: (N, 3) in FLAME canonical coordinate frame
        """
        if use_pnp_projection is None:
            use_pnp_projection = self.use_pnp

        if use_pnp_projection:
            # NEW: PnP-based projection (uses only X,Y)
            # Extract only X,Y coordinates
            if landmarks_2d.shape[1] == 3:
                landmarks_2d = landmarks_2d[:, :2]  # Drop Z

            # Project 2D landmarks to 3D camera space
            landmarks_3d_camera = self._project_landmarks_to_3d(landmarks_2d, R, t)

            # Transform from camera space to FLAME canonical space
            # Camera space: landmarks_3d_camera = R @ landmarks_canonical + t
            # Solve for canonical: landmarks_canonical = R^T @ (landmarks_3d_camera - t)

            landmarks_centered = landmarks_3d_camera - t
            landmarks_canonical = landmarks_centered @ R.T  # Apply inverse rotation

            # Un-flip Y and Z to match FLAME canonical space
            # (We flipped them in pose_reference_3d to match MediaPipe convention)
            landmarks_canonical[:, 1] *= -1  # Un-flip Y
            landmarks_canonical[:, 2] *= -1  # Un-flip Z

        else:
            # LEGACY: Procrustes-based (uses X,Y,Z with scale)
            # This path is for backward compatibility when use_pnp=False
            landmarks_3d = landmarks_2d  # Assume 3D input
            scale = self.head_scale  # Use stored scale

            # Remove translation
            landmarks_centered = landmarks_3d - t

            # Remove rotation (apply inverse rotation R^T)
            landmarks_unrotated = landmarks_centered @ R.T

            # Remove scale
            landmarks_descaled = landmarks_unrotated / scale

            # Un-flip Y and Z to convert back to FLAME convention
            landmarks_canonical = landmarks_descaled.copy()
            landmarks_canonical[:, 1] *= -1  # Un-flip Y
            landmarks_canonical[:, 2] *= -1  # Un-flip Z

        # Sanity check
        if not np.isfinite(landmarks_canonical).all():
            logger.warning("[FLAME] NaNs detected in canonical landmarks! Using zeros.")
            landmarks_canonical = np.zeros_like(landmarks_canonical)

        return landmarks_canonical

    def _apply_head_pose(self, vertices, scale, R, t, render_scale=200.0, screen_center=(320, 240, 0)):
        """
        Apply head pose and render scaling to canonical FLAME vertices.

        Args:
            vertices: (N,3) FLAME vertices in canonical space
            scale: uniform scale from head pose estimation
            R: (3,3) rotation matrix from head pose estimation
            t: (3,) translation vector from head pose estimation
            render_scale: float, scale factor to map canonical units to screen pixels
            screen_center: (3,) tuple, offset to move mesh to screen center

        Returns:
            vertices_camera: (N,3) vertices in camera/screen space
        """

        screen_center = np.array(screen_center, dtype=np.float32)

        # --- Step 1: Flip FLAME coordinates to MediaPipe convention ---
        vertices_flipped = vertices.copy()
        vertices_flipped[:, 1] *= -1  # Y-up → Y-down
        vertices_flipped[:, 2] *= -1  # -Z → +Z
        # logger.info("vertices_flipped [first 5]:\n%s", vertices_flipped[:5])

        # --- Step 2: Apply head pose scale ---
        vertices_scaled = vertices_flipped * scale
        # logger.info("vertices_scaled [first 5]:\n%s", vertices_scaled[:5])
        # logger.info("Applied head scale: %.3f", scale)

        # --- Step 3: Apply rotation ---
        vertices_rotated = vertices_scaled @ R
        # logger.info("vertices_rotated [first 5]:\n%s", vertices_rotated[:5])
        # logger.info("Rotation matrix R:\n%s", R)
        # logger.info("Rotation determinant: %.6f", np.linalg.det(R))

        # --- Step 4: Apply translation ---
        vertices_translated = vertices_rotated + t
        # logger.info("vertices_translated [first 5]:\n%s", vertices_translated[:5])
        # logger.info("Translation vector t: %s", t.tolist())

        # --- Step 5: Apply render scale and screen center offset ---
        vertices_camera = vertices_translated * render_scale + screen_center
        # logger.info("vertices_camera (rendered) [first 5]:\n%s", vertices_camera[:5])
        # logger.info("Render scale: %.3f, Screen center: %s", render_scale, screen_center.tolist())

        # --- Step 6: Clamp to avoid extreme values ---
        vertices_camera = np.clip(vertices_camera, 0, max(screen_center)*2 + render_scale*2)

        return vertices_camera


    def _optimize_expression_only(self, target_landmarks_canonical):
        """
        STAGE 2: Optimize expression parameters only.
        Input landmarks are already in canonical space (head pose removed).

        Optimizes:
        - Expression blendshape weights
        - Jaw rotation (part of pose)
        """
        # Convert to tensors
        target = torch.tensor(target_landmarks_canonical, dtype=torch.float32, device=self.device)

        # Initialize parameters
        expr_params = torch.tensor(
            self.current_expression.copy(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )
        jaw_params = torch.tensor(
            self.current_jaw_pose.copy(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        # Optimizer
        optimizer = torch.optim.Adam([
            {'params': expr_params, 'lr': 0.05},
            {'params': jaw_params, 'lr': 0.02}
        ])

        # Optimization loop
        for _ in range(self.expression_iterations):
            optimizer.zero_grad()

            # Build pose vector with current jaw
            pose = torch.zeros(12, dtype=torch.float32, device=self.device)
            pose[6:9] = jaw_params  # Jaw rotation

            # Generate vertices
            vertices = self.flame.forward(
                shape_params=self.current_shape,
                expression_params=expr_params,
                pose_params=pose
            )

            # Extract landmarks
            pred_landmarks = vertices[self.flame_expr_indices]

            # Landmark loss (L2)
            landmark_loss = torch.mean((pred_landmarks - target) ** 2)

            # Regularization
            expr_reg = 0.001 * torch.sum(expr_params ** 2)
            jaw_reg = 0.01 * torch.sum(jaw_params ** 2)

            # Total loss
            loss = landmark_loss + expr_reg + jaw_reg

            # Backprop
            loss.backward()
            optimizer.step()

            # Clamp parameters
            with torch.no_grad():
                expr_params.data = torch.clamp(expr_params, -3.0, 3.0)
                jaw_params.data = torch.clamp(jaw_params, -0.5, 0.5)  # Jaw has limited range

        return expr_params.detach().cpu().numpy(), jaw_params.detach().cpu().numpy()

    def calibrate_shape(self, mediapipe_landmarks_3d, num_iterations=50):
        """
        Calibrate shape parameters for a specific person.
        Call this once at the start of a session with a neutral expression.
        """
        logger.info("[FLAME] Calibrating shape parameters...")

        # First get head pose
        pose_landmarks = mediapipe_landmarks_3d[self.pose_mp_indices]
        scale, R, t = self._estimate_head_pose(
            pose_landmarks,
            image_size=self.image_size,
            use_pnp=self.use_pnp
        )

        # Get expression landmarks in canonical space
        expr_landmarks_mp = mediapipe_landmarks_3d[self.mp_expr_indices]
        expr_landmarks_canonical = self._remove_head_pose(expr_landmarks_mp, R, t)

        # Convert to tensor
        target = torch.tensor(expr_landmarks_canonical, dtype=torch.float32, device=self.device)

        # Initialize shape parameters
        shape_params = torch.tensor(
            self.current_shape.copy(),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True
        )

        optimizer = torch.optim.Adam([shape_params], lr=0.01)

        for _ in range(num_iterations):
            optimizer.zero_grad()

            # Generate vertices (no expression, neutral pose)
            vertices = self.flame.forward(
                shape_params=shape_params,
                expression_params=np.zeros(self.flame.num_expressions),
                pose_params=np.zeros(12)
            )

            # Extract landmarks
            pred_landmarks = vertices[self.flame_expr_indices]

            # Loss
            loss = torch.mean((pred_landmarks - target) ** 2)
            loss += 0.001 * torch.sum(shape_params ** 2)  # Regularization

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                shape_params.data = torch.clamp(shape_params, -2.0, 2.0)

        self.current_shape = shape_params.detach().cpu().numpy()
        logger.info(f"[FLAME] Shape calibration complete. Loss: {loss.item():.6f}")

        return self.current_shape


def test_flame_fitter():
    """Test the two-stage FLAME fitter"""
    logger.info("Testing two-stage FLAME fitter...")

    try:
        from flame_model import get_flame_model

        logger.info("\n1. Loading FLAME model...")
        flame = get_flame_model(use_gpu=False)

        logger.info("\n2. Creating fitter...")
        fitter = FLAMEFitter(flame)

        logger.info("\n3. Generating synthetic landmarks...")
        # Create semi-realistic face landmarks
        landmarks = np.zeros((468, 3))
        # Distribute points roughly in face shape
        for i in range(468):
            angle = (i / 468) * 2 * np.pi
            r = 100 + 50 * np.sin(angle * 3)
            landmarks[i] = [
                320 + r * np.cos(angle),  # x: around 320 (center of 640 frame)
                240 + r * np.sin(angle) * 0.8,  # y: around 240 (center of 480 frame)
                50 * np.cos(angle * 2)  # z: some depth variation
            ]

        logger.info("\n4. Running two-stage fitting...")
        result = fitter.fit(landmarks, optimize_expression=True)

        logger.info(f"\n✓ Fitting successful!")
        logger.info(f"  - Vertices: {result['vertices'].shape}")
        logger.info(f"  - Faces: {result['faces'].shape}")
        logger.info(f"  - Head scale: {result['head_pose']['scale']:.3f}")
        logger.info(f"  - Fit time: {result['fit_time_ms']:.2f} ms")
        logger.info(f"  - Jaw pose: {result['jaw_pose']}")

        return True

    except Exception as e:
        logger.info(f"\n✗ Test failed: {e}")
        import traceback
        traceback.logger.info_exc()
        return False


if __name__ == '__main__':
    test_flame_fitter()
