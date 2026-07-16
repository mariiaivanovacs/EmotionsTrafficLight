# map_mediapipe_to_flame.py
"""
Generate a mapping from MediaPipe FaceMesh landmarks (468 points) to FLAME
template vertex indices (e.g. 5023 vertices).

Usage (example):
    cd EmotionsTrafficLight
    python map_mediapipe_to_flame.py --mp backend/mp_neutral_frame.npy --out backend/mapping.json

Inputs:
- A single neutral-frame MediaPipe landmarks numpy file (shape (468,3)).
  You can dump this from your camera pipeline once (save np.array(landmarks)).
- FLAME model is loaded via your existing get_flame_model() function.
Outputs:
- JSON mapping file { "mp_index": flame_vertex_index, ... }
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Import FLAME model loader
try:
    from flame_model import get_flame_model
    print("✓ Successfully imported get_flame_model from backend/flame_model.py")
except Exception as e:
    print(f"✗ Failed to import get_flame_model: {e}")
    get_flame_model = None


def normalize_pointcloud(pc):
    """Center and scale a point cloud to unit RMS (safe for NN matching)."""
    pc = np.asarray(pc, dtype=np.float64)
    cen = pc.mean(axis=0)
    pc_centered = pc - cen
    norm = np.linalg.norm(pc_centered)
    if norm < 1e-8:
        return pc_centered, cen, 1.0
    scale = norm
    return pc_centered / scale, cen, scale


def prepare_mediapipe_points(mp_points, img_w=640, img_h=480, z_scale=100.0, from_normalized_coords=False):
    """
    Convert MediaPipe landmarks to 3D points suitable for mapping.

    If MediaPipe returns pixel coords (x,y in pixels), set from_normalized_coords=False.
    If MediaPipe returns normalized coords (0..1), set from_normalized_coords=True and pass image size.
    z_scale is used to bring the Z component to similar magnitude (tune per camera).
    """
    mp = np.asarray(mp_points, dtype=np.float64).copy()
    if from_normalized_coords:
        mp[:, 0] = mp[:, 0] * img_w
        mp[:, 1] = mp[:, 1] * img_h
        # z may already be in a small scale; treat as pixels-like
    # Convert to centered coordinates (origin at image center) so matching is translation-invariant
    mp[:, 0] = (mp[:, 0] - img_w / 2.0)
    mp[:, 1] = (mp[:, 1] - img_h / 2.0)
    # scale z to be on same order as x/y (pixel units). Tune z_scale if necessary.
    mp[:, 2] = mp[:, 2] * (img_w / z_scale)
    return mp


def map_mediapipe_to_flame(mp_landmarks,
                           flame_vertices,
                           flip_flame_yz=True,
                           img_w=640,
                           img_h=480,
                           from_normalized_coords=False,
                           verbose=True):
    """
    Returns:
      mapping: dict mp_index -> flame_vertex_index
      stats: dict with average distance etc.
    """
    # Prepare point clouds in roughly comparable coordinates
    mp_pts = prepare_mediapipe_points(mp_landmarks, img_w=img_w, img_h=img_h,
                                      z_scale=100.0, from_normalized_coords=from_normalized_coords)

    flame_pts = np.asarray(flame_vertices, dtype=np.float64).copy()  # (V,3)
    # Optionally flip FLAME Y/Z to MediaPipe convention (MediaPipe Y-down typically)
    if flip_flame_yz:
        flame_pts[:, 1] *= -1.0
        flame_pts[:, 2] *= -1.0

    # Center & normalize both clouds (so nearest neighbor compares shape, not scale)
    mp_norm, mp_cen, mp_scale = normalize_pointcloud(mp_pts)
    flame_norm, flame_cen, flame_scale = normalize_pointcloud(flame_pts)

    if verbose:
        print(f"[map] mp centroid={mp_cen}, mp_scale={mp_scale:.3f}")
        print(f"[map] flame centroid={flame_cen}, flame_scale={flame_scale:.3f}")

    # KD-tree on flame_norm
    kdt = cKDTree(flame_norm)

    # Query nearest flame vertex for each MP landmark
    dists, idxs = kdt.query(mp_norm, k=1)

    mapping = {int(mp_i): int(flame_i) for mp_i, flame_i in enumerate(idxs)}

    stats = {
        "mean_distance": float(dists.mean()),
        "median_distance": float(np.median(dists)),
        "max_distance": float(dists.max()),
    }

    if verbose:
        print(f"[map] distances (normalized units) mean={stats['mean_distance']:.6f} "
              f"median={stats['median_distance']:.6f} max={stats['max_distance']:.6f}")

    return mapping, stats


def save_mapping(mapping, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[map] Saved mapping to: {out_path}")


def main(args):
    print("=" * 60)
    print("MediaPipe to FLAME Mapping Generator")
    print("=" * 60)

    # Load MP landmarks (npy file expected shape (468,3))
    mp_path = Path(args.mp)
    print(f"\n1. Loading MediaPipe landmarks from: {mp_path}")

    if not mp_path.exists():
        raise FileNotFoundError(f"MediaPipe landmarks file not found: {mp_path}")

    mp_landmarks = np.load(mp_path)  # shape (468,3)
    print(f"   ✓ Loaded landmarks shape: {mp_landmarks.shape}")
    
    mp_landmarks = mp_landmarks[:468, :]  # Ignore iris landmarks


    if mp_landmarks.shape != (468, 3):
        raise ValueError(f"Expected shape (468, 3), got {mp_landmarks.shape}")

    # Load FLAME model
    print("\n2. Loading FLAME model...")
    if get_flame_model is None:
        raise RuntimeError("Could not import get_flame_model(). Ensure backend/flame_model.py is accessible.")

    try:
        flame = get_flame_model(use_gpu=False)
        print(f"   ✓ FLAME model loaded successfully")
        print(f"   - Device: {flame.device}")
        print(f"   - Template vertices: {flame.v_template.shape}")

        flame_vertices = flame.v_template.cpu().numpy()
        print(f"   ✓ Extracted template vertices: {flame_vertices.shape}")
    except Exception as e:
        raise RuntimeError(f"Failed to load FLAME model: {e}")

    # Generate mapping
    print("\n3. Generating MediaPipe → FLAME mapping...")
    print(f"   - Image size: {args.img_w}x{args.img_h}")
    print(f"   - Flip FLAME Y/Z: {not args.no_flip}")
    print(f"   - From normalized coords: {args.from_normalized}")

    mapping, stats = map_mediapipe_to_flame(
        mp_landmarks,
        flame_vertices,
        flip_flame_yz=not args.no_flip,
        img_w=args.img_w,
        img_h=args.img_h,
        from_normalized_coords=args.from_normalized
    )

    # Save mapping
    if args.out:
        print(f"\n4. Saving mapping to: {args.out}")
        save_mapping(mapping, args.out)

    # Print statistics
    print("\n" + "=" * 60)
    print("MAPPING STATISTICS")
    print("=" * 60)
    print(f"Total MediaPipe landmarks: {len(mapping)}")
    print(f"Mean distance (normalized): {stats['mean_distance']:.6f}")
    print(f"Median distance (normalized): {stats['median_distance']:.6f}")
    print(f"Max distance (normalized): {stats['max_distance']:.6f}")

    # Print example correspondences
    print("\nExample correspondences:")
    print("-" * 40)
    for i in range(0, 468, 47):
        print(f"  MediaPipe {i:3d} → FLAME vertex {mapping[i]:4d}")

    print("\n" + "=" * 60)
    print("✓ Mapping generation complete!")
    print("=" * 60)

    return mapping, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mp", required=True, help="Path to MediaPipe landmarks .npy file (shape (468,3))")
    parser.add_argument("--out", default="mapping.json", help="Output JSON mapping file")
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--img-h", type=int, default=480)
    parser.add_argument("--from-normalized", action="store_true",
                        help="If MediaPipe landmarks are 0..1 normalized coords, set this flag")
    parser.add_argument("--no-flip", action="store_true", help="Do not flip FLAME Y/Z before mapping")
    args = parser.parse_args()
    main(args)
