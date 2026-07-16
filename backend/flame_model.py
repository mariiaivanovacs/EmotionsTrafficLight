"""
FLAME Model Wrapper
Loads and interfaces with the FLAME (Faces Learned with an Articulated Model and Expressions)
parametric 3D face model.

FLAME Parameters:
- Shape (β): ~100-300 parameters controlling face identity
- Expression (ψ): ~50-100 parameters for facial expressions
- Pose (θ): Global rotation (3) + jaw (3) + eyes (6) = 12 total
"""

import numpy as np
import torch
import pickle
import os
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Go up one level to project root

# Check multiple possible locations for FLAME model
MODEL_PATHS = [
    SCRIPT_DIR / 'models' / 'flame' / 'generic_model.pkl',  # backend/models/flame/
    PROJECT_ROOT / 'models' / 'flame' / 'generic_model.pkl',  # models/flame/ (project root)
]

# Find first existing model path
MODEL_PATH = None
for path in MODEL_PATHS:
    if path.exists():
        MODEL_PATH = path
        break

if MODEL_PATH is None:
    MODEL_PATH = MODEL_PATHS[0]  # Default to first path for error message


class FLAMEModel:
    """FLAME parametric face model wrapper"""

    def __init__(self, model_path=None, use_gpu=False):
        """
        Initialize FLAME model

        Args:
            model_path: Path to generic_model.pkl (default: backend/models/flame/generic_model.pkl)
            use_gpu: Use CUDA GPU if available
        """
        if model_path is None:
            model_path = MODEL_PATH

        self.model_path = Path(model_path)
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

        # Check if model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"FLAME model not found at: {self.model_path}\n"
                f"Please download generic_model.pkl from https://flame.is.tue.mpg.de/\n"
                f"and place it in: {SCRIPT_DIR / 'models' / 'flame' / 'generic_model.pkl'}"
            )

        # Load FLAME model
        print(f"Loading FLAME model from: {self.model_path}")
        self._load_model()
        print(f"✓ FLAME model loaded successfully")
        print(f"  - Vertices: {self.v_template.shape[0]}")
        print(f"  - Faces: {self.faces.shape[0]}")
        print(f"  - Shape parameters: {self.num_betas}")
        print(f"  - Expression parameters: {self.num_expressions}")
        print(f"  - Device: {self.device}")
        
        
    # def get_template_vertices(self):
    #     """
    #     Return the FLAME canonical (neutral) mesh vertices.

    #     Returns:
    #         np.ndarray of shape (V,3) in meters
    #     """
    #     return np.asarray(self.v_template)
    
    def get_template_vertices(self):
        """
        Return the FLAME canonical (neutral) mesh vertices, rotated so face is upright.

        Returns:
            np.ndarray of shape (V,3) in meters
            
        """
        
        V = self.rotate_y_90_clockwise()
        # V = np.asarray(self.v_template)  # (V,3)

        # ---------- Define rotations ----------
        # 180 deg clockwise about X-axis
        Rx = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])

        # 180 deg about Y-axis
        Ry = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ])

        # Total rotation: first Rx, then Ry
        R_total = Ry @ Rx  # (3,3)

        # Apply rotation to all vertices
        V_rotated = (R_total @ V.T).T  # (V,3)
        
        

        return V_rotated
    
    def rotate_y_90_clockwise(self):
        V = np.asarray(self.v_template)  # (5023,3)
        
        Ry_90 = np.array([
            [0, 0, -1],
            [0, 1,  0],
            [1, 0,  0]
        ])
        
        V_rotated = (Ry_90 @ V.T).T  # (5023,3)
        return V_rotated



    def _load_model(self):
        """Load FLAME model from pickle file"""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f, encoding='latin1')

        # Extract core components
        # Template mesh (neutral face shape)
        self.v_template = torch.tensor(
            np.array(model_data['v_template']),
            dtype=torch.float32,
            device=self.device
        )  # Shape: (5023, 3)

        # Faces (triangle connectivity)
        self.faces = torch.tensor(
            np.array(model_data['f']),
            dtype=torch.long,
            device=self.device
        )  # Shape: (~9976, 3)

        # Shape blend shapes (identity variations)
        self.shapedirs = torch.tensor(
            np.array(model_data['shapedirs']),
            dtype=torch.float32,
            device=self.device
        )  # Shape: (5023, 3, num_betas)

        # Expression blend shapes (facial expressions)
        self.exprdirs = torch.tensor(
            np.array(model_data['posedirs']),
            dtype=torch.float32,
            device=self.device
        )  # Note: 'posedirs' often contains expression blendshapes in FLAME

        # Joint regressor and kinematic tree (for pose)
        self.J_regressor = torch.tensor(
            np.array(model_data['J_regressor'].toarray()),
            dtype=torch.float32,
            device=self.device
        ) if hasattr(model_data['J_regressor'], 'toarray') else torch.tensor(
            np.array(model_data['J_regressor']),
            dtype=torch.float32,
            device=self.device
        )

        # Skinning weights (for pose deformation)
        self.weights = torch.tensor(
            np.array(model_data['weights']),
            dtype=torch.float32,
            device=self.device
        )

        # Get dimensions
        self.num_vertices = self.v_template.shape[0]
        self.num_faces = self.faces.shape[0]
        self.num_betas = self.shapedirs.shape[2]  # Shape parameters

        # Expression parameters - check multiple possible keys
        if 'posedirs' in model_data:
            expr_data = model_data['posedirs']
            if len(expr_data.shape) == 3:
                self.num_expressions = expr_data.shape[2]
            else:
                # Reshape if necessary
                expr_flat = expr_data.reshape(self.num_vertices, 3, -1)
                self.num_expressions = expr_flat.shape[2]
                self.exprdirs = torch.tensor(expr_flat, dtype=torch.float32, device=self.device)
        else:
            # Fallback to smaller number if not found
            self.num_expressions = 50
            self.exprdirs = torch.zeros(
                (self.num_vertices, 3, self.num_expressions),
                dtype=torch.float32,
                device=self.device
            )

        # Store model data for reference
        self.model_data = model_data

    def forward(self, shape_params=None, expression_params=None, pose_params=None):
        """
        Generate FLAME mesh vertices from parameters

        Args:
            shape_params: Shape coefficients β (num_betas,) - controls identity
            expression_params: Expression coefficients ψ (num_expressions,) - controls facial expressions
            pose_params: Pose parameters θ (12,) - global rotation + jaw + eyes

        Returns:
            vertices: Generated mesh vertices (5023, 3)
        """
        # Default to zero parameters if not provided
        if shape_params is None:
            shape_params = torch.zeros(self.num_betas, dtype=torch.float32, device=self.device)
        else:
            shape_params = torch.tensor(shape_params, dtype=torch.float32, device=self.device)

        if expression_params is None:
            expression_params = torch.zeros(self.num_expressions, dtype=torch.float32, device=self.device)
        else:
            expression_params = torch.tensor(expression_params, dtype=torch.float32, device=self.device)

        if pose_params is None:
            pose_params = torch.zeros(12, dtype=torch.float32, device=self.device)
        else:
            pose_params = torch.tensor(pose_params, dtype=torch.float32, device=self.device)

        # Start with template
        vertices = self.v_template.clone()

        # Add shape blend shapes
        # shapedirs: (V, 3, num_betas)
        # shape_params: (num_betas,)
        # Result: (V, 3)
        shape_blend = torch.einsum('vck,k->vc', self.shapedirs, shape_params)
        vertices = vertices + shape_blend

        # Add expression blend shapes
        # exprdirs: (V, 3, num_expressions)
        # expression_params: (num_expressions,)
        if self.exprdirs.shape[2] >= len(expression_params):
            expr_blend = torch.einsum('vck,k->vc', self.exprdirs[:, :, :len(expression_params)], expression_params)
            vertices = vertices + expr_blend

        # TODO: Add pose-dependent deformations (jaw, eyes, global rotation)
        # For now, we skip pose deformation for simplicity
        # This can be added later using Linear Blend Skinning (LBS)

        return vertices

    def get_vertices_numpy(self, shape_params=None, expression_params=None, pose_params=None):
        """
        Get vertices as numpy array

        Returns:
            vertices: (5023, 3) numpy array
        """
        vertices = self.forward(shape_params, expression_params, pose_params)
        return vertices.cpu().numpy()

    def get_faces_numpy(self):
        """
        Get face indices as numpy array

        Returns:
            faces: (N, 3) numpy array of triangle vertex indices
        """
        return self.faces.cpu().numpy()

    def compute_normals(self, vertices):
        """
        Compute per-vertex normals for lighting

        Args:
            vertices: (V, 3) tensor or numpy array

        Returns:
            normals: (V, 3) numpy array of unit normals
        """
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)

        # Get faces
        faces = self.faces

        # Get triangle vertices
        v0 = vertices[faces[:, 0]]  # (F, 3)
        v1 = vertices[faces[:, 1]]  # (F, 3)
        v2 = vertices[faces[:, 2]]  # (F, 3)

        # Compute face normals
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = torch.cross(edge1, edge2, dim=1)  # (F, 3)

        # Normalize face normals
        face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)

        # Accumulate face normals to vertices
        vertex_normals = torch.zeros_like(vertices)
        for i in range(3):
            vertex_normals.index_add_(0, faces[:, i], face_normals)

        # Normalize vertex normals
        vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)

        return vertex_normals.cpu().numpy()

    def compute_centroid(self, vertices):
        """
        Compute mesh centroid

        Args:
            vertices: (V, 3) tensor or numpy array

        Returns:
            centroid: (3,) numpy array
        """
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)

        centroid = torch.mean(vertices, dim=0)
        return centroid.cpu().numpy()

    def center_mesh(self, vertices):
        """
        Center mesh at origin

        Args:
            vertices: (V, 3) numpy array or tensor

        Returns:
            centered_vertices: (V, 3) numpy array
            centroid: (3,) numpy array (original center)
        """
        centroid = self.compute_centroid(vertices)

        if isinstance(vertices, np.ndarray):
            centered = vertices - centroid
        else:
            centered = vertices - torch.tensor(centroid, device=vertices.device)
            centered = centered.cpu().numpy()

        return centered, centroid

    def get_landmark_indices(self):
        """
        Get vertex indices corresponding to facial landmarks

        Returns a subset of FLAME vertex indices that correspond to
        common facial landmarks (eyes, nose, mouth, etc.)

        These are approximate and may need refinement based on your
        specific FLAME model version.

        Returns:
            landmark_indices: dict of {landmark_name: vertex_index}
        """
        # These are approximate FLAME vertex indices for key landmarks
        # You may need to adjust these based on your specific FLAME model
        landmarks = {
            # Eyes
            'left_eye_outer': 2536,
            'left_eye_inner': 1463,
            'right_eye_outer': 4207,
            'right_eye_inner': 3293,

            # Eyebrows
            'left_eyebrow_outer': 2212,
            'left_eyebrow_inner': 1984,
            'right_eyebrow_outer': 3885,
            'right_eyebrow_inner': 3657,

            # Nose
            'nose_tip': 3509,
            'nose_bridge': 3508,
            'nose_left': 2794,
            'nose_right': 3543,

            # Mouth
            'mouth_left': 2760,
            'mouth_right': 3508,
            'mouth_top': 2797,
            'mouth_bottom': 2842,
            'upper_lip_top': 2797,
            'lower_lip_bottom': 2842,

            # Face outline
            'chin': 2848,
            'left_cheek': 2199,
            'right_cheek': 3872,
        }

        return landmarks


# Global instance (lazy loaded)
_flame_model_instance = None


def get_flame_model(use_gpu=False):
    """
    Get global FLAME model instance (singleton pattern)

    Args:
        use_gpu: Use CUDA GPU if available

    Returns:
        FLAMEModel instance
    """
    global _flame_model_instance

    if _flame_model_instance is None:
        _flame_model_instance = FLAMEModel(use_gpu=use_gpu)

    return _flame_model_instance


def test_flame_model():
    """Test FLAME model loading and basic functionality"""
    print("Testing FLAME model...")

    try:
        # Load model
        flame = get_flame_model(use_gpu=False)

        # Test with default parameters (neutral face)
        print("\n1. Testing neutral face generation...")
        vertices = flame.get_vertices_numpy()
        print(f"   Generated {vertices.shape[0]} vertices")

        # Test with random shape parameters
        print("\n2. Testing with random shape parameters...")
        shape_params = np.random.randn(flame.num_betas) * 0.5
        vertices_shaped = flame.get_vertices_numpy(shape_params=shape_params)
        print(f"   Shape variation magnitude: {np.linalg.norm(vertices_shaped - vertices):.4f}")

        # Test with expression parameters (smile)
        print("\n3. Testing with expression parameters (simulated smile)...")
        expr_params = np.zeros(flame.num_expressions)
        expr_params[0] = 2.0  # First expression component
        vertices_expr = flame.get_vertices_numpy(expression_params=expr_params)
        print(f"   Expression variation magnitude: {np.linalg.norm(vertices_expr - vertices):.4f}")

        # Test normals
        print("\n4. Testing normal computation...")
        normals = flame.compute_normals(vertices)
        print(f"   Computed {normals.shape[0]} normals")
        print(f"   Normal magnitude range: [{np.linalg.norm(normals, axis=1).min():.4f}, {np.linalg.norm(normals, axis=1).max():.4f}]")

        # Test centroid
        print("\n5. Testing centroid computation...")
        centroid = flame.compute_centroid(vertices)
        print(f"   Centroid: [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")

        # Test centering
        print("\n6. Testing mesh centering...")
        centered, orig_centroid = flame.center_mesh(vertices)
        new_centroid = flame.compute_centroid(centered)
        print(f"   Original centroid: [{orig_centroid[0]:.4f}, {orig_centroid[1]:.4f}, {orig_centroid[2]:.4f}]")
        print(f"   New centroid: [{new_centroid[0]:.4f}, {new_centroid[1]:.4f}, {new_centroid[2]:.4f}]")

        print("\n✓ All tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # Run tests when script is executed directly
    test_flame_model()
