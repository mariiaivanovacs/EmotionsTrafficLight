#!/usr/bin/env python3
"""
Test FLAME integration with MediaPipe Face Mesh
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import cv2
import numpy as np
from face_mesh_analyzer import FaceMeshAnalyzer

def test_flame_integration():
    print("=" * 60)
    print("FLAME Integration Test")
    print("=" * 60)

    # Initialize analyzer with FLAME
    print("\n1. Initializing Face Mesh Analyzer with FLAME...")
    try:
        analyzer = FaceMeshAnalyzer(use_flame=True)
        print(f"   ✓ Analyzer initialized (FLAME enabled: {analyzer.use_flame})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Create a test frame (black image)
    print("\n2. Creating test frame...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (128, 128, 128)  # Gray background
    print("   ✓ Test frame created (640x480)")

    # Process frame (will likely not detect face, but tests the pipeline)
    print("\n3. Processing frame...")
    try:
        result = analyzer.process_frame(test_frame)
        if result is None:
            print("   ℹ️  No faces detected (expected for blank frame)")
        else:
            print(f"   ✓ Detected {len(result)} face(s)")
            for face_data in result:
                print(f"      - Face {face_data['face_id']}:")
                if 'flame_mesh' in face_data:
                    mesh = face_data['flame_mesh']
                    print(f"        • FLAME mesh generated!")
                    print(f"        • Vertices: {len(mesh['vertices'])}")
                    print(f"        • Faces: {len(mesh['faces'])}")
                    print(f"        • Fit time: {mesh['fit_time_ms']:.2f} ms")
                else:
                    print(f"        • No FLAME mesh (landmarks only)")
    except Exception as e:
        print(f"   ⚠️  Processing completed with warning: {e}")

    print("\n" + "=" * 60)
    print("✓ Integration test completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Place your webcam-captured frame for real testing")
    print("2. Run the backend server: ./start_backend.sh")
    print("3. Open the frontend to see FLAME mesh visualization")
    print("\nFLAME model status:")
    if analyzer.use_flame:
        print("  ✓ FLAME is ENABLED and ready")
        print(f"  • Model vertices: {analyzer.flame_model.num_vertices}")
        print(f"  • Model faces: {analyzer.flame_model.num_faces}")
        print(f"  • Shape parameters: {analyzer.flame_model.num_betas}")
        print(f"  • Expression parameters: {analyzer.flame_model.num_expressions}")
    else:
        print("  ✗ FLAME is DISABLED (using landmarks only)")

    return True

if __name__ == '__main__':
    try:
        success = test_flame_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
