#!/usr/bin/env python3
"""
Quick diagnostic test for face mesh analyzer
"""
import os
import sys

print("="*60)
print("FACE MESH DIAGNOSTIC TEST")
print("="*60)

# Check working directory
print(f"\n1. Current directory: {os.getcwd()}")

# Check if model file exists
model_path = 'face_landmarker.task'
backend_model = 'backend/face_landmarker.task'

print(f"\n2. Model file check:")
print(f"   - {model_path}: {os.path.exists(model_path)}")
print(f"   - {backend_model}: {os.path.exists(backend_model)}")

if os.path.exists(model_path):
    size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✓ Model found: {size:.1f} MB")
elif os.path.exists(backend_model):
    size = os.path.getsize(backend_model) / (1024 * 1024)
    print(f"   ✓ Model found in backend/: {size:.1f} MB")
else:
    print(f"   ❌ Model NOT found!")

# Try importing mediapipe
print(f"\n3. MediaPipe import:")
try:
    import mediapipe as mp
    print(f"   ✓ MediaPipe version: {mp.__version__}")
except ImportError as e:
    print(f"   ❌ MediaPipe not installed: {e}")
    sys.exit(1)

# Try importing face mesh analyzer
print(f"\n4. Face mesh analyzer import:")
try:
    from face_mesh_analyzer import analyzer
    print(f"   ✓ Face mesh analyzer imported successfully")
    print(f"   ✓ Analyzer type: {type(analyzer)}")
except Exception as e:
    print(f"   ❌ Error importing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with a dummy frame
print(f"\n5. Testing with dummy frame:")
try:
    import numpy as np
    import cv2

    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = analyzer.process_frame(test_frame)

    if result is None:
        print(f"   ✓ No face detected (expected for black frame)")
    else:
        print(f"   ✓ Analyzer working! Detected {len(result)} face(s)")

except Exception as e:
    print(f"   ❌ Error processing frame: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")
print("="*60)
