#!/usr/bin/env python3
"""
Comprehensive test suite for emotion detection system
Tests all core functions and common error cases
"""

import cv2
import numpy as np
from fer.fer import FER
import sys

def test_color_functions():
    """Test color-related functions"""
    print("\n" + "="*60)
    print("TEST 1: Color Functions")
    print("="*60)

    # Import the functions from main file
    sys.path.insert(0, '/Users/mariaivanova/Desktop/Side_projects/EmotionsTrafficLight')

    from emotion_traffic_light import emotion_to_color, smooth_color, get_dominant_emotion

    # Test emotion_to_color
    print("\n1.1 Testing emotion_to_color...")
    test_emotions = ['happy', 'sad', 'angry', 'neutral', 'surprise', 'fear']
    for emotion in test_emotions:
        color = emotion_to_color(emotion)
        print(f"  {emotion:10} → {color}")
        assert isinstance(color, tuple), f"Color should be tuple, got {type(color)}"
        assert len(color) == 3, f"Color should have 3 values, got {len(color)}"
        assert all(isinstance(c, int) for c in color), "Color values should be integers"
        assert all(0 <= c <= 255 for c in color), "Color values should be 0-255"

    print("  ✓ emotion_to_color working correctly")

    # Test smooth_color
    print("\n1.2 Testing smooth_color...")
    test_color = (0, 255, 0)
    face_id = "test_face"

    # Add multiple colors and check smoothing
    for i in range(10):
        smoothed = smooth_color(face_id, test_color)
        print(f"  Iteration {i}: {smoothed}")
        assert isinstance(smoothed, tuple), f"Smoothed color should be tuple, got {type(smoothed)}"
        assert len(smoothed) == 3, f"Smoothed color should have 3 values"
        assert all(isinstance(c, int) for c in smoothed), "Smoothed color values should be Python ints"

    print("  ✓ smooth_color working correctly")

    # Test get_dominant_emotion
    print("\n1.3 Testing get_dominant_emotion...")
    emotions = {
        'happy': 0.7,
        'sad': 0.1,
        'angry': 0.05,
        'neutral': 0.15
    }
    dominant, confidence = get_dominant_emotion(emotions)
    print(f"  Dominant: {dominant} (confidence: {confidence})")
    assert dominant == 'happy', f"Expected 'happy', got '{dominant}'"
    assert confidence == 0.7, f"Expected 0.7, got {confidence}"

    print("  ✓ get_dominant_emotion working correctly")

    print("\n✓ ALL COLOR FUNCTION TESTS PASSED")
    return True

def test_camera_access():
    """Test camera access and basic capture"""
    print("\n" + "="*60)
    print("TEST 2: Camera Access")
    print("="*60)

    print("\n2.1 Testing camera enumeration...")
    available_cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  ✓ Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()

    if not available_cameras:
        print("  ❌ No cameras found!")
        return False

    print(f"\n  ✓ Found {len(available_cameras)} camera(s)")

    # Test capture from first available camera
    print("\n2.2 Testing frame capture...")
    cap = cv2.VideoCapture(available_cameras[0])

    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"  ❌ Failed to capture frame {i}")
            cap.release()
            return False
        print(f"  Frame {i}: {frame.shape} - dtype: {frame.dtype}")

    cap.release()
    print("  ✓ Frame capture working correctly")

    print("\n✓ ALL CAMERA TESTS PASSED")
    return True

def test_fer_detection():
    """Test FER emotion detector"""
    print("\n" + "="*60)
    print("TEST 3: FER Emotion Detection")
    print("="*60)

    print("\n3.1 Initializing FER detector...")
    try:
        detector = FER(mtcnn=False)
        print("  ✓ FER initialized successfully")
    except Exception as e:
        print(f"  ❌ Failed to initialize FER: {e}")
        return False

    print("\n3.2 Testing with dummy frame...")
    # Create a test frame (random noise)
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    try:
        result = detector.detect_emotions(test_frame)
        print(f"  Detected {len(result)} faces in random noise (expected 0)")
        assert isinstance(result, list), "Result should be a list"
    except Exception as e:
        print(f"  ❌ Error during detection: {e}")
        return False

    print("\n3.3 Testing with real camera frame...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  ⚠ Cannot open camera for FER test")
        return True  # Skip but don't fail

    ret, frame = cap.read()
    cap.release()

    if ret:
        try:
            result = detector.detect_emotions(frame)
            print(f"  Detected {len(result)} face(s)")

            if result:
                for i, face in enumerate(result):
                    print(f"    Face {i}:")
                    print(f"      Box: {face['box']}")
                    emotions = face['emotions']
                    top_emotion = max(emotions.items(), key=lambda x: x[1])
                    print(f"      Top emotion: {top_emotion[0]} ({top_emotion[1]:.2f})")

        except Exception as e:
            print(f"  ❌ Error during FER detection: {e}")
            return False

    print("\n✓ ALL FER TESTS PASSED")
    return True

def test_opencv_drawing():
    """Test OpenCV drawing functions with color tuples"""
    print("\n" + "="*60)
    print("TEST 4: OpenCV Drawing Functions")
    print("="*60)

    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    print("\n4.1 Testing cv2.rectangle with different color types...")

    # Test with Python tuple (correct)
    try:
        color_tuple = (0, 255, 0)
        cv2.rectangle(test_frame, (10, 10), (100, 100), color_tuple, 3)
        print(f"  ✓ Python tuple {color_tuple}: SUCCESS")
    except Exception as e:
        print(f"  ❌ Python tuple failed: {e}")
        return False

    # Test with numpy array (should fail or be converted)
    try:
        color_array = np.array([0, 255, 0], dtype=int)
        cv2.rectangle(test_frame, (120, 10), (210, 100), color_array, 3)
        print(f"  ⚠ Numpy array {color_array}: Worked but not recommended")
    except Exception as e:
        print(f"  ✓ Numpy array correctly rejected: {e}")

    # Test with explicit Python ints
    try:
        color_ints = (int(0), int(255), int(0))
        cv2.rectangle(test_frame, (230, 10), (320, 100), color_ints, 3)
        print(f"  ✓ Explicit ints {color_ints}: SUCCESS")
    except Exception as e:
        print(f"  ❌ Explicit ints failed: {e}")
        return False

    print("\n4.2 Testing cv2.putText...")
    try:
        cv2.putText(test_frame, "TEST", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print("  ✓ cv2.putText working correctly")
    except Exception as e:
        print(f"  ❌ cv2.putText failed: {e}")
        return False

    print("\n4.3 Testing cv2.circle...")
    try:
        cv2.circle(test_frame, (50, 200), 20, (0, 0, 255), -1)
        print("  ✓ cv2.circle working correctly")
    except Exception as e:
        print(f"  ❌ cv2.circle failed: {e}")
        return False

    print("\n✓ ALL OPENCV DRAWING TESTS PASSED")
    return True

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("EMOTION DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*60)

    results = {}

    # Run tests
    results['color_functions'] = test_color_functions()
    results['camera_access'] = test_camera_access()
    results['fer_detection'] = test_fer_detection()
    results['opencv_drawing'] = test_opencv_drawing()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {test_name:20} {status}")

    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please fix errors before using.")
        return 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
