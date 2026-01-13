#!/usr/bin/env python3
"""
Test script to verify FER import and basic functionality
without requiring camera access.
"""

import sys
import numpy as np

def test_fer_import():
    """Test that FER can be imported successfully"""
    print("Testing FER import...")
    try:
        from fer.fer import FER
        print("✓ FER imported successfully from fer.fer")
        return True
    except ImportError as e:
        print(f"❌ Failed to import FER: {e}")
        return False

def test_fer_initialization():
    """Test that FER can be initialized"""
    print("\nTesting FER initialization...")
    try:
        from fer.fer import FER
        detector = FER(mtcnn=False)
        print("✓ FER detector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize FER: {e}")
        return False

def test_emotion_detection_with_dummy_image():
    """Test emotion detection with a dummy image"""
    print("\nTesting emotion detection with dummy image...")
    try:
        from fer.fer import FER
        import cv2
        
        # Create a dummy image (black image)
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        detector = FER(mtcnn=False)
        result = detector.detect_emotions(dummy_image)
        
        print(f"✓ Emotion detection completed. Result: {result}")
        print("  (Empty result is expected for a black image)")
        return True
    except Exception as e:
        print(f"❌ Failed emotion detection test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_traffic_light_imports():
    """Test that emotion_traffic_light.py can be imported"""
    print("\nTesting emotion_traffic_light.py imports...")
    try:
        # Add the current directory to path
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try importing the module (this will execute module-level code)
        # We'll use importlib to avoid executing the main() function
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "emotion_traffic_light", 
            os.path.join(os.path.dirname(__file__), "emotion_traffic_light.py")
        )
        module = importlib.util.module_from_spec(spec)
        
        # This will execute the module but not the if __name__ == "__main__" block
        spec.loader.exec_module(module)
        
        print("✓ emotion_traffic_light.py imported successfully")
        print("✓ All module-level code executed without errors")
        return True
    except Exception as e:
        print(f"❌ Failed to import emotion_traffic_light.py: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_helper_functions():
    """Test helper functions from emotion_traffic_light.py"""
    print("\nTesting helper functions...")
    try:
        import importlib.util
        import os
        
        spec = importlib.util.spec_from_file_location(
            "emotion_traffic_light", 
            os.path.join(os.path.dirname(__file__), "emotion_traffic_light.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Test emotion_to_color function
        green = module.emotion_to_color('happy')
        yellow = module.emotion_to_color('neutral')
        red = module.emotion_to_color('sad')
        
        assert green == (0, 255, 0), "Happy should be green"
        assert yellow == (0, 255, 255), "Neutral should be yellow"
        assert red == (0, 0, 255), "Sad should be red"
        
        print("✓ emotion_to_color() works correctly")
        
        # Test get_dominant_emotion function
        emotions = {'happy': 0.8, 'sad': 0.1, 'neutral': 0.1}
        emotion, score = module.get_dominant_emotion(emotions)
        assert emotion == 'happy', "Dominant emotion should be happy"
        assert score == 0.8, "Score should be 0.8"
        
        print("✓ get_dominant_emotion() works correctly")
        
        # Test get_top_emotions function
        top_emotions = module.get_top_emotions(emotions, 2)
        assert len(top_emotions) == 2, "Should return 2 emotions"
        assert top_emotions[0][0] == 'happy', "First should be happy"
        
        print("✓ get_top_emotions() works correctly")
        
        return True
    except Exception as e:
        print(f"❌ Helper function tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("FER Import and Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_fer_import,
        test_fer_initialization,
        test_emotion_detection_with_dummy_image,
        test_emotion_traffic_light_imports,
        test_helper_functions,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! The emotion_traffic_light.py file is ready to use.")
        print("\nTo run the application:")
        print("  1. Make sure your camera is connected")
        print("  2. Grant camera permissions to your terminal/IDE")
        print("  3. Run: python emotion_traffic_light.py")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

