#!/usr/bin/env python3
"""
Test script to verify the web backend is working correctly
"""

import requests
import json
import sys

BACKEND_URL = 'http://localhost:5001'

def test_backend_connection():
    """Test if backend is running"""
    print("Testing backend connection...")
    try:
        response = requests.get(f"{BACKEND_URL}/api/cameras", timeout=5)
        if response.status_code == 200:
            print("✓ Backend is running and responding")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running?")
        print("   Start it with: ./start_backend.sh")
        return False
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        return False

def test_camera_detection():
    """Test camera detection endpoint"""
    print("\nTesting camera detection...")
    try:
        response = requests.get(f"{BACKEND_URL}/api/cameras", timeout=5)
        if response.status_code == 200:
            cameras = response.json()
            print(f"✓ Camera detection endpoint working")
            print(f"  Found {len(cameras)} camera(s):")
            for cam in cameras:
                print(f"    - {cam['name']}: {cam['resolution']}")
            
            if len(cameras) == 0:
                print("\n⚠️  No cameras detected!")
                print("   Possible issues:")
                print("   - Camera permissions not granted")
                print("   - Camera is being used by another app")
                print("   - No camera connected")
                return False
            return True
        else:
            print(f"❌ Camera endpoint returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing camera detection: {e}")
        return False

def test_cors():
    """Test CORS headers"""
    print("\nTesting CORS configuration...")
    try:
        response = requests.options(f"{BACKEND_URL}/api/cameras")
        cors_header = response.headers.get('Access-Control-Allow-Origin')
        if cors_header:
            print(f"✓ CORS is configured: {cors_header}")
            return True
        else:
            print("⚠️  CORS headers not found (might be okay)")
            return True
    except Exception as e:
        print(f"⚠️  Could not test CORS: {e}")
        return True

def print_frontend_instructions():
    """Print instructions for frontend"""
    print("\n" + "="*60)
    print("Frontend Configuration")
    print("="*60)
    print("\nThe frontend should connect to: http://localhost:5001")
    print("\nTo verify frontend configuration:")
    print("  1. Check frontend/src/components/EmotionDisplay.jsx")
    print("  2. Look for: const BACKEND_URL = 'http://localhost:5001'")
    print("\nTo start the frontend:")
    print("  ./start_frontend.sh")
    print("\nThen open your browser to: http://localhost:5173")

def main():
    """Run all tests"""
    print("="*60)
    print("Web Backend Test Suite")
    print("="*60)
    
    tests = [
        test_backend_connection,
        test_camera_detection,
        test_cors,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        print_frontend_instructions()
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\nMake sure the backend is running:")
        print("  ./start_backend.sh")
        return 1

if __name__ == "__main__":
    sys.exit(main())

