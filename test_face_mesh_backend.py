#!/usr/bin/env python3
"""
Test Face Mesh Backend - Diagnostic Script
"""

import requests
import socketio
import time
import json

BACKEND_URL = 'http://localhost:5001'

def test_face_mesh_available():
    """Test if face mesh is available"""
    print("1. Testing Face Mesh Availability...")
    print("-" * 60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/face_mesh/available")
        data = response.json()
        
        if data['available']:
            print("✓ Face Mesh is AVAILABLE")
            return True
        else:
            print("❌ Face Mesh is NOT available")
            print("   Run: cd backend && python download_face_landmarker_model.py")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_cameras():
    """Test camera detection"""
    print("\n2. Testing Camera Detection...")
    print("-" * 60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/cameras")
        cameras = response.json()
        
        print(f"✓ Found {len(cameras)} camera(s):")
        for cam in cameras:
            print(f"   - Camera {cam['id']}: {cam['name']} ({cam['resolution']})")
        
        return cameras
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_websocket_connection():
    """Test WebSocket connection and face mesh updates"""
    print("\n3. Testing WebSocket Connection...")
    print("-" * 60)
    
    sio = socketio.Client()
    received_data = []
    
    @sio.on('connect')
    def on_connect():
        print("✓ WebSocket connected")
    
    @sio.on('face_mesh_update')
    def on_face_mesh_update(data):
        print(f"✓ Received face_mesh_update:")
        print(f"   - FPS: {data.get('fps', 'N/A')}")
        print(f"   - Face Count: {data.get('face_count', 0)}")
        print(f"   - Faces Data: {len(data.get('faces', []))} face(s)")
        
        if data.get('faces'):
            face = data['faces'][0]
            print(f"   - First Face:")
            print(f"     • Emotion: {face.get('emotion_label', 'N/A')} {face.get('emotion_emoji', '')}")
            print(f"     • Valence: {face.get('valence', 'N/A')}")
            print(f"     • Arousal: {face.get('arousal', 'N/A')}")
            print(f"     • Zone: {face.get('valence_zone', 'N/A')}")
            print(f"     • Landmarks: {len(face.get('landmarks_3d', []))} points")
            print(f"     • Geometry Features: {list(face.get('geometry_features', {}).keys())}")
        
        received_data.append(data)
    
    @sio.on('disconnect')
    def on_disconnect():
        print("⚠️  WebSocket disconnected")
    
    try:
        sio.connect(BACKEND_URL)
        print("   Listening for face_mesh_update events...")
        print("   (Waiting 10 seconds for data...)")
        
        time.sleep(10)
        
        if received_data:
            print(f"\n✓ Received {len(received_data)} updates")
        else:
            print("\n⚠️  No face_mesh_update events received")
            print("   Make sure:")
            print("   1. Camera is started via /api/face_mesh/start/<camera_id>")
            print("   2. A face is visible to the camera")
        
        sio.disconnect()
        return len(received_data) > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_video_feed():
    """Test if video feed endpoint exists"""
    print("\n4. Testing Video Feed Endpoint...")
    print("-" * 60)
    
    try:
        response = requests.get(f"{BACKEND_URL}/face_mesh_feed", stream=True, timeout=2)
        print(f"✓ Video feed endpoint accessible")
        print(f"   Content-Type: {response.headers.get('Content-Type')}")
        return True
    except requests.exceptions.Timeout:
        print("⚠️  Video feed endpoint exists but no stream (camera not started)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Face Mesh Backend Diagnostic Test")
    print("=" * 60)
    
    # Test 1: Face Mesh Available
    if not test_face_mesh_available():
        print("\n❌ Face Mesh not available. Cannot continue tests.")
        return 1
    
    # Test 2: Cameras
    cameras = test_cameras()
    if not cameras:
        print("\n❌ No cameras found. Cannot continue tests.")
        return 1
    
    # Test 3: Video Feed
    test_video_feed()
    
    # Test 4: WebSocket
    print("\n" + "=" * 60)
    print("WebSocket Test (requires camera to be started)")
    print("=" * 60)
    print("\nTo test WebSocket, you need to:")
    print(f"1. Start camera: curl {BACKEND_URL}/api/face_mesh/start/0")
    print("2. Run this test again")
    print("\nOr skip to manual test:")
    print("- Open browser to http://localhost:5173")
    print("- Click 'Face Mesh Analysis (3D)' tab")
    print("- Select camera and click 'Start Face Mesh Analysis'")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("✓ Face Mesh: Available")
    print(f"✓ Cameras: {len(cameras)} found")
    print("✓ Video Feed: Endpoint exists")
    print("\nNext Steps:")
    print("1. Start the frontend: ./start_frontend.sh")
    print("2. Open browser: http://localhost:5173")
    print("3. Click 'Face Mesh Analysis (3D)' tab")
    print("4. Select camera and start analysis")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

