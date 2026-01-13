#!/usr/bin/env python3
"""
Download MediaPipe Face Landmarker model
"""

import urllib.request
import os
import sys

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
MODEL_PATH = "face_landmarker.task"

def download_model():
    """Download the face landmarker model if it doesn't exist"""
    
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists: {MODEL_PATH}")
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        print(f"  Size: {file_size:.2f} MB")
        return True
    
    print(f"Downloading Face Landmarker model...")
    print(f"URL: {MODEL_URL}")
    print(f"Destination: {MODEL_PATH}")
    print()
    
    try:
        def progress_hook(count, block_size, total_size):
            """Show download progress"""
            percent = int(count * block_size * 100 / total_size)
            mb_downloaded = count * block_size / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rProgress: {percent}% ({mb_downloaded:.2f} / {mb_total:.2f} MB)")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, progress_hook)
        print()
        print(f"✓ Model downloaded successfully!")
        
        file_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)  # MB
        print(f"  Size: {file_size:.2f} MB")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)

