#!/usr/bin/env python3
"""
Performance test for emotion detection system
Tests different resolutions and frame skip settings
"""

import cv2
import time
from fer import FER
import numpy as np

def test_performance():
    """Test emotion detection performance at different settings"""

    print("=" * 60)
    print("EMOTION DETECTION PERFORMANCE TEST")
    print("=" * 60)

    # Initialize
    print("\n1. Initializing FER detector...")
    detector = FER(mtcnn=False)
    print("   ✓ FER initialized (Haar Cascade mode)")

    # Test camera
    print("\n2. Testing camera connection...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("   ❌ ERROR: Cannot open webcam")
        print("\n   Please check:")
        print("      - Camera permissions (System Preferences → Security → Camera)")
        print("      - Camera not in use by another app")
        return

    print("   ✓ Camera opened successfully")

    # Get original resolution
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"   ✓ Original resolution: {orig_width}x{orig_height}")

    # Test configurations
    configs = [
        {"name": "Full Res (Every Frame)", "width": orig_width, "skip": 1},
        {"name": "Full Res (Every 3rd)", "width": orig_width, "skip": 3},
        {"name": "640px (Every Frame)", "width": 640, "skip": 1},
        {"name": "640px (Every 3rd)", "width": 640, "skip": 3},
        {"name": "320px (Every Frame)", "width": 320, "skip": 1},
        {"name": "320px (Every 3rd)", "width": 320, "skip": 3},
    ]

    print("\n3. Testing different configurations...")
    print("-" * 60)

    results = []

    for config in configs:
        width = config["width"]
        skip = config["skip"]

        frame_count = 0
        process_count = 0
        total_time = 0
        test_frames = 30  # Test with 30 frames

        print(f"\n   Testing: {config['name']} ({width}px, skip={skip})")

        start_time = time.time()

        while frame_count < test_frames:
            ret, frame = cap.read()

            if not ret:
                break

            # Resize if needed
            if width != orig_width:
                height = int(frame.shape[0] * (width / frame.shape[1]))
                frame = cv2.resize(frame, (width, height))

            # Process every Nth frame
            if frame_count % skip == 0:
                process_start = time.time()
                emotions = detector.detect_emotions(frame)
                process_time = time.time() - process_start
                total_time += process_time
                process_count += 1

            frame_count += 1

        elapsed = time.time() - start_time
        avg_fps = test_frames / elapsed if elapsed > 0 else 0
        avg_process_time = (total_time / process_count * 1000) if process_count > 0 else 0

        result = {
            "config": config["name"],
            "fps": avg_fps,
            "process_ms": avg_process_time,
            "processes": process_count
        }
        results.append(result)

        print(f"      FPS: {avg_fps:.1f}")
        print(f"      Avg processing time: {avg_process_time:.0f}ms")
        print(f"      Frames processed: {process_count}/{test_frames}")

    cap.release()

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Configuration':<30} {'FPS':<10} {'Process Time':<15}")
    print("-" * 60)

    for result in results:
        print(f"{result['config']:<30} {result['fps']:>6.1f}     {result['process_ms']:>8.0f}ms")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    best = max(results, key=lambda x: x['fps'])
    print(f"\n✓ Best performance: {best['config']}")
    print(f"  - FPS: {best['fps']:.1f}")
    print(f"  - Processing time: {best['process_ms']:.0f}ms")

    print("\n✓ Recommended for real-time: 320px with frame skip 3")
    print("  - Good balance of speed and accuracy")
    print("  - Suitable for interactive applications")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_performance()
