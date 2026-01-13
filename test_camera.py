import cv2

def test_camera():
    """Quick smoke test for webcam connection"""
    print("Testing webcam connection...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam")
        print("Possible issues:")
        print("  - Camera is being used by another application")
        print("  - Camera permissions not granted")
        print("  - No camera connected")
        return False

    print("✓ Webcam connected successfully!")

    # Try to read a frame
    ret, frame = cap.read()

    if not ret:
        print("❌ ERROR: Cannot read frame from webcam")
        cap.release()
        return False

    print(f"✓ Frame captured successfully! Resolution: {frame.shape[1]}x{frame.shape[0]}")

    # Test Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if face_cascade.empty():
        print("❌ ERROR: Cannot load Haar Cascade")
        cap.release()
        return False

    print("✓ Haar Cascade loaded successfully!")

    cap.release()
    print("\n✓ All tests passed! Camera is ready to use.")
    return True

if __name__ == "__main__":
    test_camera()
