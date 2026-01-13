import cv2
import numpy as np
from fer.fer import FER
import pyttsx3
from collections import deque, defaultdict
import threading
import time

# Initialize
emotion_detector = FER(mtcnn=False)

# TTS in separate thread to avoid blocking
tts_queue = []
tts_lock = threading.Lock()

def tts_worker():
    """Background thread for text-to-speech"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    while True:
        with tts_lock:
            if tts_queue:
                text = tts_queue.pop(0)
                engine.say(text)
                engine.runAndWait()
        time.sleep(0.1)

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Configuration
PROCESS_EVERY_N_FRAMES = 3  # Process emotions every 3 frames for real-time feel
INFERENCE_WIDTH = 320       # Lower resolution for faster inference
ROLLING_WINDOW_SIZE = 5     # Smaller window for more responsive colors
MIN_CONFIDENCE = 0.25       # Minimum confidence to display

# Face tracking - stores emotion data per face
face_emotions = {}
face_id_counter = 0
color_histories = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW_SIZE))

def emotion_to_color(emotion):
    """Map emotion to traffic light color"""
    positive_emotions = ['happy', 'surprise']
    neutral_emotions = ['neutral']

    if emotion in positive_emotions:
        return (0, 255, 0)  # Green
    elif emotion in neutral_emotions:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

def get_dominant_emotion(emotions):
    """Get the emotion with highest confidence"""
    if not emotions:
        return 'neutral', 0.0
    max_emotion = max(emotions.items(), key=lambda x: x[1])
    return max_emotion[0], max_emotion[1]

def get_top_emotions(emotions, top_n=3):
    """Get top N emotions sorted by confidence"""
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions[:top_n]

def smooth_color(face_id, new_color):
    """Apply rolling average to stabilize color per face"""
    color_histories[face_id].append(new_color)
    avg_color = np.mean(color_histories[face_id], axis=0)
    # Convert to tuple of Python ints (not numpy ints) for OpenCV
    return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

def resize_for_inference(frame, target_width=INFERENCE_WIDTH):
    """Resize frame maintaining aspect ratio for faster processing"""
    height, width = frame.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    return cv2.resize(frame, (target_width, new_height)), scale

def match_face_to_detected(x, y, w, h, detected_faces, tolerance=50):
    """Match Haar cascade face to FER detected face"""
    for i, det_face in enumerate(detected_faces):
        box = det_face['box']
        dx, dy, dw, dh = box

        # Check if boxes overlap significantly
        if (abs(x - dx) < tolerance and abs(y - dy) < tolerance):
            return i
    return None

def speak_emotion(emotion):
    """Add emotion to TTS queue"""
    with tts_lock:
        tts_queue.append(emotion)

def list_cameras():
    """List available cameras"""
    print("\nScanning for available cameras...")
    available_cameras = []

    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                # Try to get camera name (if available)
                backend = cap.getBackendName()
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  [{i}] Camera {i} - {width}x{height} ({backend})")
            cap.release()

    return available_cameras

def select_camera():
    """Allow user to select camera"""
    cameras = list_cameras()

    if not cameras:
        print("\n❌ No cameras found!")
        return None

    if len(cameras) == 1:
        print(f"\n✓ Using camera {cameras[0]}")
        return cameras[0]

    print(f"\nFound {len(cameras)} cameras")
    print("Tip: Camera 0 is usually built-in webcam, higher numbers might be iPhone/external cameras")

    while True:
        try:
            choice = input(f"\nSelect camera [0-{max(cameras)}] (or press Enter for camera 0): ").strip()
            if choice == "":
                return 0
            choice = int(choice)
            if choice in cameras:
                return choice
            else:
                print(f"Invalid choice. Please select from: {cameras}")
        except ValueError:
            print("Please enter a number")

def main():
    # Select camera
    print("=" * 60)
    print("EMOTION TRAFFIC LIGHT - Camera Setup")
    print("=" * 60)

    camera_index = select_camera()
    if camera_index is None:
        return

    print(f"\n✓ Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print("✓ Camera connected successfully!")
    print("Configuration:")
    print(f"  - Inference resolution: {INFERENCE_WIDTH}px width")
    print(f"  - Processing every {PROCESS_EVERY_N_FRAMES} frames")
    print(f"  - Rolling window: {ROLLING_WINDOW_SIZE} frames")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to toggle TTS\n")

    frame_count = 0
    emotion_results = []
    last_spoken = {}
    tts_enabled = True
    fps_history = deque(maxlen=30)
    last_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("ERROR: Failed to grab frame")
            break

        display_frame = frame.copy()

        # Process emotions every N frames using lower resolution
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            # Resize for faster inference
            small_frame, scale = resize_for_inference(frame)

            # Detect emotions on small frame
            emotion_results = emotion_detector.detect_emotions(small_frame)

            # Scale back coordinates for display
            if emotion_results:
                for result in emotion_results:
                    box = result['box']
                    result['box'] = [
                        int(box[0] / scale),
                        int(box[1] / scale),
                        int(box[2] / scale),
                        int(box[3] / scale)
                    ]

        # Draw rectangles and labels for each detected face
        for i, result in enumerate(emotion_results):
            try:
                box = result['box']
                emotions = result['emotions']

                x, y, w, h = box

                # Get dominant emotion and color
                dominant_emotion, confidence = get_dominant_emotion(emotions)
                raw_color = emotion_to_color(dominant_emotion)
                face_id = f"face_{i}"
                color = smooth_color(face_id, raw_color)

                # Ensure color is a valid tuple
                if not isinstance(color, tuple) or len(color) != 3:
                    print(f"Warning: Invalid color {color}, using default")
                    color = (0, 255, 255)  # Yellow as fallback

                # Only display if confidence is above threshold
                if confidence > MIN_CONFIDENCE:
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)

                    # Prepare emotion text with score
                    emotion_text = f"{dominant_emotion.upper()}: {confidence:.2f}"

                    # Get top 3 emotions for detailed view
                    top_emotions = get_top_emotions(emotions, 3)

                    # Draw main emotion label above face
                    label_y = max(y - 10, 20)
                    cv2.putText(display_frame, emotion_text, (x, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Draw top emotions on the side of face
                    detail_x = x + w + 10
                    detail_y = y + 20

                    for idx, (emotion, score) in enumerate(top_emotions):
                        detail_text = f"{emotion}: {score:.2f}"
                        cv2.putText(display_frame, detail_text,
                                   (detail_x, detail_y + idx * 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # Traffic light indicator for this face
                    circle_x = x + w - 20
                    circle_y = y + 20
                    cv2.circle(display_frame, (circle_x, circle_y), 15, color, -1)

                    # Speak emotion if changed (non-blocking)
                    face_key = f"face_{i}"
                    if (tts_enabled and
                        confidence > 0.4 and
                        last_spoken.get(face_key) != dominant_emotion):
                        speak_emotion(dominant_emotion)
                        last_spoken[face_key] = dominant_emotion

            except Exception as e:
                print(f"Error processing face {i}: {e}")
                continue

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        last_time = current_time

        # Display info panel
        info_y = 30
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display_frame, f"Faces: {len(emotion_results)}", (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        tts_status = "ON" if tts_enabled else "OFF"
        cv2.putText(display_frame, f"TTS: {tts_status}", (10, info_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Emotion Traffic Light - Real-time Detection', display_frame)

        frame_count += 1

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            tts_enabled = not tts_enabled
            print(f"TTS {'enabled' if tts_enabled else 'disabled'}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Camera released successfully")
    print(f"Average FPS: {avg_fps:.1f}")

if __name__ == "__main__":
    main()
