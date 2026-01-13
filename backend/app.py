from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from fer.fer import FER
import base64
import time
from collections import deque, defaultdict
import threading
import os
import warnings
# adder logger
import logging

# Suppress macOS AVFoundation warnings for continuity camera
# This warning is harmless - it's just Apple deprecating the external camera type in favor of continuity cameras
os.environ.setdefault('OPENCV_AVFOUNDATION_SKIP_AUTH', '1')
warnings.filterwarnings('ignore', category=DeprecationWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import face mesh analyzer (optional feature)
FACE_MESH_AVAILABLE = False
face_mesh_analyzer = None
try:
    from face_mesh_analyzer import analyzer as face_mesh_analyzer
    FACE_MESH_AVAILABLE = True
    print("✓ Face mesh analyzer loaded successfully")
except (ImportError, AttributeError) as e:
    print(f"⚠️  Face mesh analyzer not available: {e}")
    print("   Continuing with FER-only emotion detection...")

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize emotion detector
emotion_detector = FER(mtcnn=False)

# Configuration
PROCESS_EVERY_N_FRAMES = 3
INFERENCE_WIDTH = 320
ROLLING_WINDOW_SIZE = 5
MIN_CONFIDENCE = 0.25

# Global state
camera = None
camera_lock = threading.Lock()
color_histories = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW_SIZE))

# Background thread state
camera_thread = None
camera_thread_stop_flag = threading.Event()

# Face mesh thread state
mesh_thread = None
mesh_thread_stop_flag = threading.Event()

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
    return (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))

def resize_for_inference(frame, target_width=INFERENCE_WIDTH):
    """Resize frame maintaining aspect ratio for faster processing"""
    height, width = frame.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    return cv2.resize(frame, (target_width, new_height)), scale

def initialize_camera(camera_index=0):
    """Initialize camera"""
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                time.sleep(0.2)  # Give camera time to release
            except Exception as e:
                logger.error(f"Error releasing camera in initialize: {e}")

        # Try multiple times to open camera
        for attempt in range(3):
            camera = cv2.VideoCapture(camera_index)
            if camera.isOpened():
                logger.info(f"✓ Camera {camera_index} opened on attempt {attempt + 1}")
                return True
            else:
                logger.warning(f"Attempt {attempt + 1} to open camera {camera_index} failed")
                time.sleep(0.3)
                if camera is not None:
                    camera.release()

        logger.error(f"✗ Failed to open camera {camera_index} after 3 attempts")
        return False

def camera_loop():
    """Background thread for camera processing with emotion detection"""
    global camera
    logger.info("camera_loop() started in background thread")

    frame_count = 0
    emotion_results = []
    fps_history = deque(maxlen=30)
    last_time = time.time()

    try:
        while not camera_thread_stop_flag.is_set():
            with camera_lock:
                if camera is None or not camera.isOpened():
                    logger.error("Camera is None or not opened, breaking from loop")
                    break

                success, frame = camera.read()
                if not success:
                    logger.error("Failed to read frame from camera")
                    break

            display_frame = frame.copy()

            # Process emotions every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                small_frame, scale = resize_for_inference(frame)
                emotion_results = emotion_detector.detect_emotions(small_frame)

                # Scale back coordinates
                if emotion_results:
                    for result in emotion_results:
                        box = result['box']
                        result['box'] = [
                            int(box[0] / scale),
                            int(box[1] / scale),
                            int(box[2] / scale),
                            int(box[3] / scale)
                        ]

            # Prepare emotion data for frontend
            emotions_data = []

            # Draw rectangles and labels
            for i, result in enumerate(emotion_results):
                try:
                    box = result['box']
                    emotions = result['emotions']
                    x, y, w, h = box

                    dominant_emotion, confidence = get_dominant_emotion(emotions)
                    raw_color = emotion_to_color(dominant_emotion)
                    face_id = f"face_{i}"
                    color = smooth_color(face_id, raw_color)

                    if confidence > MIN_CONFIDENCE:
                        # Draw rectangle
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)

                        # Draw emotion label
                        emotion_text = f"{dominant_emotion.upper()}: {confidence:.2f}"
                        label_y = max(y - 10, 20)
                        cv2.putText(display_frame, emotion_text, (x, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Traffic light indicator
                        circle_x = x + w - 20
                        circle_y = y + 20
                        cv2.circle(display_frame, (circle_x, circle_y), 15, color, -1)

                        # Collect emotion data for WebSocket
                        top_emotions = get_top_emotions(emotions, 3)
                        emotions_data.append({
                            'id': i,
                            'dominant': dominant_emotion,
                            'confidence': round(confidence, 2),
                            'top_emotions': [
                                {'emotion': e, 'score': round(s, 2)}
                                for e, s in top_emotions
                            ],
                            'color': f"rgb({color[2]}, {color[1]}, {color[0]})"  # Convert BGR to RGB
                        })
                except Exception as e:
                    print(f"Error processing face {i}: {e}")
                    continue

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            last_time = current_time

            # Draw FPS
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Faces: {len(emotion_results)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

            # Emit frame and emotion data via WebSocket
            socketio.emit('video_frame', {
                'frame': frame_base64,
                'emotions': emotions_data,
                'fps': round(avg_fps, 1),
                'face_count': len(emotion_results)
            })

            frame_count += 1

            # Log first frame
            if frame_count == 1:
                logger.info(f"✓ First frame emitted successfully (size: {len(frame_bytes)} bytes)")

            # Small sleep to prevent CPU overload
            time.sleep(0.01)

    except Exception as e:
        logger.error(f"Error in camera loop: {e}")
    finally:
        logger.info("camera_loop() exiting")
        with camera_lock:
            if camera is not None:
                try:
                    camera.release()
                    logger.info("Camera released in camera_loop()")
                except Exception as e:
                    logger.error(f"Error releasing camera in camera_loop: {e}")
                camera = None
        logger.info("camera_loop() cleanup complete")

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'face_mesh_available': FACE_MESH_AVAILABLE,
        'camera_thread_active': camera_thread.is_alive() if camera_thread else False,
        'mesh_thread_active': mesh_thread.is_alive() if mesh_thread else False
    })

@app.route('/api/cameras')
def list_cameras():
    """List available cameras"""
    logger.info("Scanning for cameras...")
    cameras = []

    for i in range(5):
        try:
            logger.info(f"Trying camera {i}...")
            cap = cv2.VideoCapture(i)

            # Set timeout for camera opening
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cameras.append({
                        'id': i,
                        'name': f'Camera {i}',
                        'resolution': f'{width}x{height}'
                    })
                    logger.info(f"✓ Camera {i} found: {width}x{height}")
                cap.release()
            else:
                logger.info(f"  Camera {i} not available")
        except Exception as e:
            logger.error(f"Error checking camera {i}: {e}")
            continue

    logger.info(f"Found {len(cameras)} camera(s)")
    return jsonify(cameras)

@app.route('/api/start/<int:camera_id>')
def start_camera(camera_id):
    """Start camera stream in background thread"""
    logger.info("=" * 50)
    logger.info("START CAMERA ENDPOINT CALLED")
    logger.info(f"Camera ID: {camera_id}")
    logger.info("=" * 50)

    global camera_thread

    # Stop existing thread if running
    if camera_thread and camera_thread.is_alive():
        logger.info("Existing camera thread found, stopping it first")
        camera_thread_stop_flag.set()
        camera_thread.join(timeout=2.0)

    # Clear stop flag and start fresh
    camera_thread_stop_flag.clear()

    if initialize_camera(camera_id):
        # Start background thread
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        logger.info(f"✓ Camera {camera_id} started in background thread")
        return jsonify({'status': 'success', 'message': f'Camera {camera_id} started'})

    logger.error(f"✗ Failed to open camera {camera_id}")
    return jsonify({'status': 'error', 'message': 'Failed to open camera'}), 400

@app.route('/api/stop')
def stop_camera():
    """Stop camera stream - lightweight, returns immediately"""
    logger.info("Stopping camera")

    # Set stop flag - thread will check it and exit
    camera_thread_stop_flag.set()

    # Return immediately without waiting
    logger.info("Stop flag set, returning immediately")
    return jsonify({'status': 'success', 'message': 'Camera stop initiated'})

# ===== FACE MESH ENDPOINTS =====

def face_mesh_loop():
    """Background thread for face mesh analysis"""
    global camera
    logger.info("face_mesh_loop() started in background thread")

    if not FACE_MESH_AVAILABLE:
        return

    frame_count = 0
    fps_history = deque(maxlen=30)
    last_time = time.time()

    try:
        while not mesh_thread_stop_flag.is_set():
            with camera_lock:
                if camera is None or not camera.isOpened():
                    logger.error("Camera is None or not opened in mesh loop")
                    break

                success, frame = camera.read()
                if not success:
                    logger.error("Failed to read frame in mesh loop")
                    break

            # Process with face mesh analyzer
            try:
                face_data = face_mesh_analyzer.process_frame(frame)
            except Exception as e:
                print(f"Face mesh error: {e}")
                face_data = None

            display_frame = frame.copy()

            # Draw face mesh overlays
            if face_data:
                for face in face_data:
                    landmarks_3d = np.array(face['landmarks_3d'])
                    features = face['geometry_features']
                    color = tuple(face['color'])

                    # Draw landmarks
                    for landmark in landmarks_3d:
                        x, y = int(landmark[0]), int(landmark[1])
                        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
                            cv2.circle(display_frame, (x, y), 1, color, -1)

                    # Draw feature text
                    text_y = 30
                    text_lines = [
                        f"Valence: {face['valence']:.2f} ({face['valence_zone']})",
                        f"Arousal: {face['arousal']:.2f}",
                        f"Smile: {features['smile_amplitude']:.2f}",
                        f"Eyes: {features['eye_openness']:.2f}",
                    ]

                    for line in text_lines:
                        cv2.putText(display_frame, line, (10, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        text_y += 25

            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            fps_history.append(fps)
            avg_fps = np.mean(fps_history)
            last_time = current_time

            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Encode frame
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')

            # Prepare mesh data
            mesh_data = []
            if face_data:
                for face in face_data:
                    mesh_data.append({
                        'face_id': face['face_id'],
                        'landmarks_3d': face['landmarks_3d'],
                        'geometry_features': face['geometry_features'],
                        'temporal_features': face['temporal_features'],
                        'valence': face['valence'],
                        'arousal': face['arousal'],
                        'valence_zone': face['valence_zone'],
                        'emotion_label': face.get('emotion_label', 'Unknown'),
                        'emotion_emoji': face.get('emotion_emoji', '😐'),
                        'zone_changed': face['zone_changed'],
                        'color': f"rgb({face['color'][2]}, {face['color'][1]}, {face['color'][0]})"
                    })

            # Emit frame and mesh data via WebSocket
            try:
                socketio.emit('face_mesh_frame', {
                    'frame': frame_base64,
                    'faces': mesh_data,
                    'fps': round(avg_fps, 1),
                    'face_count': len(face_data) if face_data else 0
                })
            except Exception as e:
                print(f"Error emitting face mesh data: {e}")

            frame_count += 1

            # Log first frame
            if frame_count == 1:
                logger.info(f"✓ First mesh frame emitted successfully")

            # Small sleep to prevent CPU overload
            time.sleep(0.01)

    except Exception as e:
        logger.error(f"Error in face_mesh_loop: {e}")
    finally:
        logger.info("face_mesh_loop() exiting")
        with camera_lock:
            if camera is not None:
                try:
                    camera.release()
                    logger.info("Camera released in face_mesh_loop()")
                except Exception as e:
                    logger.error(f"Error releasing camera in face_mesh_loop: {e}")
                camera = None
        logger.info("face_mesh_loop() cleanup complete")

@app.route('/api/face_mesh/start/<int:camera_id>')
def start_face_mesh(camera_id):
    """Start face mesh analysis in background thread"""
    logger.info("=" * 50)
    logger.info("START FACE MESH ENDPOINT CALLED")
    logger.info(f"Camera ID: {camera_id}")
    logger.info("=" * 50)

    global mesh_thread

    if not FACE_MESH_AVAILABLE:
        logger.error("✗ Face mesh not available - MediaPipe not installed")
        return jsonify({'status': 'error', 'message': 'Face mesh not available. Install mediapipe.'}), 400

    # Stop existing thread if running
    if mesh_thread and mesh_thread.is_alive():
        logger.info("Existing mesh thread found, stopping it first")
        mesh_thread_stop_flag.set()
        mesh_thread.join(timeout=2.0)

    # Clear stop flag and start fresh
    mesh_thread_stop_flag.clear()

    if initialize_camera(camera_id):
        # Start background thread
        mesh_thread = threading.Thread(target=face_mesh_loop, daemon=True)
        mesh_thread.start()
        logger.info(f"✓ Face mesh started in background thread on camera {camera_id}")
        return jsonify({'status': 'success', 'message': f'Face mesh started on camera {camera_id}'})

    logger.error(f"✗ Failed to open camera {camera_id}")
    return jsonify({'status': 'error', 'message': 'Failed to open camera'}), 400

@app.route('/api/face_mesh/stop')
def stop_face_mesh():
    """Stop face mesh analysis - lightweight, returns immediately"""
    logger.info("Stopping face mesh")

    # Set stop flag - thread will check it and exit
    mesh_thread_stop_flag.set()

    # Return immediately without waiting
    logger.info("Mesh stop flag set, returning immediately")
    return jsonify({'status': 'success', 'message': 'Face mesh stop initiated'})

@app.route('/api/face_mesh/available')
def face_mesh_available():
    """Check if face mesh is available"""
    return jsonify({'available': FACE_MESH_AVAILABLE})

# ===== SOCKETIO HANDLERS =====

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Emotion Traffic Light Backend...")
    print("Backend running on http://localhost:5001")
    print("=" * 60)
    print("\n📝 NOTE: If you see a warning about AVCaptureDeviceTypeExternal")
    print("   being deprecated for Continuity Cameras, you can safely ignore it.")
    print("   This is a harmless macOS system warning from the camera driver.\n")
    print("=" * 60)
    socketio.run(app, host='0.0.0.0', port=5001, debug=True, allow_unsafe_werkzeug=True)
