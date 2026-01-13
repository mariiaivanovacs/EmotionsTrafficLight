"""
Face Mesh endpoints - Add these to app.py or run as separate module
"""

from flask import Response
from flask_socketio import emit
import cv2
import numpy as np
import time
from collections import deque

# Global state for face mesh mode
is_mesh_streaming = False
mesh_camera = None
mesh_camera_lock = None

def draw_face_mesh_on_frame(frame, face_data):
    """Draw face mesh landmarks and features on frame"""
    if not face_data:
        return frame

    display_frame = frame.copy()

    for face in face_data:
        landmarks_3d = np.array(face['landmarks_3d'])
        features = face['geometry_features']
        color = tuple(face['color'])

        # Draw landmarks as points
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
            f"Mouth Open: {features['mouth_openness']:.2f}",
            f"Eye Open: {features['eye_openness']:.2f}",
            f"Eyebrow: {features['eyebrow_raise']:.2f}",
        ]

        for line in text_lines:
            cv2.putText(display_frame, line, (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            text_y += 25

    return display_frame

def generate_face_mesh_frames(camera, camera_lock, socketio):
    """Generate frames with face mesh analysis"""
    global is_mesh_streaming

    frame_count = 0
    fps_history = deque(maxlen=30)
    last_time = time.time()

    while is_mesh_streaming:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break

            success, frame = camera.read()
            if not success:
                break

        # Process with face mesh analyzer
        try:
            if FACE_MESH_AVAILABLE:
                face_data = face_mesh_analyzer.process_frame(frame)
            else:
                face_data = None
        except Exception as e:
            print(f"Face mesh error: {e}")
            face_data = None

        # Draw overlays
        display_frame = draw_face_mesh_on_frame(frame, face_data)

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        last_time = current_time

        # Draw FPS
        cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Emit data via WebSocket
        if face_data and frame_count % 2 == 0:  # Send every other frame
            # Prepare data for frontend
            mesh_data = []
            for face in face_data:
                mesh_data.append({
                    'face_id': face['face_id'],
                    'landmarks_3d': face['landmarks_3d'],
                    'geometry_features': face['geometry_features'],
                    'temporal_features': face['temporal_features'],
                    'valence': face['valence'],
                    'arousal': face['arousal'],
                    'valence_zone': face['valence_zone'],
                    'zone_changed': face['zone_changed'],
                    'color': f"rgb({face['color'][2]}, {face['color'][1]}, {face['color'][0]})"
                })

            socketio.emit('face_mesh_update', {
                'faces': mesh_data,
                'fps': round(avg_fps, 1),
                'face_count': len(face_data)
            })

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        frame_count += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Add these routes to your main app.py:

# @app.route('/api/face_mesh/start/<int:camera_id>')
# def start_face_mesh(camera_id):
#     """Start face mesh analysis"""
#     global is_mesh_streaming, mesh_camera, mesh_camera_lock
#
#     if not FACE_MESH_AVAILABLE:
#         return jsonify({'status': 'error', 'message': 'Face mesh not available. Install mediapipe.'}), 400
#
#     # Initialize camera
#     with camera_lock:
#         if mesh_camera is not None:
#             mesh_camera.release()
#         mesh_camera = cv2.VideoCapture(camera_id)
#         mesh_camera_lock = camera_lock
#
#         if mesh_camera.isOpened():
#             is_mesh_streaming = True
#             return jsonify({'status': 'success', 'message': f'Face mesh started on camera {camera_id}'})
#
#     return jsonify({'status': 'error', 'message': 'Failed to open camera'}), 400
#
# @app.route('/api/face_mesh/stop')
# def stop_face_mesh():
#     """Stop face mesh analysis"""
#     global is_mesh_streaming, mesh_camera
#
#     is_mesh_streaming = False
#     with camera_lock:
#         if mesh_camera is not None:
#             mesh_camera.release()
#             mesh_camera = None
#
#     return jsonify({'status': 'success', 'message': 'Face mesh stopped'})
#
# @app.route('/face_mesh_feed')
# def face_mesh_feed():
#     """Face mesh video streaming route"""
#     return Response(generate_face_mesh_frames(mesh_camera, mesh_camera_lock, socketio),
#                    mimetype='multipart/x-mixed-replace; boundary=frame')
