from flask import Flask, render_template, request, jsonify, redirect, session, url_for, Response
from flask_socketio import SocketIO, emit
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase
import os
import logging
import threading
from threading import Lock
import time
from time import time
import uuid
import re
import cv2
import gc
import numpy as np
import traceback
import mediapipe as mp
import math
from queue import Queue
from collections import deque
import torch
from ultralytics import YOLO
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)  # Generate a random secret key
socketio = SocketIO(app, cors_allowed_origins="*")

try:
    # Firebase Configuration
    cred = credentials.Certificate("examguard/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    firebase_config = {
        "apiKey": "AIzaSyAK8aMoxmT7hpbxl8akPYitfxFM6Ytl76I",
        "authDomain": "examguard-login.firebaseapp.com",
        "databaseURL": "https://firestore.googleapis.com/v1/projects/examguard-login/databases/(default)/documents/",
        "projectId": "examguard-login",
        "storageBucket": "examguard-login.firebasestorage.app",
        "messagingSenderId": "229046359229",
        "appId": "1:229046359229:web:5f67beec4538e13b36d3d4"
    }

    firebase = pyrebase.initialize_app(firebase_config)
    auth_firebase = firebase.auth()
    logger.info("Firebase initialized successfully")

except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    db = None
    raise

# Store active camera streams with thread-safe dictionary
active_streams = {}
stream_lock = threading.Lock()

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Constants
HEAD_TURN_THRESHOLD = 40
HEAD_TURN_RESET_THRESHOLD = 10
MAX_HEAD_TURNS = 5
WARNING_DURATION = 30
SMOOTHING_WINDOW_SIZE = 5

# Color Constants (BGR Format)
COLORS = {
    'YOLO_BOX': (255, 0, 0),        # Red
    'HEAD_WARNING': (0, 0, 255),    # Red
    'HAND_WARNING': (0, 0, 255),    # Red
    'SAFE': (0, 255, 0),            # Green
    'TEXT': (255, 255, 255),        # White
    'BACKGROUND': (0, 0, 0),        # Black
    'FPS': (0, 255, 0),             # Green
    'LANDMARK_POSE': (66, 117, 245),  # Coral
    'LANDMARK_HAND_LEFT': (76, 22, 121),  # Purple
    'LANDMARK_HAND_RIGHT': (10, 22, 80)   # Dark Blue
}

class HeadTurnTracker:
    def __init__(self, person_id):
        self.person_id = person_id
        self.turn_count = 0
        self.last_angle = None
        self.turning = False
        self.warning_start_time = None
        self.turns_history = deque(maxlen=100)
        self.angle_history = deque(maxlen=SMOOTHING_WINDOW_SIZE)
        self.steady_center_frames = 0
        self.FRAMES_FOR_STABLE = 3
        self.calibrated = False
        self.initial_position = None
        self.last_seen = time.time()
        
    def get_smoothed_angle(self, new_angle):
        self.angle_history.append(new_angle)
        return np.mean(self.angle_history) if len(self.angle_history) >= 3 else new_angle
        
    def update(self, raw_angle):
        self.last_seen = time.time()
        current_time = time.time()
        
        if raw_angle is None:
            return False, 0
            
        if not self.calibrated and len(self.angle_history) >= self.FRAMES_FOR_STABLE:
            self.initial_position = self.get_smoothed_angle(raw_angle)
            self.calibrated = True
            
        relative_angle = raw_angle - (self.initial_position if self.calibrated else 0)
        smoothed_angle = self.get_smoothed_angle(relative_angle)
        
        if self.last_angle is not None:
            if abs(smoothed_angle) < HEAD_TURN_RESET_THRESHOLD:
                self.steady_center_frames += 1
            else:
                self.steady_center_frames = 0
                
            if (abs(smoothed_angle) > HEAD_TURN_THRESHOLD and 
                not self.turning and 
                abs(smoothed_angle - self.last_angle) < 15):
                
                self.turning = True
                self.turn_count += 1
                self.turns_history.append(current_time)
                
                if self.turn_count >= MAX_HEAD_TURNS and self.warning_start_time is None:
                    self.warning_start_time = current_time
            
            elif self.steady_center_frames >= self.FRAMES_FOR_STABLE:
                self.turning = False
                self.steady_center_frames = 0
        
        self.last_angle = smoothed_angle
        
        if (self.warning_start_time is not None and 
            current_time - self.warning_start_time >= WARNING_DURATION):
            self.warning_start_time = None
            self.turn_count = 0
            self.turns_history.clear()
        
        is_warning = self.warning_start_time is not None or self.turn_count >= MAX_HEAD_TURNS
        return is_warning, smoothed_angle

class CameraStream:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.thread = None
        self.frame = None
        self.running = threading.Event()
        self.frame_lock = threading.Lock()
        self.video_capture = None
        
        # Initialize MediaPipe components
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        logger.info(f"CameraStream initialized for camera {camera_id}")

    def _initialize_capture(self):
        try:
            # Try to open the camera directly first
            cap = cv2.VideoCapture(0)  # Use 0 explicitly for default camera
            
            if not cap.isOpened():
                # If direct access fails, try DSHOW backend
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if cap.isOpened():
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Test reading a frame
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    logger.error("Failed to read test frame from camera")
                    cap.release()
                    return None
                    
                logger.info("Successfully initialized camera")
                return cap
            else:
                logger.error("Failed to open camera")
                return None

        except Exception as e:
            logger.error(f"Error initializing capture: {str(e)}")
            return None

    def _update_frame(self):
        try:
            self.video_capture = self._initialize_capture()
            if self.video_capture is None:
                logger.error("Failed to initialize video capture")
                return

            while self.running.is_set():
                try:
                    ret, frame = self.video_capture.read()
                    if not ret or frame is None:
                        logger.warning("Failed to read frame")
                        time.sleep(0.1)
                        continue

                    # Store the frame immediately
                    with self.frame_lock:
                        self.frame = frame

                except Exception as e:
                    logger.error(f"Error in _update_frame: {str(e)}")
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in _update_frame main loop: {str(e)}")
        finally:
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None

    def get_frame(self):
        try:
            with self.frame_lock:
                if self.frame is None:
                    return None
                
                # Create a copy of the frame to avoid modification issues
                frame_copy = self.frame.copy()
                
                try:
                    # Process frame with MediaPipe
                    frame_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(frame_rgb)

                    # Draw landmarks
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            frame_copy, 
                            results.pose_landmarks, 
                            mp_holistic.POSE_CONNECTIONS
                        )
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                
                # Encode frame
                ret, jpeg = cv2.imencode('.jpg', frame_copy)
                if not ret:
                    return None
                    
                return b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
                
        except Exception as e:
            logger.error(f"Error in get_frame: {str(e)}")
            return None

    def start(self):
        try:
            if not self.running.is_set():
                self.running.set()
                self.thread = threading.Thread(target=self._update_frame)
                self.thread.daemon = True
                self.thread.start()
                logger.info(f"Started camera stream thread for camera {self.camera_id}")
        except Exception as e:
            logger.error(f"Error starting camera stream: {str(e)}")
            raise

    def stop(self):
        logger.info(f"Stopping stream for camera {self.camera_id}")
        self.running.clear()
        if self.thread:
            self.thread.join()
        if self.video_capture:
            self.video_capture.release()
        self.video_capture = None
        logger.info(f"Stream stopped for camera {self.camera_id}")

    def __del__(self):
        self.stop()

def get_connected_cameras():
    """Test and get all available cameras with detailed information"""
    camera_info = []
    max_cameras_to_check = 10

    for i in range(max_cameras_to_check):
        try:
            logger.info(f"Testing camera index {i}")
            # Try different backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(i + backend)
                
                if cap.isOpened():
                    # Get camera properties
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    backend_name = cap.getBackendName()
                    
                    # Try to read a frame
                    ret, frame = cap.read()
                    if ret:
                        camera_info.append({
                            "index": i,
                            "name": f"Camera {i} ({width}x{height})",
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "backend": backend_name
                        })
                        logger.info(f"Successfully detected camera {i}")
                        break  # Break if camera is successfully detected
                    
                cap.release()
                
        except Exception as e:
            logger.error(f"Error testing camera {i}: {str(e)}")
    
    return camera_info

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    # Check if user is logged in
    if "user" not in session:
        logger.warning("Unauthorized access attempt to dashboard - redirecting to login")
        return redirect(url_for('login'))
    
    try:
        # Get cameras for the dashboard
        if db:
            cameras_ref = db.collection("cameras").stream()
            cameras = []
            for doc in cameras_ref:
                camera_data = doc.to_dict()
                camera_data['id'] = doc.id
                cameras.append(camera_data)
        else:
            cameras = []
            
        # Get basic stats
        stats = {
            "people_detected": 5,
            "incidents_detected": 2,
            "seats_available": 10,
            "alerts_sent": 1
        }
        
        return render_template('dashboard.html', user=session["user"], cameras=cameras, stats=stats)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}\n{traceback.format_exc()}")
        return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if "user" in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            first_name = request.form.get('firstName')
            last_name = request.form.get('lastName')
            
            # Create user in Firebase
            user = auth_firebase.create_user_with_email_and_password(email, password)
            
            # Update user profile with name
            auth_firebase.update_profile(user['idToken'], 
                                      display_name=f"{first_name} {last_name}")
            
            # Store additional user info in Firestore
            if db:
                db.collection("users").document(user['localId']).set({
                    "email": email,
                    "firstName": first_name,
                    "lastName": last_name,
                    "created": firestore.SERVER_TIMESTAMP
                })
            
            logger.info(f"User created successfully: {email}")
            return redirect(url_for('login'))
            
        except Exception as e:
            logger.error(f"Signup error: {str(e)}")
            return render_template('signup.html', error="Failed to create account. Please try again.")
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if "user" in session:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        try:
            email = request.form.get('email')
            password = request.form.get('password')
            
            # Authenticate with Firebase
            user = auth_firebase.sign_in_with_email_and_password(email, password)
            
            # Create session
            session["user"] = email
            session["uid"] = user['localId']
            
            logger.info(f"User logged in successfully: {email}")
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return render_template('login.html', error="Invalid email or password")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    user_email = session.get("user", "unknown")
    session.clear()
    logger.info(f"User {user_email} logged out")
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/add_camera', methods=['GET', 'POST'])
def add_camera():
    if "user" not in session:
        logger.warning("Unauthorized access attempt to add_camera")
        return redirect(url_for('login'))
        
    if db is None:
        return "Firebase not initialized", 500

    if request.method == "POST":
        try:
            input_type = request.form["input_type"]
            camera_name = request.form["camera_name"]
            input_value = request.form["input_value"]
            google_maps_link = request.form.get("google_maps_link", "")
            mobile_number = request.form.get("mobile_number", "")

            # Generate a unique ID for the camera based on name with random suffix
            # Clean the camera name (remove special chars, convert to lowercase)
            clean_name = re.sub(r'[^a-zA-Z0-9]', '', camera_name.lower())
            # Add random suffix
            camera_id = f"{clean_name}_{str(uuid.uuid4())[:8]}"

            camera_data = {
                "camera_id": camera_id,  # Store the generated ID
                "input_type": input_type,
                "camera_name": camera_name,
                "input_value": input_value,
                "google_maps_link": google_maps_link,
                "mobile_number": mobile_number,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "user_id": session.get("uid", "unknown")
            }
            
            # Save to Firestore with the generated ID as document ID
            db.collection("cameras").document(camera_id).set(camera_data)

            with stream_lock:
                if camera_name not in active_streams:
                    # Initialize stream but don't start processing (visible=False)
                    # Pass the camera_id instead of input_value as the camera identifier
                    stream = CameraStream(camera_id)
                    stream.start()
                    active_streams[camera_name] = stream

            return redirect(url_for("dashboard"))
        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return render_template("add_camera.html", error="Failed to add camera")

    return render_template("add_camera.html")

@app.route('/get_cameras')
def get_cameras_endpoint():
    """API endpoint to get list of connected cameras"""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        cameras = get_connected_cameras()
        return jsonify(cameras)
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        return jsonify([]), 500

@app.route('/stats')
def get_stats():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        stats = {
            "people_detected": 5,
            "incidents_detected": 2,
            "seats_available": 10,
            "alerts_sent": 1
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({"status": "error", "message": "Failed to fetch stats"}), 500

@app.route('/video_feed')
def video_feed():
    try:
        # Check if user is logged in
        if "user" not in session:
            logger.warning("Unauthorized access attempt to video feed")
            return "Unauthorized", 401
        
        camera_id = 0  # Default camera
        logger.info(f"Initializing video feed for camera {camera_id}")
        
        # Clean up existing stream if any
        with stream_lock:
            if camera_id in active_streams:
                logger.info("Cleaning up existing stream")
                active_streams[camera_id].stop()
                del active_streams[camera_id]
            
            # Create new stream
            logger.info("Creating new camera stream")
            stream = CameraStream(camera_id)
            stream.start()
            active_streams[camera_id] = stream
        
        def generate():
            while True:
                try:
                    if camera_id not in active_streams:
                        logger.warning("Stream no longer active")
                        break
                    
                    frame = active_streams[camera_id].get_frame()
                    if frame is None:
                        logger.warning("No frame received")
                        time.sleep(0.1)
                        continue
                    
                    yield frame
                    
                except Exception as e:
                    logger.error(f"Error in frame generation: {str(e)}")
                    time.sleep(0.1)
        
        return Response(
            generate(),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            direct_passthrough=True
        )
        
    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        with stream_lock:
            if camera_id in active_streams:
                active_streams[camera_id].stop()
                del active_streams[camera_id]
        return f"Error: {str(e)}", 500

@app.route('/camera/<camera_name>/pause', methods=['POST'])
def pause_camera_stream(camera_name):
    """Pause processing for a camera when not visible in the UI"""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        with stream_lock:
            if camera_name in active_streams:
                logger.info(f"Pausing stream for camera: {camera_name}")
                active_streams[camera_name].stop()
                return jsonify({"success": True})
            else:
                logger.warning(f"Attempted to pause non-existent stream: {camera_name}")
                return jsonify({"success": False, "error": "Stream not found"}), 404
    except Exception as e:
        logger.error(f"Error pausing camera {camera_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/camera/<camera_name>/resume', methods=['POST'])
def resume_camera_stream(camera_name):
    """Resume processing for a camera when visible in the UI"""
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    try:
        with stream_lock:
            if camera_name in active_streams:
                logger.info(f"Resuming stream for camera: {camera_name}")
                active_streams[camera_name].start()
                return jsonify({"success": True})
            else:
                logger.warning(f"Attempted to resume non-existent stream: {camera_name}")
                return jsonify({"success": False, "error": "Stream not found"}), 404
    except Exception as e:
        logger.error(f"Error resuming camera {camera_name}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/camera/<camera_name>/remove', methods=['POST'])
def remove_camera(camera_name):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
        
    if db is None:
        return jsonify({"error": "Firebase not initialized"}), 500

    logger.info(f"Starting camera removal process for: {camera_name}")

    try:
        # 1. Stop the video feed first to prevent new connections
        with stream_lock:
            if camera_name in active_streams:
                logger.info(f"Stopping stream for camera: {camera_name}")
                stream = active_streams[camera_name]
                # Force stop all processing
                stream.stop()
                # Remove from active streams
                del active_streams[camera_name]

        # 2. Then remove from database
        camera_ref = db.collection("cameras").where("camera_name", "==", camera_name).limit(1).get()
        camera_doc = None
        
        for doc in camera_ref:
            camera_doc = doc
            break
            
        if camera_doc:
            camera_doc.reference.delete()
            logger.info(f"Deleted camera {camera_name} from database")
        else:
            logger.warning(f"Camera {camera_name} not found in database")

        # 3. Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        logger.info(f"Successfully removed camera {camera_name}")
        
        # 4. Add a small delay to ensure cleanup is complete
        time.sleep(0.5)
        
        return jsonify({"success": True})

    except Exception as e:
        logger.error(f"Error removing camera {camera_name}: {e}")
        # Emergency cleanup
        try:
            if camera_name in active_streams:
                stream = active_streams[camera_name]
                stream.stop()
                del active_streams[camera_name]
                gc.collect()
        except Exception as cleanup_error:
            logger.error(f"Emergency cleanup failed: {cleanup_error}")
        
        return jsonify({"success": False, "error": str(e)}), 500

def cleanup_streams():
    """Clean up all active camera streams when the application shuts down"""
    logger.info("Starting global cleanup of all camera streams")
    try:
        with stream_lock:
            camera_names = list(active_streams.keys())
            
            for camera_name in camera_names:
                try:
                    logger.info(f"Force cleaning up stream for camera: {camera_name}")
                    if camera_name in active_streams:
                        stream = active_streams[camera_name]
                        stream.stop()
                
                except Exception as e:
                    logger.error(f"Error cleaning up stream for {camera_name}: {e}")
            
            # Clear all streams
            active_streams.clear()
            
            # Force Python garbage collection
            gc.collect()
            
        logger.info("Global cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during global cleanup: {e}")
        # Emergency cleanup
        try:
            active_streams.clear()
            gc.collect()
        except:
            pass

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal Server Error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        "status": "error",
        "message": str(error)
    }), 500

@app.errorhandler(Exception)
def handle_error(error):
    logger.error(f"Unhandled error: {str(error)}\n{traceback.format_exc()}")
    return jsonify({
        "status": "error",
        "message": str(error)
    }), 500

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def emit_detection_update(detection_data):
    socketio.emit('detection_update', detection_data)

@socketio.on('process_frame')
def process_frame(data):
    try:
        # Convert frame data to numpy array
        frame_data = np.frombuffer(data['frame'], dtype=np.uint8)
        frame = frame_data.reshape((data['height'], data['width'], 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Process with YOLO
        results = yolo_model.predict(frame, conf=0.25)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_name = results.names[int(class_id)]
            
            detections.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                'class': class_name,
                'confidence': score
            })

        # Process with MediaPipe
        with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_mp = holistic.process(frame_rgb)

            # Get head angle if face landmarks are detected
            head_angle = None
            if results_mp.face_landmarks:
                # Calculate head angle using nose and eyes
                nose = results_mp.face_landmarks.landmark[1]
                left_eye = results_mp.face_landmarks.landmark[33]
                right_eye = results_mp.face_landmarks.landmark[263]
                
                # Calculate angle
                head_angle = calculate_head_angle(nose, left_eye, right_eye)

            # Check for hand gestures
            hands_detected = bool(results_mp.left_hand_landmarks or results_mp.right_hand_landmarks)

        # Emit detection results
        emit('detection_result', {
            'detections': detections,
            'head_angle': head_angle,
            'hands_detected': hands_detected
        })

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        emit('detection_result', {'error': str(e)})

def calculate_head_angle(nose, left_eye, right_eye):
    """Calculate head angle from facial landmarks"""
    try:
        # Calculate midpoint between eyes
        eye_mid_x = (left_eye.x + right_eye.x) / 2
        eye_mid_y = (left_eye.y + right_eye.y) / 2

        # Calculate angle between nose and eye midpoint
        angle = math.degrees(math.atan2(nose.y - eye_mid_y, nose.x - eye_mid_x))
        
        # Normalize angle
        if angle < 0:
            angle += 360
            
        return angle

    except Exception as e:
        logger.error(f"Error calculating head angle: {str(e)}")
        return 0

@socketio.on('start_monitoring')
def handle_start_monitoring():
    try:
        logger.info(f"Starting monitoring for user: {session.get('user', 'unknown')}")
        # Additional setup if needed
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    try:
        logger.info(f"Stopping monitoring for user: {session.get('user', 'unknown')}")
        # Cleanup if needed
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")

if __name__ == '__main__':
    app.debug = True
    logger.info("Starting application")
    
    try:
        # Ensure the static/assets directory exists
        os.makedirs('examguard/static/assets', exist_ok=True)
        
        # Clean up any existing streams
        cleanup_streams()
        
        # Initialize active streams dictionary
        active_streams.clear()
        
        # Register cleanup function to be called on shutdown
        import atexit
        atexit.register(cleanup_streams)
        
        # Start the application
        socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        # Clean up camera streams
        cleanup_streams()
