import base64
import cv2
import numpy as np
import torch
# Change from ultralytics import to YOLOv5 import
import os
import logging
import threading
from threading import Lock
import time
import uuid
import re
import gc
import traceback
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, session, url_for, Response
from flask_socketio import SocketIO, emit
import firebase_admin
from firebase_admin import credentials, auth, firestore
import pyrebase
from pathlib import Path
import sys
import mediapipe as mp
import math
from collections import deque
from student_tracker import StudentTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask and SocketIO with proper configuration
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = os.urandom(24)
socketio = SocketIO(app, 
                    cors_allowed_origins="*",
                    async_mode='threading',
                    ping_timeout=60,
                    ping_interval=25)

# Configure Firebase if credentials are available
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

# Store active camera streams with thread-safe dictionary
active_streams = {}
stream_lock = threading.Lock()

# Constants
HEAD_TURN_THRESHOLD = 40
HEAD_TURN_RESET_THRESHOLD = 10
MAX_HEAD_TURNS = 5
WARNING_DURATION = 30
SMOOTHING_WINDOW_SIZE = 5

# Color Constants (BGR Format)
COLORS = {
    'DETECTION_BOX': (0, 255, 0),    # Green
    'HEAD_WARNING': (0, 0, 255),     # Red
    'HAND_WARNING': (0, 0, 255),     # Red
    'SAFE': (0, 255, 0),            # Green
    'TEXT': (255, 255, 255),        # White
    'BACKGROUND': (0, 0, 0),        # Black
    'FPS': (0, 255, 0),             # Green
    'LANDMARK_POSE': (66, 117, 245),  # Coral
    'LANDMARK_HAND_LEFT': (76, 22, 121),  # Purple
    'LANDMARK_HAND_RIGHT': (10, 22, 80)   # Dark Blue
}

# Class labels
CLASS_LABELS = {
    0: "Head Turn",
    1: "Paper Exchange",
    2: "Hand Gesture",
    3: "Phone Usage"
}

# Class-specific confidence thresholds
CLASS_THRESHOLDS = {
    0: 0.10,  # Head turn
    1: 0.05,  # Paper exchange
    2: 0.05,  # Hand gestures
    3: 0.05,  # Smartphone usage
}

# Global variables
output_dir = None
student_tracker = None
frame_lock = threading.Lock()
latest_frame = None
detection_active = False
detection_thread = None
session_timestamp = None

# Model variables
yolo_model = None
mediapipe_detector = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Detection statistics
detection_stats = {
    'head_turns': 0,
    'paper_exchange': 0,
    'hand_gestures': 0,
    'phone_usage': 0,
    'total_students': 0,
    'fps': 0
}

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands

class HeadTurnTracker:
    def __init__(self, person_id):
        self.person_id = person_id
        self.turn_count = 0
        self.last_angle = None
        self.turning = False
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
                
            elif self.steady_center_frames >= self.FRAMES_FOR_STABLE:
                self.turning = False
                self.steady_center_frames = 0
        
        self.last_angle = smoothed_angle
        is_warning = abs(smoothed_angle) > HEAD_TURN_THRESHOLD
        return is_warning, smoothed_angle

class MediaPipeDetector:
    def __init__(self):
        """Initialize MediaPipe detection models"""
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        self.head_trackers = {}
        self.last_cleanup = time.time()
        logger.info("MediaPipe detector initialized")
    
    def cleanup_trackers(self):
        """Remove inactive head trackers"""
        current_time = time.time()
        if current_time - self.last_cleanup > 5:
            inactive_ids = []
            for person_id, tracker in self.head_trackers.items():
                if current_time - tracker.last_seen > 2:
                    inactive_ids.append(person_id)
            
            for person_id in inactive_ids:
                del self.head_trackers[person_id]
            
            self.last_cleanup = current_time
            
    def detect_head_turn_angle(self, pose_landmarks):
        """Calculate head turn angle using pose landmarks"""
        if not pose_landmarks:
            return None
        
        try:
            nose = pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
            left_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR]
            right_ear = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR]
            
            ear_midpoint_x = (left_ear.x + right_ear.x) / 2
            ear_midpoint_y = (left_ear.y + right_ear.y) / 2
            
            dx = nose.x - ear_midpoint_x
            dy = nose.y - ear_midpoint_y
            angle = math.degrees(math.atan2(dx, -dy))
            
            return angle
        except Exception:
            return None
            
    def detect_upright_hand(self, hand_landmarks, pose_landmarks):
        """Detect if hand is raised in an upright position"""
        if not hand_landmarks or not pose_landmarks:
            return False
                
        try:
            left_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y
            shoulder_height = min(left_shoulder, right_shoulder)
            
            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            if wrist.y > shoulder_height:
                return False
                    
            fingertips_above_wrist = all(
                tip.y < wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
            )
            
            vertical_threshold = 0.1
            fingers_vertical = all(
                abs(tip.x - wrist.x) < vertical_threshold
                for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
            )
            
            return fingertips_above_wrist and fingers_vertical
                
        except Exception:
            return False
    
    def process_frame(self, frame):
        """Process frame with MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        holistic_results = self.holistic.process(frame_rgb)
        face_results = self.face_detection.process(frame_rgb)
        
        mediapipe_boxes = []
        self.cleanup_trackers()
        
        # Process faces
        if face_results.detections:
            h, w = frame.shape[:2]
            for i, face in enumerate(face_results.detections):
                bbox = face.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                person_id = f"mp_{i}"
                if person_id not in self.head_trackers:
                    self.head_trackers[person_id] = HeadTurnTracker(person_id)
                
                head_angle = self.detect_head_turn_angle(holistic_results.pose_landmarks)
                warning_active, smoothed_angle = self.head_trackers[person_id].update(head_angle)
                
                # Check for raised hands
                left_hand_active = False
                right_hand_active = False
                if holistic_results.left_hand_landmarks and holistic_results.pose_landmarks:
                    left_hand_active = self.detect_upright_hand(
                        holistic_results.left_hand_landmarks,
                        holistic_results.pose_landmarks
                    )
                if holistic_results.right_hand_landmarks and holistic_results.pose_landmarks:
                    right_hand_active = self.detect_upright_hand(
                        holistic_results.right_hand_landmarks,
                        holistic_results.pose_landmarks
                    )
                
                # Determine detection type
                detection_type = None
                confidence = 0.7  # Base confidence for MediaPipe
                
                if abs(smoothed_angle) > HEAD_TURN_THRESHOLD:
                    detection_type = 0  # Head turn
                    confidence = 0.7 + (min(abs(smoothed_angle) - HEAD_TURN_THRESHOLD, 30) / 100)
                elif left_hand_active or right_hand_active:
                    detection_type = 2  # Hand gesture
                    confidence = 0.7
                
                if detection_type is not None:
                    # Convert to format compatible with YOLO boxes
                    box = torch.tensor([
                        [x, y, x + width, y + height]
                    ])
                    
                    mediapipe_boxes.append({
                        'cls': torch.tensor([detection_type]),
                        'conf': torch.tensor([confidence]),
                        'xyxy': box
                    })
        
        # Return processed results
        return mediapipe_boxes, holistic_results

# Modified to not draw landmarks (per your request)
def draw_mediapipe_landmarks(frame, holistic_results):
    """
    Function is maintained but NO LONGER draws MediaPipe landmarks on frame,
    as requested - the detection still happens but landmarks are not visualized
    """
    # We're intentionally not drawing the landmarks as per the requested change
    return frame

def get_class_threshold(cls):
    """Get confidence threshold for specific class"""
    return CLASS_THRESHOLDS.get(int(cls), 0.05)  # Default threshold for unknown classes

def initialize_models():
    """Initialize YOLOv5 and MediaPipe models"""
    global yolo_model, mediapipe_detector, device
    
    try:
        # Try different possible model paths with priority to the requested path
        model_paths = [
            "C:\\Users\\vst04\\Desktop\\Exam Guard\\examguard\\best.pt",  # Requested path
            "best.pt",
            os.path.join(os.path.dirname(__file__), "best.pt"),
            os.path.join(os.path.dirname(__file__), "examguard", "best.pt"),
            "examguard/best.pt"
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            logger.error("Model file not found in any of the expected locations")
            return False
        
        # Select device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load YOLOv5 model
        logger.info(f"Loading YOLOv5 model from {model_path}")
        # Changed from ultralytics YOLO to torch load for YOLOv5
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        
        # Configure model parameters for YOLOv5
        yolo_model.conf = 0.3  # Confidence threshold
        yolo_model.iou = 0.45  # NMS IOU threshold
        yolo_model.agnostic = False  # NMS class-agnostic
        yolo_model.multi_label = False  # Multiple labels per box
        yolo_model.max_det = 300  # Maximum number of detections
        
        # Initialize MediaPipe detector
        logger.info("Initializing MediaPipe detector")
        mediapipe_detector = MediaPipeDetector()
        
        # Test model with a dummy image
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        test_results = yolo_model(dummy_img)
        logger.info("Model test successful")
        
        logger.info("Models loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        traceback.print_exc()
        return False

def setup_session():
    """Set up a new detection session"""
    global output_dir, student_tracker, session_timestamp
    
    try:
        # Create timestamp for the session
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_base = os.path.join(os.path.expanduser("~"), "ExamGuard_Web")
        output_dir = os.path.join(output_base, f"session_{session_timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize student tracker
        student_tracker = StudentTracker(output_dir)
        
        logger.info(f"Session setup complete. Output directory: {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error setting up session: {str(e)}")
        raise

def detection_loop():
    """Main detection loop that processes webcam feed"""
    global latest_frame, detection_active, student_tracker, detection_stats
    
    cap = None
    
    try:
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error: Could not open webcam")
            socketio.emit('monitoring_status', {'status': 'error', 'message': 'Could not open webcam'})
            detection_active = False
            return
        
        # Set webcam properties - use higher resolution if available
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Try to get higher resolution
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # HD resolution
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify what resolution we actually got (some cameras may not support requested resolution)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera actual resolution: {actual_width}x{actual_height}")
        
        # Warm up camera
        logger.info("Warming up camera...")
        for _ in range(10):
            ret, _ = cap.read()
            if not ret:
                logger.error("Camera warm-up failed")
                raise Exception("Failed to read frames during camera warm-up")
            time.sleep(0.1)
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Send camera dimensions to client for proper aspect ratio
        socketio.emit('camera_dimensions', {'width': width, 'height': height})
        
        logger.info(f"Webcam initialized: {width}x{height} @ {fps}fps")
        
        # Note: Video recording is removed, using snapshots only for incidents
        
        frame_count = 0
        fps_start_time = time.time()
        last_emit_time = time.time()
        frames_processed = 0
        
        while detection_active:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                continue
            
            frame_count += 1
            frames_processed += 1
            current_time = time.time()
            
            # Calculate FPS every second
            if current_time - fps_start_time >= 1.0:
                detection_stats['fps'] = frames_processed / (current_time - fps_start_time)
                frames_processed = 0
                fps_start_time = current_time
                
            # Process every other frame for performance
            if frame_count % 2 != 0:
                continue
                
            try:
                # Create a copy of the frame for processing
                process_frame = frame.copy()
                
                # Run YOLOv5 detection (primary detection method)
                # Changed from YOLOv8 to YOLOv5 processing
                yolo_results = yolo_model(process_frame)
                
                # YOLOv5 detection processing is different from YOLOv8
                yolo_detections = []
                
                for i, (x1, y1, x2, y2, conf, cls) in enumerate(yolo_results.xyxy[0].cpu().numpy()):
                    if conf >= get_class_threshold(cls):
                        yolo_detections.append({
                            'cls': torch.tensor([cls]),
                            'conf': torch.tensor([conf]),
                            'xyxy': torch.tensor([[x1, y1, x2, y2]])
                        })
                
                # If no YOLO detections, use MediaPipe as fallback
                holistic_results = None
                if len(yolo_detections) == 0 and mediapipe_detector:
                    logger.debug("No YOLO detections, using MediaPipe fallback")
                    mediapipe_boxes, holistic_results = mediapipe_detector.process_frame(process_frame)
                    
                    # If MediaPipe found something, add these detections
                    if mediapipe_boxes:
                        yolo_detections.extend(mediapipe_boxes)
                
                # Create copy for drawing
                result_frame = frame.copy()
                
                # Process detections
                frame_detections = {
                    'head_turns': 0,
                    'paper_exchange': 0,
                    'hand_gestures': 0,
                    'phone_usage': 0
                }
                
                # Draw detections
                for box in yolo_detections:
                    cls = int(box['cls'][0]) if 'cls' in box else 0
                    conf = float(box['conf'][0]) if 'conf' in box else 0.0
                    
                    if conf >= get_class_threshold(cls):
                        xyxy = box['xyxy'].cpu().numpy()[0]
                        
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Draw box with class-specific color
                        color = COLORS['DETECTION_BOX']
                        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with confidence score
                        label = f"{CLASS_LABELS.get(cls, 'Unknown')} {conf:.2f}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        cv2.rectangle(result_frame, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                        cv2.putText(result_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['TEXT'], 2)
                        
                        # Update student tracker and counters
                        if student_tracker is not None:
                            student_id = student_tracker.get_student_id(xyxy)
                            student_tracker.update_student_data(student_id, cls, result_frame, xyxy)
                        
                        # Update counters
                        if cls == 0:
                            frame_detections['head_turns'] += 1
                        elif cls == 1:
                            frame_detections['paper_exchange'] += 1
                        elif cls == 2:
                            frame_detections['hand_gestures'] += 1
                        elif cls == 3:
                            frame_detections['phone_usage'] += 1
                
                # Add FPS and detection counts to frame
                fps_text = f"FPS: {detection_stats['fps']:.1f}"
                cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['FPS'], 2)
                
                # Display detection counts
                y_offset = 60
                for category, count in frame_detections.items():
                    if count > 0:
                        detection_stats[category] = detection_stats.get(category, 0) + count
                        text = f"{category.replace('_', ' ').title()}: {detection_stats[category]}"
                        cv2.putText(result_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['TEXT'], 2)
                        y_offset += 30
                
                # Update count of tracked students
                if student_tracker:
                    detection_stats['total_students'] = student_tracker.get_student_count()
                
                # Update latest frame
                with frame_lock:
                    latest_frame = result_frame.copy()
                
                # Emit updates at most every 100ms for responsiveness
                if current_time - last_emit_time >= 0.1:
                    # Encode frame
                    _, buffer = cv2.imencode('.jpg', result_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit frame and stats
                    socketio.emit('video_frame', {'frame': frame_b64})
                    socketio.emit('detection_update', {
                        'stats': detection_stats,
                        'frame_detections': frame_detections
                    })
                    
                    last_emit_time = current_time
                
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                traceback.print_exc()
                continue
            
            # Prevent CPU overload
            time.sleep(0.01)
            
    except Exception as e:
        logger.error(f"Fatal error in detection loop: {str(e)}")
        traceback.print_exc()
        socketio.emit('monitoring_status', {
            'status': 'error',
            'message': f'Detection error: {str(e)}'
        })
        
    finally:
        detection_active = False
        if cap is not None:
            cap.release()
        logger.info("Detection loop ended")
        socketio.emit('monitoring_status', {'status': 'stopped'})

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
            "people_detected": 0,
            "incidents_detected": 0,
            "seats_available": 10,
            "alerts_sent": 0
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

@app.route('/video_feed')
def video_feed():
    """Route for streaming the processed webcam feed"""
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    # If no frame is available, provide a blank frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    # Add text to indicate waiting for camera
                    cv2.putText(frame, "Waiting for camera...", (50, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    frame = latest_frame.copy()
                    
            # Encode the frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            # Yield the frame in multipart format
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # Brief pause to control CPU usage
            time.sleep(0.04)  # ~25 FPS
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('start_monitoring_request')
def handle_start_monitoring_request():
    """Handle request to start monitoring from client"""
    global detection_active, detection_thread, student_tracker
    
    try:
        logger.info("Received start monitoring request")
        
        if detection_active:
            logger.info("Monitoring already active")
            return
        
        # Initialize models if not already done
        if yolo_model is None:
            logger.info("Initializing models...")
            if not initialize_models():
                logger.error("Failed to initialize detection models")
                socketio.emit('monitoring_status', {
                    'status': 'error', 
                    'message': 'Failed to initialize detection models'
                })
                return
        
        # Setup session and directories
        logger.info("Setting up session...")
        setup_session()
        
        # Reset detection statistics
        detection_stats.update({
            'head_turns': 0,
            'paper_exchange': 0,
            'hand_gestures': 0,
            'phone_usage': 0,
            'total_students': 0,
            'fps': 0
        })
        
        # Start detection in a separate thread
        logger.info("Starting detection thread...")
        detection_active = True
        detection_thread = threading.Thread(target=detection_loop, daemon=True)
        detection_thread.start()
        
        # Emit success status
        socketio.emit('monitoring_status', {'status': 'started'})
        logger.info("Monitoring started successfully")
        
    except Exception as e:
        logger.error(f"Error starting monitoring: {str(e)}")
        traceback.print_exc()
        detection_active = False
        socketio.emit('monitoring_status', {
            'status': 'error', 
            'message': f'Error starting monitoring: {str(e)}'
        })

@socketio.on('stop_monitoring_request')
def handle_stop_monitoring_request():
    """Handle request to stop monitoring from client"""
    global detection_active
    
    if not detection_active:
        logger.info("Monitoring not active")
        return
    
    try:
        logger.info("Stopping monitoring")
        detection_active = False
        
        # Wait a bit for the thread to finish
        if detection_thread and detection_thread.is_alive():
            detection_thread.join(timeout=2.0)
            
        # Generate report if student_tracker exists
        if student_tracker is not None:
            try:
                logger.info("Generating detection report...")
                report_path = student_tracker.generate_pdf_report()
                logger.info(f"Report generated successfully at {report_path}")
                
                # Emit report ready event
                socketio.emit('report_ready', {
                    'path': os.path.join(output_dir, "malpractice_report.pdf")
                })
            except Exception as e:
                logger.error(f"Error generating report: {str(e)}")
                traceback.print_exc()
        
        logger.info("Monitoring stopped successfully")
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {str(e)}")
        socketio.emit('monitoring_status', {
            'status': 'error', 
            'message': f'Error stopping monitoring: {str(e)}'
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection from Socket.IO"""
    logger.info('Client disconnected from WebSocket')

def cleanup_resources():
    """Clean up resources when the application shuts down"""
    global detection_active
    
    logger.info("Cleaning up resources")
    detection_active = False
    
    # Force garbage collection
    gc.collect()

if __name__ == '__main__':
    try:
        # Register cleanup function
        import atexit
        atexit.register(cleanup_resources)
        
        # Start the application
        logger.info("Starting Exam Guard application")
        socketio.run(app, 
                    host='0.0.0.0',
                    port=8080,
                    debug=True,
                    allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        traceback.print_exc()
