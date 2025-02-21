import mediapipe as mp
import cv2
import time
import numpy as np
import math
import queue
import threading
from queue import Queue
from collections import deque
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
import os
from flask import Flask, Response
from flask_cors import CORS

# Initialize Flask for streaming
stream_app = Flask(__name__)
CORS(stream_app)

# Global variables for frame sharing
global_frame = None
frame_lock = threading.Lock()

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
print(f"PyTorch current device: {torch.cuda.current_device()}")
print(f"PyTorch device name: {torch.cuda.get_device_name(0)}")

# Initialize MediaPipe components
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Initialize queues for async processing
frame_queue = Queue(maxsize=4)
result_queue = Queue(maxsize=4)

# Constants
PROCESS_WIDTH = 1280
PROCESS_HEIGHT = 720
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
HEAD_TURN_THRESHOLD = 25
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

# CUDA setup
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    torch.backends.cudnn.benchmark = True
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("No CUDA devices available")

def set_global_frame(frame):
    global global_frame
    with frame_lock:
        global_frame = frame.copy() if frame is not None else None

@stream_app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            with frame_lock:
                if global_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', global_frame)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def run_stream_server():
    stream_app.run(host='0.0.0.0', port=5001)

def list_available_cameras(max_tested=10):
    available_cameras = []
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available_cameras.append(i)
        cap.release()
    return available_cameras

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
            print(f"Person {self.person_id}: Initial head position calibrated at {self.initial_position:.1f}°")
            
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
                print(f"Person {self.person_id}: Head turn #{self.turn_count} detected! Angle: {smoothed_angle:.1f}°")
                
                if self.turn_count >= MAX_HEAD_TURNS and self.warning_start_time is None:
                    self.warning_start_time = current_time
                    print(f"Person {self.person_id}: Maximum head turns reached - Warning activated!")
            
            elif self.steady_center_frames >= self.FRAMES_FOR_STABLE:
                self.turning = False
                self.steady_center_frames = 0
        
        self.last_angle = smoothed_angle
        
        if (self.warning_start_time is not None and 
            current_time - self.warning_start_time >= WARNING_DURATION):
            self.warning_start_time = None
            self.turn_count = 0
            self.turns_history.clear()
            print(f"Person {self.person_id}: Warning period ended, resetting turn count")
        
        is_warning = self.warning_start_time is not None or self.turn_count >= MAX_HEAD_TURNS
        return is_warning, smoothed_angle

class VideoStreamThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.src = src
        self.stopped = False
        self.cap = None
        self.last_frame = None
        self.lock = threading.Lock()

    def run(self):
        print(f"Attempting to open camera {self.src}")
        self.cap = cv2.VideoCapture(self.src)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.src}")
            return

        print(f"Successfully opened camera {self.src}")
        print(f"Camera resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"Camera FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Warning: Failed to grab frame, retrying...")
                self.cap.release()
                time.sleep(0.1)
                self.cap = cv2.VideoCapture(self.src)
                continue
                
            with self.lock:
                self.last_frame = frame.copy()
            
            try:
                if not frame_queue.full():
                    frame_queue.put(frame.copy(), timeout=0.1)
            except queue.Full:
                continue
            
            time.sleep(0.001)

        if self.cap is not None:
            self.cap.release()
            print("Camera released")

    def get_latest_frame(self):
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def stop(self):
        self.stopped = True

def detect_head_turn_angle(pose_landmarks):
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
    except Exception as e:
        print(f"Error calculating head angle: {e}")
        return None

class HolisticDetectionThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.stopped = False
        
        # Initialize YOLOv5
        print("Initializing YOLOv5 model...")
        self.device = torch.device('cuda:0' if USE_CUDA else 'cpu')
        model_path = 'yolov5/runs/train/exp9/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"Error: YOLOv5 model not found at {model_path}")
            raise FileNotFoundError(f"YOLOv5 model not found at {model_path}")
            
        self.yolo_model = attempt_load(model_path, device=self.device)
        self.yolo_model.to(self.device).half()
        self.yolo_model.eval()
        print("YOLOv5 model initialized successfully")
        
        print("Initializing MediaPipe Holistic...")
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        self.head_trackers = {}
        self.person_counter = 0
        self.last_cleanup = time.time()
        print("MediaPipe Holistic initialized successfully")

    def detect_upright_hand(self, hand_landmarks, pose_landmarks):
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
            
        except Exception as e:
            print(f"Error in hand position detection: {e}")
            return False

    def cleanup_trackers(self):
        current_time = time.time()
        if current_time - self.last_cleanup > 5:
            inactive_ids = []
            for person_id, tracker in self.head_trackers.items():
                if current_time - tracker.last_seen > 2:
                    inactive_ids.append(person_id)
            
            for person_id in inactive_ids:
                del self.head_trackers[person_id]
                print(f"Removed inactive tracker for person {person_id}")
            
            self.last_cleanup = current_time

    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # YOLOv5 Detection
        img = cv2.resize(frame_rgb, (640, 640))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = self.yolo_model(img.half())[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        yolo_detections = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame_rgb.shape).round()
                det = det.cpu().numpy()
                for *xyxy, conf, cls in det:
                    label = f'{self.yolo_model.names[int(cls)]} {conf:.2f}'
                    xyxy = [int(x) for x in xyxy]
                    yolo_detections.append((xyxy, label, conf))
        
        # MediaPipe processing
        results = self.holistic.process(frame_rgb)
        self.cleanup_trackers()
        multi_results = []
        
        # Process face detections
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
        faces = face_detection.process(frame_rgb)
        
        if faces.detections:
            for i, face in enumerate(faces.detections):
                bbox = face.location_data.relative_bounding_box
                center_x = bbox.xmin + bbox.width/2
                
                person_id = None
                min_dist = float('inf')
                for pid, tracker in self.head_trackers.items():
                    if tracker.last_angle is not None:
                        dist = abs(center_x - 0.5)
                        if dist < min_dist and dist < 0.3:
                            min_dist = dist
                            person_id = pid
                
                if person_id is None:
                    person_id = self.person_counter
                    self.head_trackers[person_id] = HeadTurnTracker(person_id)
                    self.person_counter += 1
                
                head_angle = detect_head_turn_angle(results.pose_landmarks)
                warning_active, smoothed_angle = self.head_trackers[person_id].update(head_angle)
                
                left_hand_upright = False
                right_hand_upright = False
                
                if results.left_hand_landmarks and results.pose_landmarks:
                    left_hand_upright = self.detect_upright_hand(
                        results.left_hand_landmarks,
                        results.pose_landmarks
                    )
                    
                if results.right_hand_landmarks and results.pose_landmarks:
                    right_hand_upright = self.detect_upright_hand(
                        results.right_hand_landmarks,
                        results.pose_landmarks
                    )
                
                hands_active = left_hand_upright or right_hand_upright
                
                person_result = {
                    'person_id': person_id,
                    'warning_active': warning_active,
                    'head_angle': smoothed_angle,
                    'hands_active': hands_active,
                    'left_hand_upright': left_hand_upright,
                    'right_hand_upright': right_hand_upright,
                    'bbox': bbox
                }
                multi_results.append(person_result)
        
        return results, multi_results, yolo_detections

    def run(self):
        while not self.stopped:
            if not frame_queue.empty():
                frame = frame_queue.get()
                holistic_results, multi_results, yolo_detections = self.process_frame(frame)
                if not result_queue.full():
                    result_queue.put((frame, (holistic_results, multi_results, yolo_detections)))
            else:
                time.sleep(0.001)

    def stop(self):
        self.stopped = True

def main():
    print("Checking available cameras...")
    cameras = list_available_cameras()
    if not cameras:
        print("No cameras available!")
        return
    
    print(f"Available camera indices: {cameras}")
    
    # Start video streaming server
    stream_thread = threading.Thread(target=run_stream_server)
    stream_thread.daemon = True
    stream_thread.start()
    print("Video streaming server started on port 5001")
    
    vs_thread = VideoStreamThread(src=cameras[0])
    holistic_thread = HolisticDetectionThread()
    
    vs_thread.start()
    holistic_thread.start()
    
    # Give threads time to initialize
    time.sleep(1)
    
    cv2.namedWindow('Holistic Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Holistic Detection', DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            time.sleep(0.01)
            
            if vs_thread.stopped or holistic_thread.stopped:
                print("One of the threads has stopped unexpectedly")
                break
                
            try:
                frame, results = result_queue.get(timeout=1.0)
                holistic_results, multi_results, yolo_detections = results
            except queue.Empty:
                print("No frames in queue")
                continue
            except Exception as e:
                print(f"Error getting frame from queue: {e}")
                continue
                
            if frame is None:
                print("Received empty frame")
                continue
                
            # Draw YOLO detections
            for (xyxy, label, conf) in yolo_detections:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS['YOLO_BOX'], 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['TEXT'], 2)
            
            # Draw MediaPipe landmarks
            if holistic_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLORS['LANDMARK_POSE'], thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=COLORS['LANDMARK_POSE'], thickness=2, circle_radius=2)
                )
            
            if holistic_results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLORS['LANDMARK_HAND_LEFT'], thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=COLORS['LANDMARK_HAND_LEFT'], thickness=2, circle_radius=2)
                )
            
            if holistic_results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, holistic_results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLORS['LANDMARK_HAND_RIGHT'], thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=COLORS['LANDMARK_HAND_RIGHT'], thickness=2, circle_radius=2)
                )
            
            # Draw person statuses
            for person in multi_results:
                person_id = person['person_id']
                warning_active = person['warning_active']
                head_angle = person['head_angle']
                hands_active = person['hands_active']
                bbox = person['bbox']
                
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Determine box color
                box_color = COLORS['SAFE']
                if abs(head_angle) > HEAD_TURN_THRESHOLD:
                    box_color = COLORS['HEAD_WARNING']
                if hands_active:
                    box_color = COLORS['HAND_WARNING']
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + width, y + height), box_color, 2)
                
                # Status panel
                status_width = 300
                status_height = 90
                status_x = 10
                status_y = 10 + (person_id * (status_height + 10))
                
                cv2.rectangle(frame, 
                            (status_x, status_y),
                            (status_x + status_width, status_y + status_height),
                            COLORS['BACKGROUND'], -1)
                
                # Update status text
                text_y = status_y + 25
                cv2.putText(frame, f'Person {person_id}', (status_x + 10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['TEXT'], 2)
                
                text_y += 25
                cv2.putText(frame, f'Head Angle: {head_angle:.1f}°', 
                           (status_x + 10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           COLORS['HEAD_WARNING'] if abs(head_angle) > HEAD_TURN_THRESHOLD else COLORS['TEXT'], 2)
                
                text_y += 25
                cv2.putText(frame, f'Hands Up: {"Yes" if hands_active else "No"}',
                           (status_x + 10, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           COLORS['HAND_WARNING'] if hands_active else COLORS['TEXT'], 2)
            
            # Update FPS counter
            fps_counter += 1
            if (time.time() - fps_start_time) > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(frame, f'FPS: {fps}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['FPS'], 2)
            
            # Display people count
            cv2.putText(frame, f'People Detected: {len(multi_results)}', 
                      (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['TEXT'], 2)
            
            # Update global frame for streaming
            set_global_frame(frame)
            
            # Display frame locally
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.imshow('Holistic Detection', display_frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
                
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("Cleaning up...")
        vs_thread.stop()
        holistic_thread.stop()
        vs_thread.join()
        holistic_thread.join()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()