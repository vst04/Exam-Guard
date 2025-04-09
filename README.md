# ExamGuard

ExamGuard is an AI-powered exam proctoring system designed to detect and report malpractice during exams through computer vision and machine learning techniques.

## Project Overview

ExamGuard uses a combination of YOLOv8 and MediaPipe to detect suspicious activities during exams including:
- Head turning (looking at other students' papers)
- Hand gestures (signaling or communicating)
- Phone usage (accessing unauthorized materials)
- Paper exchange (sharing answers or notes)

The system captures snapshots of detected violations rather than continuous video recording, providing evidence while preserving privacy and conserving storage space.

## Key Features

- **Real-time Monitoring**: Process webcam feed to detect suspicious activities 
- **Snapshot Evidence**: Capture both cropped and full-frame images when violations are detected
- **Student Tracking**: Identify and track individual students throughout the session
- **Dashboard Interface**: Modern web interface for monitoring and reviewing detections
- **PDF Report Generation**: Automatically generate comprehensive violation reports
- **Firebase Integration**: User authentication and data storage

## Technical Architecture

The project consists of several key components:

### Backend (Python)
- **app.py**: Main Flask application with:
  - Socket.IO for real-time communication
  - Firebase integration for authentication
  - Webcam feed processing
  - Detection algorithms
  
- **student_tracker.py**: Handles:
  - Student identification and tracking
  - Violation recording
  - Snapshot capture and management
  - PDF report generation

### AI Models
- YOLOv8 for primary object detection (head turning, paper exchange, etc.)
- MediaPipe for secondary detection (face landmarks, hand tracking, etc.)

### Frontend
- Interactive dashboard with real-time updates
- Grid and analytics views of student violations
- Full-frame video display without cropping

## How It Works

1. The system initializes the webcam feed and loads AI models
2. Each frame is processed to detect potential malpractice
3. When violations are detected:
   - The system assigns the detection to a specific student
   - It captures both a cropped image of the student and a full-frame image
   - Violation counts are updated in real-time
4. At the end of the session, a comprehensive PDF report is generated with violation evidence

## Technical Details

### Detection Methods

- **Head Turning**: Uses both YOLO detection and MediaPipe face landmarks to detect head orientation and track turning
- **Hand Gestures**: Combines YOLO object detection with MediaPipe hand tracking for comprehensive gesture recognition
- **Phone Usage**: YOLO-based detection specifically trained to recognize smartphone usage
- **Paper Exchange**: YOLO-based detection for unusual paper movement

### Data Storage

- Student violations are tracked in memory during the session
- Evidence snapshots are saved to a session-specific directory
- Violation data is exported to JSON for potential further analysis
- PDF reports compile all evidence and statistics

## Installation & Setup

(Instructions for installation and setup would go here)

## Requirements

- Python 3.8+
- Flask
- PyTorch
- OpenCV
- MediaPipe
- Firebase Admin SDK
- ReportLab
- Socket.IO
