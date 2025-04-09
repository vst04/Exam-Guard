"""student_tracker.py - Updated version for snapshot-only recording"""
import os
import cv2
import numpy as np
from datetime import datetime
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import logging

logger = logging.getLogger(__name__)

class StudentTracker:
    def __init__(self, output_dir):
        """Initialize the StudentTracker with output directory for reports and images"""
        self.output_dir = output_dir
        self.students = {}  # Dictionary to store student data
        self.student_count = 0  # Counter for total students
        self.assignment_threshold = 0.5  # IOU threshold for assigning detections to students
        
        # Create directories for images
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Initialize violation types
        self.violation_types = {
            0: "head_turns",
            1: "paper_exchange",
            2: "hand_gestures",
            3: "phone_usage"
        }
        
        logger.info(f"StudentTracker initialized with output directory: {output_dir}")
        logger.info(f"Snapshots directory: {self.snapshots_dir}")

    def get_student_count(self):
        """Return the current count of tracked students"""
        return self.student_count

    def get_student_id(self, bbox):
        """
        Get a student ID for a bounding box detection
        Args:
            bbox: Bounding box coordinates [x1, y1, x2, y2]
        Returns:
            str: Student ID
        """
        # Convert bbox to list if it's not already
        bbox = bbox.tolist() if hasattr(bbox, 'tolist') else bbox
        
        best_iou = 0
        best_id = None

        # Check existing students for best match
        for student_id, student_data in self.students.items():
            if 'bbox' in student_data:
                iou = self._calculate_iou(bbox, student_data['bbox'])
                if iou > best_iou and iou > self.assignment_threshold:
                    best_iou = iou
                    best_id = student_id

        # Create new student ID if no match found
        if best_id is None:
            self.student_count += 1
            best_id = f"student_{self.student_count}"
            self.students[best_id] = {
                'bbox': bbox,
                'violations': {
                    'head_turns': 0,
                    'hand_gestures': 0,
                    'phone_usage': 0,
                    'paper_exchange': 0
                },
                'snapshots': [],  # List to store snapshot details
                'full_frame_snapshots': []  # List to store full frame snapshots
            }
            logger.info(f"Created new student with ID: {best_id}")
        
        return best_id

    def update_student_data(self, student_id, violation_type, frame, bbox):
        """
        Update student data with new detection
        Args:
            student_id: ID of the student
            violation_type: Type of violation detected (0-3)
            frame: Video frame where violation was detected
            bbox: Bounding box coordinates
        """
        if student_id not in self.students:
            self.students[student_id] = {
                'bbox': bbox,
                'violations': {
                    'head_turns': 0,
                    'hand_gestures': 0,
                    'phone_usage': 0,
                    'paper_exchange': 0
                },
                'snapshots': [],
                'full_frame_snapshots': []
            }
            logger.info(f"Created new student on update with ID: {student_id}")

        # Update bbox
        self.students[student_id]['bbox'] = bbox

        # Get violation type as string
        violation_name = self.violation_types.get(int(violation_type), "unknown")
        
        # Update violation count based on type
        self.students[student_id]['violations'][violation_name] += 1
        
        # Take a snapshot of the student and the full frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self._take_student_snapshot(student_id, frame, bbox, violation_name, timestamp)
        
        logger.info(f"Updated student {student_id} with {violation_name} violation")

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
    
    def get_analytics_data(self):
        """Get analytics data for all students"""
        analytics_data = []
        for student_id, data in self.students.items():
            violations = data['violations']
            total_violations = sum(violations.values())
            
            # Calculate risk level
            risk_level = self.calculate_risk_level(data)
                
            analytics_data.append({
                'id': student_id,
                'violations': violations,
                'total_violations': total_violations,
                'risk_level': risk_level
            })
        return analytics_data
    
    def _take_student_snapshot(self, student_id, frame, bbox, incident_type, timestamp):
        """
        Take a snapshot of a student during an incident
        
        Args:
            student_id: ID of the student
            frame: Video frame
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            incident_type: Type of incident (head_turn, hand_gesture, etc.)
            timestamp: Timestamp for the snapshot
        """
        try:
            # Save the student crop
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame boundaries
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Check if crop has valid dimensions
            if x1 >= x2 or y1 >= y2:
                logger.warning(f"Invalid crop dimensions for {student_id}: [{x1}, {y1}, {x2}, {y2}]")
                return
            
            student_img = frame[y1:y2, x1:x2]
            
            # Save student crop
            student_snapshot_path = os.path.join(
                self.snapshots_dir, 
                f"{student_id}_{incident_type}_{timestamp}.jpg"
            )
            cv2.imwrite(student_snapshot_path, student_img)
            
            # Save full frame with bounding box
            full_frame = frame.copy()
            cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{incident_type} ({student_id})"
            cv2.putText(full_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            full_frame_path = os.path.join(
                self.snapshots_dir, 
                f"full_{student_id}_{incident_type}_{timestamp}.jpg"
            )
            cv2.imwrite(full_frame_path, full_frame)
            
            # Add snapshot info to student data
            self.students[student_id]['snapshots'].append({
                'path': student_snapshot_path,
                'type': incident_type,
                'timestamp': timestamp
            })
            
            # Add full frame path to student data
            self.students[student_id]['full_frame_snapshots'].append(full_frame_path)
            
            logger.info(f"Saved snapshot for {student_id} - {incident_type}")
            
        except Exception as e:
            logger.error(f"Error taking snapshot: {str(e)}")
    
    def calculate_risk_level(self, student_data):
        """Calculate risk level based on detection counts"""
        violations = student_data['violations']
        
        if violations['paper_exchange'] > 0 or violations['phone_usage'] > 1:
            return "High"
        
        total_incidents = (violations['head_turns'] + 
                         violations['hand_gestures'] + 
                         violations['phone_usage'])
        
        if total_incidents == 0:
            return "Low"
        elif total_incidents <= 3:
            return "Medium"
        else:
            return "High"
    
    def generate_pdf_report(self):
        """Generate PDF report with student activities and snapshots"""
        try:
            report_path = os.path.join(self.output_dir, "malpractice_report.pdf")
            doc = SimpleDocTemplate(
                report_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            heading_style = styles['Heading2']
            subheading_style = styles['Heading3'] 
            normal_style = styles['Normal']
            
            story = []
            
            # Report title and date
            story.append(Paragraph("Exam Malpractice Report", title_style))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Spacer(1, 20))
            
            # Summary table with malpractice counts
            story.append(Paragraph("Summary of Detections", heading_style))
            story.append(Spacer(1, 10))
            
            table_data = [['Student ID', 'Head Turns', 'Hand Gestures', 'Phone Usage', 'Paper Exchange', 'Risk Level']]
            sorted_students = sorted(self.students.items(), key=lambda x: x[0])
            
            for student_id, data in sorted_students:
                violations = data['violations']
                risk_level = self.calculate_risk_level(data)
                table_data.append([
                    student_id,
                    str(violations['head_turns']),
                    str(violations['hand_gestures']),
                    str(violations['phone_usage']),
                    str(violations['paper_exchange']),
                    risk_level
                ])
            
            # Format the summary table
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 30))
            
            # Student malpractice details with snapshots
            story.append(Paragraph("Detailed Violation Evidence", heading_style))
            story.append(Spacer(1, 20))
            
            incident_labels = {
                'head_turns': 'Head Turn',
                'hand_gestures': 'Hand Gesture',
                'phone_usage': 'Phone Usage',
                'paper_exchange': 'Paper Exchange'
            }
            
            for student_id, data in sorted_students:
                if not any(data['violations'].values()):
                    continue  # Skip students with no violations
                
                story.append(Paragraph(f"Student: {student_id}", subheading_style))
                story.append(Spacer(1, 10))
                
                # Add description of violations
                violations_text = []
                for violation_type, count in data['violations'].items():
                    if count > 0:
                        violations_text.append(f"{incident_labels.get(violation_type, violation_type)}: {count}")
                
                if violations_text:
                    story.append(Paragraph("Violations: " + ", ".join(violations_text), normal_style))
                    story.append(Spacer(1, 10))
                
                # Group snapshots by incident type
                snapshots_by_type = {}
                for snapshot in data.get('snapshots', []):
                    if isinstance(snapshot, dict) and 'type' in snapshot:
                        incident_type = snapshot['type']
                        if incident_type not in snapshots_by_type:
                            snapshots_by_type[incident_type] = []
                        
                        if os.path.exists(snapshot['path']):
                            snapshots_by_type[incident_type].append(snapshot['path'])
                
                # Display snapshots for each incident type
                for incident_type, snapshots in snapshots_by_type.items():
                    if not snapshots:
                        continue
                        
                    label = incident_labels.get(incident_type, incident_type.title())
                    story.append(Paragraph(f"{label} Incidents ({len(snapshots)} snapshots)", subheading_style))
                    story.append(Spacer(1, 8))
                    
                    # Display a subset of snapshots (max 4 to avoid huge PDFs)
                    display_snapshots = snapshots[:4]
                    
                    # Create snapshot table - format with 2 snapshots per row
                    snapshot_table_data = []
                    current_row = []
                    
                    for i, snapshot_path in enumerate(display_snapshots):
                        if i > 0 and i % 2 == 0:  # 2 images per row
                            snapshot_table_data.append(current_row)
                            current_row = []
                        
                        try:
                            img = Image(snapshot_path, width=2.5*inch, height=2*inch)
                            current_row.append(img)
                        except Exception as e:
                            logger.error(f"Error adding image {snapshot_path}: {str(e)}")
                            current_row.append(Paragraph(f"Image error: {os.path.basename(snapshot_path)}", normal_style))
                    
                    if current_row:
                        # Add empty cells to complete the row if needed
                        while len(current_row) < 2:
                            current_row.append("")
                        snapshot_table_data.append(current_row)
                    
                    if snapshot_table_data:
                        snapshot_table = Table(snapshot_table_data)
                        snapshot_table.setStyle(TableStyle([
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                        ]))
                        story.append(snapshot_table)
                        story.append(Spacer(1, 15))
                
                # Add a full frame example if available
                full_frames = data.get('full_frame_snapshots', [])
                if full_frames:
                    story.append(Paragraph("Full Frame Evidence", subheading_style))
                    story.append(Spacer(1, 8))
                    
                    # Display up to 2 full frames
                    display_frames = full_frames[:2]
                    frame_table_data = [[]]
                    
                    for i, frame_path in enumerate(display_frames):
                        try:
                            img = Image(frame_path, width=3*inch, height=2.25*inch)
                            frame_table_data[0].append(img)
                        except Exception as e:
                            logger.error(f"Error adding full frame {frame_path}: {str(e)}")
                            frame_table_data[0].append(Paragraph(f"Image error", normal_style))
                    
                    while len(frame_table_data[0]) < 2:
                        frame_table_data[0].append("")
                    
                    frame_table = Table(frame_table_data)
                    frame_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                    ]))
                    story.append(frame_table)
                
                story.append(Spacer(1, 30))
            
            # Build the PDF
            doc.build(story)
            logger.info(f"PDF report generated successfully: {report_path}")
            
            # Save student data to JSON
            self._save_student_data_json()
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _save_student_data_json(self):
        """Save student data to JSON file"""
        try:
            json_path = os.path.join(self.output_dir, "student_data.json")
            
            # Convert snapshot dictionaries to strings for JSON serialization
            json_data = {}
            for student_id, data in self.students.items():
                json_data[student_id] = data.copy()
                
                # Convert snapshot dictionaries to path strings for JSON
                if 'snapshots' in json_data[student_id]:
                    snapshot_paths = []
                    for snapshot in json_data[student_id]['snapshots']:
                        if isinstance(snapshot, dict):
                            snapshot_paths.append(snapshot['path'])
                        else:
                            snapshot_paths.append(snapshot)
                    json_data[student_id]['snapshots'] = snapshot_paths
                
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)
                
            logger.info(f"Student data saved to JSON: {json_path}")
                
        except Exception as e:
            logger.error(f"Error saving student data to JSON: {str(e)}")