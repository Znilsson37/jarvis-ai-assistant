import cv2
import numpy as np
from typing import Dict, Tuple

class VisionVisualizer:
    def __init__(self):
        """Initialize the visualization system"""
        # Colors for different types of detections (BGR format)
        self.colors = {
            'object': (0, 255, 0),    # Green
            'face': (255, 0, 0),      # Blue
            'hand': (0, 0, 255),      # Red
            'text_bg': (0, 0, 0),     # Black
            'text_fg': (255, 255, 255) # White
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_thickness = 1
        
    def draw_detection_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Draw all detection results on the frame
        
        Args:
            frame: The input frame to draw on
            results: Detection results from the AI vision system
            
        Returns:
            The frame with all detections visualized
        """
        # Create a copy of the frame to draw on
        output = frame.copy()
        
        # Draw object detections
        if 'objects' in results:
            for obj in results['objects']:
                bbox = obj['bbox']
                label = f"{obj['class']} ({obj['confidence']:.2f})"
                self._draw_bbox(output, bbox, label, self.colors['object'])
                
        # Draw facial expressions
        if 'expressions' in results:
            for face in results['expressions']:
                bbox = face['box']
                # Get the dominant emotion
                emotion = max(face['emotions'].items(), key=lambda x: x[1])[0]
                label = f"{emotion} ({face['emotions'][emotion]:.2f})"
                self._draw_bbox(output, bbox, label, self.colors['face'])
                
        # Draw hand gestures and landmarks
        if 'gestures' in results:
            for gesture in results['gestures']:
                self._draw_hand_landmarks(output, gesture['landmarks'], gesture['gesture'])
                
        return output
        
    def add_performance_metrics(self, frame: np.ndarray, metrics: Dict) -> np.ndarray:
        """Add performance metrics overlay to the frame
        
        Args:
            frame: The input frame
            metrics: Dictionary containing performance metrics
            
        Returns:
            Frame with metrics overlay
        """
        # Create copy of frame
        output = frame.copy()
        
        # Format metrics text
        fps_text = f"FPS: {metrics['fps']:.1f}"
        proc_time_text = f"Processing Time: {metrics['avg_processing_time']*1000:.1f}ms"
        
        # Get text sizes
        fps_size = cv2.getTextSize(fps_text, self.font, self.font_scale, self.font_thickness)[0]
        proc_size = cv2.getTextSize(proc_time_text, self.font, self.font_scale, self.font_thickness)[0]
        
        # Calculate positions
        margin = 10
        metrics_height = fps_size[1] + proc_size[1] + 3 * margin
        
        # Draw background rectangle
        cv2.rectangle(output, 
                     (0, 0),
                     (max(fps_size[0], proc_size[0]) + 2 * margin, metrics_height),
                     self.colors['text_bg'],
                     -1)
        
        # Draw metrics text
        cv2.putText(output,
                   fps_text,
                   (margin, margin + fps_size[1]),
                   self.font,
                   self.font_scale,
                   self.colors['text_fg'],
                   self.font_thickness)
                   
        cv2.putText(output,
                   proc_time_text,
                   (margin, 2 * margin + fps_size[1] + proc_size[1]),
                   self.font,
                   self.font_scale,
                   self.colors['text_fg'],
                   self.font_thickness)
                   
        return output
        
    def _draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  label: str, color: Tuple[int, int, int]):
        """Draw a bounding box with label on the frame"""
        x1, y1, x2, y2 = bbox
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get text size
        text_size = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)[0]
        
        # Draw text background
        cv2.rectangle(frame,
                     (x1, y1 - text_size[1] - 4),
                     (x1 + text_size[0], y1),
                     color,
                     -1)
                     
        # Draw text
        cv2.putText(frame,
                   label,
                   (x1, y1 - 2),
                   self.font,
                   self.font_scale,
                   self.colors['text_fg'],
                   self.font_thickness)
                   
    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks: list, gesture: str):
        """Draw hand landmarks and gesture label on the frame"""
        h, w = frame.shape[:2]
        
        # Draw landmarks
        for i, landmark in enumerate(landmarks):
            x = int(landmark['x'] * w)
            y = int(landmark['y'] * h)
            
            # Draw point
            cv2.circle(frame, (x, y), 4, self.colors['hand'], -1)
            
            # Connect landmarks with lines to show hand structure
            if i > 0:
                # Connect with previous landmark in same finger
                prev_x = int(landmarks[i-1]['x'] * w)
                prev_y = int(landmarks[i-1]['y'] * h)
                cv2.line(frame, (prev_x, prev_y), (x, y), self.colors['hand'], 2)
                
        # Find centroid of landmarks for label placement
        cx = int(np.mean([lm['x'] for lm in landmarks]) * w)
        cy = int(np.mean([lm['y'] for lm in landmarks]) * h)
        
        # Draw gesture label
        text_size = cv2.getTextSize(gesture, self.font, self.font_scale, self.font_thickness)[0]
        
        cv2.rectangle(frame,
                     (cx - text_size[0]//2 - 2, cy - text_size[1] - 2),
                     (cx + text_size[0]//2 + 2, cy),
                     self.colors['hand'],
                     -1)
                     
        cv2.putText(frame,
                   gesture,
                   (cx - text_size[0]//2, cy - 2),
                   self.font,
                   self.font_scale,
                   self.colors['text_fg'],
                   self.font_thickness)

# Create global instance
vision_visualizer = VisionVisualizer()
