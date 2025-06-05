# Import required packages
import cv2
import numpy as np
import time
from typing import Dict, List, Optional

# Initialize availability flags
YOLO_AVAILABLE = False
FER_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
YOLO = None
FER = None
mp = None

# Import AI models with error handling
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: YOLOv8 not available. Object detection will be disabled.")

try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    print("Warning: FER not available. Facial expression analysis will be disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("Warning: MediaPipe not available. Gesture recognition will be disabled.")

class AIVisionSystem:
    def __init__(self):
        """Initialize the AI Vision System with all required models and processors"""
        global YOLO_AVAILABLE, FER_AVAILABLE, MEDIAPIPE_AVAILABLE
        
        # Object Detection
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n.pt')  # Using nano model for speed
                print("YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"Error loading YOLOv8 model: {str(e)}")
                YOLO_AVAILABLE = False
        
        # Facial Expression Analysis
        self.emotion_detector = None
        if FER_AVAILABLE:
            try:
                self.emotion_detector = FER(mtcnn=True)
                print("FER model loaded successfully")
            except Exception as e:
                print(f"Error loading FER model: {str(e)}")
                FER_AVAILABLE = False
        
        # Gesture Recognition
        self.mp_hands = None
        self.hands = None
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                print("MediaPipe Hands model loaded successfully")
            except Exception as e:
                print(f"Error loading MediaPipe Hands: {str(e)}")
                MEDIAPIPE_AVAILABLE = False
        
        # Performance metrics
        self.fps = 0
        self.processing_time = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Latest results storage
        self.latest_results = None
        self.is_processing = False
        
    def start_processing(self):
        """Start the vision processing system"""
        self.is_processing = True
        self.start_time = time.time()
        self.frame_count = 0
        print("AI Vision System initialized and ready for processing")
        
    def stop_processing(self):
        """Stop the vision processing system and cleanup"""
        self.is_processing = False
        try:
            if self.hands and hasattr(self.hands, '_graph') and self.hands._graph is not None:
                self.hands.close()
        except Exception as e:
            print(f"Warning: Error during hands cleanup: {str(e)}")
        cv2.destroyAllWindows()
        print("AI Vision System stopped")
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame through all vision components"""
        global YOLO_AVAILABLE, FER_AVAILABLE, MEDIAPIPE_AVAILABLE
        
        if not self.is_processing:
            return None
            
        start_time = time.time()
        
        # Update performance metrics
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        self.fps = self.frame_count / elapsed_time
        
        # Create results dictionary
        results = {
            'timestamp': time.time(),
            'objects': [],
            'expressions': [],
            'gestures': [],
            'scene_3d': {'objects': []}
        }
        
        try:
            # 1. Object Detection
            if YOLO_AVAILABLE and self.yolo_model:
                yolo_results = self.yolo_model(frame, verbose=False)[0]
                for detection in yolo_results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = detection
                    results['objects'].append({
                        'class': yolo_results.names[int(cls)],
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
                
            # 2. Facial Expression Analysis
            if FER_AVAILABLE and self.emotion_detector:
                emotions = self.emotion_detector.detect_emotions(frame)
                for face in emotions:
                    results['expressions'].append({
                        'box': [int(x) for x in face['box']],
                        'emotions': face['emotions']
                    })
                
            # 3. Gesture Recognition and 3D Scene Understanding
            if MEDIAPIPE_AVAILABLE and self.hands:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    hand_results = self.hands.process(frame_rgb)
                    
                    if hand_results and hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # Convert landmarks to list of coordinates
                            landmarks = []
                            for landmark in hand_landmarks.landmark:
                                landmarks.append({
                                    'x': landmark.x,
                                    'y': landmark.y,
                                    'z': landmark.z
                                })
                            
                            # Basic gesture recognition based on landmark positions
                            gesture = self._recognize_gesture(landmarks)
                            results['gestures'].append({
                                'landmarks': landmarks,
                                'gesture': gesture
                            })
                            
                            # Add to 3D scene understanding
                            results['scene_3d']['objects'].append({
                                'type': 'hand',
                                'vertices': landmarks  # Reuse the same landmarks
                            })
                except Exception as e:
                    print(f"Error processing hand gestures: {str(e)}")
                    
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            
        # Update processing metrics
        self.processing_time = time.time() - start_time
        self.latest_results = results
        
        return results
        
    def get_latest_results(self) -> Optional[Dict]:
        """Get the latest processing results"""
        return self.latest_results
        
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            'fps': self.fps,
            'avg_processing_time': self.processing_time
        }
        
    def _recognize_gesture(self, landmarks: List[Dict]) -> str:
        """Basic gesture recognition based on hand landmarks"""
        # Get thumb tip and index finger tip positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger
        distance = np.sqrt(
            (thumb_tip['x'] - index_tip['x'])**2 +
            (thumb_tip['y'] - index_tip['y'])**2
        )
        
        # Basic gesture recognition
        if distance < 0.1:  # Close together
            return "pinch"
        else:
            # Check if fingers are extended
            if landmarks[8]['y'] < landmarks[5]['y']:  # Index up
                if landmarks[12]['y'] < landmarks[9]['y']:  # Middle up
                    if landmarks[16]['y'] < landmarks[13]['y']:  # Ring up
                        if landmarks[20]['y'] < landmarks[17]['y']:  # Pinky up
                            return "open_palm"
                        return "three_fingers"
                    return "peace"
                return "pointing"
            return "fist"

# Create global instance
ai_vision_system = AIVisionSystem()
