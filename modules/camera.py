import cv2
import time
from typing import Optional, Tuple
import numpy as np

class Camera:
    def __init__(self):
        """Initialize the camera interface"""
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0
        self.fps = 0
        
    def start(self, device_id: int = 0) -> bool:
        """Start the camera capture
        
        Args:
            device_id: Camera device index (default: 0 for primary camera)
            
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(device_id)
            if not self.cap.isOpened():
                print("Failed to open camera")
                return False
                
            # Try to set 720p resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Get actual resolution
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera initialized at {int(width)}x{int(height)}")
            
            self.is_running = True
            self.frame_count = 0
            self.start_time = time.time()
            return True
            
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
            
    def stop(self):
        """Stop the camera capture and release resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cap = None
        
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the next frame from the camera
        
        Returns:
            numpy.ndarray: BGR image if successful, None otherwise
        """
        if not self.is_running or not self.cap:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                return None
                
            # Update FPS calculation
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            self.fps = self.frame_count / elapsed_time
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {str(e)}")
            return None
            
    def get_fps(self) -> float:
        """Get the current frames per second
        
        Returns:
            float: Current FPS
        """
        return self.fps
        
    def get_resolution(self) -> Tuple[int, int]:
        """Get the current camera resolution
        
        Returns:
            tuple: Width and height in pixels
        """
        if not self.cap:
            return (0, 0)
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop()

# Create global instance
camera = Camera()
