"""Test suite for Vision-NLP integration"""
import unittest
import numpy as np
import time
from modules.vision_nlp_bridge import handle_vision_intent
from modules.nlp import recognize_intent
from modules.vision import vision_system
from modules.ai_vision import ai_vision_system

class TestVisionNLPBridge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up vision systems"""
        # Start vision processing
        vision_system.start_vision_processing()
        ai_vision_system.start_processing()
        
        # Wait for systems to initialize and capture first frame
        time.sleep(2)  # Give time for camera initialization and first frame capture
        
        # Ensure we have frames and mock results
        if not vision_system.frame_buffer:
            vision_system.frame_buffer.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Mock AI vision results for testing
        ai_vision_system._latest_results = {
            "objects": [
                {"class": "test_object", "confidence": 0.95, "bbox": [100, 100, 200, 200]}
            ],
            "timestamp": time.time()
        }

    def test_analyze_environment_intent(self):
        """Test environment analysis intent recognition and handling"""
        # Test intent recognition
        command = "analyze the environment"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "analyze_environment")

        # Test intent handling
        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("environment_analysis", result)
        self.assertIn("description", result)

    def test_detect_objects_intent(self):
        """Test object detection intent recognition and handling"""
        command = "what objects do you see"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "detect_objects")

        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("objects", result)
        self.assertIn("description", result)

    def test_track_motion_intent(self):
        """Test motion tracking intent recognition and handling"""
        command = "detect any motion"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "track_motion")

        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("motion_detected", result)
        self.assertIn("description", result)

    def test_generate_wireframe_intent(self):
        """Test wireframe generation intent recognition and handling"""
        command = "generate a wireframe view"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "generate_wireframe")

        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("wireframe", result)

    def test_depth_analysis_intent(self):
        """Test depth analysis intent recognition and handling"""
        command = "analyze depth"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "depth_analysis")

        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("depth_map", result)

    def test_spectral_analysis_intent(self):
        """Test spectral analysis intent recognition and handling"""
        command = "run spectral analysis"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "spectral_analysis")

        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("spectral_analysis", result)

    def test_save_analysis_intent(self):
        """Test analysis saving intent recognition and handling"""
        command = "save this analysis"
        intent_result = recognize_intent(command)
        self.assertEqual(intent_result["intent"], "save_analysis")

        result = handle_vision_intent(intent_result)
        self.assertEqual(result["status"], "success")
        self.assertIn("saved_files", result)

    def test_error_handling(self):
        """Test error handling for invalid intents"""
        result = handle_vision_intent({"intent": "invalid_intent"})
        self.assertEqual(result["status"], "error")
        self.assertIn("message", result)

    @classmethod
    def tearDownClass(cls):
        """Clean up vision systems"""
        vision_system.stop_vision_processing()
        ai_vision_system.stop_processing()

if __name__ == '__main__':
    unittest.main()
