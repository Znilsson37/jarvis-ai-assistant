import pytest
import asyncio
import threading
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
from pathlib import Path

from modules.vision_speech_bridge import VisionSpeechBridge
from modules.vision import vision_system
from modules.speech import speech

class TestVisionSpeechBridge:
    @classmethod
    def setup_class(cls):
        """Set up test environment"""
        cls.bridge = VisionSpeechBridge()
        
        # Create mock vision data
        cls.mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cls.mock_depth_map = np.zeros((480, 640), dtype=np.uint8)
        cls.mock_analysis = {
            "wireframe": np.zeros((480, 640, 3), dtype=np.uint8),
            "depth_map": cls.mock_depth_map,
            "motion_detected": True,
            "objects_detected": [
                {"type": "face", "confidence": 0.9, "bbox": [100, 100, 50, 50]}
            ]
        }
    
    def setup_method(self):
        """Set up each test method"""
        # Reset vision system state
        vision_system.is_processing = False
        vision_system.frame_buffer = []
        
        # Mock vision system methods
        vision_system.start_vision_processing = MagicMock(return_value="Started vision processing")
        vision_system.stop_vision_processing = MagicMock(return_value="Stopped vision processing")
        vision_system.get_environment_analysis = MagicMock(return_value=self.mock_analysis)
        vision_system.get_depth_map = MagicMock(return_value=self.mock_depth_map)
        vision_system._detect_motion = MagicMock(return_value=True)
        vision_system._detect_objects = MagicMock(return_value=[
            {"type": "face", "confidence": 0.9, "bbox": [100, 100, 50, 50]}
        ])
        vision_system.save_analysis = MagicMock(return_value={
            "wireframe": "path/to/wireframe.png",
            "depth_map": "path/to/depth_map.png"
        })
        
        # Mock speech methods
        speech.speak = MagicMock()
        speech.start_continuous_recording = MagicMock()
        speech.stop_continuous_recording = MagicMock()
        
        # Reset bridge state
        self.bridge.is_listening = False
        self.bridge.listening_thread = None
    
    def test_initialization(self):
        """Test bridge initialization"""
        assert not self.bridge.is_listening
        assert self.bridge.listening_thread is None
        assert len(self.bridge.command_handlers) > 0
    
    def test_start_stop(self):
        """Test starting and stopping voice command processing"""
        # Start bridge
        self.bridge.start()
        assert self.bridge.is_listening
        assert self.bridge.listening_thread is not None
        speech.speak.assert_called_with("Vision voice control activated")
        
        # Stop bridge
        self.bridge.stop()
        assert not self.bridge.is_listening
        speech.speak.assert_called_with("Vision voice control deactivated")
    
    @pytest.mark.parametrize("command,expected_calls", [
        ("start vision", ["Started vision processing"]),
        ("stop vision", ["Stopped vision processing"]),
        ("analyze environment", ["Motion detected in the environment. Detected 1 face."]),
        ("get depth map", ["Depth map generated successfully", "Depth map saved"]),
        ("detect motion", ["Motion detected in the environment"]),
        ("detect objects", ["Detected 1 face"]),
        ("save analysis", ["Analysis results saved successfully"]),
        ("status", ["Vision processing is inactive. No frames in buffer"]),
    ])
    def test_command_handling(self, command, expected_calls):
        """Test handling of various voice commands"""
        # Reset mock
        speech.speak.reset_mock()
        
        # Process command
        if command in self.bridge.command_handlers:
            # Add frame buffer for object detection
            if command == "detect objects":
                vision_system.frame_buffer = [self.mock_frame]
            
            self.bridge.command_handlers[command]()
            
            # Verify each expected call was made
            for expected in expected_calls:
                assert call(expected) in speech.speak.call_args_list, \
                    f"Expected call with '{expected}' not found in {speech.speak.call_args_list}"
    
    def test_error_handling(self):
        """Test error handling in command processing"""
        # Reset mock
        speech.speak.reset_mock()
        
        # Test analysis error
        vision_system.get_environment_analysis = MagicMock(return_value={"error": "Analysis failed"})
        self.bridge.command_handlers["analyze environment"]()
        speech.speak.assert_called_with("Error in analysis: Analysis failed")
        speech.speak.reset_mock()
        
        # Test depth map error
        vision_system.get_depth_map = MagicMock(return_value=np.array([]))
        self.bridge.command_handlers["get depth map"]()
        speech.speak.assert_called_with("Could not generate depth map")
        speech.speak.reset_mock()
        
        # Test save analysis error
        vision_system.get_environment_analysis = MagicMock(return_value={"error": "Save failed"})
        self.bridge.command_handlers["save analysis"]()
        speech.speak.assert_called_with("Error getting analysis: Save failed")
    
    def test_continuous_command_processing(self):
        """Test continuous voice command processing"""
        # Reset mock
        speech.speak.reset_mock()
        
        # Start bridge with mock callback
        def mock_start_recording(callback=None):
            # Simulate voice commands
            test_commands = [
                "start vision",
                "analyze environment",
                "stop vision",
                "invalid command"
            ]
            
            # Add frame buffer for commands that need it
            vision_system.frame_buffer = [self.mock_frame]
            
            # Process each command
            for command in test_commands:
                if callback:
                    callback(command)
                    time.sleep(0.1)
        
        # Replace the start_continuous_recording with our mock
        speech.start_continuous_recording = mock_start_recording
        
        # Start the bridge
        self.bridge.start()
        
        # Verify activation message
        speech.speak.assert_called_with("Vision voice control activated")
        
        # Wait a bit to allow commands to be processed
        time.sleep(1)
        
        # Verify command responses
        assert call("Started vision processing") in speech.speak.call_args_list
        assert call("Motion detected in the environment. Detected 1 face.") in speech.speak.call_args_list
        assert call("Stopped vision processing") in speech.speak.call_args_list
        assert call("Command not recognized") in speech.speak.call_args_list
    
    def teardown_method(self):
        """Clean up after each test"""
        self.bridge.stop()
        speech.speak.reset_mock()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        vision_system.cleanup()
        speech.cleanup()

if __name__ == "__main__":
    pytest.main([__file__])
