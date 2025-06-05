import logging
from typing import Optional, Dict, List
import threading
import time
from pathlib import Path

from modules.vision import vision_system
from modules.speech import speech

logger = logging.getLogger(__name__)

class VisionSpeechBridge:
    """Bridge between vision and speech systems for voice-controlled vision features"""
    
    def __init__(self):
        self.is_listening = False
        self.listening_thread = None
        self.command_handlers = {
            "start vision": self._handle_start_vision,
            "stop vision": self._handle_stop_vision,
            "analyze environment": self._handle_analyze_environment,
            "get depth map": self._handle_depth_map,
            "detect motion": self._handle_detect_motion,
            "detect objects": self._handle_detect_objects,
            "save analysis": self._handle_save_analysis,
            "status": self._handle_status
        }
    
    def start(self):
        """Start voice command processing"""
        if not self.is_listening:
            self.is_listening = True
            speech.load_models()  # Ensure speech models are loaded
            self.listening_thread = threading.Thread(target=self._voice_command_worker)
            self.listening_thread.daemon = True
            self.listening_thread.start()
            speech.speak("Vision voice control activated")
    
    def stop(self):
        """Stop voice command processing"""
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=2)
        speech.speak("Vision voice control deactivated")
    
    def _voice_command_worker(self):
        """Worker function for processing voice commands"""
        def process_command(text: str):
            text = text.lower().strip()
            logger.info(f"Processing command: {text}")
            
            # Check for exact matches first
            if text in self.command_handlers:
                self.command_handlers[text]()
                return
            
            # Check for commands within the text
            for command in self.command_handlers:
                if command in text:
                    self.command_handlers[command]()
                    return
            
            speech.speak("Command not recognized")
        
        speech.start_continuous_recording(callback=process_command)
        
        while self.is_listening:
            time.sleep(0.1)
        
        speech.stop_continuous_recording()
    
    def _handle_start_vision(self):
        """Handle 'start vision' command"""
        result = vision_system.start_vision_processing()
        speech.speak(result)
    
    def _handle_stop_vision(self):
        """Handle 'stop vision' command"""
        result = vision_system.stop_vision_processing()
        speech.speak(result)
    
    def _handle_analyze_environment(self):
        """Handle 'analyze environment' command"""
        analysis = vision_system.get_environment_analysis()
        
        if "error" in analysis:
            speech.speak(f"Error in analysis: {analysis['error']}")
            return
        
        # Prepare verbal report
        report = []
        
        if analysis.get("motion_detected"):
            report.append("Motion detected in the environment.")
        
        objects = analysis.get("objects_detected", [])
        if objects:
            obj_types = {}
            for obj in objects:
                obj_type = obj["type"]
                obj_types[obj_type] = obj_types.get(obj_type, 0) + 1
            
            for obj_type, count in obj_types.items():
                report.append(f"Detected {count} {obj_type}{'s' if count > 1 else ''}.")
        
        if not report:
            report.append("No significant changes or objects detected.")
        
        speech.speak(" ".join(report))
    
    def _handle_depth_map(self):
        """Handle 'get depth map' command"""
        depth_map = vision_system.get_depth_map()
        if depth_map.size == 0:
            speech.speak("Could not generate depth map")
        else:
            speech.speak("Depth map generated successfully")
            # Save depth map
            save_path = Path("vision_data/depth/latest_depth_map.png")
            vision_system.save_analysis({"depth_map": depth_map}, "voice_command")
            speech.speak("Depth map saved")
    
    def _handle_detect_motion(self):
        """Handle 'detect motion' command"""
        if vision_system._detect_motion():
            speech.speak("Motion detected in the environment")
        else:
            speech.speak("No motion detected")
    
    def _handle_detect_objects(self):
        """Handle 'detect objects' command"""
        if not vision_system.frame_buffer:
            speech.speak("No frames available for object detection")
            return
        
        objects = vision_system._detect_objects(vision_system.frame_buffer[-1])
        if not objects:
            speech.speak("No objects detected")
            return
        
        # Count object types
        obj_types = {}
        for obj in objects:
            obj_type = obj["type"]
            obj_types[obj_type] = obj_types.get(obj_type, 0) + 1
        
        # Report findings
        report = []
        for obj_type, count in obj_types.items():
            report.append(f"Detected {count} {obj_type}{'s' if count > 1 else ''}")
        
        speech.speak(". ".join(report))
    
    def _handle_save_analysis(self):
        """Handle 'save analysis' command"""
        analysis = vision_system.get_environment_analysis()
        if "error" in analysis:
            speech.speak(f"Error getting analysis: {analysis['error']}")
            return
        
        saved_files = vision_system.save_analysis(analysis, "voice_command")
        if "error" in saved_files:
            speech.speak(f"Error saving analysis: {saved_files['error']}")
            return
        
        speech.speak("Analysis results saved successfully")
    
    def _handle_status(self):
        """Handle 'status' command"""
        status = []
        
        # Vision processing status
        if vision_system.is_processing:
            status.append("Vision processing is active")
        else:
            status.append("Vision processing is inactive")
        
        # Frame buffer status
        if vision_system.frame_buffer:
            status.append(f"Currently holding {len(vision_system.frame_buffer)} frames")
        else:
            status.append("No frames in buffer")
        
        speech.speak(". ".join(status))

# Create global bridge instance
vision_speech_bridge = VisionSpeechBridge()
