import unittest
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modules.speech import speech
from modules.nlp import recognize_intent

nlp = type('NLPWrapper', (), {'recognize_intent': staticmethod(recognize_intent)})()
from modules.system_control import system
from modules.system_diagnostics import diagnostics
from modules.browser_control import BrowserController

browser_control = BrowserController()
from modules.ui_visualization import initialize as init_visualization
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import config
except ModuleNotFoundError:
    config = {}

class TestJarvisIntegration(unittest.TestCase):
    """Test full integration of Jarvis components"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Initialize speech
        if hasattr(speech, 'load_models'):
            speech.load_models()
        
        # Initialize NLP
        if hasattr(nlp, 'load_models'):
            nlp.load_models()
        
        # Initialize visualization
        init_visualization()
    
    async def async_setUp(self):
        """Async setup for tests requiring browser"""
        await browser_control.initialize()
    
    async def async_tearDown(self):
        """Async cleanup for browser tests"""
        await browser_control.cleanup()
    
    def test_speech_integration(self):
        """Test speech recognition and synthesis integration"""
        # Test TTS
        speech.speak("Running integration test")
        
        # Test STT with a test audio file
        test_audio = "test_data/test_command.wav"
        if os.path.exists(test_audio):
            text = speech.transcribe(test_audio)
            self.assertIsInstance(text, str)
            self.assertTrue(len(text) > 0)
    
    def test_system_control(self):
        """Test system control functionality"""
        # Test system info
        info = system.get_system_info()
        self.assertIsInstance(info, dict)
        self.assertIn('cpu_percent', info)
        
        # Test app management
        result = system.launch_app("notepad")
        self.assertIsInstance(result, str)
        
        # Clean up
        system.close_app("notepad")
    
    def test_diagnostics(self):
        """Test system diagnostics"""
        # Run diagnostics
        results = diagnostics.run_full_diagnostics()
        self.assertIsInstance(results, dict)
        
        # Check key diagnostic components
        self.assertIn('system_health', results)
        self.assertIn('disk_health', results)
        self.assertIn('memory_health', results)
    
    async def test_browser_control(self):
        """Test browser automation"""
        await self.async_setUp()
        
        try:
            # Test search
            result = await browser_control.search("python programming")
            self.assertEqual(result["status"], "success")
            self.assertIsInstance(result["results"], list)
            
            # Test navigation
            result = await browser_control.navigate("https://www.python.org")
            self.assertEqual(result["status"], "success")
            
        finally:
            await self.async_tearDown()
    
    def test_nlp_processing(self):
        """Test natural language processing"""
        # Test intent recognition
        test_commands = [
            ("open chrome", "launch_app"),
            ("what's the system status", "system_control"),
            ("search for python tutorials", "web_search"),
            ("check my email", "email")
        ]
        
        for command, expected_intent in test_commands:
            result = nlp.recognize_intent(command)
            self.assertEqual(result["intent"], expected_intent)
    
    def test_error_handling(self):
        """Test error handling across components"""
        # Test invalid app launch
        result = system.launch_app("nonexistent_app")
        self.assertTrue("Failed" in result or "failed" in result or "Error" in result or "error" in result)
        
        # Test invalid audio file
        result = speech.transcribe("nonexistent_file.wav")
        self.assertEqual(result, "")
        
        # Test system control with invalid parameters
        result = system.set_volume(-1)
        self.assertTrue("Failed" in result or "failed" in result or "Error" in result or "error" in result)
    
    @unittest.skipIf(not config.get("eleven_labs_api_key"), 
                    "Eleven Labs API key not configured")
    def test_tts_quality(self):
        """Test text-to-speech quality with Eleven Labs"""
        test_phrases = [
            "Hello, this is a test of the text to speech system.",
            "The quick brown fox jumps over the lazy dog.",
            "Testing numbers and symbols: 1234567890!@#$%"
        ]
        
        for phrase in test_phrases:
            speech.speak(phrase)
            # Add manual verification prompt if needed
    
    def test_visualization(self):
        """Test visualization components"""
        # Test initialization
        success = init_visualization()
        self.assertTrue(success)
        
        # Note: Visual testing typically requires manual verification
        # Add automated tests for non-visual components
    
def async_test(coro):
    """Decorator for async tests"""
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper

if __name__ == '__main__':
    # Create test data directory if it doesn't exist
    os.makedirs("test_data", exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
