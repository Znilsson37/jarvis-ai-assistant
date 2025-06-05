import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modules.speech import SpeechEngine
from modules.eleven_labs import ElevenLabsTTS

class TestElevenLabsIntegration(unittest.TestCase):
    """Test suite for ElevenLabs TTS integration"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.speech_engine = SpeechEngine()
        
    def setUp(self):
        """Set up each test"""
        self.test_text = "This is a test message"
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    @patch('modules.speech.ElevenLabsTTS')
    def test_tts_initialization(self, mock_tts):
        """Test TTS initialization"""
        # Configure mock
        mock_instance = mock_tts.return_value
        mock_instance.test_connection.return_value = True
        mock_instance.test_connection = MagicMock(wraps=mock_instance.test_connection)
        
        # Initialize speech engine
        engine = SpeechEngine()
        engine_type, tts = engine._initialize_tts()
        
        # Verify initialization
        self.assertEqual(engine_type, "eleven_labs")
        self.assertIsNotNone(tts)
        mock_instance.test_connection.assert_called()
        
    @patch('modules.speech.ElevenLabsTTS')
    def test_tts_synthesis(self, mock_tts):
        """Test text-to-speech synthesis"""
        # Configure mock
        mock_instance = mock_tts.return_value
        mock_instance.test_connection.return_value = True
        test_audio_path = os.path.join(self.temp_dir, "test_audio.mp3")
        mock_instance.synthesize.return_value = test_audio_path
        
        # Initialize speech engine with mock
        engine = SpeechEngine()
        engine.tts_engine = ("eleven_labs", mock_instance)
        
        # Test speech synthesis
        engine.speak(self.test_text)
        
        # Verify synthesis call
        mock_instance.synthesize.assert_called_once_with(self.test_text)
        
    @patch('modules.speech.ElevenLabsTTS')
    def test_error_handling(self, mock_tts):
        """Test error handling during synthesis"""
        # Configure mock to raise exception
        mock_instance = mock_tts.return_value
        mock_instance.test_connection.return_value = True
        mock_instance.synthesize.side_effect = Exception("TTS Error")
        
        # Initialize speech engine with mock
        engine = SpeechEngine()
        engine.tts_engine = ("eleven_labs", mock_instance)
        
        # Test error handling
        with self.assertLogs(level='ERROR') as log:
            engine.speak(self.test_text)
            self.assertIn("Error in text-to-speech", log.output[0])
            
    @patch('modules.speech.ElevenLabsTTS')
    def test_fallback_behavior(self, mock_tts):
        """Test fallback behavior when TTS fails"""
        # Configure mock to fail connection test
        mock_instance = mock_tts.return_value
        mock_instance.test_connection.return_value = False
        mock_instance.test_connection = MagicMock(wraps=mock_instance.test_connection)
        
        # Initialize speech engine
        engine = SpeechEngine()
        engine_type, tts = engine._initialize_tts()
        
        # Verify fallback to next available TTS method
        self.assertNotEqual(engine_type, "eleven_labs", "Fallback did not occur when ElevenLabsTTS connection failed")
        
    @patch('modules.speech.ElevenLabsTTS')
    def test_performance(self, mock_tts):
        """Test TTS performance with varying text lengths"""
        # Configure mock
        mock_instance = mock_tts.return_value
        mock_instance.test_connection.return_value = True
        test_audio_path = os.path.join(self.temp_dir, "test_audio.mp3")
        mock_instance.synthesize.return_value = test_audio_path
        
        # Initialize speech engine
        engine = SpeechEngine()
        engine.tts_engine = ("eleven_labs", mock_instance)
        
        # Test with different text lengths
        test_cases = [
            "Short text",
            "Medium length text with multiple words",
            "Long text " * 20  # Approximately 200 characters
        ]
        
        import time
        for text in test_cases:
            start_time = time.time()
            engine.speak(text)
            end_time = time.time()
            
            # Verify reasonable processing time (adjust threshold as needed)
            processing_time = end_time - start_time
            self.assertLess(processing_time, 5.0, 
                          f"Processing time {processing_time}s too long for text length {len(text)}")
            
    @patch('modules.speech.ElevenLabsTTS')
    def test_concurrent_requests(self, mock_tts):
        """Test handling of concurrent TTS requests"""
        # Configure mock
        mock_instance = mock_tts.return_value
        mock_instance.test_connection.return_value = True
        test_audio_path = os.path.join(self.temp_dir, "test_audio.mp3")
        mock_instance.synthesize.return_value = test_audio_path
        
        # Initialize speech engine
        engine = SpeechEngine()
        engine.tts_engine = ("eleven_labs", mock_instance)
        
        # Test concurrent requests
        import threading
        threads = []
        for i in range(3):
            thread = threading.Thread(target=engine.speak, 
                                   args=(f"Test message {i}",))
            threads.append(thread)
            
        # Start all threads
        for thread in threads:
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Verify all requests were processed
        self.assertEqual(mock_instance.synthesize.call_count, 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
