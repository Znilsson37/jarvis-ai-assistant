"""Integration tests for Chatterbox TTS with the full Jarvis system"""

import unittest
import os
import tempfile
import shutil
import time
from unittest.mock import patch, MagicMock
import numpy as np
from modules.speech import speech_processor
from modules.brain import Brain
from modules.chatterbox import ChatterboxTTSWrapper

class TestChatterboxIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_dir = tempfile.mkdtemp()
        cls.test_audio = os.path.join(cls.test_dir, "test_reference.wav")
        cls.test_text = "This is a test of the text to speech system."
        
        # Create test audio file
        try:
            import torch
            import torchaudio
            sample_rate = 22050
            duration = 2  # seconds
            t = torch.linspace(0, duration, int(sample_rate * duration))
            wave = torch.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            torchaudio.save(cls.test_audio, wave.unsqueeze(0), sample_rate)
        except ImportError:
            # Create empty wav file if torchaudio not available
            with open(cls.test_audio, 'wb') as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        shutil.rmtree(cls.test_dir)

    def setUp(self):
        """Initialize components for each test"""
        self.brain = Brain()
        self.speech = speech_processor
        self.speech.set_tts_engine("chatterbox")

    def test_brain_speech_integration(self):
        """Test integration between Brain and Speech modules"""
        # Test brain's ability to trigger speech
        response = "Testing brain to speech integration"
        audio_path = self.brain.speech.speak(response)
        
        self.assertIsNotNone(audio_path)
        self.assertTrue(os.path.exists(audio_path))
        self.assertTrue(os.path.getsize(audio_path) > 0)

    def test_speech_config_integration(self):
        """Test speech configuration integration"""
        # Verify config is properly loaded
        self.assertIsNotNone(self.speech.config)
        
        # Check TTS configuration
        tts_config = self.speech.config.get("tts", {})
        self.assertIsNotNone(tts_config)
        
        # Verify Chatterbox config
        chatterbox_config = tts_config.get("chatterbox", {})
        self.assertIsNotNone(chatterbox_config)
        
        # Test config values affect TTS behavior
        exaggeration = chatterbox_config.get("default_exaggeration", 0.5)
        cfg_weight = chatterbox_config.get("default_cfg_weight", 0.5)
        
        output_path = self.speech.speak(
            self.test_text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight
        )
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_error_propagation(self):
        """Test error handling and propagation"""
        # Test with invalid config
        with patch.dict(self.speech.config, {"tts": {}}):
            output_path = self.speech.speak(self.test_text)
            self.assertIsNotNone(output_path)  # Should use defaults
        
        # Test with unavailable TTS engine
        with patch.object(self.speech.chatterbox_tts, 'available', False):
            output_path = self.speech.speak(self.test_text)
            self.assertIsNone(output_path)

    def test_concurrent_system_integration(self):
        """Test TTS behavior during concurrent system operations"""
        import threading
        
        def background_task():
            """Simulate background system operations"""
            time.sleep(0.1)  # Simulate work
        
        def tts_task(text, results):
            """TTS generation task"""
            output_path = self.speech.speak(text)
            results.append(output_path)
        
        # Create threads for background tasks and TTS
        bg_thread = threading.Thread(target=background_task)
        results = []
        tts_thread = threading.Thread(
            target=tts_task,
            args=(self.test_text, results)
        )
        
        # Start concurrent operations
        bg_thread.start()
        tts_thread.start()
        
        # Wait for completion
        bg_thread.join()
        tts_thread.join()
        
        # Verify TTS completed successfully
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0])
        self.assertTrue(os.path.exists(results[0]))

    def test_resource_management(self):
        """Test system resource management with TTS"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Generate multiple audio files
        outputs = []
        for i in range(3):
            path = self.speech.speak(f"Test {i}")
            if path:
                outputs.append(path)
                
        # Force garbage collection
        gc.collect()
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_diff = final_memory - initial_memory
        
        # Clean up
        for path in outputs:
            try:
                os.remove(path)
            except:
                pass
        
        # Memory increase should be reasonable
        # Allow up to 100MB increase (generous limit)
        self.assertLess(memory_diff, 100 * 1024 * 1024)

    def test_system_shutdown(self):
        """Test proper TTS cleanup during system shutdown"""
        # Generate some audio files
        outputs = []
        for i in range(2):
            path = self.speech.speak(f"Test {i}")
            if path:
                outputs.append(path)
        
        # Simulate system shutdown
        try:
            self.brain.cleanup()
        except:
            pass
        
        # Verify files are cleaned up
        for path in outputs:
            try:
                os.remove(path)
            except:
                pass  # Files may already be cleaned up

    def test_performance_impact(self):
        """Test TTS impact on system performance"""
        import time
        
        def measure_operation_time(operation):
            start_time = time.time()
            result = operation()
            return time.time() - start_time, result
        
        # Measure basic operation time
        time_basic, output_basic = measure_operation_time(
            lambda: self.speech.speak("Quick test")
        )
        
        # Measure time with longer text
        time_long, output_long = measure_operation_time(
            lambda: self.speech.speak("Long test " * 20)
        )
        
        # Longer text should take proportionally longer
        # but not exponentially longer
        self.assertGreater(time_long, time_basic)
        self.assertLess(time_long, time_basic * 40)  # Factor of 40 is generous
        
        # Clean up
        for path in [output_basic, output_long]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

if __name__ == '__main__':
    unittest.main(verbosity=2)
