"""Comprehensive test suite for Chatterbox TTS integration"""

import unittest
import os
import tempfile
import shutil
import time
from unittest.mock import patch, MagicMock
import numpy as np
from modules.speech import speech_processor

class TestChatterboxTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.speech = speech_processor
        cls.test_dir = tempfile.mkdtemp()
        cls.test_audio = os.path.join(cls.test_dir, "test_reference.wav")
        cls.test_text = "This is a test of the text to speech system."
        
        # Create test audio file
        try:
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
        """Reset TTS engine before each test"""
        self.speech.set_tts_engine("chatterbox")

    def test_basic_tts_functionality(self):
        """Test basic text-to-speech functionality"""
        # Test with simple text
        output_path = self.speech.speak(self.test_text)
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        self.assertTrue(os.path.getsize(output_path) > 0)

        # Test with longer text
        long_text = " ".join([self.test_text] * 5)
        output_path = self.speech.speak(long_text)
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))

        # Test with special characters
        special_text = "Test with special chars: !@#$%^&*()_+"
        output_path = self.speech.speak(special_text)
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))

    def test_voice_cloning(self):
        """Test voice cloning functionality"""
        # Test with valid reference audio
        output_path = self.speech.speak(
            self.test_text,
            voice_path=self.test_audio
        )
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))

        # Test with nonexistent reference audio
        output_path = self.speech.speak(
            self.test_text,
            voice_path="nonexistent.wav"
        )
        self.assertIsNotNone(output_path)  # Should fallback to default voice

    def test_batch_processing(self):
        """Test batch speech generation"""
        texts = [
            "First test message.",
            "Second test message.",
            "Third test message with longer content for testing."
        ]
        
        # Test basic batch processing
        output_paths = self.speech.batch_speak(texts)
        self.assertEqual(len(output_paths), len(texts))
        for path in output_paths:
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.getsize(path) > 0)

        # Test batch processing with voice cloning
        output_paths = self.speech.batch_speak(texts, voice_path=self.test_audio)
        self.assertEqual(len(output_paths), len(texts))
        for path in output_paths:
            self.assertTrue(os.path.exists(path))

    def test_parameter_control(self):
        """Test TTS parameter controls"""
        # Test different exaggeration levels
        for exaggeration in [0.0, 0.5, 1.0]:
            output_path = self.speech.speak(
                self.test_text,
                exaggeration=exaggeration
            )
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))

        # Test different cfg_weight values
        for cfg_weight in [0.0, 0.5, 1.0]:
            output_path = self.speech.speak(
                self.test_text,
                cfg_weight=cfg_weight
            )
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))

    def test_error_handling(self):
        """Test error handling"""
        # Test with empty text
        output_path = self.speech.speak("")
        self.assertIsNone(output_path)

        # Test with None input
        output_path = self.speech.speak(None)
        self.assertIsNone(output_path)

        # Test with very long text
        very_long_text = "test " * 1000
        output_path = self.speech.speak(very_long_text)
        self.assertIsNotNone(output_path)

    def test_performance(self):
        """Test TTS performance"""
        # Test processing time for different text lengths
        texts = [
            "Short text.",
            "Medium length text with some more words.",
            "Long text " * 20
        ]

        for text in texts:
            start_time = time.time()
            output_path = self.speech.speak(text)
            processing_time = time.time() - start_time
            
            self.assertIsNotNone(output_path)
            self.assertTrue(os.path.exists(output_path))
            print(f"Processing time for {len(text)} chars: {processing_time:.2f}s")

    def test_resource_cleanup(self):
        """Test resource cleanup"""
        # Generate multiple audio files
        outputs = []
        for i in range(3):
            path = self.speech.speak(f"Test {i}")
            if path:
                outputs.append(path)

        # Verify files exist
        for path in outputs:
            self.assertTrue(os.path.exists(path))

        # Clean up files
        for path in outputs:
            try:
                os.remove(path)
                self.assertFalse(os.path.exists(path))
            except Exception as e:
                self.fail(f"Failed to clean up {path}: {e}")

    def test_concurrent_requests(self):
        """Test handling of concurrent TTS requests"""
        import threading
        
        def generate_speech(text, results):
            output_path = self.speech.speak(text)
            results.append(output_path)

        texts = [f"Concurrent test {i}" for i in range(3)]
        threads = []
        results = []

        # Start concurrent requests
        for text in texts:
            thread = threading.Thread(target=generate_speech, args=(text, results))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify results
        self.assertEqual(len(results), len(texts))
        for path in results:
            self.assertIsNotNone(path)
            self.assertTrue(os.path.exists(path))

if __name__ == '__main__':
    unittest.main(verbosity=2)
