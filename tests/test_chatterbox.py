"""Tests for Chatterbox TTS module"""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import torch
from modules.chatterbox import ChatterboxTTSWrapper

class TestChatterboxTTS(unittest.TestCase):
    """Test suite for Chatterbox TTS functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_text = "Hello, this is a test."
        cls.test_output_dir = "test_audio_output"
        os.makedirs(cls.test_output_dir, exist_ok=True)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove test output directory
        if os.path.exists(cls.test_output_dir):
            for file in os.listdir(cls.test_output_dir):
                os.remove(os.path.join(cls.test_output_dir, file))
            os.rmdir(cls.test_output_dir)
    
    def setUp(self):
        """Set up each test"""
        self.tts = ChatterboxTTSWrapper(device="cpu")  # Use CPU for testing
    
    @patch('chatterbox.tts.ChatterboxTTS.generate')
    def test_generate_speech(self, mock_generate):
        """Test basic speech generation"""
        # Setup mock
        mock_generate.return_value = torch.zeros((1, 16000))  # 1 second of silence
        
        # Test generation
        output_path = os.path.join(self.test_output_dir, "test_speech.wav")
        result = self.tts.generate_speech(self.test_text, output_path)
        
        # Verify
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        mock_generate.assert_called_once()
    
    @patch('chatterbox.tts.ChatterboxTTS.generate')
    def test_voice_cloning(self, mock_generate):
        """Test voice cloning functionality"""
        # Setup mock
        mock_generate.return_value = torch.zeros((1, 16000))
        
        # Test cloning
        reference_audio = "test_reference.wav"
        output_path = os.path.join(self.test_output_dir, "cloned_speech.wav")
        result = self.tts.clone_voice(reference_audio, self.test_text, output_path)
        
        # Verify
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        mock_generate.assert_called_once_with(
            self.test_text,
            audio_prompt_path=reference_audio,
            exaggeration=0.5,
            cfg_weight=0.5
        )
    
    @patch('chatterbox.tts.ChatterboxTTS.generate')
    def test_batch_generation(self, mock_generate):
        """Test batch speech generation"""
        # Setup mock
        mock_generate.return_value = torch.zeros((1, 16000))
        
        # Test batch generation
        texts = ["First test", "Second test", "Third test"]
        results = self.tts.batch_generate(texts, self.test_output_dir)
        
        # Verify
        self.assertEqual(len(results), len(texts))
        for path in results:
            self.assertTrue(os.path.exists(path))
        self.assertEqual(mock_generate.call_count, len(texts))
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with empty text
        result = self.tts.generate_speech("")
        self.assertIsNone(result)
        
        # Test with invalid reference audio
        result = self.tts.clone_voice("nonexistent.wav", self.test_text)
        self.assertIsNone(result)
    
    @patch('chatterbox.tts.ChatterboxTTS.generate')
    def test_expressiveness_control(self, mock_generate):
        """Test expressiveness control parameters"""
        # Setup mock
        mock_generate.return_value = torch.zeros((1, 16000))
        
        # Test with different expressiveness settings
        output_path = os.path.join(self.test_output_dir, "expressive_speech.wav")
        result = self.tts.generate_speech(
            self.test_text,
            output_path,
            exaggeration=0.8,
            cfg_weight=0.3
        )
        
        # Verify
        self.assertIsNotNone(result)
        mock_generate.assert_called_once_with(
            self.test_text,
            audio_prompt_path=None,
            exaggeration=0.8,
            cfg_weight=0.3
        )
    
    def test_output_format(self):
        """Test output audio format"""
        output_path = os.path.join(self.test_output_dir, "test_format.wav")
        with patch('chatterbox.tts.ChatterboxTTS.generate') as mock_generate:
            # Setup mock to return specific format
            mock_generate.return_value = torch.zeros((1, self.tts.sample_rate))
            
            # Generate speech
            result = self.tts.generate_speech(self.test_text, output_path)
            
            # Verify format
            self.assertTrue(os.path.exists(result))
            self.assertTrue(result.endswith('.wav'))
            
            # Verify sample rate
            import torchaudio
            waveform, sample_rate = torchaudio.load(result)
            self.assertEqual(sample_rate, self.tts.sample_rate)

if __name__ == '__main__':
    unittest.main()
