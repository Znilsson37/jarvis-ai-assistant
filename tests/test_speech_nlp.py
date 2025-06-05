import unittest
import sys
import os
import asyncio
import wave
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from modules.speech import SpeechProcessor
from modules.nlp import recognize_intent, extract_entities, plan_task

class TestSpeechNLP(unittest.TestCase):
    """Test suite for speech processing and NLP functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.speech = SpeechProcessor()
        cls.nlp = type('NLPWrapper', (), {
            'recognize_intent': staticmethod(recognize_intent),
            'extract_entities': staticmethod(extract_entities),
            'plan_task': staticmethod(plan_task)
        })()
        
        # Create test audio file
        cls._create_test_audio()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Remove test audio file
        if os.path.exists("test_data/test_audio.wav"):
            os.remove("test_data/test_audio.wav")
    
    @classmethod
    def _create_test_audio(cls):
        """Create a test audio file"""
        # Ensure test_data directory exists
        os.makedirs("test_data", exist_ok=True)
        
        # Create a simple sine wave
        sample_rate = 44100
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save as WAV file
        with wave.open("test_data/test_audio.wav", 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    def test_speech_initialization(self):
        """Test speech processor initialization"""
        self.assertIsNotNone(self.speech)
        self.assertTrue(hasattr(self.speech, 'transcribe'))
        self.assertTrue(hasattr(self.speech, 'speak'))
        self.assertTrue(hasattr(self.speech, 'record_audio'))
    
    def test_nlp_initialization(self):
        """Test NLP processor initialization"""
        self.assertIsNotNone(self.nlp)
        self.assertTrue(hasattr(self.nlp, 'recognize_intent'))
        self.assertTrue(hasattr(self.nlp, 'extract_entities'))
        self.assertTrue(hasattr(self.nlp, 'plan_task'))
    
    @patch('modules.speech.SpeechEngine.transcribe')
    def test_speech_transcription(self, mock_transcribe):
        """Test speech transcription"""
        # Set up mock
        expected_text = "test transcription result"
        mock_transcribe.return_value = expected_text
        
        # Test transcription
        result = self.speech.transcribe("test_data/test_audio.wav")
        
        # Verify result
        self.assertEqual(result, expected_text)
        mock_transcribe.assert_called_once_with("test_data/test_audio.wav")
    
    @patch('modules.speech.SpeechEngine.speak')
    def test_speech_synthesis(self, mock_speak):
        """Test speech synthesis"""
        test_text = "Hello, this is a test"
        
        # Test synthesis
        self.speech.speak(test_text)
        
        # Verify call
        mock_speak.assert_called_once_with(test_text)
    
    def test_intent_recognition(self):
        """Test intent recognition"""
        test_commands = [
            ("open chrome", "launch_app"),
            ("what's the system status", "system_control"),
            ("search for python tutorials", "web_search"),
            ("check my email", "email"),
            ("play some music", "media_control")
        ]
        
        for command, expected_intent in test_commands:
            result = self.nlp.recognize_intent(command)
            self.assertIsInstance(result, dict)
            self.assertEqual(result["intent"], expected_intent)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
    
    def test_entity_extraction(self):
        """Test entity extraction"""
        test_cases = [
            {
                "text": "open chrome browser",
                "expected": {"app_name": "chrome", "app_type": "browser"}
            },
            {
                "text": "set volume to 50 percent",
                "expected": {"setting": "volume", "value": "50", "unit": "percent"}
            },
            {
                "text": "search for python programming tutorials",
                "expected": {"query": "python programming tutorials"}
            }
        ]
        
        for case in test_cases:
            result = self.nlp.extract_entities(case["text"])
            self.assertIsInstance(result, dict)
            for key, value in case["expected"].items():
                self.assertIn(key, result)
                self.assertEqual(result[key], value)
    
    async def test_task_planning(self):
        """Test task planning"""
        test_commands = [
            "open the browser and search for weather",
            "check system status and report any issues",
            "send an email to test@example.com"
        ]
        
        for command in test_commands:
            plan = await self.nlp.plan_task(command)
            
            # Verify plan structure
            self.assertIsInstance(plan, dict)
            self.assertIn("steps", plan)
            self.assertIn("estimated_time", plan)
            
            # Verify steps
            self.assertIsInstance(plan["steps"], list)
            self.assertGreater(len(plan["steps"]), 0)
            
            # Verify each step
            for step in plan["steps"]:
                self.assertIn("action", step)
                self.assertIn("params", step)
    
    def test_error_handling(self):
        """Test error handling in speech and NLP"""
        # Test transcription with invalid file
        result = self.speech.transcribe("nonexistent_file.wav")
        self.assertEqual(result, "")
        
        # Test synthesis with invalid text
        with self.assertRaises(ValueError):
            self.speech.speak("")
        
        # Test intent recognition with invalid input
        result = self.nlp.recognize_intent("")
        self.assertEqual(result["intent"], "unknown")
        
        # Test entity extraction with invalid input
        result = self.nlp.extract_entities("")
        self.assertEqual(result, {})
    
    def test_voice_activity_detection(self):
        """Test voice activity detection"""
        # Test with silence
        silence = np.zeros(44100)  # 1 second of silence
        vad_result = self.speech.detect_voice_activity(silence)
        self.assertFalse(vad_result)
        
        # Test with speech (sine wave)
        speech_data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        vad_result = self.speech.detect_voice_activity(speech_data)
        self.assertTrue(vad_result)
    
    def test_wake_word_detection(self):
        """Test wake word detection"""
        test_phrases = [
            ("jarvis what's the weather", True),
            ("hey jarvis open chrome", True),
            ("hello computer", False),
            ("jarvis", True)
        ]
        
        for phrase, expected in test_phrases:
            result = self.speech.detect_wake_word(phrase)
            self.assertEqual(result, expected)
    
    def test_noise_reduction(self):
        """Test noise reduction"""
        # Create noisy audio
        clean_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        noise = np.random.normal(0, 0.1, 44100)
        noisy_signal = clean_signal + noise
        
        # Apply noise reduction
        cleaned_signal = self.speech.reduce_noise(noisy_signal)
        
        # Verify noise reduction
        original_noise_level = np.std(noisy_signal - clean_signal)
        cleaned_noise_level = np.std(cleaned_signal - clean_signal)
        self.assertLess(cleaned_noise_level, original_noise_level)

    @patch('modules.brain.BrainManager')
    def test_brain_integration(self, mock_brain):
        """Test integration with brain module"""
        # Setup mock brain responses
        mock_brain.process_request.return_value = "Command processed successfully"
        mock_brain.search_memory.return_value = [
            ("previous_command", "what's the weather"),
            ("previous_response", "It's sunny today")
        ]
        
        # Test speech processing with brain integration
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
        with patch.object(self.speech, 'transcribe', return_value="what's the weather"):
            # Process speech
            transcription = self.speech.transcribe(test_audio)
            intent = self.nlp.recognize_intent(transcription)
            
            # Verify brain interaction
            mock_brain.process_request.assert_called_once()
            self.assertIn("weather", str(mock_brain.process_request.call_args))
            
            # Verify context retrieval
            mock_brain.search_memory.assert_called_once()
            self.assertIn("weather", str(mock_brain.search_memory.call_args))

    def test_concurrent_processing(self):
        """Test concurrent speech and NLP processing"""
        async def process_concurrent():
            # Create multiple audio samples
            audio_samples = [
                np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)),
                np.sin(2 * np.pi * 880 * np.linspace(0, 1, 44100)),
                np.sin(2 * np.pi * 1320 * np.linspace(0, 1, 44100))
            ]
            
            # Process concurrently
            tasks = []
            for audio in audio_samples:
                task = asyncio.create_task(self._process_audio(audio))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            return results
        
        # Run concurrent processing
        results = asyncio.run(process_concurrent())
        
        # Verify results
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result)
            self.assertIn("intent", result)
            self.assertIn("confidence", result)

    async def _process_audio(self, audio_data):
        """Helper method for concurrent processing"""
        # Mock transcription for testing
        with patch.object(self.speech, 'transcribe') as mock_transcribe:
            mock_transcribe.return_value = "test command"
            transcription = self.speech.transcribe(audio_data)
            return self.nlp.recognize_intent(transcription)

    def test_streaming_integration(self):
        """Test streaming speech processing integration"""
        # Setup streaming buffer
        buffer_size = 1024
        sample_rate = 44100
        stream_duration = 1  # second
        
        # Create streaming data
        num_chunks = int(sample_rate * stream_duration / buffer_size)
        audio_chunks = []
        for i in range(num_chunks):
            t = np.linspace(i*buffer_size/sample_rate, 
                          (i+1)*buffer_size/sample_rate, 
                          buffer_size)
            chunk = np.sin(2 * np.pi * 440 * t)
            audio_chunks.append(chunk)
        
        # Process streaming audio
        results = []
        with patch.object(self.speech, 'process_stream') as mock_process:
            mock_process.return_value = {"text": "test", "is_final": True}
            
            for chunk in audio_chunks:
                result = self.speech.process_stream(chunk)
                if result and result["is_final"]:
                    intent = self.nlp.recognize_intent(result["text"])
                    results.append(intent)
        
        # Verify streaming results
        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIn("intent", result)
            self.assertIn("confidence", result)

if __name__ == '__main__':
    unittest.main(verbosity=2)
