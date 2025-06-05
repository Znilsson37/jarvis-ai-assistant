import unittest
import os
from modules.speech import speech
from modules.config import config
from modules.eleven_labs import ElevenLabsClient

class TestSpeech(unittest.TestCase):
    def setUp(self):
        self.speech = speech
        
    def test_eleven_labs_tts(self):
        """Test Eleven Labs text-to-speech functionality"""
        # Skip if no API key configured
        if not config.get("eleven_labs_api_key"):
            self.skipTest("Eleven Labs API key not configured")
            
        try:
            # Initialize client directly to test connection
            client = ElevenLabsClient(config.get("eleven_labs_api_key"))
            voices = client.get_voices()
            
            if voices:
                print("\nAvailable voices:")
                for voice in voices:
                    print(f"Voice ID: {voice.get('voice_id')}, Name: {voice.get('name')}")
            
            # Test speech synthesis
            print("\nTesting speech synthesis...")
            self.speech.load_models()
            self.speech.speak("This is a test of the Eleven Labs text to speech system.")
            
            # Verify engine type
            self.assertEqual(self.speech.tts_engine[0], "eleven_labs")
            
        except Exception as e:
            print(f"\nError during test: {str(e)}")
            raise
            
if __name__ == '__main__':
    unittest.main(verbosity=2)
