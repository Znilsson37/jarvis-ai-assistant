import os
import logging
import whisper
import numpy as np
import soundfile as sf
from modules import speech

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_test_audio():
    """Generate a test audio file with a sine wave"""
    sample_rate = 16000
    duration = 2  # seconds
    frequency = 440  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    test_file = "test_audio.wav"
    sf.write(test_file, audio, sample_rate)
    return test_file

def test_whisper_directly():
    """Test Whisper model directly"""
    try:
        logger.info("Loading Whisper model directly...")
        model = whisper.load_model("base")
        
        # Generate test audio
        test_file = generate_test_audio()
        
        logger.info(f"Transcribing test audio file: {test_file}")
        result = model.transcribe(test_file)
        
        logger.info(f"Direct Whisper transcription result: {result['text']}")
        
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        logger.error(f"Error in direct Whisper test: {e}")

def test_speech_module_transcription():
    """Test transcription through speech module"""
    try:
        logger.info("Testing speech module transcription...")
        
        # Initialize speech processor
        speech.load_models()
        
        # Generate test audio
        test_file = generate_test_audio()
        
        # Test transcription
        result = speech.transcribe(test_file)
        logger.info(f"Speech module transcription result: {result}")
        
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        logger.error(f"Error in speech module test: {e}")

if __name__ == "__main__":
    logger.info("Starting Whisper tests...")
    test_whisper_directly()
    test_speech_module_transcription()
