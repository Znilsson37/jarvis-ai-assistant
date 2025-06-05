#!/usr/bin/env python3
"""
Test script for speech recognition functionality
"""

import sys
import os
import logging

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_speech_recognition():
    """Test speech recognition functionality"""
    try:
        # Import speech module
        from modules import speech
        
        logger.info("Testing speech recognition functionality...")
        
        # Test 1: Check if models can be loaded
        logger.info("Test 1: Loading speech models...")
        try:
            speech.load_models()
            logger.info("✓ Speech models loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load speech models: {e}")
            return False
        
        # Test 2: Test TTS functionality
        logger.info("Test 2: Testing text-to-speech...")
        try:
            audio_file = speech.speak("Hello, this is a test of the speech system.")
            if audio_file:
                logger.info(f"✓ TTS generated audio file: {audio_file}")
            else:
                logger.warning("✗ TTS returned None - check TTS configuration")
        except Exception as e:
            logger.error(f"✗ TTS test failed: {e}")
        
        # Test 3: Test audio recording (short duration for testing)
        logger.info("Test 3: Testing audio recording...")
        try:
            logger.info("Recording 3 seconds of audio... Please speak something!")
            audio_file = speech.record_audio(duration=3.0, auto_stop=False)
            if audio_file:
                logger.info(f"✓ Audio recorded to: {audio_file}")
                
                # Test 4: Test transcription
                logger.info("Test 4: Testing transcription...")
                transcript = speech.transcribe(audio_file)
                if transcript:
                    logger.info(f"✓ Transcription: '{transcript}'")
                else:
                    logger.warning("✗ No transcription returned")
            else:
                logger.warning("✗ Audio recording failed")
        except Exception as e:
            logger.error(f"✗ Audio recording test failed: {e}")
        
        # Test 5: Test engine switching
        logger.info("Test 5: Testing TTS engine switching...")
        try:
            current_engine = speech.speech_processor.tts_engine
            logger.info(f"Current TTS engine: {current_engine}")
            
            # Try switching engines
            if current_engine == "chatterbox":
                if speech.set_tts_engine("eleven_labs"):
                    logger.info("✓ Switched to Eleven Labs TTS")
                    speech.set_tts_engine("chatterbox")  # Switch back
                    logger.info("✓ Switched back to Chatterbox TTS")
                else:
                    logger.info("ℹ Eleven Labs TTS not available (expected if not configured)")
            
        except Exception as e:
            logger.error(f"✗ Engine switching test failed: {e}")
        
        logger.info("Speech recognition tests completed!")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to import speech module: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring audio hardware"""
    try:
        from modules import speech
        import numpy as np
        
        logger.info("Testing basic functionality...")
        
        # Test audio analysis with dummy data
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1 second of noise
        
        # Test voice activity detection
        has_speech = speech.detect_speech(dummy_audio)
        logger.info(f"Voice activity detection result: {has_speech}")
        
        # Test emotion analysis
        emotion = speech.analyze_emotion(dummy_audio)
        logger.info(f"Emotion analysis result: {emotion}")
        
        # Test language detection
        language = speech.detect_language(dummy_audio)
        logger.info(f"Language detection result: {language}")
        
        # Test full audio analysis
        analysis = speech.analyze_audio(dummy_audio)
        logger.info(f"Full audio analysis: {analysis}")
        
        logger.info("✓ Basic functionality tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting speech recognition tests...")
    
    # Test basic functionality first
    if test_basic_functionality():
        logger.info("Basic tests passed, proceeding to full tests...")
        test_speech_recognition()
    else:
        logger.error("Basic tests failed, skipping full tests")
        sys.exit(1)
    
    logger.info("All tests completed!")
