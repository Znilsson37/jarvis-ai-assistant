
"""Speech processing module for Jarvis"""

import numpy as np
import json
import os
import tempfile
import logging
import threading
import time
import wave
from typing import Dict, Any, Optional, List, Tuple
from modules.chatterbox import tts as chatterbox_tts

# Set up logging
logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self):
        self.initialized = False
        self.config = self._load_config()
        self.tts_engine = self.config.get("tts", {}).get("default_engine", "chatterbox")
        self.chatterbox_tts = chatterbox_tts
        self.eleven_labs_tts = None
        
        # Speech recognition components
        self.whisper_model = None
        self.audio_stream = None
        self.is_recording = False
        self.recording_thread = None
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        self.format = None
        
        # Voice activity detection settings
        self.vad_threshold = 0.01
        self.silence_duration = 2.0  # seconds of silence to stop recording
        self.min_audio_length = 1.0  # minimum audio length in seconds
        
        # Initialize components
        self._initialize_audio()
        self._initialize_whisper()
        
        # Initialize Eleven Labs if configured
        if self.tts_engine == "eleven_labs" or self.config.get("eleven_labs_api_key"):
            try:
                from modules.eleven_labs import ElevenLabsClient
                self.eleven_labs_tts = ElevenLabsClient(self.config.get("eleven_labs_api_key"))
            except ImportError:
                logger.warning("Eleven Labs module not available")
        
        self.initialized = True
        logger.info("Speech processor initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        try:
            config_path = os.path.join("Config", "config.json")
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def _initialize_audio(self):
        """Initialize audio recording components"""
        try:
            import pyaudio
            self.pyaudio = pyaudio.PyAudio()
            self.format = pyaudio.paInt16
            logger.info("PyAudio initialized successfully")
        except ImportError:
            logger.warning("PyAudio not available, trying sounddevice")
            try:
                import sounddevice as sd
                self.sounddevice = sd
                logger.info("SoundDevice initialized successfully")
            except ImportError:
                logger.error("No audio library available (PyAudio or SoundDevice)")
                raise ImportError("No audio recording library available")
    
    def _initialize_whisper(self):
        """Initialize Whisper model for speech recognition"""
        try:
            import whisper
            model_name = self.config.get("whisper_model", "base")
            logger.info(f"Loading Whisper model: {model_name}")
            self.whisper_model = whisper.load_model(model_name)
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.error("Whisper not available. Install with: pip install openai-whisper")
            raise ImportError("Whisper not available")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def load_models(self):
        """Load speech recognition models (for compatibility)"""
        if not self.initialized:
            self._initialize_whisper()
            self.initialized = True
        logger.info("Speech models loaded")
    
    def record_audio(self, duration: float = 5.0, auto_stop: bool = True) -> Optional[str]:
        """
        Record audio from microphone
        
        Args:
            duration: Maximum recording duration in seconds
            auto_stop: Whether to automatically stop on silence
            
        Returns:
            Path to recorded audio file or None if failed
        """
        try:
            logger.info(f"Starting audio recording for {duration} seconds")
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            if hasattr(self, 'pyaudio'):
                return self._record_with_pyaudio(temp_file.name, duration, auto_stop)
            elif hasattr(self, 'sounddevice'):
                return self._record_with_sounddevice(temp_file.name, duration, auto_stop)
            else:
                logger.error("No audio recording method available")
                return None
                
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
    
    def _record_with_pyaudio(self, filename: str, duration: float, auto_stop: bool) -> Optional[str]:
        """Record audio using PyAudio"""
        try:
            stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            silence_start = None
            recording_start = time.time()
            
            logger.info("Recording started...")
            
            while True:
                data = stream.read(self.chunk_size)
                frames.append(data)
                
                # Check for voice activity if auto_stop is enabled
                if auto_stop:
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    if self._detect_voice_activity(audio_chunk):
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > self.silence_duration:
                            if time.time() - recording_start > self.min_audio_length:
                                logger.info("Silence detected, stopping recording")
                                break
                
                # Check maximum duration
                if time.time() - recording_start > duration:
                    logger.info("Maximum duration reached, stopping recording")
                    break
            
            stream.stop_stream()
            stream.close()
            
            # Save to WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            logger.info(f"Audio recorded to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error in PyAudio recording: {e}")
            return None
    
    def _record_with_sounddevice(self, filename: str, duration: float, auto_stop: bool) -> Optional[str]:
        """Record audio using SoundDevice"""
        try:
            import soundfile as sf
            
            logger.info("Recording started...")
            
            if auto_stop:
                # Record with voice activity detection
                recording = []
                silence_start = None
                recording_start = time.time()
                
                def callback(indata, frames, time, status):
                    nonlocal silence_start
                    if status:
                        logger.warning(f"Recording status: {status}")
                    
                    recording.extend(indata[:, 0])
                    
                    # Voice activity detection
                    if self._detect_voice_activity(indata[:, 0]):
                        silence_start = None
                    else:
                        if silence_start is None:
                            silence_start = time.inputBufferAdcTime
                        elif time.inputBufferAdcTime - silence_start > self.silence_duration:
                            if len(recording) / self.sample_rate > self.min_audio_length:
                                raise self.sounddevice.CallbackStop()
                
                with self.sounddevice.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    callback=callback,
                    dtype=np.float32
                ):
                    self.sounddevice.sleep(int(duration * 1000))
                
                audio_data = np.array(recording, dtype=np.float32)
            else:
                # Simple duration-based recording
                audio_data = self.sounddevice.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                self.sounddevice.wait()
            
            # Save to file
            sf.write(filename, audio_data, self.sample_rate)
            logger.info(f"Audio recorded to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error in SoundDevice recording: {e}")
            return None
    
    def transcribe(self, audio_file: str) -> str:
        """
        Transcribe audio file to text using Whisper
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Transcribed text or empty string if failed
        """
        try:
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return ""
            
            if not self.whisper_model:
                logger.error("Whisper model not loaded")
                return ""
            
            logger.info(f"Transcribing audio file: {audio_file}")
            result = self.whisper_model.transcribe(audio_file)
            
            text = result["text"].strip()
            confidence = result.get("confidence", 0.0)
            
            logger.info(f"Transcription: '{text}' (confidence: {confidence:.2f})")
            
            # Only clean up temporary files (those in temp directory)
            try:
                if "temp" in audio_file.lower() or "tmp" in audio_file.lower():
                    os.unlink(audio_file)
            except:
                pass
            
            return text
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return ""
    
    def _detect_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple voice activity detection based on energy threshold
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            True if voice activity detected, False otherwise
        """
        try:
            # Calculate RMS energy
            if len(audio_chunk) == 0:
                return False
            
            # Normalize audio to float32 if needed
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0
            
            rms = np.sqrt(np.mean(audio_chunk ** 2))
            return rms > self.vad_threshold
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            return False
    
    def set_tts_engine(self, engine: str) -> bool:
        """
        Switch TTS engine
        
        Args:
            engine: Either 'chatterbox' or 'eleven_labs'
            
        Returns:
            True if switch successful, False otherwise
        """
        if engine not in ["chatterbox", "eleven_labs"]:
            logger.error(f"Invalid TTS engine: {engine}")
            return False
            
        if engine == "eleven_labs" and not self.eleven_labs_tts:
            logger.error("Eleven Labs TTS not available")
            return False
            
        self.tts_engine = engine
        logger.info(f"Switched to {engine} TTS engine")
        return True
    
    def analyze_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio data and return speech recognition results and metadata"""
        try:
            # Create temporary file for analysis
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            # Save audio to temporary file
            import soundfile as sf
            sf.write(temp_file.name, audio, self.sample_rate)
            
            # Transcribe
            transcript = self.transcribe(temp_file.name)
            
            # Basic analysis
            confidence = 0.8 if transcript else 0.0
            has_speech = self.detect_speech(audio)
            emotion = self.analyze_emotion(audio)
            language = self.detect_language(audio)
            
            return {
                "transcript": transcript,
                "confidence": confidence,
                "speaker": "unknown",
                "emotion": emotion,
                "language": language,
                "timestamps": [],
                "has_speech": has_speech
            }
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            return {
                "transcript": "",
                "confidence": 0.0,
                "speaker": "unknown",
                "emotion": "neutral",
                "language": "en",
                "timestamps": [],
                "has_speech": False
            }
    
    def detect_speech(self, audio: np.ndarray) -> bool:
        """Detect presence of speech in audio"""
        try:
            # Use voice activity detection
            return self._detect_voice_activity(audio)
        except Exception as e:
            logger.error(f"Error detecting speech: {e}")
            return False
    
    def recognize_speaker(self, audio: np.ndarray) -> Optional[str]:
        """Identify speaker if known (placeholder implementation)"""
        # This would require speaker recognition models
        # For now, return None (unknown speaker)
        return None
    
    def analyze_emotion(self, audio: np.ndarray) -> str:
        """Detect emotional content in speech (basic implementation)"""
        try:
            # Basic emotion detection based on audio characteristics
            if len(audio) == 0:
                return "neutral"
            
            # Normalize audio
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            # Calculate basic features
            rms = np.sqrt(np.mean(audio ** 2))
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            
            # Simple heuristic-based emotion detection
            if rms > 0.1 and zero_crossings > len(audio) * 0.1:
                return "excited"
            elif rms < 0.02:
                return "calm"
            elif zero_crossings < len(audio) * 0.05:
                return "sad"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return "neutral"
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect spoken language (placeholder implementation)"""
        # For now, assume English
        # This would require language detection models
        return "en"
    
    def get_word_timestamps(self, audio: np.ndarray) -> list:
        """Get word-level timestamps (placeholder implementation)"""
        # This would require word-level alignment
        # For now, return empty list
        return []
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            if hasattr(self, 'pyaudio') and self.pyaudio:
                self.pyaudio.terminate()
                logger.info("PyAudio terminated")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def speak(self, text: str, voice_path: Optional[str] = None, 
             exaggeration: float = None, cfg_weight: float = None,
             stability: float = None, similarity_boost: float = None) -> Optional[str]:
        """
        Convert text to speech using the configured TTS engine
        
        Args:
            text: Text to convert to speech
            voice_path: Optional path to reference voice file for voice cloning
            exaggeration: Controls expressiveness for Chatterbox (0.0-1.0)
            cfg_weight: Controls speaking pace for Chatterbox (lower = faster)
            stability: Controls stability for Eleven Labs (0.0-1.0)
            similarity_boost: Controls similarity for Eleven Labs (0.0-1.0)
            
        Returns:
            Path to generated audio file or None if generation failed
        """
        try:
            tts_config = self.config.get("tts", {})
            
            if self.tts_engine == "chatterbox":
                chatterbox_config = tts_config.get("chatterbox", {})
                if exaggeration is None:
                    exaggeration = chatterbox_config.get("default_exaggeration", 0.5)
                if cfg_weight is None:
                    cfg_weight = chatterbox_config.get("default_cfg_weight", 0.5)
                    
                if voice_path:
                    return self.chatterbox_tts.clone_voice(voice_path, text)
                else:
                    return self.chatterbox_tts.generate_speech(
                        text,
                        exaggeration=exaggeration,
                        cfg_weight=cfg_weight
                    )
                    
            elif self.tts_engine == "eleven_labs":
                if not self.eleven_labs_tts:
                    logger.error("Eleven Labs TTS not available")
                    return None
                    
                eleven_labs_config = tts_config.get("eleven_labs", {})
                if stability is None:
                    stability = eleven_labs_config.get("default_stability", 0.5)
                if similarity_boost is None:
                    similarity_boost = eleven_labs_config.get("default_similarity_boost", 0.5)
                voice_id = voice_path or eleven_labs_config.get("default_voice_id")
                
                return self.eleven_labs_tts.generate_speech(
                    text,
                    voice_id=voice_id,
                    stability=stability,
                    similarity_boost=similarity_boost
                )
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None
    
    def batch_speak(self, texts: list[str], voice_path: Optional[str] = None) -> list[str]:
        """
        Convert multiple texts to speech using the configured TTS engine
        
        Args:
            texts: List of texts to convert
            voice_path: Optional path to reference voice file or voice ID
            
        Returns:
            List of paths to generated audio files
        """
        if self.tts_engine == "chatterbox":
            return self.chatterbox_tts.batch_generate(texts, audio_prompt_path=voice_path)
        elif self.tts_engine == "eleven_labs" and self.eleven_labs_tts:
            return [self.speak(text, voice_path) for text in texts]
        return []

# Create global instance
speech_processor = SpeechProcessor()

# Create speech interface object
class Speech:
    def __init__(self, processor: SpeechProcessor):
        self._processor = processor
    
    def load_models(self):
        """Load speech recognition models"""
        return self._processor.load_models()
    
    def record_audio(self, duration: float = 5.0, auto_stop: bool = True) -> Optional[str]:
        """Record audio from microphone"""
        return self._processor.record_audio(duration, auto_stop)
    
    def transcribe(self, audio_file: str) -> str:
        """Transcribe audio file to text"""
        return self._processor.transcribe(audio_file)
    
    def speak(self, text: str, voice_path: Optional[str] = None, 
             exaggeration: float = None, cfg_weight: float = None,
             stability: float = None, similarity_boost: float = None) -> Optional[str]:
        """Convert text to speech"""
        return self._processor.speak(text, voice_path, exaggeration, cfg_weight, stability, similarity_boost)
    
    def analyze_audio(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyze audio data and return speech recognition results and metadata"""
        return self._processor.analyze_audio(audio)
    
    def detect_speech(self, audio: np.ndarray) -> bool:
        """Detect presence of speech in audio"""
        return self._processor.detect_speech(audio)
    
    def recognize_speaker(self, audio: np.ndarray) -> Optional[str]:
        """Identify speaker if known"""
        return self._processor.recognize_speaker(audio)
    
    def analyze_emotion(self, audio: np.ndarray) -> str:
        """Detect emotional content in speech"""
        return self._processor.analyze_emotion(audio)
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect spoken language"""
        return self._processor.detect_language(audio)
    
    def get_word_timestamps(self, audio: np.ndarray) -> list:
        """Get word-level timestamps"""
        return self._processor.get_word_timestamps(audio)
    
    def set_tts_engine(self, engine: str) -> bool:
        """Switch TTS engine"""
        return self._processor.set_tts_engine(engine)
    
    def batch_speak(self, texts: list[str], voice_path: Optional[str] = None) -> list[str]:
        """Convert multiple texts to speech"""
        return self._processor.batch_speak(texts, voice_path)
    
    def cleanup(self):
        """Clean up audio resources"""
        return self._processor.cleanup()

# Create and export speech interface
speech = Speech(speech_processor)

# Keep global functions for backward compatibility
def load_models():
    """Load speech recognition models"""
    return speech_processor.load_models()

def record_audio(duration: float = 5.0, auto_stop: bool = True) -> Optional[str]:
    """Record audio from microphone"""
    return speech_processor.record_audio(duration, auto_stop)

def transcribe(audio_file: str) -> str:
    """Transcribe audio file to text"""
    return speech_processor.transcribe(audio_file)

def speak(text: str, voice_path: Optional[str] = None, 
         exaggeration: float = None, cfg_weight: float = None,
         stability: float = None, similarity_boost: float = None) -> Optional[str]:
    """Convert text to speech"""
    return speech_processor.speak(text, voice_path, exaggeration, cfg_weight, stability, similarity_boost)

def analyze_audio(audio: np.ndarray) -> Dict[str, Any]:
    """Analyze audio data and return speech recognition results and metadata"""
    return speech_processor.analyze_audio(audio)

def detect_speech(audio: np.ndarray) -> bool:
    """Detect presence of speech in audio"""
    return speech_processor.detect_speech(audio)

def recognize_speaker(audio: np.ndarray) -> Optional[str]:
    """Identify speaker if known"""
    return speech_processor.recognize_speaker(audio)

def analyze_emotion(audio: np.ndarray) -> str:
    """Detect emotional content in speech"""
    return speech_processor.analyze_emotion(audio)

def detect_language(audio: np.ndarray) -> str:
    """Detect spoken language"""
    return speech_processor.detect_language(audio)

def get_word_timestamps(audio: np.ndarray) -> list:
    """Get word-level timestamps"""
    return speech_processor.get_word_timestamps(audio)

def set_tts_engine(engine: str) -> bool:
    """Switch TTS engine"""
    return speech_processor.set_tts_engine(engine)

def batch_speak(texts: list[str], voice_path: Optional[str] = None) -> list[str]:
    """Convert multiple texts to speech"""
    return speech_processor.batch_speak(texts, voice_path)

def cleanup():
    """Clean up audio resources"""
    return speech_processor.cleanup()
