"""Enhanced speech processing with better error handling and performance"""

import asyncio
import logging
import numpy as np
import threading
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import queue

logger = logging.getLogger(__name__)

class SpeechState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class SpeechResult:
    text: str
    confidence: float
    language: str
    processing_time: float
    error: Optional[str] = None

class EnhancedSpeechProcessor:
    """Enhanced speech processor with better performance and reliability"""
    
    def __init__(self):
        self.state = SpeechState.IDLE
        self.callbacks = {}
        self.audio_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        self.is_running = False
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_processing_time": 0.0,
            "error_rate": 0.0
        }
        
        # Initialize components
        self._init_whisper()
        self._init_tts()
    
    def _init_whisper(self):
        """Initialize Whisper with error handling"""
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
    
    def _init_tts(self):
        """Initialize TTS engines"""
        self.tts_engines = {}
        
        # Initialize Chatterbox
        try:
            from modules.chatterbox import ChatterboxTTSWrapper
            self.tts_engines["chatterbox"] = ChatterboxTTSWrapper()
            logger.info("Chatterbox TTS initialized")
        except Exception as e:
            logger.warning(f"Chatterbox TTS not available: {e}")
        
        # Initialize Eleven Labs
        try:
            from modules.eleven_labs import ElevenLabsTTS
            self.tts_engines["eleven_labs"] = ElevenLabsTTS()
            logger.info("Eleven Labs TTS initialized")
        except Exception as e:
            logger.warning(f"Eleven Labs TTS not available: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for speech events"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def _emit_event(self, event: str, data: Any = None):
        """Emit event to registered callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in callback for {event}: {e}")
    
    async def transcribe_async(self, audio_file: str) -> SpeechResult:
        """Asynchronous transcription with better error handling"""
        start_time = time.time()
        self.state = SpeechState.PROCESSING
        self.metrics["total_requests"] += 1
        
        try:
            if not self.whisper_model:
                raise RuntimeError("Whisper model not available")
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self.whisper_model.transcribe, 
                audio_file
            )
            
            processing_time = time.time() - start_time
            
            speech_result = SpeechResult(
                text=result["text"].strip(),
                confidence=result.get("confidence", 0.0),
                language=result.get("language", "en"),
                processing_time=processing_time
            )
            
            self.metrics["successful_requests"] += 1
            self._update_metrics(processing_time)
            self.state = SpeechState.IDLE
            
            self._emit_event("transcription_complete", speech_result)
            return speech_result
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            
            self.state = SpeechState.ERROR
            self._emit_event("transcription_error", error_msg)
            
            return SpeechResult(
                text="",
                confidence=0.0,
                language="en",
                processing_time=time.time() - start_time,
                error=error_msg
            )
    
    def _update_metrics(self, processing_time: float):
        """Update performance metrics"""
        current_avg = self.metrics["average_processing_time"]
        total_requests = self.metrics["total_requests"]
        
        self.metrics["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
        
        self.metrics["error_rate"] = (
            (total_requests - self.metrics["successful_requests"]) / total_requests
        )
    
    async def speak_async(self, text: str, engine: str = "chatterbox") -> Optional[str]:
        """Asynchronous speech synthesis"""
        self.state = SpeechState.SPEAKING
        
        try:
            if engine not in self.tts_engines:
                raise ValueError(f"TTS engine {engine} not available")
            
            tts_engine = self.tts_engines[engine]
            
            # Run TTS in thread pool
            loop = asyncio.get_event_loop()
            audio_path = await loop.run_in_executor(
                None,
                tts_engine.generate_speech,
                text
            )
            
            self.state = SpeechState.IDLE
            self._emit_event("speech_complete", audio_path)
            return audio_path
            
        except Exception as e:
            error_msg = f"Speech synthesis failed: {str(e)}"
            logger.error(error_msg)
            self.state = SpeechState.ERROR
            self._emit_event("speech_error", error_msg)
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            "current_state": self.state.value,
            "available_engines": list(self.tts_engines.keys())
        }
    
    def start_continuous_listening(self, callback: Callable[[str], None]):
        """Start continuous listening with callback"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._continuous_listening_worker,
            args=(callback,)
        )
        self.processing_thread.start()
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
    
    def _continuous_listening_worker(self, callback: Callable[[str], None]):
        """Worker for continuous listening"""
        while self.is_running:
            try:
                # Record audio
                audio_file = self._record_audio_chunk()
                if audio_file:
                    # Transcribe
                    result = asyncio.run(self.transcribe_async(audio_file))
                    if result.text and not result.error:
                        callback(result.text)
                
                time.sleep(0.1)  # Small delay to prevent overwhelming
                
            except Exception as e:
                logger.error(f"Error in continuous listening: {e}")
                time.sleep(1)  # Longer delay on error
    
    def _record_audio_chunk(self) -> Optional[str]:
        """Record a chunk of audio"""
        # Implementation depends on your audio recording setup
        # This is a placeholder
        return None

# Global enhanced speech processor
enhanced_speech = EnhancedSpeechProcessor()