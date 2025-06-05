"""Chatterbox TTS module for Jarvis"""

import os
from typing import Optional
import numpy as np

try:
    import torch
    import torchaudio
    from chatterbox.tts import ChatterboxTTS
    CHATTERBOX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Chatterbox TTS not available: {e}")
    CHATTERBOX_AVAILABLE = False

class ChatterboxTTSWrapper:
    def __init__(self, device="cuda" if CHATTERBOX_AVAILABLE and torch.cuda.is_available() else "cpu"):
        """Initialize Chatterbox TTS"""
        self.available = CHATTERBOX_AVAILABLE
        self.output_dir = "audio_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.available:
            try:
                self.model = ChatterboxTTS.from_pretrained(device=device)
                self.sample_rate = self.model.sr
            except Exception as e:
                print(f"Error initializing Chatterbox TTS: {e}")
                self.available = False
        else:
            self.model = None
            self.sample_rate = 22050  # Default sample rate
        
    def generate_speech(self, text: str, output_path: Optional[str] = None, 
                       audio_prompt_path: Optional[str] = None,
                       exaggeration: float = 0.5,
                       cfg_weight: float = 0.5) -> Optional[str]:
        """
        Generate speech from text using Chatterbox TTS
        
        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file
            audio_prompt_path: Optional path to reference voice audio file
            exaggeration: Controls expressiveness (0.0-1.0)
            cfg_weight: Controls speaking pace (lower = faster)
            
        Returns:
            Path to generated audio file or None if generation failed
        """
        if not self.available:
            print("Chatterbox TTS is not available")
            return None
            
        if not self.model:
            print("Chatterbox TTS model not initialized")
            return None
        
        # Input validation
        if not text or not isinstance(text, str):
            print("Invalid text input: text must be a non-empty string")
            return None
            
        if len(text.strip()) == 0:
            print("Empty text provided")
            return None
            
        # Parameter validation
        if not (0.0 <= exaggeration <= 1.0):
            print(f"Warning: exaggeration {exaggeration} outside range [0.0, 1.0], clamping")
            exaggeration = max(0.0, min(1.0, exaggeration))
            
        if not (0.0 <= cfg_weight <= 1.0):
            print(f"Warning: cfg_weight {cfg_weight} outside range [0.0, 1.0], clamping")
            cfg_weight = max(0.0, min(1.0, cfg_weight))
        
        # Validate audio prompt path if provided
        if audio_prompt_path and not os.path.exists(audio_prompt_path):
            print(f"Warning: audio prompt path {audio_prompt_path} does not exist, using default voice")
            audio_prompt_path = None
            
        try:
            # Generate speech
            wav = self.model.generate(
                text,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight
            )
            
            # Validate generated audio
            if wav is None or len(wav) == 0:
                print("Generated audio is empty")
                return None
            
            # Save to file
            if output_path is None:
                # Create unique filename based on text hash and timestamp
                import time
                timestamp = int(time.time() * 1000)
                text_hash = abs(hash(text)) % 10000
                output_path = os.path.join(self.output_dir, f"speech_{text_hash}_{timestamp}.wav")
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            torchaudio.save(output_path, wav, self.sample_rate)
            
            # Verify file was created and has content
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                print(f"Failed to create valid audio file at {output_path}")
                return None
                
            return output_path
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None
    
    def clone_voice(self, reference_audio: str, text: str, output_path: Optional[str] = None) -> str:
        """
        Generate speech in the style of a reference voice
        
        Args:
            reference_audio: Path to reference voice audio file
            text: Text to speak in cloned voice
            output_path: Optional path to save generated audio
            
        Returns:
            Path to generated audio file
        """
        return self.generate_speech(text, output_path, audio_prompt_path=reference_audio)
    
    def batch_generate(self, texts: list[str], output_dir: Optional[str] = None,
                      audio_prompt_path: Optional[str] = None) -> list[str]:
        """
        Generate speech for multiple texts
        
        Args:
            texts: List of texts to convert to speech
            output_dir: Optional directory to save audio files
            audio_prompt_path: Optional reference voice audio file
            
        Returns:
            List of paths to generated audio files
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        os.makedirs(output_dir, exist_ok=True)
        
        output_paths = []
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f"speech_{i}.wav")
            result = self.generate_speech(text, output_path, audio_prompt_path)
            if result:
                output_paths.append(result)
                
        return output_paths

# Create global instance
tts = ChatterboxTTSWrapper()
