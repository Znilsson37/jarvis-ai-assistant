import os
import json
import logging
import requests
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

class ElevenLabsTTS:
    """Client for Eleven Labs API integration"""
    
    def __init__(self):
        from modules.config import config
        self.api_key = config.get("eleven_labs_api_key")
        if not self.api_key:
            raise ValueError("Eleven Labs API key is required")
            
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "application/json",
            "xi-api-key": self.api_key
        }
        
        # Test connection during initialization
        self.test_connection()
        
    def test_connection(self) -> bool:
        """Test connection to Eleven Labs API"""
        try:
            response = requests.get(f"{self.base_url}/user", headers=self.headers)
            response.raise_for_status()
            logger.info("Successfully authenticated with Eleven Labs API")
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Eleven Labs API: {e}")
            return False
            
    def synthesize(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Optional[str]:
        """
        Convert text to speech using Eleven Labs API
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (default is Josh, a male voice)
        Returns:
            Path to temporary audio file, or None if conversion failed
        """
        if not self.test_connection():
            logger.error("Cannot synthesize speech: Not connected to Eleven Labs API")
            return None
        try:
            headers = {**self.headers, "Content-Type": "application/json"}
            response = requests.post(
                f"{self.base_url}/text-to-speech/{voice_id}",
                headers=headers,
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                stream=True
            )
            response.raise_for_status()
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    temp_file.write(chunk)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return None
            
    def get_voice_settings(self, voice_id: str):
        """Get settings for a specific voice"""
        try:
            response = requests.get(
                f"{self.base_url}/voices/{voice_id}/settings",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error getting voice settings: {e}")
            return None
