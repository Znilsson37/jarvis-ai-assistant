"""Enhanced configuration management with validation and encryption"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from cryptography.fernet import Fernet
from pydantic import BaseModel, validator
import yaml

logger = logging.getLogger(__name__)

class APIConfig(BaseModel):
    """API configuration with validation"""
    openai_api_key: Optional[str] = None
    eleven_labs_api_key: Optional[str] = None
    grok_api_key: Optional[str] = None
    
    @validator('*', pre=True)
    def validate_api_keys(cls, v):
        if v and len(v) < 10:
            raise ValueError("API key appears to be invalid (too short)")
        return v

class TTSConfig(BaseModel):
    """TTS configuration with validation"""
    default_engine: str = "chatterbox"
    chatterbox: Dict[str, Any] = {}
    eleven_labs: Dict[str, Any] = {}
    
    @validator('default_engine')
    def validate_engine(cls, v):
        if v not in ["chatterbox", "eleven_labs"]:
            raise ValueError("Invalid TTS engine")
        return v

class JarvisConfig(BaseModel):
    """Main Jarvis configuration"""
    api: APIConfig = APIConfig()
    tts: TTSConfig = TTSConfig()
    vision: Dict[str, Any] = {}
    system: Dict[str, Any] = {}

class ConfigManager:
    """Enhanced configuration manager with encryption and validation"""
    
    def __init__(self, config_path: str = "Config/config.yaml"):
        self.config_path = Path(config_path)
        self.encryption_key = self._get_encryption_key()
        self.cipher = Fernet(self.encryption_key) if self.encryption_key else None
        self._config: Optional[JarvisConfig] = None
        self.load_config()
    
    def _get_encryption_key(self) -> Optional[bytes]:
        """Get or generate encryption key"""
        key_path = Path("Config/.encryption_key")
        
        if key_path.exists():
            return key_path.read_bytes()
        
        # Generate new key
        key = Fernet.generate_key()
        key_path.write_bytes(key)
        key_path.chmod(0o600)  # Restrict permissions
        return key
    
    def load_config(self):
        """Load and validate configuration"""
        try:
            if self.config_path.suffix.lower() == '.yaml':
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            else:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
            
            # Decrypt sensitive values
            config_data = self._decrypt_sensitive_data(config_data)
            
            # Validate configuration
            self._config = JarvisConfig(**config_data)
            logger.info("Configuration loaded and validated successfully")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self._config = JarvisConfig()  # Use defaults
    
    def _decrypt_sensitive_data(self, config_data: Dict) -> Dict:
        """Decrypt sensitive configuration values"""
        if not self.cipher:
            return config_data
        
        sensitive_keys = ["api_key", "password", "token"]
        
        def decrypt_recursive(obj):
            if isinstance(obj, dict):
                return {k: decrypt_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, str) and any(key in obj.lower() for key in sensitive_keys):
                try:
                    if obj.startswith("encrypted:"):
                        return self.cipher.decrypt(obj[10:].encode()).decode()
                except Exception:
                    pass
            return obj
        
        return decrypt_recursive(config_data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        if not self._config:
            return default
        
        keys = key.split('.')
        value = self._config.dict()
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        # Implementation for setting values
        pass
    
    def save_config(self):
        """Save configuration with encryption for sensitive data"""
        if not self._config:
            return
        
        config_data = self._config.dict()
        encrypted_data = self._encrypt_sensitive_data(config_data)
        
        with open(self.config_path, 'w') as f:
            if self.config_path.suffix.lower() == '.yaml':
                yaml.dump(encrypted_data, f, default_flow_style=False)
            else:
                json.dump(encrypted_data, f, indent=2)

# Global configuration instance
config_manager = ConfigManager()