import os
import sys
import json
from pathlib import Path

def get_api_key():
    """Get the OpenAI API key"""
    # Try to load from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # If not found, try to load from config file
    if not api_key:
        config_path = Path("config.json")
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                    api_key = config_data.get("openai_api_key")
            except:
                pass
    
    return api_key

# Make the module accessible as security.x
security = sys.modules[__name__]