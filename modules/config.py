import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "Config", "config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
