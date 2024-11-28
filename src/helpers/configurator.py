import os
import json

def load_config(file_path: str = 'config.json') -> dict:
    """
    Load a nested JSON configuration file.
    
    Args:
        file_path (str): Path to the JSON configuration file.
        
    Returns:
        dict: Nested dictionary of configuration settings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(file_path, "r") as f:
        return json.load(f)
