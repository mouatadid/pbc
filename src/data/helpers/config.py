import yaml
from pathlib import Path
from typing import Dict, Any

# Path of the configuration file
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml" 

def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file {config_path}: {e}")

