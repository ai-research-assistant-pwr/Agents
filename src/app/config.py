from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/app/config.yaml") -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the config file, relative to project root.

    Returns:
        A dictionary with the parsed configuration.
    """
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        config = yaml.safe_load(f)
    return config
