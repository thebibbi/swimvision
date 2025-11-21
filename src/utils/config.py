"""Configuration management utilities."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for loading YAML config files and environment variables."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize configuration manager.

        Args:
            config_dir: Directory containing config files. Defaults to project config/.
        """
        if config_dir is None:
            # Get project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            config_dir = project_root / "config"

        self.config_dir = Path(config_dir)
        self._configs: dict[str, dict[str, Any]] = {}

    def load(self, config_name: str) -> dict[str, Any]:
        """Load a configuration file.

        Args:
            config_name: Name of config file (without .yaml extension).

        Returns:
            Dictionary containing configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self.config_dir / f"{config_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self._configs[config_name] = config or {}
        return self._configs[config_name]

    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value.

        Args:
            config_name: Name of config file.
            key: Dot-separated key path (e.g., "yolo.model").
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        config = self.load(config_name)
        keys = key.split(".")

        value: Any = config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def get_env(self, key: str, default: Any = None) -> Any:
        """Get environment variable with optional default.

        Args:
            key: Environment variable name.
            default: Default value if not found.

        Returns:
            Environment variable value or default.
        """
        return os.getenv(key, default)

    def reload(self, config_name: str) -> dict[str, Any]:
        """Reload a configuration file (clears cache).

        Args:
            config_name: Name of config file.

        Returns:
            Reloaded configuration dictionary.
        """
        if config_name in self._configs:
            del self._configs[config_name]
        return self.load(config_name)


# Global config instance
_global_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance (singleton).

    Returns:
        Global Config instance.
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def load_pose_config() -> dict[str, Any]:
    """Load pose estimation configuration.

    Returns:
        Pose configuration dictionary.
    """
    return get_config().load("pose_config")


def load_camera_config() -> dict[str, Any]:
    """Load camera configuration.

    Returns:
        Camera configuration dictionary.
    """
    return get_config().load("camera_config")


def load_analysis_config() -> dict[str, Any]:
    """Load analysis configuration.

    Returns:
        Analysis configuration dictionary.
    """
    return get_config().load("analysis_config")
