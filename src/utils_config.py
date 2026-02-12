"""
Configuration management utilities.
Handles loading, parsing, and accessing configuration parameters.
"""

import yaml
import json
from pathlib import Path
from typing import Any, Dict
from src.utils_logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Manager for application configuration."""
    
    def __init__(self, config_file: str = "config/config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (YAML or JSON)
        """
        self.config_file = Path(config_file)
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        if not self.config_file.exists():
            logger.warning(f"Config file not found: {self.config_file}. Using defaults.")
            self.config = self._get_defaults()
            return
        
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.suffix.lower() == '.yaml':
                    self.config = yaml.safe_load(f) or {}
                elif self.config_file.suffix.lower() == '.json':
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_file.suffix}")
            
            logger.info(f"Configuration loaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self._get_defaults()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'robot.num_joints')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., 'robot.num_joints')
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def get_robot_config(self) -> Dict[str, Any]:
        """Get robot configuration."""
        return self.config.get('robot', {})
    
    def get_nn_config(self) -> Dict[str, Any]:
        """Get neural network configuration."""
        return self.config.get('nn_model', {})
    
    def get_gazebo_config(self) -> Dict[str, Any]:
        """Get Gazebo configuration."""
        return self.config.get('gazebo', {})
    
    def get_ik_solver_config(self) -> Dict[str, Any]:
        """Get IK solver configuration."""
        return self.config.get('ik_solver', {})
    
    def save_config(self, filepath: str = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            filepath: Path to save config (default: original file)
        """
        filepath = filepath or str(self.config_file)
        
        try:
            with open(filepath, 'w') as f:
                if filepath.endswith('.yaml'):
                    yaml.dump(self.config, f, default_flow_style=False)
                elif filepath.endswith('.json'):
                    json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @staticmethod
    def _get_defaults() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'robot': {
                'name': 'robotic_arm',
                'num_joints': 6,
                'urdf_file': 'assets/urdf/my_robot.urdf'
            },
            'nn_model': {
                'input_size': 7,
                'output_size': 6,
                'model_path': 'ik_nn_model.pth'
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs'
            }
        }


# Global config instance
_config_manager = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
