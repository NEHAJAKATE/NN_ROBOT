"""
Logging utility module for NN IK robotics project.
Provides centralized logging configuration and utilities.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


class RobotLogger:
    """Centralized logging manager for the NN IK project."""
    
    _loggers = {}
    
    @staticmethod
    def setup_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
        """
        Set up and return a logger with the specified configuration.
        
        Args:
            name: Logger name (usually __name__)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            
        Returns:
            Configured logger instance
        """
        if name in RobotLogger._loggers:
            return RobotLogger._loggers[name]
        
        # Create logs directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"nn_ik_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
        
        RobotLogger._loggers[name] = logger
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return RobotLogger.setup_logger(name)
