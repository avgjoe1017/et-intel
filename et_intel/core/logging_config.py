"""
Centralized Logging Configuration
Ensures consistent logging across all modules
"""

import logging
import sys
from pathlib import Path
from .. import config

def setup_logging(level=None, log_file=None):
    """
    Configure logging for the entire application
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Defaults to config.LOG_LEVEL
        log_file: Path to log file. Defaults to config.LOG_FILE
                 Set to None to disable file logging
    
    Returns:
        logger: Root logger instance
    """
    log_level = level or config.LOG_LEVEL
    log_file_path = log_file or config.LOG_FILE
    
    # Get root logger
    root_logger = logging.getLogger('et_intel')
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers (prevents duplicate logs)
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        config.LOG_FORMAT,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file_path:
        # Ensure log directory exists
        log_path = Path(log_file_path)
        if log_path.parent != Path('.'):
            log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Prevent propagation to root logger (avoid duplicate logs)
    root_logger.propagate = False
    
    return root_logger

def get_logger(name):
    """
    Get a logger for a specific module
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        logger: Logger instance
    """
    # Ensure root logger is configured
    root_logger = logging.getLogger('et_intel')
    if not root_logger.handlers:
        setup_logging()
    
    return logging.getLogger(f'et_intel.{name}')

