import logging
import os
from datetime import datetime
from typing import Optional

def setup_application_logging(
    log_level: str = 'INFO', 
    log_file: Optional[str] = None,
    log_dir: str = 'logs'
) -> str:
    """
    Setup logging configuration for the entire application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Custom log file name (optional)
        log_dir: Directory to store log files
        
    Returns:
        Path to the log file
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f'analysis_pipeline_{timestamp}.log'
    
    log_file_path = os.path.join(log_dir, log_file)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler with rotation
            logging.FileHandler(log_file_path, encoding='utf-8', mode='w')
        ],
        force=True  # Override any existing configuration
    )
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}, File: {log_file_path}")
    
    return log_file_path

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)