import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_empty_values(value: Any) -> Any:
    """
    Normalize empty or null-like values to consistent Python types.
    
    Args:
        value: Any value that might represent "empty" or "null"
        
    Returns:
        Normalized value (None for null, [] for empty lists, {} for empty dicts)
    """
    if value is None:
        return None
        
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ['null', 'none', 'no information available', 'not available', 
                         'not found', 'no data', 'unknown', '']:
            return None
        return value
    
    if isinstance(value, list):
        return [] if not value else value
        
    if isinstance(value, dict):
        return {} if not value else value
        
    return value