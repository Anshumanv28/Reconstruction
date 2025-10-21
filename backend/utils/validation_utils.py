"""
Validation utilities for input validation
"""
from typing import Optional, List
import re


class ValidationUtils:
    """Utility class for input validation"""
    
    @staticmethod
    def validate_model_name(model_name: str, available_models: List[str]) -> bool:
        """Validate that a model name exists in the available models list"""
        return model_name in available_models
    
    @staticmethod
    def validate_filename(filename: str) -> bool:
        """Validate filename format"""
        if not filename or len(filename) > 255:
            return False
        
        # Check for invalid characters
        invalid_chars = r'[<>:"/\\|?*]'
        if re.search(invalid_chars, filename):
            return False
            
        return True
    
    @staticmethod
    def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
        """Validate file extension"""
        if not filename:
            return False
            
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        return extension in allowed_extensions
    
    @staticmethod
    def validate_image_dimensions(width: int, height: int, max_width: int = 10000, max_height: int = 10000) -> bool:
        """Validate image dimensions"""
        return 0 < width <= max_width and 0 < height <= max_height
    
    @staticmethod
    def sanitize_string(input_string: str) -> str:
        """Sanitize string input"""
        if not input_string:
            return ""
        
        # Remove or replace potentially dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', input_string)
        return sanitized.strip()
