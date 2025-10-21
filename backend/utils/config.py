"""
Configuration utilities
"""
import os
from typing import Dict, Any


class Config:
    """Configuration class for application settings"""
    
    # API Settings
    API_TITLE = "Reconstruction Model Hub"
    API_VERSION = "0.1.0"
    API_DESCRIPTION = "Pluggable backend for layout detection and OCR models"
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS Settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_CREDENTIALS = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"
    
    # File Upload Settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 50 * 1024 * 1024))  # 50MB
    ALLOWED_FILE_TYPES = ["image/png", "image/jpeg", "image/jpg", "application/pdf"]
    
    # Model Settings
    MODEL_CACHE_SIZE = int(os.getenv("MODEL_CACHE_SIZE", 2))
    MODEL_LOAD_TIMEOUT = int(os.getenv("MODEL_LOAD_TIMEOUT", 300))  # 5 minutes
    
    # Image Processing Settings
    MAX_IMAGE_WIDTH = int(os.getenv("MAX_IMAGE_WIDTH", 10000))
    MAX_IMAGE_HEIGHT = int(os.getenv("MAX_IMAGE_HEIGHT", 10000))
    DEFAULT_IMAGE_QUALITY = int(os.getenv("DEFAULT_IMAGE_QUALITY", 95))
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get all configuration settings as a dictionary"""
        return {
            "api": {
                "title": cls.API_TITLE,
                "version": cls.API_VERSION,
                "description": cls.API_DESCRIPTION
            },
            "server": {
                "host": cls.HOST,
                "port": cls.PORT,
                "debug": cls.DEBUG
            },
            "cors": {
                "origins": cls.CORS_ORIGINS,
                "credentials": cls.CORS_CREDENTIALS
            },
            "file_upload": {
                "max_file_size": cls.MAX_FILE_SIZE,
                "allowed_types": cls.ALLOWED_FILE_TYPES
            },
            "models": {
                "cache_size": cls.MODEL_CACHE_SIZE,
                "load_timeout": cls.MODEL_LOAD_TIMEOUT
            },
            "image_processing": {
                "max_width": cls.MAX_IMAGE_WIDTH,
                "max_height": cls.MAX_IMAGE_HEIGHT,
                "default_quality": cls.DEFAULT_IMAGE_QUALITY
            }
        }
