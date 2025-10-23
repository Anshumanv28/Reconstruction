"""
Model Controller - Handles model-related operations
"""
from typing import Dict, List
from ..services.model_service import ModelService


class ModelController:
    """Controller for model-related operations"""
    
    def __init__(self):
        self.model_service = ModelService()
    
    def list_models(self) -> Dict[str, List[str]]:
        """Get list of available models"""
        return self.model_service.get_available_models()
    
    def get_model_info(self) -> Dict[str, Dict[str, str]]:
        """Get detailed information about available models"""
        return self.model_service.get_model_info()
