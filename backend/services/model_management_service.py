"""
Model Management Service - Handles dynamic model loading and management
"""
from typing import Dict, Any, List, Optional
from ..models.registry_new import layout_registry, ocr_registry
from ..models.base_model import BaseModel


class ModelManagementService:
    """Service for managing model lifecycle and operations"""
    
    def __init__(self):
        self.layout_registry = layout_registry
        self.ocr_registry = ocr_registry
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models by type"""
        return {
            "layout_models": list(self.layout_registry.get_models_by_type("layout").keys()),
            "ocr_models": list(self.ocr_registry.get_models_by_type("ocr").keys()),
        }
    
    def get_model_info(self, model_name: str, model_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model"""
        registry = self.layout_registry if model_type == "layout" else self.ocr_registry
        return registry.get_model_info(model_name)
    
    def load_model(self, model_name: str, model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseModel:
        """Load a specific model"""
        registry = self.layout_registry if model_type == "layout" else self.ocr_registry
        
        if not registry.is_model_available(model_name):
            raise ValueError(f"Model '{model_name}' of type '{model_type}' not available")
        
        return registry.load_model(model_name, config)
    
    def unload_model(self, model_name: str, model_type: str) -> None:
        """Unload a specific model"""
        registry = self.layout_registry if model_type == "layout" else self.ocr_registry
        registry.unload_model(model_name)
    
    def unload_all_models(self) -> None:
        """Unload all models to free memory"""
        self.layout_registry.unload_all_models()
        self.ocr_registry.unload_all_models()
    
    def add_model_plugin(self, model_name: str, model_type: str, 
                        class_path: str, config: Dict[str, Any] = None) -> None:
        """Add a new model plugin"""
        registry = self.layout_registry if model_type == "layout" else self.ocr_registry
        registry.add_model_plugin(model_name, model_type, class_path, config)
    
    def remove_model_plugin(self, model_name: str, model_type: str) -> None:
        """Remove a model plugin"""
        registry = self.layout_registry if model_type == "layout" else self.ocr_registry
        registry.remove_model_plugin(model_name)
    
    def load_config_file(self, config_file: str) -> None:
        """Load model configurations from a file"""
        self.layout_registry.load_config_file(config_file)
        self.ocr_registry.load_config_file(config_file)
    
    def save_config_file(self, config_file: str, config: Dict[str, Any]) -> None:
        """Save model configurations to a file"""
        self.layout_registry.save_config_file(config_file, config)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        layout_models = self.layout_registry.get_available_models()
        ocr_models = self.ocr_registry.get_available_models()
        
        return {
            "layout_models": {
                name: {
                    "available": True,
                    "loaded": name in self.layout_registry._model_instances,
                    "info": self.layout_registry.get_model_info(name)
                }
                for name in layout_models.keys()
            },
            "ocr_models": {
                name: {
                    "available": True,
                    "loaded": name in self.ocr_registry._model_instances,
                    "info": self.ocr_registry.get_model_info(name)
                }
                for name in ocr_models.keys()
            }
        }
    
    def reload_all_models(self) -> None:
        """Reload all model configurations"""
        self.layout_registry.reload_models()
        self.ocr_registry.reload_models()
    
    def validate_model_combination(self, layout_model: Optional[str], ocr_model: Optional[str]) -> bool:
        """Validate that the model combination is valid"""
        if layout_model and not self.layout_registry.is_model_available(layout_model):
            return False
        if ocr_model and not self.ocr_registry.is_model_available(ocr_model):
            return False
        return True
