"""
Model Service - Handles model registry and management operations
"""
from typing import Dict, List
from models.registry import layout_registry, ocr_registry


class ModelService:
    """Service for handling model registry and management operations"""
    
    def __init__(self):
        self.layout_registry = layout_registry
        self.ocr_registry = ocr_registry
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available layout and OCR models"""
        return {
            "layout_models": list(self.layout_registry.keys()),
            "ocr_models": list(self.ocr_registry.keys()),
        }
    
    def get_layout_models(self) -> List[str]:
        """Get list of available layout models"""
        return list(self.layout_registry.keys())
    
    def get_ocr_models(self) -> List[str]:
        """Get list of available OCR models"""
        return list(self.ocr_registry.keys())
    
    def is_layout_model_available(self, model_name: str) -> bool:
        """Check if a layout model is available"""
        return model_name in self.layout_registry
    
    def is_ocr_model_available(self, model_name: str) -> bool:
        """Check if an OCR model is available"""
        return model_name in self.ocr_registry
    
    def get_model_info(self) -> Dict[str, Dict[str, str]]:
        """Get detailed information about available models"""
        return {
            "layout_models": {
                model_name: "IBM Docling + TableFormer layout detection" 
                for model_name in self.layout_registry.keys()
            },
            "ocr_models": {
                model_name: "Tesseract OCR text extraction"
                for model_name in self.ocr_registry.keys()
            }
        }
