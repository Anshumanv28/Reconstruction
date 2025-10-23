"""
Inference Service - Handles all model inference operations
"""
from typing import Dict, Any, Optional, List
from PIL import Image
import tempfile
import os
import json
import zipfile

from ..models.registry import layout_registry, ocr_registry
from ..utils.visualize import render_overlay


class InferenceService:
    """Service for handling model inference operations"""
    
    def __init__(self):
        self.layout_registry = layout_registry
        self.ocr_registry = ocr_registry
    
    def validate_models(self, layout_model: Optional[str], ocr_model: Optional[str]) -> None:
        """Validate that the specified models exist in the registry"""
        if layout_model and layout_model not in self.layout_registry:
            raise ValueError(f"Unknown layout model: {layout_model}")
        if ocr_model and ocr_model not in self.ocr_registry:
            raise ValueError(f"Unknown OCR model: {ocr_model}")
    
    def load_models(self, layout_model: Optional[str], ocr_model: Optional[str]) -> tuple:
        """Load the specified models"""
        layout = None
        ocr = None
        
        if layout_model:
            layout = self.layout_registry[layout_model]
            layout.load()
        
        if ocr_model:
            ocr = self.ocr_registry[ocr_model]
            ocr.load()
        
        return layout, ocr
    
    def run_inference(self, image: Image.Image, layout_model: Optional[str], ocr_model: Optional[str], 
                     layout_params: Dict[str, Any] = None, ocr_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run inference on an image with the specified models"""
        # Validate models
        self.validate_models(layout_model, ocr_model)
        
        # Load models
        layout, ocr = self.load_models(layout_model, ocr_model)
        
        # Initialize outputs
        layout_out: Dict[str, Any] = {"boxes": []}
        ocr_out: Dict[str, Any] = {"tokens": []}
        
        # Run layout inference
        if layout:
            layout_out = layout.predict(image, layout_params or {})
        
        # Run OCR inference (can use layout output for guidance)
        if ocr:
            ocr_out = ocr.predict(image, ocr_params or {}, layout_out)
        
        return {
            "layout": layout_out,
            "ocr": ocr_out
        }
    
    def generate_visualization(self, image: Image.Image, layout_out: Dict[str, Any], 
                             ocr_out: Dict[str, Any]) -> Optional[Image.Image]:
        """Generate visualization overlay for the inference results"""
        return render_overlay(image, layout_out, ocr_out)
    
    def create_result_files(self, image: Image.Image, layout_out: Dict[str, Any], 
                          ocr_out: Dict[str, Any], filename: str, content_type: str,
                          layout_model: Optional[str], ocr_model: Optional[str]) -> tuple:
        """Create annotated image and JSON data files"""
        # Generate annotated image
        overlay_img = self.generate_visualization(image, layout_out, ocr_out)
        
        # Create temporary directory for output files
        temp_dir = tempfile.mkdtemp()
        
        # Save annotated image
        annotated_path = os.path.join(temp_dir, f"annotated_{filename}")
        if overlay_img is not None:
            overlay_img.save(annotated_path, format="PNG")
        else:
            # If no overlay, save original image
            image.save(annotated_path, format="PNG")
        
        # Create JSON data
        json_data = {
            "filename": filename,
            "content_type": content_type,
            "layout": layout_out,
            "ocr": ocr_out,
            "metadata": {
                "image_width": image.width,
                "image_height": image.height,
                "layout_model": layout_model,
                "ocr_model": ocr_model
            }
        }
        
        # Save JSON data
        json_path = os.path.join(temp_dir, f"data_{filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        return annotated_path, json_path, temp_dir
    
    def create_zip_file(self, annotated_path: str, json_path: str, temp_dir: str, filename: str) -> str:
        """Create a ZIP file containing the annotated image and JSON data"""
        zip_path = os.path.join(temp_dir, f"results_{filename}.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(annotated_path, f"annotated_{filename}")
            zipf.write(json_path, f"data_{filename}.json")
        
        return zip_path
