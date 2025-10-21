"""
Inference Controller - Handles inference operations
"""
from typing import Dict, Any, List, Optional
from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import base64
from io import BytesIO
import os

from services.inference_service import InferenceService
from services.file_service import FileService
from helpers.image_helper import ImageHelper


class InferenceController:
    """Controller for inference operations"""
    
    def __init__(self):
        self.inference_service = InferenceService()
        self.file_service = FileService()
        self.image_helper = ImageHelper()
    
    def infer_base64(self, inputs: List[Dict[str, Any]], layout_model: Optional[str], 
                    ocr_model: Optional[str], params: Dict[str, Any], 
                    return_visualization: bool) -> Dict[str, Any]:
        """Handle base64 image inference"""
        results: List[Dict[str, Any]] = []
        
        for item in inputs:
            try:
                # Decode base64 image
                pil_image = self.image_helper.decode_base64_image(item["image_b64"])
                
                # Run inference
                inference_result = self.inference_service.run_inference(
                    pil_image, layout_model, ocr_model, 
                    params.get("layout", {}), params.get("ocr", {})
                )
                
                # Generate visualization if requested
                viz_b64: Optional[str] = None
                if return_visualization:
                    viz_b64 = self.image_helper.create_base64_visualization(
                        pil_image, inference_result["layout"], inference_result["ocr"]
                    )
                
                results.append({
                    "image_id": item["image_id"],
                    "layout": inference_result["layout"],
                    "ocr": inference_result["ocr"],
                    "visualization": viz_b64,
                })
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error processing image {item['image_id']}: {str(e)}")
        
        return {"results": results}
    
    def infer_file(self, file: UploadFile, layout_model: Optional[str], 
                  ocr_model: Optional[str], return_visualization: bool) -> Dict[str, Any]:
        """Handle file upload inference with JSON response"""
        try:
            # Process uploaded file
            pil_image = self.file_service.process_uploaded_file(file)
            
            # Run inference
            inference_result = self.inference_service.run_inference(
                pil_image, layout_model, ocr_model, {}, {}
            )
            
            # Generate visualization if requested
            viz_b64: Optional[str] = None
            if return_visualization:
                viz_b64 = self.image_helper.create_base64_visualization(
                    pil_image, inference_result["layout"], inference_result["ocr"]
                )
            
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "layout": inference_result["layout"],
                "ocr": inference_result["ocr"],
                "visualization": viz_b64,
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    def infer_file_with_files(self, file: UploadFile, layout_model: Optional[str], 
                            ocr_model: Optional[str]) -> FileResponse:
        """Handle file upload inference with file downloads"""
        try:
            # Process uploaded file
            pil_image = self.file_service.process_uploaded_file(file)
            
            # Run inference
            inference_result = self.inference_service.run_inference(
                pil_image, layout_model, ocr_model, {}, {}
            )
            
            # Create result files
            annotated_path, json_path, temp_dir = self.inference_service.create_result_files(
                pil_image, inference_result["layout"], inference_result["ocr"],
                file.filename, file.content_type, layout_model, ocr_model
            )
            
            # Create ZIP file
            zip_path = self.inference_service.create_zip_file(
                annotated_path, json_path, temp_dir, file.filename
            )
            
            # Return the ZIP file
            return FileResponse(
                path=zip_path,
                filename=f"results_{file.filename}.zip",
                media_type="application/zip"
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    def infer_file_json(self, file: UploadFile, layout_model: Optional[str], 
                       ocr_model: Optional[str]) -> Dict[str, Any]:
        """Handle file upload inference with JSON response only"""
        try:
            # Process uploaded file
            pil_image = self.file_service.process_uploaded_file(file)
            
            # Run inference
            inference_result = self.inference_service.run_inference(
                pil_image, layout_model, ocr_model, {}, {}
            )
            
            # Return raw JSON data
            return {
                "filename": file.filename,
                "content_type": file.content_type,
                "layout": inference_result["layout"],
                "ocr": inference_result["ocr"],
                "metadata": {
                    "image_width": pil_image.width,
                    "image_height": pil_image.height,
                    "layout_model": layout_model,
                    "ocr_model": ocr_model
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
