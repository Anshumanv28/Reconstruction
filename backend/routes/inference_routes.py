"""
Inference Routes - Inference endpoints
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pydantic import BaseModel

from controllers.inference_controller import InferenceController

router = APIRouter(prefix="/infer", tags=["inference"])
inference_controller = InferenceController()


class InferenceImage(BaseModel):
    image_id: str
    image_b64: str


class InferenceRequest(BaseModel):
    inputs: List[InferenceImage]
    layout_model: Optional[str] = None
    ocr_model: Optional[str] = None
    params: Dict[str, Any] = {}
    return_visualization: bool = True


@router.post("/")
def infer_base64(req: InferenceRequest):
    """Run inference on base64 encoded images"""
    return inference_controller.infer_base64(
        req.inputs, req.layout_model, req.ocr_model, 
        req.params, req.return_visualization
    )


@router.post("/file")
async def infer_file(
    file: UploadFile = File(...),
    layout_model: Optional[str] = Form(None),
    ocr_model: Optional[str] = Form(None),
    return_visualization: bool = Form(True)
):
    """Infer from uploaded file (PNG, JPG, PDF) with JSON response"""
    return inference_controller.infer_file(
        file, layout_model, ocr_model, return_visualization
    )


@router.post("/file-with-files")
async def infer_file_with_files(
    file: UploadFile = File(...),
    layout_model: Optional[str] = Form(None),
    ocr_model: Optional[str] = Form(None)
):
    """Infer from uploaded file and return both annotated image and JSON data as files"""
    return inference_controller.infer_file_with_files(
        file, layout_model, ocr_model
    )


@router.post("/file-json")
async def infer_file_json(
    file: UploadFile = File(...),
    layout_model: Optional[str] = Form(None),
    ocr_model: Optional[str] = Form(None)
):
    """Infer from uploaded file and return only the raw JSON data"""
    return inference_controller.infer_file_json(
        file, layout_model, ocr_model
    )
