from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import json
import tempfile
import os
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF for PDF support

from .models.registry import layout_registry, ocr_registry
from .utils.visualize import render_overlay


app = FastAPI(title="Reconstruction Model Hub", version="0.1.0")

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceImage(BaseModel):
    image_id: str
    image_b64: str


class InferenceRequest(BaseModel):
    inputs: List[InferenceImage]
    layout_model: Optional[str] = None
    ocr_model: Optional[str] = None
    params: Dict[str, Any] = {}
    return_visualization: bool = True


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def list_models() -> Dict[str, List[str]]:
    return {
        "layout_models": list(layout_registry.keys()),
        "ocr_models": list(ocr_registry.keys()),
    }


def decode_image(image_b64: str) -> Image.Image:
    # Accept data URLs or raw base64
    payload = image_b64.split(",")[-1]
    data = base64.b64decode(payload)
    pil = Image.open(BytesIO(data)).convert("RGB")
    return pil


def process_uploaded_file(file: UploadFile) -> Image.Image:
    """Process uploaded file (PNG, JPG, PDF) and return PIL Image"""
    content = file.file.read()
    
    # Check file type
    if file.content_type == "application/pdf":
        # Convert PDF to image (first page)
        pdf_doc = fitz.open(stream=content, filetype="pdf")
        page = pdf_doc[0]  # First page
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        pdf_doc.close()
        return Image.open(BytesIO(img_data)).convert("RGB")
    
    elif file.content_type in ["image/png", "image/jpeg", "image/jpg"]:
        # Direct image processing
        return Image.open(BytesIO(content)).convert("RGB")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")


@app.post("/infer")
def infer(req: InferenceRequest) -> Dict[str, Any]:
    layout_model = None
    ocr_model = None

    if req.layout_model:
        if req.layout_model not in layout_registry:
            raise HTTPException(status_code=400, detail=f"Unknown layout model: {req.layout_model}")
        layout_model = layout_registry[req.layout_model]
        layout_model.load()

    if req.ocr_model:
        if req.ocr_model not in ocr_registry:
            raise HTTPException(status_code=400, detail=f"Unknown OCR model: {req.ocr_model}")
        ocr_model = ocr_registry[req.ocr_model]
        ocr_model.load()

    results: List[Dict[str, Any]] = []

    for item in req.inputs:
        pil_image = decode_image(item.image_b64)

        layout_out: Dict[str, Any] = {"boxes": []}
        ocr_out: Dict[str, Any] = {"tokens": []}

        if layout_model:
            layout_out = layout_model.predict(pil_image, req.params.get("layout", {}))

        if ocr_model:
            ocr_out = ocr_model.predict(pil_image, req.params.get("ocr", {}), layout_out)

        viz_b64: Optional[str] = None
        if req.return_visualization:
            overlay_img = render_overlay(pil_image, layout_out, ocr_out)
            if overlay_img is not None:
                buf = BytesIO()
                overlay_img.save(buf, format="PNG")
                viz_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        results.append(
            {
                "image_id": item.image_id,
                "layout": layout_out,
                "ocr": ocr_out,
                "visualization": viz_b64,
            }
        )

    return {"results": results}


@app.post("/infer-file")
async def infer_file(
    file: UploadFile = File(...),
    layout_model: Optional[str] = Form(None),
    ocr_model: Optional[str] = Form(None),
    return_visualization: bool = Form(True)
) -> Dict[str, Any]:
    """Infer from uploaded file (PNG, JPG, PDF)"""
    
    # Validate models
    if layout_model and layout_model not in layout_registry:
        raise HTTPException(status_code=400, detail=f"Unknown layout model: {layout_model}")
    if ocr_model and ocr_model not in ocr_registry:
        raise HTTPException(status_code=400, detail=f"Unknown OCR model: {ocr_model}")
    
    # Load models
    layout = None
    ocr = None
    
    if layout_model:
        layout = layout_registry[layout_model]
        layout.load()
    
    if ocr_model:
        ocr = ocr_registry[ocr_model]
        ocr.load()
    
    # Process uploaded file
    try:
        pil_image = process_uploaded_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    # Run inference
    layout_out: Dict[str, Any] = {"boxes": []}
    ocr_out: Dict[str, Any] = {"tokens": []}
    
    if layout:
        layout_out = layout.predict(pil_image, {})
    
    if ocr:
        ocr_out = ocr.predict(pil_image, {}, layout_out)
    
    # Generate visualization
    viz_b64: Optional[str] = None
    if return_visualization:
        overlay_img = render_overlay(pil_image, layout_out, ocr_out)
        if overlay_img is not None:
            buf = BytesIO()
            overlay_img.save(buf, format="PNG")
            viz_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "layout": layout_out,
        "ocr": ocr_out,
        "visualization": viz_b64,
    }


@app.post("/infer-file-with-files")
async def infer_file_with_files(
    file: UploadFile = File(...),
    layout_model: Optional[str] = Form(None),
    ocr_model: Optional[str] = Form(None)
):
    """Infer from uploaded file and return both annotated image and JSON data as files"""
    
    # Validate models
    if layout_model and layout_model not in layout_registry:
        raise HTTPException(status_code=400, detail=f"Unknown layout model: {layout_model}")
    if ocr_model and ocr_model not in ocr_registry:
        raise HTTPException(status_code=400, detail=f"Unknown OCR model: {ocr_model}")
    
    # Load models
    layout = None
    ocr = None
    
    if layout_model:
        layout = layout_registry[layout_model]
        layout.load()
    
    if ocr_model:
        ocr = ocr_registry[ocr_model]
        ocr.load()
    
    # Process uploaded file
    try:
        pil_image = process_uploaded_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    # Run inference
    layout_out: Dict[str, Any] = {"boxes": []}
    ocr_out: Dict[str, Any] = {"tokens": []}
    
    if layout:
        layout_out = layout.predict(pil_image, {})
    
    if ocr:
        ocr_out = ocr.predict(pil_image, {}, layout_out)
    
    # Create temporary directory for output files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate annotated image
        overlay_img = render_overlay(pil_image, layout_out, ocr_out)
        annotated_path = os.path.join(temp_dir, f"annotated_{file.filename}")
        
        if overlay_img is not None:
            overlay_img.save(annotated_path, format="PNG")
        else:
            # If no overlay, save original image
            pil_image.save(annotated_path, format="PNG")
        
        # Generate JSON data file
        json_data = {
            "filename": file.filename,
            "content_type": file.content_type,
            "layout": layout_out,
            "ocr": ocr_out,
            "metadata": {
                "image_width": pil_image.width,
                "image_height": pil_image.height,
                "layout_model": layout_model,
                "ocr_model": ocr_model
            }
        }
        
        json_path = os.path.join(temp_dir, f"data_{file.filename}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Create a zip file containing both files
        import zipfile
        zip_path = os.path.join(temp_dir, f"results_{file.filename}.zip")
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(annotated_path, f"annotated_{file.filename}")
            zipf.write(json_path, f"data_{file.filename}.json")
        
        # Return the zip file
        return FileResponse(
            path=zip_path,
            filename=f"results_{file.filename}.zip",
            media_type="application/zip"
        )


@app.post("/infer-file-json")
async def infer_file_json(
    file: UploadFile = File(...),
    layout_model: Optional[str] = Form(None),
    ocr_model: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """Infer from uploaded file and return only the raw JSON data"""
    
    # Validate models
    if layout_model and layout_model not in layout_registry:
        raise HTTPException(status_code=400, detail=f"Unknown layout model: {layout_model}")
    if ocr_model and ocr_model not in ocr_registry:
        raise HTTPException(status_code=400, detail=f"Unknown OCR model: {ocr_model}")
    
    # Load models
    layout = None
    ocr = None
    
    if layout_model:
        layout = layout_registry[layout_model]
        layout.load()
    
    if ocr_model:
        ocr = ocr_registry[ocr_model]
        ocr.load()
    
    # Process uploaded file
    try:
        pil_image = process_uploaded_file(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    # Run inference
    layout_out: Dict[str, Any] = {"boxes": []}
    ocr_out: Dict[str, Any] = {"tokens": []}
    
    if layout:
        layout_out = layout.predict(pil_image, {})
    
    if ocr:
        ocr_out = ocr.predict(pil_image, {}, layout_out)
    
    # Return raw JSON data
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "layout": layout_out,
        "ocr": ocr_out,
        "metadata": {
            "image_width": pil_image.width,
            "image_height": pil_image.height,
            "layout_model": layout_model,
            "ocr_model": ocr_model
        }
    }


