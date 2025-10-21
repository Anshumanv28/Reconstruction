"""
File Service - Handles file processing operations
"""
import os
from typing import Optional
from fastapi import UploadFile, HTTPException
from PIL import Image
import fitz  # PyMuPDF for PDF support
from io import BytesIO


class FileService:
    """Service for handling file processing operations"""
    
    @staticmethod
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
    
    @staticmethod
    def validate_file_type(file: UploadFile) -> bool:
        """Validate that the uploaded file is a supported type"""
        supported_types = ["application/pdf", "image/png", "image/jpeg", "image/jpg"]
        return file.content_type in supported_types
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Get file extension from filename"""
        return os.path.splitext(filename)[1].lower()
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe file operations"""
        import re
        # Remove or replace unsafe characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return safe_filename
