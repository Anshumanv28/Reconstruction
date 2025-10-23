"""
Image Helper - Utility functions for image processing
"""
from typing import Optional
from PIL import Image
import base64
from io import BytesIO

from ..utils.visualize import render_overlay


class ImageHelper:
    """Helper class for image processing operations"""
    
    @staticmethod
    def decode_base64_image(image_b64: str) -> Image.Image:
        """Decode base64 image string to PIL Image"""
        # Accept data URLs or raw base64
        payload = image_b64.split(",")[-1]
        data = base64.b64decode(payload)
        pil = Image.open(BytesIO(data)).convert("RGB")
        return pil
    
    @staticmethod
    def encode_image_to_base64(image: Image.Image, format: str = "PNG") -> str:
        """Encode PIL Image to base64 string"""
        buf = BytesIO()
        image.save(buf, format=format)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    
    @staticmethod
    def create_base64_visualization(image: Image.Image, layout_out: dict, 
                                  ocr_out: dict) -> Optional[str]:
        """Create base64 visualization from image and inference results"""
        overlay_img = render_overlay(image, layout_out, ocr_out)
        if overlay_img is not None:
            buf = BytesIO()
            overlay_img.save(buf, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        return None
    
    @staticmethod
    def resize_image(image: Image.Image, max_width: int = 1024, max_height: int = 1024) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        # Calculate new dimensions
        width, height = image.size
        ratio = min(max_width / width, max_height / height)
        
        if ratio < 1:
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    @staticmethod
    def get_image_info(image: Image.Image) -> dict:
        """Get basic information about an image"""
        return {
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "format": image.format
        }
