"""
Response utilities for API responses
"""
from typing import Dict, Any, Optional
from fastapi.responses import JSONResponse


class ResponseUtils:
    """Utility class for creating standardized API responses"""
    
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> JSONResponse:
        """Create a standardized success response"""
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": message,
                "data": data
            }
        )
    
    @staticmethod
    def error_response(message: str, status_code: int = 400, details: Optional[Dict] = None) -> JSONResponse:
        """Create a standardized error response"""
        content = {
            "success": False,
            "message": message
        }
        if details:
            content["details"] = details
            
        return JSONResponse(
            status_code=status_code,
            content=content
        )
    
    @staticmethod
    def validation_error_response(errors: list) -> JSONResponse:
        """Create a validation error response"""
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "message": "Validation error",
                "errors": errors
            }
        )
