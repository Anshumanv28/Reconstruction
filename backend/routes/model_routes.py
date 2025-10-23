"""
Model Routes - Model-related endpoints
"""
from fastapi import APIRouter
from ..controllers.model_controller import ModelController

router = APIRouter(prefix="/models", tags=["models"])
model_controller = ModelController()


@router.get("/")
def list_models():
    """Get list of available models"""
    return model_controller.list_models()


@router.get("/info")
def get_model_info():
    """Get detailed information about available models"""
    return model_controller.get_model_info()
