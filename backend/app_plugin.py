"""
Plugin-based FastAPI Application - Fully pluggable model system
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .services.model_management_service import ModelManagementService
from .services.inference_service import InferenceService
from .services.file_service import FileService
from .controllers.health_controller import HealthController
from .controllers.model_controller import ModelController
from .controllers.inference_controller import InferenceController
from .routes import health_routes, model_routes, inference_routes

# Create FastAPI app
app = FastAPI(
    title="Pluggable Reconstruction Model Hub", 
    version="0.2.0",
    description="Fully pluggable backend for layout detection and OCR models with dynamic model loading"
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_management_service = ModelManagementService()

# Include routers
app.include_router(health_routes.router)
app.include_router(model_routes.router)
app.include_router(inference_routes.router)

# Add new plugin management endpoints
@app.get("/plugins")
def list_plugins():
    """List all available model plugins"""
    return model_management_service.get_available_models()

@app.get("/plugins/status")
def get_plugin_status():
    """Get status of all model plugins"""
    return model_management_service.get_model_status()

@app.post("/plugins/reload")
def reload_plugins():
    """Reload all model plugins"""
    try:
        model_management_service.reload_all_models()
        return {"message": "All plugins reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload plugins: {str(e)}")

@app.post("/plugins/{model_type}/{model_name}/load")
def load_plugin(model_type: str, model_name: str):
    """Load a specific model plugin"""
    try:
        model = model_management_service.load_model(model_name, model_type)
        return {
            "message": f"Model '{model_name}' loaded successfully",
            "model_info": model.get_info()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {str(e)}")

@app.post("/plugins/{model_type}/{model_name}/unload")
def unload_plugin(model_type: str, model_name: str):
    """Unload a specific model plugin"""
    try:
        model_management_service.unload_model(model_name, model_type)
        return {"message": f"Model '{model_name}' unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to unload model: {str(e)}")

@app.post("/plugins/unload-all")
def unload_all_plugins():
    """Unload all model plugins"""
    try:
        model_management_service.unload_all_models()
        return {"message": "All models unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Pluggable Reconstruction Model Hub API",
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health/",
        "models": "/models/",
        "plugins": "/plugins/",
        "inference": "/infer/"
    }
