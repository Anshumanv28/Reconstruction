"""
Main FastAPI Application - Restructured with MVC architecture
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import health_routes, model_routes, inference_routes

# Create FastAPI app
app = FastAPI(
    title="Reconstruction Model Hub", 
    version="0.1.0",
    description="Pluggable backend for layout detection and OCR models"
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_routes.router)
app.include_router(model_routes.router)
app.include_router(inference_routes.router)

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Reconstruction Model Hub API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health/",
        "models": "/models/",
        "inference": "/infer/"
    }
