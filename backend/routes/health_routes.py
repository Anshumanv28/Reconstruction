"""
Health Routes - Health check endpoints
"""
from fastapi import APIRouter
from controllers.health_controller import HealthController

router = APIRouter(prefix="/health", tags=["health"])
health_controller = HealthController()


@router.get("/")
def health_check():
    """Health check endpoint"""
    return health_controller.health_check()
