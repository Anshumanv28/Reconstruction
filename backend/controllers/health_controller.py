"""
Health Controller - Handles health check operations
"""
from typing import Dict


class HealthController:
    """Controller for health check operations"""
    
    @staticmethod
    def health_check() -> Dict[str, str]:
        """Perform health check"""
        return {"status": "ok"}
