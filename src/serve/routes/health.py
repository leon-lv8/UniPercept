from fastapi import APIRouter

from ..health.health_report import health

router = APIRouter()
router.add_api_route("/health", health, methods=["GET"])
