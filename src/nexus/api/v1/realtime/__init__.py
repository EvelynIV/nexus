from fastapi import APIRouter

from nexus.application.container import configure, shutdown

from .endpoint import realtime_endpoint_worker

router = APIRouter(tags=["Realtime"])
router.websocket("/realtime")(realtime_endpoint_worker)

__all__ = ["router", "configure", "shutdown", "realtime_endpoint_worker"]
