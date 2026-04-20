from fastapi import APIRouter

from nexus.application.container import configure, shutdown

from .endpoint import realtime_endpoint_worker
from .http import create_client_secret_endpoint, create_realtime_call_endpoint

router = APIRouter(tags=["Realtime"])
router.websocket("/realtime")(realtime_endpoint_worker)
router.post("/realtime/client_secrets")(create_client_secret_endpoint)
router.post("/realtime/calls")(create_realtime_call_endpoint)

__all__ = ["router", "configure", "shutdown", "realtime_endpoint_worker"]
