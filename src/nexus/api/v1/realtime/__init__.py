from fastapi import APIRouter

from nexus.application.container import configure, shutdown

from .endpoint import realtime_endpoint_worker
from .http import (
    accept_realtime_call_endpoint,
    create_client_secret_endpoint,
    create_realtime_call_endpoint,
    hangup_realtime_call_endpoint,
    refer_realtime_call_endpoint,
    reject_realtime_call_endpoint,
)

router = APIRouter(tags=["Realtime"])
router.websocket("/realtime")(realtime_endpoint_worker)
router.post("/realtime/client_secrets")(create_client_secret_endpoint)
router.post("/realtime/calls")(create_realtime_call_endpoint)
router.post("/realtime/calls/{call_id}/accept")(accept_realtime_call_endpoint)
router.post("/realtime/calls/{call_id}/reject")(reject_realtime_call_endpoint)
router.post("/realtime/calls/{call_id}/refer")(refer_realtime_call_endpoint)
router.post("/realtime/calls/{call_id}/hangup")(hangup_realtime_call_endpoint)

__all__ = ["router", "configure", "shutdown", "realtime_endpoint_worker"]
