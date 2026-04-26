from __future__ import annotations

import logging

from fastapi import Depends, Query, WebSocket, WebSocketDisconnect

from nexus.application.container import AppContainer, get_container
from nexus.application.realtime.protocol import BroadcastRealtimeSink, RealtimeServerWriter

from .asterisk import AsteriskCallError
from .controller import RealtimeSessionController
from .http import extract_bearer_token
from .runtime import RealtimeCallError, get_realtime_api_runtime


logger = logging.getLogger(__name__)


async def realtime_endpoint_worker(
    websocket: WebSocket,
    model: str = Query(default="gpt-realtime"),
    call_id: str | None = Query(default=None),
    container: AppContainer = Depends(get_container),
):
    runtime = await get_realtime_api_runtime(container)
    try:
        bearer_token = extract_bearer_token(websocket.headers.get("authorization"))
    except Exception:
        await websocket.close(code=1008)
        return

    if runtime.api_key_required() and not runtime.check_api_key(bearer_token):
        await websocket.close(code=1008)
        return

    if call_id:
        call = await runtime.calls.get(call_id)
        if call is None:
            await websocket.close(code=1008)
            return

        await websocket.accept()
        writer = RealtimeServerWriter(websocket)
        try:
            await call.attach_sideband(writer)
        except (RealtimeCallError, AsteriskCallError):
            await websocket.close(code=1008)
            return
        try:
            while True:
                raw_text = await websocket.receive_text()
                await call.controller.enqueue_text(raw_text, writer)
        except WebSocketDisconnect:
            logger.info("Realtime sideband websocket disconnected for call %s", call_id)
        finally:
            await call.detach_sideband(writer)
        return

    await websocket.accept()
    writer = RealtimeServerWriter(websocket)
    broadcaster = BroadcastRealtimeSink([writer])
    service = container.realtime

    try:
        session = service.create_session(
            writer=broadcaster,
            output_modalities=["text"],
            tools=[],
            chat_model=model,
        )
    except Exception as exc:
        await writer.send_error(
            message=str(exc),
            error_type="server_error",
            code="session_init_failed",
        )
        await websocket.close(code=1011)
        return

    controller = RealtimeSessionController(
        session=session,
        service=service,
        model=model,
        broadcaster=broadcaster,
    )

    try:
        await controller.start()
        await service.emit_session_created(session, model, sink=writer)
        while True:
            raw_text = await websocket.receive_text()
            await controller.enqueue_text(raw_text, writer)
    except WebSocketDisconnect:
        logger.info("Realtime websocket disconnected for session %s", session.session_id)
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception("Unhandled realtime websocket error: %s", exc)
        await writer.send_error(
            message=str(exc),
            error_type="server_error",
            code="internal_server_error",
        )
    finally:
        await controller.close()
