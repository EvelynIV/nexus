from __future__ import annotations

import asyncio
import logging

from fastapi import Depends, Query, WebSocket, WebSocketDisconnect

from nexus.application.container import AppContainer, get_container
from nexus.application.realtime.dispatch import RealtimeDispatchContext, build_default_registry
from nexus.application.realtime.protocol import ClientEventParseError, RealtimeClientParser, RealtimeServerWriter

logger = logging.getLogger(__name__)


async def realtime_endpoint_worker(
    websocket: WebSocket,
    model: str = Query(default="gpt-4o-realtime-preview"),
    container: AppContainer = Depends(get_container),
):
    is_chat_model = "transcribe" not in model.lower()
    await websocket.accept()

    writer = RealtimeServerWriter(websocket)
    service = container.realtime
    parser = RealtimeClientParser()
    registry = build_default_registry()

    try:
        session = service.create_session(
            writer=writer,
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

    ctx = RealtimeDispatchContext(session=session, service=service, model=model)

    await service.emit_session_created(session, model)
    worker_task = await service.start_transcription_worker(session, is_chat_model)

    got_initial_update = False

    try:
        while True:
            raw_text = await websocket.receive_text()

            try:
                event = parser.parse_text(raw_text)
            except ClientEventParseError as exc:
                await writer.send_error(
                    message=exc.message,
                    error_type=exc.error_type,
                    code=exc.code,
                    event_ref=exc.event_id,
                )
                continue

            if not got_initial_update:
                if event.type != "session.update":
                    await writer.send_error(
                        message="First client event must be session.update",
                        error_type="invalid_request_error",
                        code="invalid_event_sequence",
                        event_ref=getattr(event, "event_id", None),
                    )
                    await websocket.close(code=1008)
                    return
                got_initial_update = True

            await registry.dispatch(event, ctx)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session.session_id)
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception("Unhandled realtime websocket error: %s", exc)
        await writer.send_error(
            message=str(exc),
            error_type="server_error",
            code="internal_server_error",
        )
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        await service.close_session(session)
