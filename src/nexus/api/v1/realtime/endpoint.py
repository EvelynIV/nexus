"""
FastAPI WebSocket 路由 - OpenAI Realtime API 兼容
处理实时语音转录的 WebSocket 连接
"""

import asyncio
import base64
import json
import logging
import queue
import threading
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import Depends, Query, WebSocket, WebSocketDisconnect
from openai.types.realtime import RealtimeFunctionTool
from nexus.servicers.realtime.build_events import (
    build_error_event,
    build_response_audio_delta,
    build_response_audio_done,
    build_response_done,
    build_session_created,
    build_session_updated,
)
from nexus.servicers.realtime.servicer import RealtimeServicer
from nexus.sessions import RealtimeSession

from .depends import get_realtime_servicer

logger = logging.getLogger(__name__)


TTS_SEND_INTERVAL = 0.01  # 10ms


async def realtime_endpoint_worker(
    websocket: WebSocket,
    model: str = Query(default="gpt-4o-realtime-preview"),
    realtime_servicer: RealtimeServicer = Depends(get_realtime_servicer),
):
    is_chat_model = "transcribe" not in model.lower()
    logger.info(
        f"WebSocket connected for model: {model}, is_chat_model: {is_chat_model}"
    )
    await websocket.accept()
    # 创建会话
    update_event = await receive_update_event(websocket)
    # 如果首先发送的不是配置请求，则报错并关闭连接
    if not update_event:
        logger.error("Failed to receive initial session.update event")
        await _send_event(
            websocket,
            build_error_event("invalid_request", "First event must be session.update"),
        )
        await websocket.close()
        return
    # 初始化 RealtimeSession
    tools = update_event.get("session", {}).get("tools", [])
    if tools:
        logger.info(f"Initializing session with tools: {tools}")
        tools = [RealtimeFunctionTool(**tool) for tool in tools]
    realtime_session = realtime_servicer.create_realtime_session(
        output_modalities=update_event.get("session", {}).get(
            "output_modalities", ["text"]
        ),
        tools=tools,
        chat_model=model,
    )

    await _send_event(
        websocket,
        build_session_created(realtime_session.session_id, model),
    )
    # 启动后台工作线程
    threading.Thread(
        target=realtime_servicer.realtime_worker,
        args=(realtime_session, is_chat_model),
        daemon=True,
    ).start()
    # 启动异步任务：从 result_queue 读取结果并发送
    send_results_task = asyncio.create_task(_send_results(websocket, realtime_session))
    # 启动异步任务：从 tts_audio_queue 读取音频并发送
    send_audio_task = asyncio.create_task(
        _send_audio_results(websocket, realtime_session)
    )

    try:
        while True:
            # 同时处理：接收消息 和 发送转录结果
            receive_task = asyncio.create_task(websocket.receive_text())
            # 构建等待任务列表
            pending_tasks = {receive_task}
            if send_results_task and not send_results_task.done():
                pending_tasks.add(send_results_task)
            if send_audio_task and not send_audio_task.done():
                pending_tasks.add(send_audio_task)
            # 同时等待消息和发送结果
            done, pending = await asyncio.wait(
                pending_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            # 检查发送任务是否完成
            if send_results_task in done:
                send_results_task = None
            if send_audio_task in done:
                send_audio_task = None
            # 检查是否收到消息
            if receive_task in done:
                data = receive_task.result()
                event = json.loads(data)
                event_type = event.get("type", "")
                logger.debug(f"Received event: {event_type}")
                if event_type == "input_audio_buffer.append":
                    # 追加音频数据到队列
                    audio_base64 = event.get("audio", "")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
                        realtime_session.audio_queue.put(audio_chunk)
                elif event_type == "session.update":
                    session_settings = event.get("session", {})
                    output_modalities = (
                        session_settings.get("output_modalities", ["text"])
                        if session_settings
                        else ["text"]
                    )
                    realtime_session.update_output_modalities(output_modalities)
                    logger.info(
                        f"Session update received. output_modalities: {output_modalities}"
                    )
                    # 发送 session.updated 事件
                    await _send_event(
                        websocket,
                        build_session_updated(
                            realtime_session.session_id, model, output_modalities
                        ),
                    )
                elif event_type == "response.cancel":
                    logger.info("Response cancelled by client")
                elif event_type == "conversation.item.create":
                    item = event.get("item", {})
                    if item.get("type", "") == "function_call_output":
                        item_id = item["call_id"]
                        output = item["output"]
                        realtime_servicer.use_tool(
                            realtime_session, tool_call_id=item_id, content=output
                        )
                else:
                    logger.warning(f"Unknown event type received: {event_type}")
    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected for session: {realtime_session.session_id}"
        )
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await _send_event(
            websocket,
            build_error_event("server_error", str(e)),
        )


async def _send_event(websocket: WebSocket, event: dict):
    """发送事件到 WebSocket"""
    if not isinstance(event, dict):
        event = event.model_dump()
    await websocket.send_text(json.dumps(event, ensure_ascii=False))


async def receive_update_event(websocket: WebSocket) -> Dict:
    """接收 session.update 事件"""
    try:
        data = await websocket.receive_text()
        event = json.loads(data)
        event_type = event.get("type", "")
        if event_type == "session.update":
            logger.info("Received session.update event")
            return event
        else:
            logger.warning(f"Expected session.update, but received: {event_type}")
            return None
    except Exception as e:
        logger.error(f"Error receiving session.update event: {e}")
        return None


async def _send_results(websocket: WebSocket, session: RealtimeSession):
    """异步任务：从 result_queue 读取转录结果并发送到 WebSocket"""
    loop = asyncio.get_event_loop()
    while True:
        try:
            # 使用 run_in_executor 异步等待同步队列
            event = await loop.run_in_executor(
                None, lambda: session.result_queue.get(timeout=0.1)
            )
            await _send_event(websocket, event)
        except queue.Empty:
            # 队列为空，继续等待
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.exception(f"Error sending transcription result: {e}")
            break


async def _send_audio_results(websocket: WebSocket, session: RealtimeSession):
    """异步任务：从 tts_audio_queue 读取 TTS 音频并发送到 WebSocket"""
    loop = asyncio.get_event_loop()
    while True:
        try:
            # 使用 run_in_executor 异步等待同步队列
            audio_data, is_last = await loop.run_in_executor(
                None, lambda: session.tts_audio_queue.get(timeout=0.1)
            )

            # 获取当前 response_id 和 item_id
            response_id = session.current_response_id or str(uuid.uuid4())
            item_id = session.current_item_id or str(uuid.uuid4())

            if audio_data:
                # 发送音频增量事件
                event = build_response_audio_delta(
                    response_id, item_id, base64.b64encode(audio_data).decode("utf-8")
                )
                await _send_event(websocket, event)

                # 控制发送速率，略小于音频时长，便于打断
                await asyncio.sleep(TTS_SEND_INTERVAL)

            # 发送完成事件
            if is_last:
                await _send_event(
                    websocket,
                    build_response_audio_done(response_id, item_id),
                )
                await _send_event(
                    websocket,
                    build_response_done(response_id),
                )
        except queue.Empty:
            # 队列为空，继续等待
            await asyncio.sleep(0.01)
        except Exception as e:
            logger.exception(f"Error sending audio result: {e}")
            break
