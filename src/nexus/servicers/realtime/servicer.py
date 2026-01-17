import logging
import threading
import time
from collections.abc import Iterable
from typing import Iterator, List, Optional

import numpy as np
from openai.types.realtime.realtime_function_tool import RealtimeFunctionTool

from nexus.inferencers.asr.inferencer import Inferencer
from nexus.inferencers.chat.inferencer import Inferencer as ChatInferencer
from nexus.inferencers.tts.inferencer import Inferencer as TTSInferencer
from nexus.inferencers.tts.text_normalizer import split_text_by_punctuation
from nexus.sessions import ChatSession, RealtimeSession

from .build_events import (
    build_input_audio_transcription_completed,
    build_response_text_done,
    build_response_text_delta,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个有帮助的助手。根据用户提供的语音内容，生成简洁明了的回答。"""

# TTS 音频分片大小 (pcm16 @ 24kHz, ~80ms per chunk)
TTS_CHUNK_SIZE = 3840  # 24000 * 0.08 * 2 bytes

# 淡入淡出参数 (采样点数, 24kHz下约5ms)
FADE_SAMPLES = 120


def apply_fade_in(audio: np.ndarray, fade_samples: int = FADE_SAMPLES) -> np.ndarray:
    """对音频块应用淡入效果（在线处理）"""
    if len(audio) == 0:
        return audio
    fade_len = min(fade_samples, len(audio))
    fade_curve = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    audio = audio.astype(np.float32)
    audio[:fade_len] *= fade_curve
    return audio.astype(np.int16)


def apply_fade_out(audio: np.ndarray, fade_samples: int = FADE_SAMPLES) -> np.ndarray:
    """对音频块应用淡出效果（在线处理）"""
    if len(audio) == 0:
        return audio
    fade_len = min(fade_samples, len(audio))
    fade_curve = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    audio = audio.astype(np.float32)
    audio[-fade_len:] *= fade_curve
    return audio.astype(np.int16)


class RealtimeServicer:
    def __init__(
        self,
        grpc_addr: str,
        interim_results: bool = False,
        chat_base_url: Optional[str] = None,
        chat_api_key: Optional[str] = None,
        tts_base_url: Optional[str] = None,
        tts_api_key: Optional[str] = None,
    ):
        self.grpc_addr = grpc_addr
        self.interim_results = interim_results
        self.inferencer = Inferencer(self.grpc_addr)
        self.chat_inferencer = (
            ChatInferencer(api_key=chat_api_key, base_url=chat_base_url)
            if chat_api_key
            else None
        )
        self.tts_inferencer = (
            TTSInferencer(base_url=tts_base_url, api_key=tts_api_key)
            if tts_api_key
            else None
        )

    def create_realtime_session(
        self,
        output_modalities: List[str],
        tools: List[RealtimeFunctionTool],
        chat_model: str,
    ) -> RealtimeSession:
        chat_session = ChatSession(chat_inferencer=self.chat_inferencer)
        return RealtimeSession(
            chat_session=chat_session,
            chat_model=chat_model,
            output_modalities=output_modalities,
            tools=tools,
        )

    def realtime_worker(
        self, session: RealtimeSession, is_chat: bool = False
    ) -> Iterable[str]:
        """后台工作线程，将转录结果放入 result_queue"""
        audio_iterator = self._audio_iterator(session)
        for asr_result in self.inferencer.transcribe(
            audio_iterator,
            sample_rate=session.sample_rate,
            interim_results=self.interim_results,
        ):
            transcript = asr_result.transcript
            is_final = asr_result.is_final
            if not is_final or not transcript.strip():
                continue
            session.result_queue.put(
                build_input_audio_transcription_completed(transcript)
            )
            logger.info(f"Transcript: {transcript}")
            # 未配置对话功能，不返回对话结果
            if not is_chat:
                continue
            chat_stream_resp = session.chat(transcript + " /no_think")
            output_text = ""
            for resp_chunk in chat_stream_resp:
                if isinstance(resp_chunk, str):
                    output_text += resp_chunk
                    resp_chunk = build_response_text_delta(resp_chunk)
                session.result_queue.put(resp_chunk)
            session.result_queue.put(build_response_text_done(output_text))
            logger.info(f"Chat response done: {output_text}")

    def use_tool(self, session: RealtimeSession, tool_call_id: str, content: str):
        chat_stream_resp = session.use_tool(tool_call_id=tool_call_id, content=content)
        output_text = ""
        for resp_chunk in chat_stream_resp:
            if isinstance(resp_chunk, str):
                output_text += resp_chunk
                resp_chunk = build_response_text_delta(resp_chunk)
            session.result_queue.put(resp_chunk)
        session.result_queue.put(build_response_text_done(output_text))
        logger.info(f"Chat response done: {output_text}")

    def process_chat_response(
        self,
        session: RealtimeSession,
        transcript: str,
    ):
        pass

    def _process_chat_response(
        self,
        session: RealtimeSession,
        transcript: str,
        response_id: str,
        item_id: str,
        has_audio_output: bool,
    ):
        """处理 Chat 响应，支持文本和音频输出"""
        # 重置 TTS 取消标志
        session.reset_tts_cancel()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript + " /no_think"},
        ]

        # 流式获取 Chat 响应
        text_buffer = ""
        full_response = ""

        for chunk in self.chat_inferencer.chat_stream(
            messages=messages,
            # model=CHAT_MODEL,
            max_tokens=512,
        ):
            # 检查是否被打断
            if session.tts_cancel_event.is_set():
                logger.info("Chat response cancelled due to interruption")
                break

            # 清理 think 标签
            chunk = chunk.replace("<think>", "").replace("</think>", "")
            if not chunk:
                continue

            text_buffer += chunk
            full_response += chunk

            # 按标点分句，发送完整句子
            sentences = split_text_by_punctuation(text_buffer)
            if len(sentences) > 1:
                # 有完整的句子，发送文本并送入 TTS
                for sentence in sentences[:-1]:
                    if session.tts_cancel_event.is_set():
                        break
                    # 发送完整句子的文本
                    session.result_queue.put((sentence, False))
                    # 如果需要音频输出，送入 TTS
                    if has_audio_output and self.tts_inferencer:
                        self._tts_sentence(session, sentence)
                # 保留未完成的部分
                text_buffer = sentences[-1] if sentences else ""
            elif len(sentences) == 1 and text_buffer.endswith(tuple("。！？.!?；;")):
                # 单个完整句子
                session.result_queue.put((sentences[0], False))
                if has_audio_output and self.tts_inferencer:
                    self._tts_sentence(session, sentences[0])
                text_buffer = ""

        # 处理剩余的文本
        if text_buffer.strip():
            if not session.tts_cancel_event.is_set():
                # 发送剩余文本
                session.result_queue.put((text_buffer.strip(), False))
                if has_audio_output and self.tts_inferencer:
                    self._tts_sentence(session, text_buffer.strip(), is_last=True)

        # 发送最终标记
        if not session.tts_cancel_event.is_set():
            session.result_queue.put(("", True))
            if has_audio_output:
                # 发送音频结束标记
                session.tts_audio_queue.put((b"", True))

        logger.info(f"Chat response: {full_response}")

    def _tts_sentence(self, session: RealtimeSession, text: str, is_last: bool = False):
        """将一个句子转换为 TTS 音频并放入队列，带淡入淡出"""
        if not text.strip() or not self.tts_inferencer:
            return

        logger.info(f"TTS for sentence: {text}")
        session.set_tts_streaming(True)

        try:
            # 使用 pcm 格式以便直接播放
            audio_buffer = b""
            is_first_chunk = True
            chunks_to_send = []  # 缓存块以便对最后一块做淡出

            for audio_chunk in self.tts_inferencer.speech_stream(
                input=text,
                model="tts-1",
                voice="rita",
                response_format="wav",  # 24kHz, 16-bit, mono
            ):
                # 检查是否被打断
                if session.tts_cancel_event.is_set():
                    logger.info(f"TTS cancelled for: {text}")
                    break

                audio_buffer += audio_chunk

                # 分片发送，略小于完整 TTS 块以便打断
                while len(audio_buffer) >= TTS_CHUNK_SIZE:
                    chunk_bytes = audio_buffer[:TTS_CHUNK_SIZE]
                    audio_buffer = audio_buffer[TTS_CHUNK_SIZE:]

                    # 转换为numpy数组处理
                    chunk_array = np.frombuffer(chunk_bytes, dtype=np.int16).copy()

                    # 对第一个块应用淡入
                    if is_first_chunk:
                        chunk_array = apply_fade_in(chunk_array)
                        is_first_chunk = False

                    # 先发送上一个缓存的块（不是最后一块）
                    if chunks_to_send:
                        session.tts_audio_queue.put((chunks_to_send.pop(0), False))

                    # 缓存当前块（可能是最后一块，需要淡出）
                    chunks_to_send.append(chunk_array.tobytes())

            # 处理剩余的音频
            if audio_buffer and not session.tts_cancel_event.is_set():
                chunk_array = np.frombuffer(audio_buffer, dtype=np.int16).copy()
                if is_first_chunk:
                    chunk_array = apply_fade_in(chunk_array)
                # 缓存当前块
                if chunks_to_send:
                    session.tts_audio_queue.put((chunks_to_send.pop(0), False))
                chunks_to_send.append(chunk_array.tobytes())

            # 发送最后一个块，带淡出
            if chunks_to_send and not session.tts_cancel_event.is_set():
                last_chunk = np.frombuffer(chunks_to_send[0], dtype=np.int16).copy()
                last_chunk = apply_fade_out(last_chunk)
                session.tts_audio_queue.put((last_chunk.tobytes(), is_last))

        except Exception as e:
            logger.error(f"TTS error: {e}")
        finally:
            session.set_tts_streaming(False)

    def _audio_iterator(self, session: RealtimeSession) -> Iterable[np.ndarray]:
        while True:
            chunk: np.ndarray = session.audio_queue.get()
            if chunk is None:
                break
            yield chunk
