import logging
from typing import Optional

from nexus.configs.config import NexusConfig
from nexus.servicers.realtime.servicer import RealtimeServicer

logger = logging.getLogger(__name__)

realtime_servicer: Optional[RealtimeServicer] = None


def configure(
    engine_config: NexusConfig,
):
    logger.info("Used Realtime settings: %s", engine_config)
    global realtime_servicer
    if realtime_servicer is None:
        realtime_servicer = RealtimeServicer(
            grpc_addr=engine_config.asr_grpc_addr,
            interim_results=engine_config.asr_interim_results,
            chat_base_url=engine_config.chat_base_url,
            chat_api_key=engine_config.chat_api_key,
            tts_base_url=engine_config.tts_base_url,
            tts_api_key=engine_config.tts_api_key,
        )


def get_realtime_servicer() -> RealtimeServicer:
    if realtime_servicer is None:
        raise RuntimeError("RealtimeServicer not configured. Call configure() first.")
    return realtime_servicer
