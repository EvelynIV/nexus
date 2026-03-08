from fastapi import APIRouter

from . import chat, transcribe, tts
from .realtime import router as realtime_router

router = APIRouter(prefix="/v1")
router.include_router(chat.router)
router.include_router(transcribe.router)
router.include_router(tts.router)
router.include_router(realtime_router)

__all__ = ["router"]
