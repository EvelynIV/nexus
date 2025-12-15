from __future__ import annotations

import inspect
from contextlib import asynccontextmanager

from fastapi import FastAPI
from .core.middleware import CustomMiddleware
from .api.v1.routes_transcriptions import router as transcriptions_router
from .clients.grpc.stubs import UxSpeechClient
from .core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
     # Add custom middleware
    app.add_middleware(CustomMiddleware)

    app.include_router(transcriptions_router)
    return app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ===== Startup：在 yield 之前执行 =====
    client = UxSpeechClient(
        host=settings.default_grpc_host,
        port=settings.default_grpc_port,
    )
    app.state.default_grpc_client = client
    
    try:
        yield  # 应用开始接收请求；直到关闭时才会继续往下走
    finally:
        # ===== Shutdown：在 yield 之后执行 =====
        client = getattr(app.state, "default_grpc_client", None)
        if client is not None:
            # 兼容 close() 可能是同步或异步的两种实现
            ret = client.close()
            if inspect.isawaitable(ret):
                await ret


# 作用：构造 FastAPI 应用对象。
#通过lifespan