from __future__ import annotations

from fastapi import FastAPI

from .api.v1.routes_transcriptions import router as transcriptions_router
from .clients.grpc.stubs import UxSpeechClient
from .core.config import settings


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name)

    app.include_router(transcriptions_router)

    @app.on_event("startup")
    async def on_startup():
        # 初始化一个默认上游 gRPC 客户端
        app.state.default_grpc_client = UxSpeechClient(
            host=settings.default_grpc_host,
            port=settings.default_grpc_port,
        )

    @app.on_event("shutdown")
    async def on_shutdown():
        client = getattr(app.state, "default_grpc_client", None)
        if client:
            client.close()

    return app

