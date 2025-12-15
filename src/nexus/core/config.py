from __future__ import annotations
from logging.config import dictConfig

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # basic
    app_name: str = "UxSpeech HTTP -> gRPC proxy"
    env: str = "dev"

    # http
    http_host: str = "0.0.0.0"
    http_port: int = 8000

    # default upstream gRPC
    default_grpc_host: str = "39.106.1.132"
    default_grpc_port: int = 30029
    grpc_timeout_s: float = 180.0

    # 关于grpc请求的设置
    grpc_keepalive_time_ms: int = 600000
    grpc_keepalive_timeout_ms: int = 10000
    grpc_keepalive_permit_without_calls: int = 1
    grpc_http2_max_pings_without_data: int = 0


settings = Settings()
#在一开始就把一切处理完，减少反复调用，中间报错等问题