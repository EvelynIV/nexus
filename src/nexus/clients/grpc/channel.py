from __future__ import annotations

import grpc

from ...core.config import settings


def make_target(host: str, port: int) -> str:
    return f"{host}:{port}"


def create_insecure_channel(host: str, port: int) -> grpc.Channel:
    """
    Create an insecure gRPC channel with keepalive options.
    """
    target = make_target(host, port)

    options = [
        ("grpc.keepalive_time_ms", settings.grpc_keepalive_time_ms),
        ("grpc.keepalive_timeout_ms", settings.grpc_keepalive_timeout_ms),
        ("grpc.http2.max_pings_without_data", settings.grpc_http2_max_pings_without_data),
        ("grpc.keepalive_permit_without_calls", settings.grpc_keepalive_permit_without_calls),
    ]

    return grpc.insecure_channel(target, options=options)

