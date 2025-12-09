from __future__ import annotations

import uvicorn

from .app import create_app
from .core.config import settings

app = create_app()


def run():
    uvicorn.run(
        "ux_speech_gateway.main:app",
        host=settings.http_host,
        port=settings.http_port,
        reload=(settings.env == "dev"),
    )


if __name__ == "__main__":
    run()
