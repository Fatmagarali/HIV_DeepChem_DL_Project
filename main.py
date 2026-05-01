"""Cloud Run entry point for the HIV prediction API."""

from __future__ import annotations

from app.api import app
from app.config import get_settings


def main() -> None:
    """Start the ASGI server on the configured Cloud Run port."""

    import os

    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=int(os.getenv("WEB_CONCURRENCY", "1")),
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()