"""Per-IP rate limiting for the Face Detection API.

Uses slowapi (built on limits) with an in-memory back-end.
Override the default with RATE_LIMIT_PER_MINUTE (e.g. "60/minute").
"""

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address


def get_rate_limit() -> str:
    """Return the per-minute rate-limit string, evaluated lazily per request."""
    return os.environ.get("RATE_LIMIT_PER_MINUTE", "30/minute")


limiter: Limiter = Limiter(key_func=get_remote_address)


async def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded,
) -> JSONResponse:
    """Return a structured 429 with retry_after instead of slowapi's default HTML."""
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limit_exceeded", "retry_after": 60},
        headers={"Retry-After": "60"},
    )
