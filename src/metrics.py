"""Prometheus metrics and structured logging for the Face Detection API.

Exposes:
  GET /metrics  — Prometheus text format (auto-mounted by instrumentator)
  face_detection_backend_total  — counter by acceleration backend
  face_detection_errors_total   — counter by error type

Structured logs are emitted as JSON via structlog; the stdlib bridge means
existing logging.getLogger(...) calls in other modules continue to work.
"""

import logging
import uuid

import structlog
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

face_detection_backend_total = Counter(
    "face_detection_backend_total",
    "Total face detections grouped by compute backend",
    ["backend"],
)

face_detection_errors_total = Counter(
    "face_detection_errors_total",
    "Total face detection errors grouped by error type",
    ["error_type"],
)


def build_instrumentator() -> Instrumentator:
    return Instrumentator(
        excluded_handlers=["/metrics", "/health"],
        should_group_status_codes=False,
    )


def configure_structlog() -> None:
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def new_request_id() -> str:
    return str(uuid.uuid4())


logger = structlog.get_logger("face_tracker.api")


def record_backend(backend_name: str) -> None:
    face_detection_backend_total.labels(backend=backend_name).inc()


def record_error(error_type: str) -> None:
    face_detection_errors_total.labels(error_type=error_type).inc()
