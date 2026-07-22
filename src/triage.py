"""Optional Claude-powered triage for low-confidence detections.

Mirrors the log-analyzer ai_summary.py pattern: ANTHROPIC_API_KEY unset
disables the feature silently. The model and retry logic are identical to
log-analyzer so both features stay in sync.

Environment
-----------
ANTHROPIC_API_KEY          : str (required to enable triage)
TRIAGE_CONFIDENCE_THRESHOLD: float (default 0.6)
    Faces with confidence below this threshold are triaged when ?triage=true.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

TRIAGE_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("TRIAGE_CONFIDENCE_THRESHOLD", "0.6")
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

_client = None

try:
    import anthropic as _anthropic_sdk

    if ANTHROPIC_API_KEY:
        _client = _anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        logger.info("ANTHROPIC_API_KEY not set — triage disabled")
except ImportError:
    _anthropic_sdk = None  # type: ignore[assignment]
    logger.info("anthropic package not installed — triage disabled")

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 200
_MAX_RETRIES = 3


def triage_detection(face: dict) -> str | None:
    """Ask Claude whether a low-confidence face detection warrants attention.

    Returns a short advisory string, or None on failure or when disabled.
    Only face confidence is sent — never coordinates or image bytes.
    """
    if _client is None:
        return None

    confidence = face.get("confidence", 0.0)
    prompt = (
        f"A face detector returned a detection with confidence {confidence:.3f} "
        f"(threshold: {TRIAGE_CONFIDENCE_THRESHOLD:.2f}). "
        "In one sentence, suggest what might cause a low-confidence detection "
        "and whether the operator should investigate."
    )

    for attempt in range(_MAX_RETRIES):
        try:
            msg = _client.messages.create(
                model=_MODEL,
                max_tokens=_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text.strip()
        except Exception as exc:
            # Catch all SDK errors (RateLimitError, APITimeoutError, etc.)
            # and any unexpected errors — triage must never break the request.
            backoff = 0.5 * (2 ** attempt)
            logger.warning(
                "triage attempt %d/%d failed: %s — retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                exc,
                backoff,
            )
            if attempt < _MAX_RETRIES - 1:
                time.sleep(backoff)

    return None
