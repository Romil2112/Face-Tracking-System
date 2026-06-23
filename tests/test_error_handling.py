"""Tests for the CircuitBreaker and retry primitives (pure logic)."""
import pytest

from error_handling import CircuitBreaker, retry


def test_circuit_breaker_opens_after_max_failures():
    cb = CircuitBreaker(max_failures=3, reset_timeout=60)
    assert not cb.is_open()
    for _ in range(3):
        cb.record_failure()
    assert cb.is_open()


def test_circuit_breaker_reset_closes_it():
    cb = CircuitBreaker(max_failures=2, reset_timeout=60)
    cb.record_failure()
    cb.record_failure()
    assert cb.is_open()
    cb.reset()
    assert not cb.is_open()


def test_circuit_breaker_auto_resets_after_timeout_window():
    cb = CircuitBreaker(max_failures=1, reset_timeout=10)
    cb.record_failure()
    assert cb.is_open()
    cb.last_failure -= 20  # simulate the reset window having elapsed
    assert not cb.is_open()


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    @retry(max_attempts=3, delay=0, jitter=0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise ValueError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


def test_retry_propagates_after_exhausting_attempts():
    calls = {"n": 0}

    @retry(max_attempts=2, delay=0, jitter=0)
    def always_fails():
        calls["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        always_fails()
    assert calls["n"] >= 2
