"""
Scanner Prime - Prometheus Metrics
Application-level metrics for monitoring and alerting.

Exposes /metrics endpoint compatible with Prometheus scraping.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import time
from typing import Optional

from core.logging_config import get_logger

logger = get_logger("metrics")

# Try to import prometheus client
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed, metrics endpoint disabled")

if PROMETHEUS_AVAILABLE:
    # --- Counters ---
    REQUESTS_TOTAL = Counter(
        "scanner_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    ANALYSES_TOTAL = Counter(
        "scanner_analyses_total",
        "Total analysis requests",
        ["type", "verdict"],
    )

    AUTH_EVENTS = Counter(
        "scanner_auth_events_total",
        "Authentication events",
        ["event", "success"],
    )

    ERRORS_TOTAL = Counter(
        "scanner_errors_total",
        "Total errors",
        ["error_code"],
    )

    # --- Histograms ---
    REQUEST_LATENCY = Histogram(
        "scanner_request_duration_seconds",
        "Request latency in seconds",
        ["endpoint"],
        buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
    )

    ANALYSIS_DURATION = Histogram(
        "scanner_analysis_duration_seconds",
        "Analysis processing time in seconds",
        ["type"],
        buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    )

    # --- Gauges ---
    ACTIVE_ANALYSES = Gauge(
        "scanner_active_analyses",
        "Currently running analyses",
    )

    QUEUE_LENGTH = Gauge(
        "scanner_queue_length",
        "Celery task queue length",
    )

    # --- Info ---
    APP_INFO = Info(
        "scanner",
        "Scanner application info",
    )
    APP_INFO.info({
        "version": "3.3.0",
        "architecture": "PRIME_HYBRID",
    })


def get_metrics_response() -> Optional[tuple]:
    """
    Generate Prometheus metrics response.

    Returns:
        Tuple of (content_bytes, content_type) or None if unavailable.
    """
    if not PROMETHEUS_AVAILABLE:
        return None
    return generate_latest(), CONTENT_TYPE_LATEST


def record_analysis(analysis_type: str, verdict: str, duration_seconds: float) -> None:
    """Record an analysis completion in metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    ANALYSES_TOTAL.labels(type=analysis_type, verdict=verdict).inc()
    ANALYSIS_DURATION.labels(type=analysis_type).observe(duration_seconds)


def record_request(method: str, endpoint: str, status_code: int, duration_seconds: float) -> None:
    """Record an HTTP request in metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_seconds)


def record_error(error_code: str) -> None:
    """Record an error in metrics."""
    if not PROMETHEUS_AVAILABLE:
        return
    ERRORS_TOTAL.labels(error_code=error_code).inc()


def record_auth_event(event: str, success: bool) -> None:
    """Record an auth event."""
    if not PROMETHEUS_AVAILABLE:
        return
    AUTH_EVENTS.labels(event=event, success=str(success)).inc()
