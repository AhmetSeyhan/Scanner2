"""
Scanner - Detector Framework (v5.0.0)
Pluggable detection engine abstraction layer.

Provides:
- BaseDetector ABC for all detection engines
- DetectorResult / DetectorInput standardized data types
- DetectorRegistry for runtime discovery and health checks

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from detectors.base import (
    BaseDetector,
    DetectorCapability,
    DetectorInput,
    DetectorResult,
    DetectorStatus,
    DetectorType,
)
from detectors.registry import DetectorRegistry

__all__ = [
    "BaseDetector",
    "DetectorResult",
    "DetectorInput",
    "DetectorType",
    "DetectorCapability",
    "DetectorStatus",
    "DetectorRegistry",
]

__version__ = "5.0.0"
