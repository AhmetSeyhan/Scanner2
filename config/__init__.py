"""
Scanner - Configuration Module (v5.0.0)
Centralized, typed settings via Pydantic BaseSettings.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from config.settings import ScannerSettings, get_settings

__all__ = ["ScannerSettings", "get_settings"]
