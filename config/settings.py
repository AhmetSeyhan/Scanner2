"""
Scanner - Centralized Settings (v6.0.0)
Replaces scattered os.getenv() calls with typed, validated configuration.

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.secret_key)

All values can be overridden via environment variables with the SCANNER_ prefix
(except where a custom env= is specified).

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class ScannerSettings(BaseSettings):
    """Typed, validated application settings."""

    model_config = {"env_prefix": "SCANNER_", "env_file": ".env", "extra": "ignore"}

    # --- Core ---
    env: str = Field("development", description="Runtime environment")
    version: str = Field("6.0.0", description="Application version")
    debug: bool = Field(False, description="Debug mode flag")

    # --- Authentication ---
    secret_key: str = Field("change-this-to-a-secure-random-string", description="JWT signing secret")
    api_key: str = Field("change-this-to-a-secure-random-string", description="Service API key")
    access_token_expire_minutes: int = Field(30, description="JWT token lifetime in minutes")
    admin_password: str = Field("", description="Admin user password")
    analyst_password: str = Field("", description="Analyst user password")
    viewer_password: str = Field("", description="Viewer user password")

    # --- Server ---
    host: str = Field("0.0.0.0", description="API bind host")
    port: int = Field(8000, description="API bind port")
    cors_origins: str = Field("*", description="Comma-separated CORS origins")

    # --- Redis ---
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL", description="Redis connection URL")

    # --- S3 / MinIO ---
    s3_endpoint_url: str = Field("http://localhost:9000", env="S3_ENDPOINT_URL")
    s3_bucket_name: str = Field("scanner-analyses", env="S3_BUCKET_NAME")
    aws_access_key_id: str = Field("", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field("", env="AWS_SECRET_ACCESS_KEY")

    # --- Rate Limiting ---
    rate_limit: int = Field(10, description="Requests per minute per IP")

    # --- Logging ---
    log_level: str = Field("INFO", description="Log level")
    log_json: bool = Field(True, description="Emit JSON-structured logs")

    # --- Model / Inference ---
    weights_dir: str = Field("./weights", description="Path to model weight files")
    device: str = Field("cpu", description="Inference device (cpu, cuda, cuda:0)")
    max_upload_size: int = Field(524_288_000, description="Max upload size in bytes")

    # --- Webhooks ---
    webhook_url: str = Field("", description="Webhook notification URL")

    # --- PentaShield (v6.0.0) ---
    pentashield_enabled: bool = Field(True, description="Enable PentaShield defense system")
    hydra_heads: int = Field(5, description="Number of Hydra detection heads")
    hydra_agreement_threshold: float = Field(0.6, description="Min head agreement ratio for verdict")
    zero_day_manifold_dim: int = Field(64, description="Authenticity manifold latent dimension")
    zero_day_ood_threshold: float = Field(0.85, description="OOD detection threshold")
    forensic_dna_db_path: str = Field("./data/generator_db.json", description="Generator fingerprint DB path")
    active_probe_timeout_ms: int = Field(5000, description="Challenge-response timeout in ms")
    active_probe_max_challenges: int = Field(3, description="Max challenges per session")
    ghost_protocol_model_size_mb: int = Field(10, description="Max edge model size in MB")
    ghost_protocol_quantization: str = Field("int8", description="Quantization mode (int8, fp16, dynamic)")
    federated_learning_enabled: bool = Field(False, description="Enable federated learning client")
    federated_server_url: str = Field("", description="Federated aggregation server URL")
    differential_privacy_epsilon: float = Field(1.0, description="DP epsilon value")
    qdrant_url: str = Field("http://localhost:6333", description="Qdrant vector DB URL")

    # --- Derived helpers ---
    @property
    def is_production(self) -> bool:
        return self.env.lower() == "production"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> ScannerSettings:
    """Return a cached singleton of the application settings."""
    return ScannerSettings()
