"""
Scanner Prime - Webhook Manager
Async webhook notifications for scan results.

Sends POST requests to configured callback URLs
when scan events occur (completion, errors).

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from core.forensic_types import FusionVerdict

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Webhook configuration."""
    url: str
    secret: Optional[str] = None
    events: List[str] = field(default_factory=lambda: ["scan.complete"])
    timeout: int = 30
    retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)


class WebhookManager:
    """
    Async webhook notification manager.

    Sends POST requests to configured callback URLs
    when scan events occur.

    Events:
    - scan.complete: Analysis completed successfully
    - scan.error: Analysis failed
    - scan.started: Analysis started (optional)
    """

    def __init__(self, webhooks: Optional[List[WebhookConfig]] = None):
        """
        Initialize webhook manager.

        Args:
            webhooks: List of WebhookConfig objects
        """
        if not HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for webhooks. "
                "Install with: pip install httpx"
            )

        self.webhooks = webhooks or []
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def add_webhook(self, config: WebhookConfig):
        """Add webhook configuration."""
        self.webhooks.append(config)

    def remove_webhook(self, url: str):
        """Remove webhook by URL."""
        self.webhooks = [w for w in self.webhooks if w.url != url]

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """List configured webhooks."""
        return [
            {"url": w.url, "events": w.events, "timeout": w.timeout}
            for w in self.webhooks
        ]

    async def notify_scan_complete(
        self,
        session_id: str,
        filename: str,
        verdict: FusionVerdict,
        sha256_hash: str,
        user: str = "anonymous"
    ):
        """
        Send scan.complete notification to all webhooks.

        Args:
            session_id: Unique session ID
            filename: Original filename
            verdict: FusionVerdict from analysis
            sha256_hash: File hash
            user: Username
        """
        payload = {
            "event": "scan.complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "session_id": session_id,
                "filename": filename,
                "sha256_hash": sha256_hash,
                "user": user,
                "verdict": verdict.verdict,
                "integrity_score": verdict.integrity_score,
                "confidence": verdict.confidence,
                "core_scores": {
                    "biosignal": verdict.biosignal_score,
                    "artifact": verdict.artifact_score,
                    "alignment": verdict.alignment_score
                },
                "weights": verdict.weights,
                "leading_core": verdict.leading_core
            }
        }

        await self._send_to_all("scan.complete", payload)

    async def notify_scan_complete_dict(
        self,
        session_id: str,
        filename: str,
        verdict_dict: Dict[str, Any],
        sha256_hash: str,
        user: str = "anonymous"
    ):
        """
        Send scan.complete notification using dictionary (no FusionVerdict needed).
        """
        payload = {
            "event": "scan.complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "session_id": session_id,
                "filename": filename,
                "sha256_hash": sha256_hash,
                "user": user,
                **verdict_dict
            }
        }

        await self._send_to_all("scan.complete", payload)

    async def notify_scan_error(
        self,
        session_id: str,
        filename: str,
        error: str,
        user: str = "anonymous"
    ):
        """Send scan.error notification."""
        payload = {
            "event": "scan.error",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "session_id": session_id,
                "filename": filename,
                "error": error,
                "user": user
            }
        }

        await self._send_to_all("scan.error", payload)

    async def notify_scan_started(
        self,
        session_id: str,
        filename: str,
        user: str = "anonymous"
    ):
        """Send scan.started notification."""
        payload = {
            "event": "scan.started",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "session_id": session_id,
                "filename": filename,
                "user": user
            }
        }

        await self._send_to_all("scan.started", payload)

    async def _send_to_all(self, event: str, payload: Dict[str, Any]):
        """Send payload to all webhooks subscribed to event."""
        tasks = []
        for webhook in self.webhooks:
            if event in webhook.events:
                tasks.append(self._send_with_retry(webhook, payload))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Webhook failed: {result}")

    async def _send_with_retry(
        self,
        webhook: WebhookConfig,
        payload: Dict[str, Any]
    ) -> bool:
        """Send with retry logic."""
        client = await self._get_client()

        headers = {"Content-Type": "application/json"}
        headers.update(webhook.headers)

        # Add signature header if secret is configured
        if webhook.secret:
            payload_bytes = json.dumps(payload, separators=(',', ':')).encode()
            signature = hmac.new(
                webhook.secret.encode(),
                payload_bytes,
                hashlib.sha256
            ).hexdigest()
            headers["X-Scanner-Signature"] = f"sha256={signature}"

        for attempt in range(webhook.retries):
            try:
                response = await client.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=webhook.timeout
                )

                if response.status_code < 400:
                    logger.info(f"Webhook delivered: {webhook.url}")
                    return True

                logger.warning(
                    f"Webhook returned {response.status_code}: {webhook.url}"
                )

            except httpx.TimeoutException:
                logger.error(f"Webhook timeout (attempt {attempt + 1}): {webhook.url}")
            except httpx.RequestError as e:
                logger.error(f"Webhook request error (attempt {attempt + 1}): {e}")
            except Exception as e:
                logger.error(f"Webhook failed (attempt {attempt + 1}): {e}")

            if attempt < webhook.retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Webhook exhausted retries: {webhook.url}")
        return False

    async def test_webhook(self, url: str) -> Dict[str, Any]:
        """
        Test webhook connectivity.

        Args:
            url: Webhook URL to test

        Returns:
            Test result with status and latency
        """
        client = await self._get_client()

        test_payload = {
            "event": "test",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"message": "Scanner webhook test"}
        }

        try:
            import time
            start = time.time()
            response = await client.post(
                url,
                json=test_payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            latency = (time.time() - start) * 1000

            return {
                "success": response.status_code < 400,
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "url": url
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url
            }


# Synchronous wrapper for non-async contexts
class SyncWebhookManager:
    """Synchronous wrapper for WebhookManager."""

    def __init__(self, webhooks: Optional[List[WebhookConfig]] = None):
        self._async_manager = WebhookManager(webhooks)

    def add_webhook(self, config: WebhookConfig):
        self._async_manager.add_webhook(config)

    def remove_webhook(self, url: str):
        self._async_manager.remove_webhook(url)

    def notify_scan_complete(
        self,
        session_id: str,
        filename: str,
        verdict: FusionVerdict,
        sha256_hash: str,
        user: str = "anonymous"
    ):
        """Send notification synchronously."""
        asyncio.run(
            self._async_manager.notify_scan_complete(
                session_id, filename, verdict, sha256_hash, user
            )
        )

    def notify_scan_error(
        self,
        session_id: str,
        filename: str,
        error: str,
        user: str = "anonymous"
    ):
        """Send error notification synchronously."""
        asyncio.run(
            self._async_manager.notify_scan_error(
                session_id, filename, error, user
            )
        )
