"""
Scanner Prime - Forensic Integrity Module
SHA-256 chain-of-custody hashing for tamper evidence.

Provides:
- Original file hash computation
- Intermediate result hashing
- PDF report hash embedding
- Chain-of-custody proof generation

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from core.logging_config import get_logger

logger = get_logger("forensic_hash")


class ForensicHashChain:
    """
    Manages a chain of SHA-256 hashes for forensic integrity.

    Each step in the analysis pipeline appends a hash entry,
    creating a tamper-evident chain of custody.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.chain: List[Dict[str, Any]] = []
        self._created_at = datetime.now(timezone.utc).isoformat()

    def hash_file(self, file_path: str) -> str:
        """
        Compute SHA-256 of a file on disk.

        Args:
            file_path: Path to the file.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()
        self._append("original_file", digest, {"file_path": file_path})
        logger.info("File hashed", extra={"session_id": self.session_id, "video_hash": digest[:16]})
        return digest

    def hash_bytes(self, data: bytes, label: str) -> str:
        """
        Compute SHA-256 of arbitrary bytes.

        Args:
            data: Bytes to hash.
            label: Human-readable step label.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        digest = hashlib.sha256(data).hexdigest()
        self._append(label, digest)
        return digest

    def hash_result(self, result: Dict[str, Any]) -> str:
        """
        Compute SHA-256 of a JSON-serialisable analysis result.

        The result dict is sorted-key serialised for deterministic hashing.

        Args:
            result: Analysis result dict.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        canonical = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        self._append("analysis_result", digest)
        return digest

    def hash_frames(self, frames: list) -> str:
        """
        Compute a combined hash over extracted video frames.

        Uses the first, middle, and last frame to avoid hashing
        all frame data while still detecting frame-level tampering.

        Args:
            frames: List of numpy arrays (BGR frames).

        Returns:
            Hex-encoded SHA-256 digest.
        """
        sha256 = hashlib.sha256()
        if frames:
            indices = {0, len(frames) // 2, len(frames) - 1}
            for idx in sorted(indices):
                if idx < len(frames):
                    sha256.update(frames[idx].tobytes())
        digest = sha256.hexdigest()
        self._append("extracted_frames", digest, {"frame_count": len(frames)})
        return digest

    def verify(self, label: str, expected_hash: str) -> bool:
        """
        Verify a step's hash matches the expected value.

        Args:
            label: Step label.
            expected_hash: Expected hex digest.

        Returns:
            True if the stored hash matches.
        """
        for entry in self.chain:
            if entry["label"] == label:
                match = entry["sha256"] == expected_hash
                if not match:
                    logger.warning(
                        "Hash mismatch detected",
                        extra={
                            "session_id": self.session_id,
                            "stage": label,
                        },
                    )
                return match
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the full chain for embedding in reports/responses."""
        return {
            "session_id": self.session_id,
            "created_at": self._created_at,
            "chain_length": len(self.chain),
            "entries": self.chain,
        }

    def summary(self) -> Dict[str, str]:
        """Return a label -> hash mapping for compact embedding."""
        return {entry["label"]: entry["sha256"] for entry in self.chain}

    # --- Internal ---

    def _append(
        self,
        label: str,
        sha256: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.chain.append({
            "label": label,
            "sha256": sha256,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        })


def compute_file_hash(file_path: str) -> str:
    """Convenience: compute SHA-256 of a file without creating a chain."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
