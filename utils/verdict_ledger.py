"""
Scanner Prime - Blockchain-Ready Verdict Ledger
Immutable, cryptographically chained verdict log for forensic audit trails.

Each verdict is hashed and linked to the previous entry, creating a
tamper-evident chain (similar to blockchain block linking).

This provides:
1. Court-admissible evidence chain
2. Tamper detection for historical verdicts
3. Export format compatible with Ethereum/Polygon attestation

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import json
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class VerdictBlock:
    """Single block in the verdict chain."""
    block_index: int
    timestamp: str
    session_id: str
    file_hash: str         # SHA-256 of analyzed file
    verdict: str
    integrity_score: float
    core_scores: Dict[str, float]
    weights_used: Dict[str, float]
    previous_hash: str     # Hash of previous block
    block_hash: str = ""   # This block's hash (computed)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of this block (excluding block_hash itself)."""
        data = {
            "block_index": self.block_index,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "file_hash": self.file_hash,
            "verdict": self.verdict,
            "integrity_score": self.integrity_score,
            "core_scores": self.core_scores,
            "weights_used": self.weights_used,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()


class VerdictLedger:
    """
    Append-only verdict ledger with cryptographic chaining.

    Each verdict is stored as a block with a hash linking to the
    previous block, creating a tamper-evident chain.
    """

    def __init__(self, db_path: str = "verdict_ledger.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verdict_chain (
                block_index INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                verdict TEXT NOT NULL,
                integrity_score REAL NOT NULL,
                core_scores TEXT NOT NULL,
                weights_used TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                block_hash TEXT NOT NULL UNIQUE
            )
        """)
        conn.commit()
        conn.close()

    def append_verdict(self, analysis_result: Dict[str, Any]) -> VerdictBlock:
        """
        Append a new verdict to the chain.

        Args:
            analysis_result: Output from AnalysisService.analyze_video_v2().

        Returns:
            The new VerdictBlock with computed hash.
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get previous block
            cursor.execute(
                "SELECT block_index, block_hash FROM verdict_chain ORDER BY block_index DESC LIMIT 1"
            )
            row = cursor.fetchone()

            if row:
                prev_index, prev_hash = row
                new_index = prev_index + 1
            else:
                new_index = 0
                prev_hash = "0" * 64  # Genesis block

            block = VerdictBlock(
                block_index=new_index,
                timestamp=datetime.now(timezone.utc).isoformat(),
                session_id=analysis_result.get("session_id", ""),
                file_hash=analysis_result.get("forensic_hashes", {}).get("file_hash", ""),
                verdict=analysis_result.get("verdict", "UNKNOWN"),
                integrity_score=analysis_result.get("integrity_score", 0),
                core_scores=analysis_result.get("core_scores", {}),
                weights_used=analysis_result.get("weights", {}),
                previous_hash=prev_hash,
            )
            block.block_hash = block.compute_hash()

            cursor.execute(
                """INSERT INTO verdict_chain
                   (block_index, timestamp, session_id, file_hash, verdict,
                    integrity_score, core_scores, weights_used, previous_hash, block_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (block.block_index, block.timestamp, block.session_id,
                 block.file_hash, block.verdict, block.integrity_score,
                 json.dumps(block.core_scores), json.dumps(block.weights_used),
                 block.previous_hash, block.block_hash),
            )
            conn.commit()
            conn.close()

            return block

    def verify_chain(self) -> Dict[str, Any]:
        """Verify the integrity of the entire verdict chain."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verdict_chain ORDER BY block_index")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"valid": True, "blocks": 0, "message": "Empty chain"}

        prev_hash = "0" * 64
        for row in rows:
            block = VerdictBlock(
                block_index=row[0], timestamp=row[1], session_id=row[2],
                file_hash=row[3], verdict=row[4], integrity_score=row[5],
                core_scores=json.loads(row[6]), weights_used=json.loads(row[7]),
                previous_hash=row[8], block_hash=row[9],
            )

            # Verify previous hash link
            if block.previous_hash != prev_hash:
                return {
                    "valid": False, "blocks": len(rows),
                    "broken_at": block.block_index,
                    "message": f"Chain broken at block {block.block_index}: previous_hash mismatch",
                }

            # Verify self hash
            computed = block.compute_hash()
            if computed != block.block_hash:
                return {
                    "valid": False, "blocks": len(rows),
                    "broken_at": block.block_index,
                    "message": f"Block {block.block_index} hash mismatch (tampered)",
                }

            prev_hash = block.block_hash

        return {"valid": True, "blocks": len(rows), "message": "Chain integrity verified"}

    def get_block(self, block_index: int) -> Optional[VerdictBlock]:
        """Get a single block by index."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verdict_chain WHERE block_index = ?", (block_index,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return VerdictBlock(
            block_index=row[0], timestamp=row[1], session_id=row[2],
            file_hash=row[3], verdict=row[4], integrity_score=row[5],
            core_scores=json.loads(row[6]), weights_used=json.loads(row[7]),
            previous_hash=row[8], block_hash=row[9],
        )

    def chain_length(self) -> int:
        """Return the number of blocks in the chain."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM verdict_chain")
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def export_chain(self, format: str = "json") -> str:
        """Export the full chain for external verification."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM verdict_chain ORDER BY block_index")
        rows = cursor.fetchall()
        conn.close()

        blocks = []
        for row in rows:
            blocks.append({
                "block_index": row[0], "timestamp": row[1],
                "session_id": row[2], "file_hash": row[3],
                "verdict": row[4], "integrity_score": row[5],
                "core_scores": json.loads(row[6]),
                "weights_used": json.loads(row[7]),
                "previous_hash": row[8], "block_hash": row[9],
            })

        return json.dumps({"chain": blocks, "verification": self.verify_chain()}, indent=2)
