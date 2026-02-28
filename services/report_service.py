"""
Scanner Prime - Report Service
PDF report generation and export logic.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""


from core.exceptions import ProcessingError
from core.forensic_types import FusionVerdict
from core.logging_config import get_logger

logger = get_logger("report_service")

# Lazy import — reportlab may not be installed
_reporter = None


def _get_reporter():
    """Lazy-load ForensicReporter to avoid import errors when reportlab is missing."""
    global _reporter
    if _reporter is None:
        try:
            from utils.forensic_reporter import ForensicReporter
            _reporter = ForensicReporter()
        except ImportError:
            raise ProcessingError(
                "pdf_generation",
                "reportlab is not installed. Install with: pip install reportlab",
            )
    return _reporter


class ReportService:
    """Generates forensic PDF reports from scan results."""

    @staticmethod
    def is_available() -> bool:
        """Check if PDF generation is available."""
        try:
            from reportlab.lib.pagesizes import A4  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def generate_pdf_from_verdict(
        verdict: FusionVerdict,
        filename: str,
        sha256_hash: str,
        resolution: str,
        duration: float,
        session_id: str,
    ) -> bytes:
        """
        Generate a PDF report from a FusionVerdict.

        Returns:
            Raw PDF bytes.
        """
        reporter = _get_reporter()
        pdf_bytes = reporter.generate_report(
            verdict=verdict,
            filename=filename,
            sha256_hash=sha256_hash,
            resolution=resolution,
            duration=duration,
            session_id=session_id,
        )
        logger.info("PDF report generated", extra={"session_id": session_id})
        return pdf_bytes

    @staticmethod
    def generate_pdf_from_history_entry(entry, session_id: str) -> bytes:
        """
        Generate a PDF from a ScanHistoryEntry.

        Constructs a minimal FusionVerdict from history data.

        Args:
            entry: ScanHistoryEntry instance.
            session_id: Session identifier.

        Returns:
            Raw PDF bytes.
        """
        verdict = FusionVerdict(
            verdict=entry.verdict,
            integrity_score=entry.integrity_score,
            confidence=0.8,
            biosignal_score=entry.biosignal_score,
            artifact_score=entry.artifact_score,
            alignment_score=entry.alignment_score,
            weights={"biosignal": 0.33, "artifact": 0.33, "alignment": 0.34},
            leading_core="N/A",
        )

        return ReportService.generate_pdf_from_verdict(
            verdict=verdict,
            filename=entry.filename,
            sha256_hash=entry.sha256_hash,
            resolution=entry.resolution,
            duration=entry.duration_seconds,
            session_id=session_id,
        )
