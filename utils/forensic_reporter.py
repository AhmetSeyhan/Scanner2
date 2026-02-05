"""
Scanner Prime - Forensic PDF Reporter
Generates detailed PDF reports for scan results.

Creates professional forensic reports including:
- SHA-256 hash verification
- FusionVerdict summary
- Per-core analysis breakdown
- Recommendations

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import io
from datetime import datetime
from typing import Optional, Dict, Any

# PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from core.forensic_types import FusionVerdict


class ForensicReporter:
    """
    Generates professional PDF forensic reports.

    Includes:
    - SHA-256 hash verification
    - FusionVerdict summary
    - Per-core analysis breakdown
    - Recommendations
    """

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError(
                "reportlab is required for PDF generation. "
                "Install with: pip install reportlab"
            )

        # Brand colors (initialized after reportlab availability check)
        self.COLOR_PRIMARY = colors.HexColor("#0066FF")
        self.COLOR_SECONDARY = colors.HexColor("#FFB800")
        self.COLOR_DANGER = colors.HexColor("#FF3D57")
        self.COLOR_SUCCESS = colors.HexColor("#00C853")
        self.COLOR_DARK = colors.HexColor("#0B0B0C")
        self.COLOR_LIGHT = colors.HexColor("#F5F5F7")

        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='ScannerTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=self.COLOR_PRIMARY,
            spaceAfter=20
        ))

        self.styles.add(ParagraphStyle(
            name='ScannerSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=self.COLOR_DARK,
            spaceBefore=15,
            spaceAfter=10
        ))

        self.styles.add(ParagraphStyle(
            name='VerdictAuthentic',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=self.COLOR_SUCCESS,
            alignment=1  # Center
        ))

        self.styles.add(ParagraphStyle(
            name='VerdictManipulated',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=self.COLOR_DANGER,
            alignment=1
        ))

        self.styles.add(ParagraphStyle(
            name='VerdictUncertain',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=self.COLOR_SECONDARY,
            alignment=1
        ))

    def generate_report(
        self,
        verdict: FusionVerdict,
        filename: str,
        sha256_hash: str,
        resolution: str,
        duration: float,
        session_id: str,
    ) -> bytes:
        """
        Generate PDF report.

        Args:
            verdict: FusionVerdict from analysis
            filename: Original filename
            sha256_hash: SHA-256 hash of file
            resolution: Video resolution
            duration: Video duration in seconds
            session_id: Unique session ID

        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        story = []

        # Header
        story.append(Paragraph(
            "SCANNER FORENSIC REPORT",
            self.styles['ScannerTitle']
        ))
        story.append(HRFlowable(width="100%", color=self.COLOR_PRIMARY))
        story.append(Spacer(1, 20))

        # Metadata table
        hash_display = sha256_hash[:32] + "..." if len(sha256_hash) > 32 else sha256_hash
        metadata = [
            ["Session ID:", session_id],
            ["Filename:", filename[:50] + "..." if len(filename) > 50 else filename],
            ["SHA-256:", hash_display],
            ["Resolution:", resolution],
            ["Duration:", f"{duration:.2f} seconds"],
            ["Generated:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")]
        ]

        meta_table = Table(metadata, colWidths=[120, 350])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), self.COLOR_PRIMARY),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 30))

        # Verdict Section
        story.append(Paragraph("VERDICT", self.styles['ScannerSubtitle']))

        if verdict.verdict == "AUTHENTIC":
            verdict_style = 'VerdictAuthentic'
        elif verdict.verdict == "MANIPULATED":
            verdict_style = 'VerdictManipulated'
        else:
            verdict_style = 'VerdictUncertain'

        story.append(Paragraph(
            f"<b>{verdict.verdict}</b>",
            self.styles[verdict_style]
        ))
        story.append(Spacer(1, 10))
        story.append(Paragraph(
            f"Integrity Score: {verdict.integrity_score:.1f}/100",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Confidence: {verdict.confidence:.1%}",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Leading Core: {verdict.leading_core}",
            self.styles['Normal']
        ))
        story.append(Spacer(1, 20))

        # Core Scores Table
        story.append(Paragraph("CORE ANALYSIS", self.styles['ScannerSubtitle']))

        core_data = [
            ["Core", "Score", "Weight", "Status"],
            ["BIOSIGNAL", f"{verdict.biosignal_score:.2%}",
             f"{verdict.weights.get('biosignal', 0.33):.1%}",
             self._get_status(verdict.biosignal_score)],
            ["ARTIFACT", f"{verdict.artifact_score:.2%}",
             f"{verdict.weights.get('artifact', 0.33):.1%}",
             self._get_status(verdict.artifact_score)],
            ["ALIGNMENT", f"{verdict.alignment_score:.2%}",
             f"{verdict.weights.get('alignment', 0.34):.1%}",
             self._get_status(verdict.alignment_score)],
        ]

        core_table = Table(core_data, colWidths=[120, 100, 100, 100])
        core_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.COLOR_PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(core_table)
        story.append(Spacer(1, 30))

        # Transparency Report
        if verdict.transparency_report:
            story.append(Paragraph("ANALYSIS DETAILS", self.styles['ScannerSubtitle']))
            story.append(Paragraph(
                f"<b>Summary:</b> {verdict.transparency_report.summary}",
                self.styles['Normal']
            ))

            if verdict.transparency_report.primary_concern:
                story.append(Paragraph(
                    f"<b>Primary Concern:</b> {verdict.transparency_report.primary_concern}",
                    self.styles['Normal']
                ))

            if verdict.transparency_report.environmental_factors:
                story.append(Spacer(1, 10))
                story.append(Paragraph("<b>Environmental Factors:</b>", self.styles['Normal']))
                for factor in verdict.transparency_report.environmental_factors[:5]:
                    story.append(Paragraph(f"  - {factor}", self.styles['Normal']))

            if verdict.transparency_report.supporting_evidence:
                story.append(Spacer(1, 10))
                story.append(Paragraph("<b>Supporting Evidence:</b>", self.styles['Normal']))
                for evidence in verdict.transparency_report.supporting_evidence[:5]:
                    story.append(Paragraph(f"  - {evidence}", self.styles['Normal']))

        # Recommendations
        story.append(Spacer(1, 20))
        story.append(Paragraph("RECOMMENDATIONS", self.styles['ScannerSubtitle']))
        recommendations = self._get_recommendations(verdict)
        for rec in recommendations:
            story.append(Paragraph(f"  - {rec}", self.styles['Normal']))

        # Footer
        story.append(Spacer(1, 50))
        story.append(HRFlowable(width="100%", color=self.COLOR_LIGHT))
        story.append(Paragraph(
            "Generated by SCANNER ELITE v3.2.0 | scanner.ai",
            self.styles['Normal']
        ))
        story.append(Paragraph(
            f"Full SHA-256: {sha256_hash}",
            self.styles['Normal']
        ))

        doc.build(story)
        return buffer.getvalue()

    def generate_report_from_dict(
        self,
        result: Dict[str, Any],
        filename: str,
        sha256_hash: str,
        resolution: str,
        duration: float,
        session_id: str,
    ) -> bytes:
        """
        Generate PDF report from dictionary (no FusionVerdict needed).

        Args:
            result: Dictionary with verdict info
            filename: Original filename
            sha256_hash: SHA-256 hash of file
            resolution: Video resolution
            duration: Video duration in seconds
            session_id: Unique session ID

        Returns:
            PDF bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        story = []

        # Header
        story.append(Paragraph(
            "SCANNER FORENSIC REPORT",
            self.styles['ScannerTitle']
        ))
        story.append(HRFlowable(width="100%", color=self.COLOR_PRIMARY))
        story.append(Spacer(1, 20))

        # Metadata
        hash_display = sha256_hash[:32] + "..." if len(sha256_hash) > 32 else sha256_hash
        metadata = [
            ["Session ID:", session_id],
            ["Filename:", filename[:50] + "..." if len(filename) > 50 else filename],
            ["SHA-256:", hash_display],
            ["Resolution:", resolution],
            ["Duration:", f"{duration:.2f} seconds"],
            ["Generated:", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")]
        ]

        meta_table = Table(metadata, colWidths=[120, 350])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), self.COLOR_PRIMARY),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 30))

        # Verdict
        verdict_str = result.get("verdict", "UNKNOWN")
        integrity_score = result.get("integrity_score", 0)
        confidence = result.get("confidence", 0)

        story.append(Paragraph("VERDICT", self.styles['ScannerSubtitle']))

        if verdict_str == "AUTHENTIC":
            verdict_style = 'VerdictAuthentic'
        elif verdict_str == "MANIPULATED":
            verdict_style = 'VerdictManipulated'
        else:
            verdict_style = 'VerdictUncertain'

        story.append(Paragraph(f"<b>{verdict_str}</b>", self.styles[verdict_style]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Integrity Score: {integrity_score:.1f}/100", self.styles['Normal']))
        story.append(Paragraph(f"Confidence: {confidence:.1%}", self.styles['Normal']))
        story.append(Spacer(1, 30))

        # Footer
        story.append(HRFlowable(width="100%", color=self.COLOR_LIGHT))
        story.append(Paragraph(
            "Generated by SCANNER ELITE v3.2.0",
            self.styles['Normal']
        ))

        doc.build(story)
        return buffer.getvalue()

    def _get_status(self, score: float) -> str:
        """Convert score to status text."""
        if score > 0.6:
            return "FAIL"
        elif score > 0.3:
            return "WARN"
        return "PASS"

    def _get_recommendations(self, verdict: FusionVerdict) -> list:
        """Generate recommendations based on verdict."""
        recommendations = []

        if verdict.verdict == "AUTHENTIC":
            recommendations.append("No manipulation detected. Media appears genuine.")
            recommendations.append("Archive this report for audit trail purposes.")
        elif verdict.verdict == "MANIPULATED":
            recommendations.append("Do NOT use this media for official purposes.")
            recommendations.append("Investigate the source of this media.")
            recommendations.append("Consider notifying relevant stakeholders.")
            if verdict.leading_core:
                recommendations.append(f"Primary detection by {verdict.leading_core}.")
        elif verdict.verdict == "UNCERTAIN":
            recommendations.append("Manual review recommended.")
            recommendations.append("Consider analyzing with higher quality source.")
        else:  # INCONCLUSIVE
            recommendations.append("Results are inconclusive - additional analysis needed.")
            recommendations.append("Try with higher resolution video if available.")

        return recommendations
