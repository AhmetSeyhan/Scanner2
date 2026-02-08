"""
Scanner Prime - Forensic Heatmap Renderer
Generates visual overlay images for forensic analysis results.

Produces:
1. Anomaly heatmap overlay on original frame
2. PPG map visualization (Intel FakeCatcher style)
3. Combined forensic overview image

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional


def render_anomaly_heatmap(
    frame: np.ndarray,
    heatmap_data: Dict[str, Any],
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay artifact anomaly heatmap on a video frame.

    Args:
        frame: Original BGR frame.
        heatmap_data: HeatmapAnalysis.to_dict() output.
        alpha: Overlay transparency (0=transparent, 1=opaque).
        colormap: OpenCV colormap (default JET).

    Returns:
        BGR frame with heatmap overlay.
    """
    h, w = frame.shape[:2]
    cells = heatmap_data.get("cells", [])
    grid_rows, grid_cols = heatmap_data.get("grid_size", (8, 8))

    # Build anomaly grid
    grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for cell in cells:
        x, y = cell.get("x", 0), cell.get("y", 0)
        if 0 <= y < grid_rows and 0 <= x < grid_cols:
            grid[y, x] = cell.get("score", 0.0)

    # Resize to frame dimensions
    heatmap = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-255 and apply colormap
    heatmap_norm = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)

    # Blend
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)

    # Add grid lines
    cell_h, cell_w = h // grid_rows, w // grid_cols
    for r in range(1, grid_rows):
        cv2.line(overlay, (0, r * cell_h), (w, r * cell_h), (255, 255, 255), 1)
    for c in range(1, grid_cols):
        cv2.line(overlay, (c * cell_w, 0), (c * cell_w, h), (255, 255, 255), 1)

    # Label hotspots
    for cell in cells:
        if cell.get("score", 0) > 0.5:
            x, y = cell["x"], cell["y"]
            cx = x * cell_w + cell_w // 2
            cy = y * cell_h + cell_h // 2
            label = cell.get("type", "?")
            cv2.putText(
                overlay, label, (cx - 15, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
            )

    return overlay


def render_ppg_map(
    frame: np.ndarray,
    ppg_data: Dict[str, Any],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Render PPG (photoplethysmography) map overlay - Intel FakeCatcher style.

    Green regions: strong biological pulse signal (authentic)
    Red regions: weak/absent pulse signal (suspicious)

    Args:
        frame: Original BGR frame.
        ppg_data: Output from BioSignalCore.generate_ppg_map().
        alpha: Overlay transparency.

    Returns:
        BGR frame with PPG overlay.
    """
    h, w = frame.shape[:2]
    ppg_map = ppg_data.get("ppg_map")
    if ppg_map is None:
        return frame.copy()

    if isinstance(ppg_map, list):
        ppg_map = np.array(ppg_map, dtype=np.float32)

    # Resize to frame dimensions
    ppg_resized = cv2.resize(ppg_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Create custom colormap: red (no pulse) -> green (strong pulse)
    ppg_norm = np.clip(ppg_resized, 0, 1)
    overlay = frame.copy()

    # Green channel for strong signal, red for weak
    color_overlay = np.zeros_like(frame)
    color_overlay[:, :, 1] = (ppg_norm * 255).astype(np.uint8)        # Green = pulse present
    color_overlay[:, :, 2] = ((1 - ppg_norm) * 200).astype(np.uint8)  # Red = no pulse

    overlay = cv2.addWeighted(frame, 1 - alpha, color_overlay, alpha, 0)

    # Add PPG strength label
    mean_strength = ppg_data.get("mean_ppg_strength", 0)
    coverage = ppg_data.get("ppg_coverage", 0)
    label = f"PPG: {mean_strength:.2f} | Coverage: {coverage:.0%}"
    cv2.putText(
        overlay, label, (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )

    return overlay


def render_forensic_overview(
    frame: np.ndarray,
    heatmap_data: Optional[Dict] = None,
    ppg_data: Optional[Dict] = None,
    verdict: str = "UNKNOWN",
    integrity_score: float = 0.0,
) -> np.ndarray:
    """
    Generate combined forensic overview image (2x2 grid).

    Layout:
    +---------------+---------------+
    |  Original     |  Anomaly      |
    |  Frame        |  Heatmap      |
    +---------------+---------------+
    |  PPG Map      |  Summary      |
    |  Overlay      |  Panel        |
    +---------------+---------------+
    """
    # Standardize panel size
    panel_h, panel_w = 360, 480
    frame_resized = cv2.resize(frame, (panel_w, panel_h))

    # Panel 1: Original
    panel1 = frame_resized.copy()
    cv2.putText(panel1, "ORIGINAL", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Panel 2: Anomaly Heatmap
    if heatmap_data:
        panel2 = render_anomaly_heatmap(frame_resized, heatmap_data, alpha=0.5)
    else:
        panel2 = frame_resized.copy()
    cv2.putText(panel2, "ARTIFACT HEATMAP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Panel 3: PPG Map
    if ppg_data:
        panel3 = render_ppg_map(frame_resized, ppg_data, alpha=0.5)
    else:
        panel3 = frame_resized.copy()
    cv2.putText(panel3, "PPG MAP", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    # Panel 4: Summary
    panel4 = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel4[:] = (20, 20, 25)  # Dark background

    # Verdict color
    verdict_colors = {
        "AUTHENTIC": (0, 200, 0),
        "MANIPULATED": (0, 0, 220),
        "UNCERTAIN": (0, 180, 255),
        "INCONCLUSIVE": (180, 180, 0),
    }
    color = verdict_colors.get(verdict, (200, 200, 200))

    cv2.putText(panel4, "FORENSIC SUMMARY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel4, f"Verdict: {verdict}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(panel4, f"Integrity: {integrity_score:.1f}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

    # Draw integrity bar
    bar_w = int((panel_w - 40) * integrity_score / 100)
    cv2.rectangle(panel4, (20, 140), (panel_w - 20, 165), (60, 60, 60), -1)
    bar_color = (0, 200, 0) if integrity_score > 70 else (0, 180, 255) if integrity_score > 40 else (0, 0, 200)
    cv2.rectangle(panel4, (20, 140), (20 + bar_w, 165), bar_color, -1)

    cv2.putText(panel4, "SCANNER PRIME v4.0.0", (20, panel_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    # Combine into 2x2 grid
    top_row = np.hstack([panel1, panel2])
    bottom_row = np.hstack([panel3, panel4])
    overview = np.vstack([top_row, bottom_row])

    return overview
