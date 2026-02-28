"""
Scanner Prime - Video Profiler Service
Extracts video metadata and creates VideoProfile for analysis pipeline.

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import os
from typing import List

import cv2
import numpy as np

from core.exceptions import NoFramesExtractedError, VideoDecodeError
from core.forensic_types import ResolutionTier, VideoProfile
from core.logging_config import get_logger

logger = get_logger("video_profiler")

# Allowed extensions for upload validation
ALLOWED_VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})
ALLOWED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def get_resolution_tier(height: int) -> ResolutionTier:
    """Map pixel height to resolution tier."""
    if height <= 360:
        return ResolutionTier.ULTRA_LOW
    elif height <= 480:
        return ResolutionTier.LOW
    elif height <= 720:
        return ResolutionTier.MEDIUM
    elif height <= 1080:
        return ResolutionTier.HIGH
    return ResolutionTier.ULTRA_HIGH


class VideoProfiler:
    """Handles video metadata extraction and frame sampling."""

    DEFAULT_MAX_FRAMES = 60

    def profile_video(self, video_path: str) -> VideoProfile:
        """
        Open a video file and build its VideoProfile.

        Args:
            video_path: Path to video on disk.

        Returns:
            VideoProfile with resolution tier, fps, duration etc.

        Raises:
            VideoDecodeError: If the video cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0.0
            tier = get_resolution_tier(height)

            profile = VideoProfile(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration_seconds=duration,
                resolution_tier=tier,
                pixel_count=width * height,
                aspect_ratio=width / height if height > 0 else 1.0,
                rppg_viable=height >= 480 and fps >= 24,
                mesh_viable=height >= 720,
                recommended_analysis="PRIME HYBRID",
            )

            logger.info(
                "Video profiled",
                extra={
                    "stage": "profiling",
                    "duration_ms": 0,
                },
            )
            return profile
        finally:
            cap.release()

    def extract_frames(
        self,
        video_path: str,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ) -> List[np.ndarray]:
        """
        Extract evenly-spaced frames from a video file.

        Args:
            video_path: Path to video on disk.
            max_frames: Maximum number of frames to extract.

        Returns:
            List of BGR numpy arrays.

        Raises:
            VideoDecodeError: If the video cannot be opened.
            NoFramesExtractedError: If no frames could be read.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoDecodeError(f"Cannot open video: {video_path}")

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = max(1, frame_count // max_frames)
            frames: List[np.ndarray] = []
            idx = 0

            while len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % interval == 0:
                    frames.append(frame)
                idx += 1

            if not frames:
                raise NoFramesExtractedError()

            logger.info(
                f"Extracted {len(frames)} frames",
                extra={"stage": "frame_extraction"},
            )
            return frames
        finally:
            cap.release()

    def validate_extension(self, filename: str, media_type: str = "video") -> str:
        """
        Validate file extension and return it.

        Args:
            filename: Original filename.
            media_type: "video" or "image".

        Returns:
            Lowercased extension (e.g. ".mp4").

        Raises:
            UnsupportedFileTypeError: If extension not allowed.
        """
        from core.exceptions import UnsupportedFileTypeError

        ext = os.path.splitext(filename or "")[1].lower()
        allowed = ALLOWED_VIDEO_EXTENSIONS if media_type == "video" else ALLOWED_IMAGE_EXTENSIONS

        if ext not in allowed:
            raise UnsupportedFileTypeError(ext, sorted(allowed))

        return ext
