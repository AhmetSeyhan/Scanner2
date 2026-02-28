"""
Preprocessing module for deepfake detection.
Uses MediaPipe Tasks API for ultra-fast face detection and cropping.
"""

from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceExtractor:
    """Extracts and preprocesses faces from images/frames using MediaPipe Tasks API."""

    def __init__(self, min_detection_confidence: float = 0.5, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the face extractor.

        Args:
            min_detection_confidence: Minimum confidence for face detection.
            target_size: Output size for cropped faces (width, height).
        """
        self.target_size = target_size
        self.min_detection_confidence = min_detection_confidence

        # Initialize MediaPipe Face Detector with Tasks API
        base_options = python.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=min_detection_confidence
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)

    def _get_model_path(self) -> str:
        """Get the path to the face detection model, downloading if necessary."""
        import os
        import urllib.request

        # Model path in the project directory
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, "blaze_face_short_range.tflite")

        # Download model if not present
        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            print("Downloading face detection model...")
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded to {model_path}")

        return model_path

    def extract_faces(self, frame: np.ndarray, padding: float = 0.2) -> List[np.ndarray]:
        """
        Extract all faces from a frame.

        Args:
            frame: BGR image as numpy array.
            padding: Percentage of padding around the detected face.

        Returns:
            List of cropped and resized face images.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect faces
        detection_result = self.face_detector.detect(mp_image)

        faces = []
        h, w = frame.shape[:2]

        for detection in detection_result.detections:
            bbox = detection.bounding_box

            # Calculate coordinates with padding
            pad_w = int(bbox.width * padding)
            pad_h = int(bbox.height * padding)

            x_min = max(0, bbox.origin_x - pad_w)
            y_min = max(0, bbox.origin_y - pad_h)
            x_max = min(w, bbox.origin_x + bbox.width + pad_w)
            y_max = min(h, bbox.origin_y + bbox.height + pad_h)

            # Crop and resize face
            face = frame[y_min:y_max, x_min:x_max]
            if face.size > 0:
                face = cv2.resize(face, self.target_size)
                faces.append(face)

        return faces

    def extract_primary_face(self, frame: np.ndarray, padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract the largest/most prominent face from a frame.

        Args:
            frame: BGR image as numpy array.
            padding: Percentage of padding around the detected face.

        Returns:
            Cropped and resized face image, or None if no face detected.
        """
        faces = self.extract_faces(frame, padding)
        return faces[0] if faces else None

    def preprocess_for_model(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess a face image for the detection model.

        Args:
            face: BGR face image as numpy array.

        Returns:
            Normalized float32 array in CHW format.
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        face_normalized = (face_normalized - mean) / std

        # Convert HWC to CHW format
        face_chw = np.transpose(face_normalized, (2, 0, 1))

        return face_chw

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, 'face_detector') and self.face_detector:
            self.face_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoProcessor:
    """Processes video files for deepfake detection."""

    def __init__(self, face_extractor: FaceExtractor, fps_sample_rate: int = 1):
        """
        Initialize the video processor.

        Args:
            face_extractor: FaceExtractor instance for face detection.
            fps_sample_rate: Process 1 frame per this many seconds.
        """
        self.face_extractor = face_extractor
        self.fps_sample_rate = fps_sample_rate

    def extract_frames(self, video_path: str) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames from video at the specified sample rate.

        Args:
            video_path: Path to the video file.

        Returns:
            List of tuples (frame_number, frame).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * self.fps_sample_rate)
        frame_interval = max(1, frame_interval)

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frames.append((frame_count, frame))

            frame_count += 1

        cap.release()
        return frames

    def process_video(self, video_path: str) -> List[Tuple[int, np.ndarray]]:
        """
        Extract faces from video frames at the sample rate.

        Args:
            video_path: Path to the video file.

        Returns:
            List of tuples (frame_number, preprocessed_face).
        """
        frames = self.extract_frames(video_path)
        processed = []

        for frame_num, frame in frames:
            face = self.face_extractor.extract_primary_face(frame)
            if face is not None:
                preprocessed = self.face_extractor.preprocess_for_model(face)
                processed.append((frame_num, preprocessed))

        return processed
