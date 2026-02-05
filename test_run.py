"""
Test script to verify the deepfake detection pipeline works correctly.
Tests preprocessing, model, and integration without requiring real video files.
"""

import sys
import numpy as np
import cv2


def test_preprocessing():
    """Test the face extraction and preprocessing pipeline."""
    print("\n[TEST 1] Testing Preprocessing Module...")

    from preprocessing import FaceExtractor

    # Create a synthetic image with a face-like pattern
    # This tests that the pipeline works, even if MediaPipe doesn't detect a face
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    with FaceExtractor() as extractor:
        # Test face extraction (may return empty if no real face)
        faces = extractor.extract_faces(test_image)
        print(f"  - Face extraction: OK (found {len(faces)} faces in synthetic image)")

        # Test preprocessing on a mock face
        mock_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        preprocessed = extractor.preprocess_for_model(mock_face)

        assert preprocessed.shape == (3, 224, 224), f"Wrong shape: {preprocessed.shape}"
        assert preprocessed.dtype == np.float32, f"Wrong dtype: {preprocessed.dtype}"
        print(f"  - Face preprocessing: OK (shape: {preprocessed.shape}, dtype: {preprocessed.dtype})")

    print("[TEST 1] Preprocessing Module: PASSED")
    return True


def test_model():
    """Test the deepfake detection model."""
    print("\n[TEST 2] Testing Model Module...")

    import torch
    from model import DeepfakeDetector, DeepfakeInference

    # Test model architecture
    model = DeepfakeDetector(pretrained=True)
    print(f"  - Model created: OK (backbone features: {model.feature_dim})")

    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_proba(dummy_input)

    assert logits.shape == (1, 1), f"Wrong logits shape: {logits.shape}"
    assert probs.shape == (1, 1), f"Wrong probs shape: {probs.shape}"
    assert 0 <= probs.item() <= 1, f"Probability out of range: {probs.item()}"
    print(f"  - Forward pass: OK (logit: {logits.item():.4f}, prob: {probs.item():.4f})")

    # Test inference wrapper
    inference = DeepfakeInference()
    mock_face = np.random.randn(3, 224, 224).astype(np.float32)
    prob, label = inference.predict_single(mock_face)

    assert isinstance(prob, float), f"Probability should be float: {type(prob)}"
    assert label in ["REAL", "FAKE"], f"Invalid label: {label}"
    print(f"  - Inference wrapper: OK (prob: {prob:.4f}, label: {label})")

    # Test batch inference
    batch = [mock_face, mock_face, mock_face]
    results = inference.predict_batch(batch)
    assert len(results) == 3, f"Wrong batch results length: {len(results)}"
    print(f"  - Batch inference: OK ({len(results)} predictions)")

    # Test results aggregation
    frame_results = [(0, 0.8, "FAKE"), (30, 0.7, "FAKE"), (60, 0.3, "REAL")]
    analysis = inference.analyze_video_results(frame_results)
    assert "verdict" in analysis, "Missing verdict in analysis"
    assert "confidence" in analysis, "Missing confidence in analysis"
    print(f"  - Results aggregation: OK (verdict: {analysis['verdict']}, confidence: {analysis['confidence']:.2f})")

    print("[TEST 2] Model Module: PASSED")
    return True


def test_integration():
    """Test the full pipeline integration."""
    print("\n[TEST 3] Testing Integration...")

    from preprocessing import FaceExtractor, VideoProcessor
    from model import DeepfakeInference

    # Create components
    face_extractor = FaceExtractor()
    video_processor = VideoProcessor(face_extractor, fps_sample_rate=1)
    inference = DeepfakeInference()

    # Create a mock "frame" and process it
    mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Draw a simple face-like circle to potentially trigger detection
    cv2.circle(mock_frame, (320, 240), 100, (200, 180, 170), -1)  # Face
    cv2.circle(mock_frame, (290, 220), 15, (50, 50, 50), -1)  # Left eye
    cv2.circle(mock_frame, (350, 220), 15, (50, 50, 50), -1)  # Right eye
    cv2.ellipse(mock_frame, (320, 280), (30, 15), 0, 0, 180, (100, 80, 80), -1)  # Mouth

    face = face_extractor.extract_primary_face(mock_frame)

    if face is not None:
        preprocessed = face_extractor.preprocess_for_model(face)
        prob, label = inference.predict_single(preprocessed)
        print(f"  - Full pipeline with detected face: OK (prob: {prob:.4f}, label: {label})")
    else:
        # If no face detected, test with mock data
        mock_preprocessed = np.random.randn(3, 224, 224).astype(np.float32)
        prob, label = inference.predict_single(mock_preprocessed)
        print(f"  - Full pipeline with mock face: OK (prob: {prob:.4f}, label: {label})")

    face_extractor.close()

    print("[TEST 3] Integration: PASSED")
    return True


def test_api_imports():
    """Test that API module imports correctly."""
    print("\n[TEST 4] Testing API Module Imports...")

    # Test imports
    from fastapi import FastAPI
    from api import app

    assert app is not None, "FastAPI app not created"
    assert app.title == "Scanner API", f"Wrong app title: {app.title}"

    # Check routes exist
    routes = [route.path for route in app.routes]
    assert "/" in routes, "Missing root route"
    assert "/health" in routes, "Missing health route"
    assert "/analyze-video" in routes, "Missing analyze-video route"
    assert "/analyze-image" in routes, "Missing analyze-image route"
    assert "/analyze-video-v2" in routes, "Missing analyze-video-v2 route"

    print(f"  - FastAPI app: OK (title: {app.title})")
    print(f"  - Routes registered: {len(routes)} routes")
    print(f"  - Key endpoints: /, /health, /analyze-video, /analyze-video-v2, /analyze-image")

    print("[TEST 4] API Module: PASSED")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("DEEPFAKE DETECTION ENGINE - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Preprocessing", test_preprocessing),
        ("Model", test_model),
        ("Integration", test_integration),
        ("API Imports", test_api_imports),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"\n[ERROR] {name} test failed: {str(e)}")
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "PASSED" if success else f"FAILED: {error}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
