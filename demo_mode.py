"""
Demo Mode for Deepfake Detector.

Allows setting mock results for specific files to demonstrate the UI
to investors without requiring actual trained model predictions.

Usage:
    # Set a file as FAKE with 95% confidence
    python demo_mode.py --set video.mp4 --fake 0.95

    # Set a file as REAL with 90% confidence
    python demo_mode.py --set video.mp4 --real 0.90

    # Clear a mock result
    python demo_mode.py --clear video.mp4

    # List all mock results
    python demo_mode.py --list

    # Clear all mock results
    python demo_mode.py --clear-all
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Mock results storage file
MOCK_RESULTS_FILE = Path(__file__).parent / ".demo_mock_results.json"


def load_mock_results() -> dict:
    """Load mock results from file."""
    if MOCK_RESULTS_FILE.exists():
        with open(MOCK_RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_mock_results(results: dict) -> None:
    """Save mock results to file."""
    with open(MOCK_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def set_mock_result(filename: str, fake_probability: float, label: str) -> None:
    """
    Set a mock result for a specific file.

    Args:
        filename: Name of the file (will match by basename)
        fake_probability: Probability that the content is fake (0.0 to 1.0)
        label: "FAKE" or "REAL"
    """
    results = load_mock_results()

    # Normalize filename to basename for flexible matching
    basename = os.path.basename(filename)

    results[basename] = {
        "fake_probability": fake_probability,
        "label": label,
        "confidence": fake_probability if label == "FAKE" else (1 - fake_probability),
    }

    save_mock_results(results)
    print(f"Mock result set for '{basename}':")
    print(f"  Label: {label}")
    print(f"  Fake Probability: {fake_probability:.1%}")
    print(f"  Confidence: {results[basename]['confidence']:.1%}")


def get_mock_result(filename: str) -> Optional[dict]:
    """
    Get mock result for a file if it exists.

    Args:
        filename: Full path or basename of the file

    Returns:
        Mock result dict or None if no mock result exists
    """
    results = load_mock_results()
    basename = os.path.basename(filename)

    # Try exact match first
    if basename in results:
        return results[basename]

    # Try lowercase match
    for key, value in results.items():
        if key.lower() == basename.lower():
            return value

    return None


def is_demo_mode_active() -> bool:
    """Check if demo mode has any active mock results."""
    results = load_mock_results()
    return len(results) > 0


def clear_mock_result(filename: str) -> bool:
    """Clear mock result for a specific file."""
    results = load_mock_results()
    basename = os.path.basename(filename)

    if basename in results:
        del results[basename]
        save_mock_results(results)
        print(f"Mock result cleared for '{basename}'")
        return True
    else:
        print(f"No mock result found for '{basename}'")
        return False


def clear_all_mock_results() -> None:
    """Clear all mock results."""
    save_mock_results({})
    print("All mock results cleared.")


def list_mock_results() -> None:
    """List all mock results."""
    results = load_mock_results()

    if not results:
        print("No mock results configured.")
        print("Use --set to add mock results for demo mode.")
        return

    print("Current Mock Results:")
    print("=" * 60)
    for filename, data in results.items():
        print(f"\n  {filename}:")
        print(f"    Label: {data['label']}")
        print(f"    Fake Probability: {data['fake_probability']:.1%}")
        print(f"    Confidence: {data['confidence']:.1%}")


def generate_mock_frame_results(num_frames: int, fake_prob: float, label: str) -> list:
    """
    Generate mock frame-by-frame results for video analysis.

    Args:
        num_frames: Number of frames to generate
        fake_prob: Base fake probability
        label: "FAKE" or "REAL"

    Returns:
        List of frame results matching the real API format
    """
    import random

    results = []
    for i in range(num_frames):
        # Add some variance to make it look realistic
        variance = random.uniform(-0.05, 0.05)
        frame_prob = max(0.0, min(1.0, fake_prob + variance))
        frame_label = "FAKE" if frame_prob > 0.5 else "REAL"

        results.append({
            "frame": i * 30,  # Assuming 30 FPS, 1 sample per second
            "fake_probability": round(frame_prob, 4),
            "label": frame_label
        })

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demo Mode - Set mock results for deepfake detection demos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mark a video as FAKE with 95% fake probability
  python demo_mode.py --set demo_video.mp4 --fake 0.95

  # Mark an image as REAL with 85% real probability
  python demo_mode.py --set photo.jpg --real 0.85

  # List all mock results
  python demo_mode.py --list

  # Clear a specific mock result
  python demo_mode.py --clear demo_video.mp4
        """
    )

    parser.add_argument(
        "--set",
        metavar="FILENAME",
        help="Set mock result for a file"
    )
    parser.add_argument(
        "--fake",
        type=float,
        metavar="PROB",
        help="Set as FAKE with this probability (0.0-1.0)"
    )
    parser.add_argument(
        "--real",
        type=float,
        metavar="PROB",
        help="Set as REAL with this probability (0.0-1.0)"
    )
    parser.add_argument(
        "--clear",
        metavar="FILENAME",
        help="Clear mock result for a file"
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear all mock results"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all mock results"
    )

    args = parser.parse_args()

    if args.list:
        list_mock_results()
        return 0

    if args.clear_all:
        clear_all_mock_results()
        return 0

    if args.clear:
        clear_mock_result(args.clear)
        return 0

    if args.set:
        if args.fake is not None:
            if not 0.0 <= args.fake <= 1.0:
                print("Error: --fake must be between 0.0 and 1.0")
                return 1
            set_mock_result(args.set, args.fake, "FAKE")
            return 0
        elif args.real is not None:
            if not 0.0 <= args.real <= 1.0:
                print("Error: --real must be between 0.0 and 1.0")
                return 1
            # Convert real probability to fake probability
            fake_prob = 1.0 - args.real
            set_mock_result(args.set, fake_prob, "REAL")
            return 0
        else:
            print("Error: --set requires either --fake or --real")
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
