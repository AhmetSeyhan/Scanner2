"""
Scanner Prime - Weight Download Utility
Original Implementation by Scanner Prime Team based on Public Academic Research.

This script downloads publicly available pre-trained weights from open-source
repositories. All weights are from permissively licensed sources.

Weight Sources:
- DeepfakeBench (MIT License): https://github.com/SCLBD/DeepfakeBench
  - Xception trained on FaceForensics++ (c23)
  - EfficientNet-B4 trained on FaceForensics++
  - UCF detector trained on FaceForensics++
  - F3Net trained on FaceForensics++

Alternative (Built-in):
- timm library (Apache 2.0): EfficientNet-B0 with ImageNet weights
- torchvision (BSD-3): EfficientNet-B0 with ImageNet weights

The Scanner model uses EfficientNet-B0 from timm (Apache 2.0 licensed) and will
work without any external weights by falling back to ImageNet initialization.
External weights are optional and provided for improved deepfake detection accuracy.

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

# Weight configurations - URLs for publicly available deepfake detection weights
# Source: DeepfakeBench (MIT License)
# Repository: https://github.com/SCLBD/DeepfakeBench/releases/tag/v1.0.1
#
# NOTE: These weights are OPTIONAL. Scanner works with ImageNet-initialized
# EfficientNet-B0 from timm (Apache 2.0) without any external downloads.
# These weights can improve deepfake detection accuracy when available.
WEIGHT_CONFIGS = {
    "xception_ff++": {
        "filename": "xception_best.pth",
        "urls": [
            "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/xception_best.pth",
        ],
        "description": "Xception trained on FaceForensics++ (c23) - DeepfakeBench",
        "input_size": 299,
        "architecture": "xception",
    },
    "efficientnet_b4_ff++": {
        "filename": "effnb4_best.pth",
        "urls": [
            "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/effnb4_best.pth",
        ],
        "description": "EfficientNet-B4 trained on FaceForensics++ (c23) - DeepfakeBench",
        "input_size": 380,
        "architecture": "efficientnet_b4",
    },
    "ucf_ff++": {
        "filename": "ucf_best.pth",
        "urls": [
            "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/ucf_best.pth",
        ],
        "description": "UCF detector trained on FaceForensics++ (c23) - DeepfakeBench",
        "input_size": 299,
        "architecture": "ucf",
    },
    "f3net_ff++": {
        "filename": "f3net_best.pth",
        "urls": [
            "https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/f3net_best.pth",
        ],
        "description": "F3Net trained on FaceForensics++ (c23) - DeepfakeBench",
        "input_size": 299,
        "architecture": "f3net",
    },
}

# Default model to download
DEFAULT_MODEL = "xception_ff++"


def get_weights_dir() -> Path:
    """Get the weights directory path."""
    return Path(__file__).parent / "weights"


def compute_md5(filepath: Path) -> str:
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> bool:
    """
    Download a file from URL to destination path.

    Args:
        url: Source URL
        dest_path: Destination file path
        show_progress: Whether to show download progress

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Downloading from: {url}")

        # Create request with user agent
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (DeepfakeDetector/1.0)"}
        )

        with urllib.request.urlopen(request, timeout=120) as response:
            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0
            block_size = 8192

            with open(dest_path, "wb") as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)

                    if show_progress and total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        bar_length = 40
                        filled = int(bar_length * downloaded / total_size)
                        bar = "=" * filled + "-" * (bar_length - filled)
                        sys.stdout.write(f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
                        sys.stdout.flush()

            if show_progress:
                print()  # New line after progress bar

        print(f"Successfully downloaded to: {dest_path}")
        return True

    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def download_weights(model_key: str = DEFAULT_MODEL, force: bool = False) -> Path | None:
    """
    Download pre-trained weights for the specified model.

    Args:
        model_key: Key from WEIGHT_CONFIGS
        force: Force re-download even if file exists

    Returns:
        Path to downloaded weights, or None if failed
    """
    if model_key not in WEIGHT_CONFIGS:
        print(f"Unknown model: {model_key}")
        print(f"Available models: {list(WEIGHT_CONFIGS.keys())}")
        return None

    config = WEIGHT_CONFIGS[model_key]
    weights_dir = get_weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)

    dest_path = weights_dir / config["filename"]

    # Check if already exists
    if dest_path.exists() and not force:
        file_size = dest_path.stat().st_size / (1024 * 1024)
        print(f"Weights already exist: {dest_path} ({file_size:.1f} MB)")
        return dest_path

    print(f"Downloading: {config['description']}")
    print(f"Input size: {config['input_size']}x{config['input_size']}")
    print(f"Architecture: {config['architecture']}")
    print("-" * 60)

    # Try each URL until one works
    for url in config["urls"]:
        if download_file(url, dest_path):
            return dest_path

    print("Failed to download weights from all sources.")
    return None


def list_available_models():
    """Print available model configurations."""
    print("Available pre-trained models (from DeepfakeBench):")
    print("=" * 60)
    for key, config in WEIGHT_CONFIGS.items():
        print(f"\n{key}:")
        print(f"  Description: {config['description']}")
        print(f"  Input size: {config['input_size']}x{config['input_size']}")
        print(f"  Architecture: {config['architecture']}")
        print(f"  Filename: {config['filename']}")


def download_all():
    """Download all available weights."""
    print("Downloading all available weights...")
    print("=" * 60)

    results = {}
    for key in WEIGHT_CONFIGS:
        print(f"\n>>> Downloading {key}...")
        result = download_weights(key)
        results[key] = result is not None

    print("\n" + "=" * 60)
    print("Download Summary:")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {key}: {status}")

    return all(results.values())


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Download pre-trained deepfake detection weights")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists"
    )

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return 0

    if args.all:
        success = download_all()
        return 0 if success else 1

    result = download_weights(args.model, args.force)
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
