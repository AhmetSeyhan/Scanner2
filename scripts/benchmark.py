#!/usr/bin/env python3
"""
Scanner Benchmark Suite
=======================

Evaluates the PRIME HYBRID detection engine against standard deepfake datasets.

Supported Datasets:
    - FaceForensics++ (FF++)
    - Celeb-DF v2
    - DFDC (Deepfake Detection Challenge)
    - WildDeepfake
    - Custom datasets

Metrics Computed:
    - Accuracy, Precision, Recall, F1-Score
    - AUC-ROC, AUC-PR
    - False Positive Rate (FPR) at fixed True Positive Rates
    - Per-core breakdown (BioSignal, Artifact, Alignment)
    - Confusion matrix and ROC curve plots

Usage:
    # Full benchmark on FaceForensics++
    python scripts/benchmark.py --dataset ff++ --data_dir /path/to/ff++ --output_dir docs/assets

    # Quick benchmark (subset)
    python scripts/benchmark.py --dataset ff++ --data_dir /path/to/ff++ --max_samples 100

    # Custom dataset
    python scripts/benchmark.py --dataset custom --data_dir /path/to/data --output_dir results/

    # Generate plots only from saved results
    python scripts/benchmark.py --from_results results/benchmark_results.json --output_dir docs/assets

Dataset Directory Structure:
    data_dir/
    ├── real/
    │   ├── video001.mp4
    │   ├── video002.mp4
    │   └── ...
    └── fake/
        ├── video001.mp4
        ├── video002.mp4
        └── ...

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def import_dependencies():
    """Import heavy dependencies with availability checks."""
    deps = {}
    try:
        import cv2
        deps["cv2"] = cv2
    except ImportError:
        print("ERROR: opencv-python required. Install: pip install opencv-python")
        sys.exit(1)

    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )
        deps["sklearn"] = {
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "roc_auc_score": roc_auc_score,
            "average_precision_score": average_precision_score,
            "confusion_matrix": confusion_matrix,
            "roc_curve": roc_curve,
            "precision_recall_curve": precision_recall_curve,
            "classification_report": classification_report,
        }
    except ImportError:
        print("ERROR: scikit-learn required. Install: pip install scikit-learn")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        deps["plt"] = plt
    except ImportError:
        print("WARNING: matplotlib not found. Plots will be skipped.")
        deps["plt"] = None

    try:
        from tqdm import tqdm
        deps["tqdm"] = tqdm
    except ImportError:
        deps["tqdm"] = lambda x, **kw: x

    return deps


class BenchmarkRunner:
    """Runs comprehensive benchmarks on deepfake detection datasets."""

    # Dataset configurations
    DATASET_CONFIGS = {
        "ff++": {
            "name": "FaceForensics++",
            "manipulations": ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
            "compression": ["c23", "c40"],
        },
        "celeb_df": {
            "name": "Celeb-DF v2",
            "manipulations": ["CelebDF"],
            "compression": ["default"],
        },
        "dfdc": {
            "name": "DFDC (Preview)",
            "manipulations": ["DFDC"],
            "compression": ["default"],
        },
        "wild": {
            "name": "WildDeepfake",
            "manipulations": ["Wild"],
            "compression": ["default"],
        },
        "custom": {
            "name": "Custom Dataset",
            "manipulations": ["custom"],
            "compression": ["default"],
        },
    }

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        output_dir: str = "docs/assets",
        max_samples: Optional[int] = None,
        max_frames: int = 30,
        verbose: bool = True,
    ):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.max_frames = max_frames
        self.verbose = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = self.DATASET_CONFIGS.get(dataset_name, self.DATASET_CONFIGS["custom"])

        # Results storage
        self.results: Dict[str, Any] = {
            "metadata": {
                "dataset": self.config["name"],
                "dataset_key": dataset_name,
                "data_dir": str(self.data_dir),
                "max_samples": max_samples,
                "max_frames": max_frames,
                "timestamp": datetime.utcnow().isoformat(),
                "scanner_version": "4.0.0",
            },
            "samples": [],
            "metrics": {},
        }

    def discover_videos(self) -> Tuple[List[Path], List[Path]]:
        """Discover real and fake video files in the dataset directory."""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        real_dir = self.data_dir / "real"
        fake_dir = self.data_dir / "fake"

        if not real_dir.exists() or not fake_dir.exists():
            print(f"ERROR: Expected directories: {real_dir} and {fake_dir}")
            print("Dataset structure should be:")
            print("  data_dir/real/*.mp4")
            print("  data_dir/fake/*.mp4")
            sys.exit(1)

        real_videos = sorted([
            f for f in real_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ])
        fake_videos = sorted([
            f for f in fake_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ])

        if self.max_samples:
            half = self.max_samples // 2
            real_videos = real_videos[:half]
            fake_videos = fake_videos[:half]

        if self.verbose:
            print(f"Found {len(real_videos)} real videos, {len(fake_videos)} fake videos")

        return real_videos, fake_videos

    def extract_frames(self, video_path: Path, cv2_module) -> List[np.ndarray]:
        """Extract frames from a video file."""
        cap = cv2_module.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2_module.CAP_PROP_FPS) or 30.0

        interval = max(1, frame_count // self.max_frames)
        frames = []
        idx = 0

        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append(frame)
            idx += 1

        cap.release()
        return frames

    def analyze_video(
        self,
        video_path: Path,
        cv2_module,
        biosignal_core,
        artifact_core,
        alignment_core,
        fusion_engine,
        audio_analyzer,
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single video with PRIME HYBRID and return results."""
        from core.forensic_types import ResolutionTier, VideoProfile

        frames = self.extract_frames(video_path, cv2_module)
        if len(frames) < 10:
            return None

        h, w = frames[0].shape[:2]
        cap = cv2_module.VideoCapture(str(video_path))
        fps = cap.get(cv2_module.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2_module.CAP_PROP_FRAME_COUNT))
        cap.release()

        duration = frame_count / fps if fps > 0 else 0

        if h <= 360:
            tier = ResolutionTier.ULTRA_LOW
        elif h <= 480:
            tier = ResolutionTier.LOW
        elif h <= 720:
            tier = ResolutionTier.MEDIUM
        elif h <= 1080:
            tier = ResolutionTier.HIGH
        else:
            tier = ResolutionTier.ULTRA_HIGH

        video_profile = VideoProfile(
            width=w, height=h, fps=fps,
            frame_count=frame_count, duration_seconds=duration,
            resolution_tier=tier, pixel_count=w * h,
            aspect_ratio=w / h if h > 0 else 1.0,
            rppg_viable=h >= 480 and fps >= 24,
            mesh_viable=h >= 720,
            recommended_analysis="PRIME HYBRID"
        )

        # Run cores
        start_time = time.time()
        biosignal_result = biosignal_core.analyze(frames, fps, video_profile)
        artifact_result = artifact_core.analyze(frames, video_profile)
        alignment_result = alignment_core.analyze(frames, fps, str(video_path), video_profile)

        audio_profile = None
        try:
            audio_profile = audio_analyzer.analyze(str(video_path))
        except Exception:
            pass

        verdict = fusion_engine.get_final_integrity_score(
            biosignal_result, artifact_result, alignment_result,
            video_profile, audio_profile
        )
        elapsed = time.time() - start_time

        return {
            "file": video_path.name,
            "verdict": verdict.verdict,
            "integrity_score": verdict.integrity_score,
            "confidence": verdict.confidence,
            "fusion_score": 1.0 - (verdict.integrity_score / 100.0),
            "biosignal_score": verdict.biosignal_score,
            "artifact_score": verdict.artifact_score,
            "alignment_score": verdict.alignment_score,
            "leading_core": verdict.leading_core,
            "consensus_type": verdict.consensus_type,
            "elapsed_seconds": round(elapsed, 3),
            "resolution": f"{h}p",
            "frames_analyzed": len(frames),
        }

    def run(self, deps) -> Dict[str, Any]:
        """Run the full benchmark suite."""
        cv2_module = deps["cv2"]
        tqdm = deps["tqdm"]
        sk = deps["sklearn"]

        print(f"\n{'='*70}")
        print("  SCANNER BENCHMARK SUITE v3.2.0")
        print(f"  Dataset: {self.config['name']}")
        print(f"  Data Directory: {self.data_dir}")
        print(f"{'='*70}\n")

        # Initialize cores
        print("Initializing PRIME HYBRID cores...")
        from core.alignment_core import AlignmentCore
        from core.artifact_core import ArtifactCore
        from core.audio_analyzer import AudioAnalyzer
        from core.biosignal_core import BioSignalCore
        from core.fusion_engine import FusionEngine

        biosignal_core = BioSignalCore()
        artifact_core = ArtifactCore()
        alignment_core = AlignmentCore()
        fusion_engine = FusionEngine()
        audio_analyzer = AudioAnalyzer()
        print("All cores initialized.\n")

        # Discover videos
        real_videos, fake_videos = self.discover_videos()

        y_true = []
        y_scores = []
        y_pred = []
        per_core_scores = {"biosignal": [], "artifact": [], "alignment": []}
        timings = []

        # Process real videos
        print(f"\nProcessing {len(real_videos)} REAL videos...")
        for video_path in tqdm(real_videos, desc="Real"):
            result = self.analyze_video(
                video_path, cv2_module,
                biosignal_core, artifact_core, alignment_core,
                fusion_engine, audio_analyzer
            )
            if result:
                result["ground_truth"] = "REAL"
                result["label"] = 0
                self.results["samples"].append(result)

                y_true.append(0)
                y_scores.append(result["fusion_score"])
                y_pred.append(1 if result["fusion_score"] >= 0.5 else 0)
                per_core_scores["biosignal"].append(result["biosignal_score"])
                per_core_scores["artifact"].append(result["artifact_score"])
                per_core_scores["alignment"].append(result["alignment_score"])
                timings.append(result["elapsed_seconds"])

        # Process fake videos
        print(f"\nProcessing {len(fake_videos)} FAKE videos...")
        for video_path in tqdm(fake_videos, desc="Fake"):
            result = self.analyze_video(
                video_path, cv2_module,
                biosignal_core, artifact_core, alignment_core,
                fusion_engine, audio_analyzer
            )
            if result:
                result["ground_truth"] = "FAKE"
                result["label"] = 1
                self.results["samples"].append(result)

                y_true.append(1)
                y_scores.append(result["fusion_score"])
                y_pred.append(1 if result["fusion_score"] >= 0.5 else 0)
                per_core_scores["biosignal"].append(result["biosignal_score"])
                per_core_scores["artifact"].append(result["artifact_score"])
                per_core_scores["alignment"].append(result["alignment_score"])
                timings.append(result["elapsed_seconds"])

        if len(y_true) == 0:
            print("ERROR: No videos were successfully processed.")
            return self.results

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = np.array(y_pred)

        # Compute metrics
        metrics = {
            "total_samples": len(y_true),
            "real_samples": int(np.sum(y_true == 0)),
            "fake_samples": int(np.sum(y_true == 1)),
            "accuracy": float(sk["accuracy_score"](y_true, y_pred)),
            "precision": float(sk["precision_score"](y_true, y_pred, zero_division=0)),
            "recall": float(sk["recall_score"](y_true, y_pred, zero_division=0)),
            "f1_score": float(sk["f1_score"](y_true, y_pred, zero_division=0)),
            "auc_roc": float(sk["roc_auc_score"](y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0,
            "auc_pr": float(sk["average_precision_score"](y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0,
            "confusion_matrix": sk["confusion_matrix"](y_true, y_pred).tolist(),
            "avg_inference_time_sec": float(np.mean(timings)),
            "median_inference_time_sec": float(np.median(timings)),
            "p95_inference_time_sec": float(np.percentile(timings, 95)),
        }

        # FPR at various TPR thresholds
        if len(np.unique(y_true)) > 1:
            fpr_arr, tpr_arr, _ = sk["roc_curve"](y_true, y_scores)
            for target_tpr in [0.90, 0.95, 0.99]:
                idx = np.argmin(np.abs(tpr_arr - target_tpr))
                metrics[f"fpr_at_tpr_{int(target_tpr*100)}"] = float(fpr_arr[idx])

        # Per-core metrics
        for core_name, scores in per_core_scores.items():
            if scores:
                core_scores = np.array(scores)
                core_pred = (core_scores >= 0.5).astype(int)
                metrics[f"{core_name}_accuracy"] = float(sk["accuracy_score"](y_true, core_pred))
                metrics[f"{core_name}_auc_roc"] = float(
                    sk["roc_auc_score"](y_true, core_scores)
                ) if len(np.unique(y_true)) > 1 else 0.0

        self.results["metrics"] = metrics

        # Print summary
        self._print_summary(metrics)

        # Save results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")

        # Generate plots
        if deps["plt"] is not None:
            self._generate_plots(y_true, y_scores, y_pred, per_core_scores, sk, deps["plt"])

        return self.results

    def _print_summary(self, metrics: Dict[str, Any]):
        """Print formatted benchmark summary."""
        print(f"\n{'='*70}")
        print(f"  BENCHMARK RESULTS: {self.config['name']}")
        print(f"{'='*70}")
        print(f"  Total Samples:   {metrics['total_samples']} ({metrics['real_samples']} real, {metrics['fake_samples']} fake)")
        print(f"{'─'*70}")
        print(f"  {'Metric':<30} {'Value':>10}")
        print(f"{'─'*70}")
        print(f"  {'Accuracy':<30} {metrics['accuracy']:>10.4f}")
        print(f"  {'Precision':<30} {metrics['precision']:>10.4f}")
        print(f"  {'Recall':<30} {metrics['recall']:>10.4f}")
        print(f"  {'F1-Score':<30} {metrics['f1_score']:>10.4f}")
        print(f"  {'AUC-ROC':<30} {metrics['auc_roc']:>10.4f}")
        print(f"  {'AUC-PR':<30} {metrics['auc_pr']:>10.4f}")
        print(f"{'─'*70}")
        if "fpr_at_tpr_95" in metrics:
            print(f"  {'FPR @ 90% TPR':<30} {metrics.get('fpr_at_tpr_90', 'N/A'):>10.4f}")
            print(f"  {'FPR @ 95% TPR':<30} {metrics.get('fpr_at_tpr_95', 'N/A'):>10.4f}")
            print(f"  {'FPR @ 99% TPR':<30} {metrics.get('fpr_at_tpr_99', 'N/A'):>10.4f}")
            print(f"{'─'*70}")
        print(f"  {'Avg Inference Time':<30} {metrics['avg_inference_time_sec']:>10.3f}s")
        print(f"  {'Median Inference Time':<30} {metrics['median_inference_time_sec']:>10.3f}s")
        print(f"  {'P95 Inference Time':<30} {metrics['p95_inference_time_sec']:>10.3f}s")
        print(f"{'─'*70}")

        # Per-core breakdown
        for core in ["biosignal", "artifact", "alignment"]:
            acc_key = f"{core}_accuracy"
            auc_key = f"{core}_auc_roc"
            if acc_key in metrics:
                print(f"  {core.upper() + ' CORE Accuracy':<30} {metrics[acc_key]:>10.4f}")
                print(f"  {core.upper() + ' CORE AUC-ROC':<30} {metrics[auc_key]:>10.4f}")
        print(f"{'='*70}\n")

    def _generate_plots(self, y_true, y_scores, y_pred, per_core_scores, sk, plt):
        """Generate ROC curve and confusion matrix plots."""
        # --- ROC Curve ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Main ROC
        fpr, tpr, _ = sk["roc_curve"](y_true, y_scores)
        auc = sk["roc_auc_score"](y_true, y_scores)

        axes[0].plot(fpr, tpr, color="#0066FF", lw=2, label=f"PRIME HYBRID (AUC = {auc:.4f})")

        # Per-core ROC
        colors = {"biosignal": "#00C853", "artifact": "#FF3D57", "alignment": "#FFB800"}
        for core_name, scores in per_core_scores.items():
            if scores:
                core_fpr, core_tpr, _ = sk["roc_curve"](y_true, np.array(scores))
                core_auc = sk["roc_auc_score"](y_true, np.array(scores))
                axes[0].plot(
                    core_fpr, core_tpr, color=colors[core_name], lw=1.5, linestyle="--",
                    label=f"{core_name.upper()} (AUC = {core_auc:.4f})"
                )

        axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
        axes[0].set_xlabel("False Positive Rate", fontsize=12)
        axes[0].set_ylabel("True Positive Rate", fontsize=12)
        axes[0].set_title(f"ROC Curve - {self.config['name']}", fontsize=14, fontweight="bold")
        axes[0].legend(loc="lower right", fontsize=9)
        axes[0].grid(True, alpha=0.3)

        # --- Confusion Matrix ---
        cm = sk["confusion_matrix"](y_true, y_pred)
        im = axes[1].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[1].set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=axes[1], shrink=0.8)

        classes = ["Real", "Fake"]
        tick_marks = np.arange(len(classes))
        axes[1].set_xticks(tick_marks)
        axes[1].set_xticklabels(classes)
        axes[1].set_yticks(tick_marks)
        axes[1].set_yticklabels(classes)
        axes[1].set_xlabel("Predicted", fontsize=12)
        axes[1].set_ylabel("Actual", fontsize=12)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1].text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold"
                )

        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_roc_confusion.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"ROC curve and confusion matrix saved to: {plot_path}")

        # --- Precision-Recall Curve ---
        fig, ax = plt.subplots(figsize=(7, 6))
        precision_arr, recall_arr, _ = sk["precision_recall_curve"](y_true, y_scores)
        auc_pr = sk["average_precision_score"](y_true, y_scores)

        ax.plot(recall_arr, precision_arr, color="#0066FF", lw=2, label=f"PRIME HYBRID (AP = {auc_pr:.4f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve - {self.config['name']}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        pr_path = self.output_dir / "benchmark_precision_recall.png"
        plt.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Precision-Recall curve saved to: {pr_path}")

    def run_ablation(self, deps) -> Dict[str, Any]:
        """
        Ablation study: measure each core's standalone and fusion-combined performance.

        Produces a table showing individual and combined core contributions.
        """
        sk = deps["sklearn"]
        if not self.results.get("samples"):
            print("ERROR: Run benchmark first (no samples found)")
            return {}

        samples = self.results["samples"]
        y_true = np.array([s["label"] for s in samples])

        configs = {
            "BIOSIGNAL only": [("biosignal_score", 1.0)],
            "ARTIFACT only": [("artifact_score", 1.0)],
            "ALIGNMENT only": [("alignment_score", 1.0)],
            "BIO + ART": [("biosignal_score", 0.5), ("artifact_score", 0.5)],
            "BIO + ALIGN": [("biosignal_score", 0.5), ("alignment_score", 0.5)],
            "ART + ALIGN": [("artifact_score", 0.5), ("alignment_score", 0.5)],
            "FULL FUSION": [("fusion_score", 1.0)],
        }

        ablation_results = {}
        print(f"\n{'='*70}")
        print(f"  ABLATION STUDY: {self.config['name']}")
        print(f"{'='*70}")
        print(f"  {'Config':<22} {'AUC-ROC':>10} {'Accuracy':>10} {'FPR@95TPR':>10}")
        print(f"{'─'*70}")

        for config_name, score_keys in configs.items():
            y_scores = np.zeros(len(samples))
            for key, weight in score_keys:
                y_scores += np.array([s[key] for s in samples]) * weight

            y_pred = (y_scores >= 0.5).astype(int)
            auc = float(sk["roc_auc_score"](y_true, y_scores)) if len(np.unique(y_true)) > 1 else 0.0
            acc = float(sk["accuracy_score"](y_true, y_pred))

            fpr_at_95 = 0.0
            if len(np.unique(y_true)) > 1:
                fpr_arr, tpr_arr, _ = sk["roc_curve"](y_true, y_scores)
                idx = np.argmin(np.abs(tpr_arr - 0.95))
                fpr_at_95 = float(fpr_arr[idx])

            ablation_results[config_name] = {
                "auc_roc": auc, "accuracy": acc, "fpr_at_tpr_95": fpr_at_95,
            }
            print(f"  {config_name:<22} {auc:>10.4f} {acc:>10.4f} {fpr_at_95:>10.4f}")

        print(f"{'='*70}")

        # Fusion delta (how much fusion improves over best single core)
        best_single = max(
            ablation_results[k]["auc_roc"]
            for k in ["BIOSIGNAL only", "ARTIFACT only", "ALIGNMENT only"]
        )
        fusion_auc = ablation_results["FULL FUSION"]["auc_roc"]
        print(f"\n  Fusion Delta: +{(fusion_auc - best_single):.4f} AUC over best single core")

        self.results["ablation"] = ablation_results

        # Save updated results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        return ablation_results


def run_cross_dataset(data_dirs: List[str], output_dir: str, max_samples: int = 100):
    """Evaluate across multiple datasets - measures generalization."""
    deps = import_dependencies()
    sk = deps["sklearn"]

    results_matrix = {}
    for eval_dir in data_dirs:
        name = Path(eval_dir).name
        runner = BenchmarkRunner(
            dataset_name="custom", data_dir=eval_dir,
            output_dir=output_dir, max_samples=max_samples,
        )
        result = runner.run(deps)
        results_matrix[name] = result.get("metrics", {})

    # Print generalization matrix
    print(f"\n{'='*70}")
    print("  CROSS-DATASET GENERALIZATION MATRIX")
    print(f"{'='*70}")
    print(f"  {'Dataset':<20} {'AUC-ROC':>10} {'Accuracy':>10} {'F1':>10}")
    print(f"{'─'*70}")
    for name, m in results_matrix.items():
        print(f"  {name:<20} {m.get('auc_roc',0):>10.4f} {m.get('accuracy',0):>10.4f} {m.get('f1_score',0):>10.4f}")
    print(f"{'='*70}")

    out_path = Path(output_dir) / "cross_dataset_results.json"
    with open(out_path, "w") as f:
        json.dump(results_matrix, f, indent=2)
    print(f"Saved to {out_path}")


def generate_plots_from_results(results_path: str, output_dir: str):
    """Generate plots from previously saved benchmark results."""
    deps = import_dependencies()
    plt = deps["plt"]
    sk = deps["sklearn"]

    if plt is None:
        print("ERROR: matplotlib required for plot generation.")
        return

    with open(results_path) as f:
        results = json.load(f)

    y_true = [s["label"] for s in results["samples"]]
    y_scores = [s["fusion_score"] for s in results["samples"]]
    y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

    per_core_scores = {
        "biosignal": [s["biosignal_score"] for s in results["samples"]],
        "artifact": [s["artifact_score"] for s in results["samples"]],
        "alignment": [s["alignment_score"] for s in results["samples"]],
    }

    dataset_name = results["metadata"].get("dataset", "Unknown")
    runner = BenchmarkRunner("custom", ".", output_dir)
    runner.config = {"name": dataset_name}

    runner._generate_plots(
        np.array(y_true), np.array(y_scores), np.array(y_pred),
        per_core_scores, sk, plt
    )


def main():
    parser = argparse.ArgumentParser(
        description="Scanner Benchmark Suite - Evaluate deepfake detection performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/benchmark.py --dataset ff++ --data_dir /data/faceforensics
    python scripts/benchmark.py --dataset celeb_df --data_dir /data/celeb_df --max_samples 200
    python scripts/benchmark.py --from_results results/benchmark_results.json
        """
    )
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["ff++", "celeb_df", "dfdc", "wild", "custom"],
                        help="Dataset to benchmark against")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="docs/assets",
                        help="Directory for output files (plots, JSON)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max total samples (for quick testing)")
    parser.add_argument("--max_frames", type=int, default=30,
                        help="Max frames per video")
    parser.add_argument("--from_results", type=str, default=None,
                        help="Generate plots from saved results JSON")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study after benchmark")
    parser.add_argument("--cross_dataset", nargs="+", default=None,
                        help="Run cross-dataset eval (list of data_dirs)")

    args = parser.parse_args()

    if args.cross_dataset:
        run_cross_dataset(args.cross_dataset, args.output_dir, args.max_samples or 100)
        return

    if args.from_results:
        generate_plots_from_results(args.from_results, args.output_dir)
        return

    if args.data_dir is None:
        parser.error("--data_dir is required when running benchmarks")

    deps = import_dependencies()

    runner = BenchmarkRunner(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        max_frames=args.max_frames,
        verbose=not args.quiet,
    )

    results = runner.run(deps)

    # Run ablation study if requested
    if args.ablation:
        runner.run_ablation(deps)

    # Print final metrics as markdown table for README
    metrics = results.get("metrics", {})
    if metrics:
        print("\n\nMarkdown table for documentation:")
        print("| Metric | Value |")
        print("|--------|-------|")
        for key in ["accuracy", "precision", "recall", "f1_score", "auc_roc", "auc_pr"]:
            if key in metrics:
                print(f"| {key.replace('_', ' ').title()} | {metrics[key]:.4f} |")


if __name__ == "__main__":
    main()
