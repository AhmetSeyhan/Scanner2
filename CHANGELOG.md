# Changelog

All notable changes to Scanner are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.2.0] - 2026-02-05

### Added
- **InputSanityGuard**: Adversarial input detection with frame-sequence validation, gradient analysis, and frequency-domain checks.
- **WeightManager**: Hot-reload capability for model weights without service restart.
- **HistoryManager**: SQLite-backed scan history with last-100 retention and per-day statistics.
- **ForensicReporter**: Professional PDF report generation via ReportLab with branded styles.
- **StorageManager**: S3/MinIO cloud storage abstraction for analysis artifacts.
- **WebhookManager**: Async HTTP webhook notifications for scan events.
- **HeatmapAnalysis**: 8x8 spatial anomaly grid in ArtifactCore for forensic visualization.
- Rate limiting via slowapi + Redis (10 req/min per IP, configurable).
- Enterprise API endpoints: `/history`, `/history/stats`, `/export/pdf/{session_id}`, `/admin/dashboard`.
- Unit tests for BioSignalCore, ArtifactCore, AlignmentCore, FusionEngine, and InputSanityGuard.
- GitHub Actions CI/CD pipeline: lint, test with coverage, Docker build, Trivy security scan.
- Docker-compose services for Redis and MinIO.
- `SanityCheckResult` and `ScanHistoryEntry` dataclasses.

### Changed
- API version bumped to 3.2.0.
- `requirements.txt` updated with 8 new enterprise dependencies.
- `docker-compose.yml` extended with Redis and MinIO services.

## [3.0.0] - 2026-02-04

### Changed
- **BREAKING**: Renamed all core modules for trademark compliance:
  - INTEL CORE -> BIOSIGNAL CORE
  - SENTINEL CORE -> ARTIFACT CORE
  - DEFENDER CORE -> ALIGNMENT CORE
  - THE ARBITER -> FUSION ENGINE
- **BREAKING**: API response keys changed (`intel` -> `biosignal`, `sentinel` -> `artifact`, `defender` -> `alignment`).
- Added copyright headers to all core modules.
- Updated `download_weights.py` with open-source license documentation.

## [2.0.0] - 2026-02-03

### Added
- **PRIME HYBRID Architecture**: Modular forensic engine with 4 independent cores + Fusion Engine.
- **BioSignalCore**: 32-ROI rPPG biological signal analysis with Butterworth bandpass filter.
- **ArtifactCore**: GAN/Diffusion/VAE fingerprint detection with FFT grid analysis.
- **AlignmentCore**: Phoneme-Viseme mapping with lip-closure detection and speech rhythm analysis.
- **AudioAnalyzer**: SNR estimation for adaptive weight adjustment.
- **FusionEngine**: Unified decision engine with dynamic weight redistribution and conflict resolution.
- `forensic_types.py` with shared dataclasses for all modules.
- High Precision Mode with INCONCLUSIVE verdict for ambiguous cases.
- Dynamic weight redistribution when core confidence < 0.4.
- Audio-aware ALIGNMENT weighting based on SNR.
- Resolution-aware BIOSIGNAL weighting for low-res input.
- TransparencyReport for detailed verdict explanations.
- Confidence-product fusion mode as alternative to weighted average.

### Changed
- API extended with `/analyze-video-v2` endpoint using full PRIME HYBRID analysis.

## [1.2.0] - 2026-02-01

### Added
- **SCANNER ELITE v5.0 "Tactical Command Suite"**: Professional Streamlit dashboard.
- War Room interface with animated liquid borders and dual-tone corner glows.
- Professional navbar with radar pulse logo and system status.
- Dual-pane layout: Analysis Zone + Intelligence Logs.
- Live Sentinel Mode with real-time webcam analysis.
- Trust Score System (0-100%) with adaptive weighting.
- Blockchain-Ready Origin Hash (SHA-256).
- Batch processing for multiple video files.

## [1.1.0] - 2026-01-31

### Added
- JWT Bearer token authentication (`auth.py`).
- API key support (`X-API-Key` header).
- Dockerfile for production deployment.
- `docker-compose.yml` with API + Dashboard services.
- `train.py` for custom dataset training (FF++/Celeb-DF/DFDC).
- Demo credentials for investor presentations.

### Changed
- All protected endpoints now require authentication.

## [1.0.0] - 2026-01-25

### Added
- EfficientNet-B0 backbone with deepfake-trained weight loading.
- `download_weights.py` for weight management.
- Streamlit dashboard for video/image analysis.
- Project rebranded to **Scanner**.

## [0.1.0] - 2026-01-24

### Added
- Initial project structure.
- MediaPipe Tasks API face extraction (`preprocessing.py`).
- DeepfakeDetector model (`model.py`) with ImageNet initialization.
- FastAPI backend (`api.py`) with video and image analysis endpoints.
- Test suite with 4 passing tests.

[Unreleased]: https://github.com/AhmetSeyhan/Scanner2/compare/v3.2.0...HEAD
[3.2.0]: https://github.com/AhmetSeyhan/Scanner2/compare/v3.0.0...v3.2.0
[3.0.0]: https://github.com/AhmetSeyhan/Scanner2/compare/v2.0.0...v3.0.0
[2.0.0]: https://github.com/AhmetSeyhan/Scanner2/compare/v1.2.0...v2.0.0
[1.2.0]: https://github.com/AhmetSeyhan/Scanner2/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/AhmetSeyhan/Scanner2/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/AhmetSeyhan/Scanner2/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/AhmetSeyhan/Scanner2/releases/tag/v0.1.0
