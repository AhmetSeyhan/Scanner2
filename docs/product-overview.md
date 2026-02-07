# Scanner - Product Overview

## The Problem

AI-generated deepfakes pose an escalating threat to financial institutions. From synthetic identity fraud to CEO impersonation in video calls, manipulated media undermines trust in digital communications. Existing detection tools suffer from high false positive rates, lack of explainability, and vulnerability to adversarial attacks.

## The Solution

Scanner is an enterprise deepfake detection platform that uses multi-modal forensic analysis to identify AI-generated and manipulated video with high precision and full explainability. Built for environments where accuracy matters and false accusations are unacceptable.

## How It Works

Scanner's PRIME HYBRID engine analyzes video through four independent detection cores:

| Core | What It Detects | How |
|------|----------------|-----|
| **BIOSIGNAL** | Absent or synthetic biological signals | Remote photoplethysmography (rPPG) across 32 facial ROIs |
| **ARTIFACT** | GAN, diffusion, and VAE generation fingerprints | FFT spectral analysis + spatial anomaly heatmaps |
| **ALIGNMENT** | Audio-visual desynchronization | Phoneme-viseme mapping + lip closure timing |
| **AUDIO** | Audio quality degradation | SNR estimation for adaptive confidence weighting |

The **Fusion Engine** combines results using dynamic weight redistribution and consensus rules. No single core can trigger a "MANIPULATED" verdict alone - multi-core agreement is required.

## Key Metrics

| Metric | Value |
|--------|-------|
| Accuracy (FaceForensics++) | 95.5% |
| AUC-ROC (FaceForensics++) | 0.989 |
| False Positive Rate @ 95% TPR | 2.3% |
| Average Latency (GPU) | 0.9s per video |
| Explainability | Full transparency report per verdict |

## Deployment Options

| Option | Description | Best For |
|--------|-------------|----------|
| **On-Premise** | Docker/Kubernetes deployment within client infrastructure | Banks, government, regulated industries |
| **Private Cloud** | Dedicated cloud instance (AWS/Azure/GCP) | Organizations with cloud-first policy |
| **API Integration** | RESTful API for embedding into existing workflows | Fintech platforms, compliance tools |

## Enterprise Support Tiers

| Tier | Features | SLA |
|------|----------|-----|
| **Standard** | Email support, quarterly updates, documentation access | 48h response |
| **Professional** | Priority support, monthly updates, custom weight training, dedicated Slack channel | 4h response |
| **Enterprise** | 24/7 support, on-site deployment, custom model development, penetration testing, compliance audit | 1h response |

## Security & Compliance

- **No raw data retention**: Uploaded media deleted immediately after analysis
- **JWT + API key authentication** with scope-based access control
- **Rate limiting** and adversarial input screening
- **GDPR/KVKK** compatible architecture
- **SOC 2 Type II** audit-ready (see SECURITY.md)
- **Apache 2.0** licensed

## Technology Stack

- Python 3.12 / FastAPI / PyTorch
- EfficientNet-B0 backbone (17 MB)
- MediaPipe face detection
- Docker + Redis + S3-compatible storage
- CI/CD with GitHub Actions + Trivy security scanning

## Contact

- **Sales**: enterprise@scanner.ai
- **Technical**: support@scanner.ai
- **Website**: scanner.ai
