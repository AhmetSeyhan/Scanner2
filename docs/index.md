# Scanner - Enterprise Deepfake Detection

Scanner is an enterprise-grade deepfake detection platform that uses multi-modal forensic analysis to identify AI-generated and manipulated video content.

## Why Scanner?

| Capability | Description |
|-----------|-------------|
| **Multi-Core Detection** | Four independent analysis engines (biological, artifact, alignment, audio) provide defense-in-depth |
| **Explainable AI** | Every verdict includes a transparency report explaining which signals drove the decision |
| **High Precision** | Conservative thresholds minimize false positives - critical for financial and legal use cases |
| **Adversarial Defense** | InputSanityGuard screens for evasion attacks before analysis begins |
| **Privacy-First** | No raw media storage; uploaded files deleted immediately after analysis |
| **Enterprise Infrastructure** | JWT auth, rate limiting, PDF reports, scan history, webhook notifications |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.12+ | 3.12 |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB | 10 GB |
| GPU | Not required | NVIDIA CUDA-capable |
| OS | Linux, macOS, Windows (WSL2) | Ubuntu 22.04+ |

## Quick Links

- [Installation Guide](installation.md)
- [Architecture Overview](architecture.md)
- [API Reference](api.md)
- [Benchmark Results](benchmarks.md)
- [Deployment Guide](deployment.md)
- [Security Policy](security.md)
