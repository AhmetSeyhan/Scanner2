"""
Scanner - PentaShield Defense System (v6.0.0)

Five-layer adversarial-immune deepfake detection:
- HYDRA ENGINE: Multi-head adversarial-immune detection
- ZERO-DAY SENTINEL: Unknown deepfake type detection
- FORENSIC DNA: Generator fingerprinting and attribution
- ACTIVE PROBE: Challenge-response liveness verification
- GHOST PROTOCOL: Edge deployment and federated learning

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

__version__ = "6.0.0"

__all__ = [
    "HydraEngine",
    "ZeroDaySentinel",
    "ForensicDNA",
    "ActiveProbe",
    "GhostProtocol",
    "PentaShieldOrchestrator",
]


def get_version() -> str:
    return __version__
