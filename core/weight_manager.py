"""
Scanner Prime - Weight Manager
Hot-reload support for model and fusion weights.

Monitors weight files/configs for changes and triggers
reload callbacks when updates are detected.

Copyright (c) 2026 Scanner Prime Team. All rights reserved.
"""

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class WeightConfiguration:
    """Fusion weight configuration."""
    biosignal: float = 0.33
    artifact: float = 0.33
    alignment: float = 0.34
    version: str = "3.2.0"
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = "Default weights"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "biosignal": self.biosignal,
            "artifact": self.artifact,
            "alignment": self.alignment,
            "version": self.version,
            "updated_at": self.updated_at,
            "description": self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WeightConfiguration":
        return cls(
            biosignal=data.get("biosignal", 0.33),
            artifact=data.get("artifact", 0.33),
            alignment=data.get("alignment", 0.34),
            version=data.get("version", "3.2.0"),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            description=data.get("description", "")
        )

    def validate(self) -> bool:
        """Validate that weights sum to 1.0 and are non-negative."""
        weights = [self.biosignal, self.artifact, self.alignment]
        if any(w < 0 for w in weights):
            return False
        total = sum(weights)
        return 0.99 <= total <= 1.01  # Allow small floating point error


class WeightManager:
    """
    Model weight hot-reload manager.

    Monitors weight files for changes and triggers
    reload callbacks when updates are detected.
    """

    DEFAULT_CONFIG_PATH = Path("config/weights.json")

    def __init__(
        self,
        weights_dir: str = "weights",
        config_path: Optional[str] = None
    ):
        """
        Initialize weight manager.

        Args:
            weights_dir: Directory containing model weight files
            config_path: Path to fusion weights config file
        """
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)

        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self._file_hashes: Dict[str, str] = {}
        self._callbacks: Dict[str, Callable] = {}
        self._watch_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_config: Optional[WeightConfiguration] = None
        self._config_hash: str = ""

    def get_fusion_weights(self) -> Dict[str, float]:
        """
        Get current fusion weights.

        Loads from config file if exists, otherwise returns defaults.
        """
        if self._current_config is None:
            self._load_config()

        return {
            "biosignal": self._current_config.biosignal,
            "artifact": self._current_config.artifact,
            "alignment": self._current_config.alignment,
        }

    def _load_config(self) -> bool:
        """Load weight configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                config = WeightConfiguration.from_dict(data)
                if config.validate():
                    self._current_config = config
                    self._config_hash = self._compute_file_hash(self.config_path)
                    logger.info(f"Loaded weight config v{config.version}")
                    return True
                else:
                    logger.warning("Invalid weight config, using defaults")
            except Exception as e:
                logger.error(f"Error loading weight config: {e}")

        # Use defaults
        self._current_config = WeightConfiguration()
        return False

    def save_config(self, config: WeightConfiguration) -> bool:
        """
        Save weight configuration to file.

        Args:
            config: WeightConfiguration to save

        Returns:
            True if successful
        """
        if not config.validate():
            logger.error("Invalid configuration - weights must sum to 1.0")
            return False

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            self._current_config = config
            self._config_hash = self._compute_file_hash(self.config_path)
            logger.info(f"Saved weight config v{config.version}")
            return True
        except Exception as e:
            logger.error(f"Error saving weight config: {e}")
            return False

    def update_weights(
        self,
        biosignal: float,
        artifact: float,
        alignment: float,
        description: str = ""
    ) -> bool:
        """
        Update fusion weights.

        Args:
            biosignal: BIOSIGNAL CORE weight
            artifact: ARTIFACT CORE weight
            alignment: ALIGNMENT CORE weight
            description: Optional description

        Returns:
            True if successful
        """
        config = WeightConfiguration(
            biosignal=biosignal,
            artifact=artifact,
            alignment=alignment,
            version=self._current_config.version if self._current_config else "3.2.0",
            description=description
        )
        return self.save_config(config)

    def register_model_file(
        self,
        filename: str,
        callback: Callable[[str], None]
    ):
        """
        Register a model weight file for monitoring.

        Args:
            filename: Weight filename (relative to weights_dir)
            callback: Function to call when file changes
        """
        filepath = self.weights_dir / filename
        if filepath.exists():
            self._file_hashes[filename] = self._compute_file_hash(filepath)
        self._callbacks[filename] = callback
        logger.info(f"Registered model file for monitoring: {filename}")

    def _compute_file_hash(self, filepath: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def check_for_updates(self) -> Dict[str, bool]:
        """
        Check all registered files for updates.

        Returns:
            Dict mapping filename to whether it was updated
        """
        updates = {}

        # Check config file
        if self.config_path.exists():
            current_hash = self._compute_file_hash(self.config_path)
            if current_hash != self._config_hash:
                logger.info("Weight config changed, reloading...")
                self._load_config()
                updates["config"] = True

        # Check model weight files
        for filename, callback in self._callbacks.items():
            filepath = self.weights_dir / filename

            if not filepath.exists():
                updates[filename] = False
                continue

            current_hash = self._compute_file_hash(filepath)
            previous_hash = self._file_hashes.get(filename)

            if current_hash != previous_hash:
                logger.info(f"Model weight file changed: {filename}")
                self._file_hashes[filename] = current_hash

                try:
                    callback(str(filepath))
                    updates[filename] = True
                except Exception as e:
                    logger.error(f"Error reloading weights {filename}: {e}")
                    updates[filename] = False
            else:
                updates[filename] = False

        return updates

    def start_watching(self, interval: float = 60.0):
        """
        Start background thread to watch for changes.

        Args:
            interval: Check interval in seconds
        """
        if self._watch_thread and self._watch_thread.is_alive():
            return

        self._stop_event.clear()
        self._watch_thread = threading.Thread(
            target=self._watch_loop,
            args=(interval,),
            daemon=True
        )
        self._watch_thread.start()
        logger.info(f"Started weight watcher (interval: {interval}s)")

    def stop_watching(self):
        """Stop the background watcher thread."""
        self._stop_event.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=5)
        logger.info("Stopped weight watcher")

    def _watch_loop(self, interval: float):
        """Background loop for watching weight files."""
        while not self._stop_event.is_set():
            try:
                self.check_for_updates()
            except Exception as e:
                logger.error(f"Error in weight watcher: {e}")

            self._stop_event.wait(interval)

    def list_available_weights(self) -> list:
        """List available weight files in weights directory."""
        if not self.weights_dir.exists():
            return []
        return [
            f.name for f in self.weights_dir.iterdir()
            if f.is_file() and f.suffix in ('.pth', '.pt', '.onnx', '.tflite')
        ]

    def get_weight_path(self, filename: str) -> Optional[Path]:
        """
        Get full path for a weight file.

        Returns None if the file does not exist.
        """
        filepath = self.weights_dir / filename
        return filepath if filepath.exists() else None

    def get_status(self) -> Dict[str, Any]:
        """Get current status of weight manager."""
        return {
            "config_loaded": self._current_config is not None,
            "config_path": str(self.config_path),
            "current_weights": self.get_fusion_weights(),
            "watched_files": list(self._callbacks.keys()),
            "watcher_active": self._watch_thread is not None and self._watch_thread.is_alive()
        }
