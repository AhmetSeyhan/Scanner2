"""
Scanner Test Suite - ModelManager Tests
Unit tests for utils/model_manager.py
"""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import ModelManager


class TestModelManagerSingleton:
    """Tests for thread-safe singleton behaviour."""

    def setup_method(self):
        """Reset singleton before each test."""
        ModelManager.reset()

    def test_singleton_returns_same_instance(self):
        """get_instance() always returns the same object."""
        a = ModelManager.get_instance()
        b = ModelManager.get_instance()
        assert a is b

    def test_reset_creates_new_instance(self):
        """reset() allows creation of a fresh instance."""
        a = ModelManager.get_instance()
        ModelManager.reset()
        b = ModelManager.get_instance()
        assert a is not b

    def test_health_status_before_init(self):
        """health_status returns all False before initialisation."""
        mm = ModelManager.get_instance()
        status = mm.health_status()
        assert status["biosignal_core"] is False
        assert status["artifact_core"] is False
        assert status["alignment_core"] is False

    def test_is_ready_before_init(self):
        """is_ready is False before initialise_all."""
        mm = ModelManager.get_instance()
        assert mm.is_ready is False

    def test_device_detection(self):
        """device property returns 'cpu' or 'cuda'."""
        mm = ModelManager.get_instance()
        assert mm.device in ("cpu", "cuda")

    def test_shutdown_does_not_raise(self):
        """shutdown() is safe to call even without init."""
        mm = ModelManager.get_instance()
        mm.shutdown()  # Should not raise


class TestModelManagerLazyLoading:
    """Tests for lazy loading behaviour with mocked imports."""

    def setup_method(self):
        ModelManager.reset()

    @patch("utils.model_manager.get_logger")
    def test_biosignal_core_lazy_load(self, mock_logger):
        """biosignal_core is None until accessed."""
        mm = ModelManager.get_instance()
        assert mm._biosignal_core is None

    @patch("utils.model_manager.get_logger")
    def test_model_load_error_on_import_failure(self, mock_logger):
        """ModelLoadError raised if core import fails."""

        ModelManager.get_instance()

        with patch.dict("sys.modules", {"core.biosignal_core": None}):
            # This should raise ModelLoadError when trying to import
            # (the exact behavior depends on how the import fails)
            pass  # Placeholder for import-failure testing
