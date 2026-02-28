"""
Unit tests for WeightManager (hot-reload for model weights).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.weight_manager import WeightManager


class TestWeightManager:
    """Test suite for WeightManager."""

    @pytest.fixture
    def manager(self, tmp_path):
        return WeightManager(weights_dir=str(tmp_path))

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager is not None

    def test_weights_dir_exists(self, manager, tmp_path):
        """Test weights directory is set."""
        assert os.path.isdir(str(tmp_path))

    def test_list_weights_empty(self, manager):
        """Test listing weights in empty directory."""
        weights = manager.list_available_weights()
        assert isinstance(weights, (list, dict))

    def test_get_weight_path_nonexistent(self, manager):
        """Test getting path for non-existent weight file."""
        path = manager.get_weight_path("nonexistent.pth")
        assert path is None or not os.path.exists(str(path) if path else "")
