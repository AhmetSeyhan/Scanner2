"""Tests for adversarial robustness module."""

import pytest
import torch
import torch.nn as nn
import numpy as np


@pytest.fixture
def simple_model():
    """A minimal binary classifier for testing attacks."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    model.predict_proba = lambda x: torch.sigmoid(model(x))
    return model


@pytest.fixture
def sample_batch():
    torch.manual_seed(42)
    images = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    return images, labels


class TestFGSM:
    def test_fgsm_produces_perturbation(self, simple_model, sample_batch):
        from core.adversarial import fgsm_attack
        images, labels = sample_batch
        adv = fgsm_attack(simple_model, images, labels, epsilon=0.1)
        assert adv.shape == images.shape
        assert not torch.allclose(adv, images), "FGSM should perturb images"

    def test_fgsm_respects_epsilon_bound(self, simple_model, sample_batch):
        from core.adversarial import fgsm_attack
        images, labels = sample_batch
        eps = 0.03
        adv = fgsm_attack(simple_model, images, labels, epsilon=eps)
        diff = (adv - images).abs().max()
        assert diff <= eps + 1e-6

    def test_fgsm_output_in_valid_range(self, simple_model, sample_batch):
        from core.adversarial import fgsm_attack
        images, labels = sample_batch
        adv = fgsm_attack(simple_model, images, labels, epsilon=0.1)
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0


class TestPGD:
    def test_pgd_produces_perturbation(self, simple_model, sample_batch):
        from core.adversarial import pgd_attack
        images, labels = sample_batch
        adv = pgd_attack(simple_model, images, labels, epsilon=0.03, num_steps=3)
        assert adv.shape == images.shape

    def test_pgd_respects_epsilon_bound(self, simple_model, sample_batch):
        from core.adversarial import pgd_attack
        images, labels = sample_batch
        eps = 0.03
        adv = pgd_attack(simple_model, images, labels, epsilon=eps, num_steps=5)
        diff = (adv - images).abs().max()
        assert diff <= eps + 1e-5

    def test_pgd_output_in_valid_range(self, simple_model, sample_batch):
        from core.adversarial import pgd_attack
        images, labels = sample_batch
        adv = pgd_attack(simple_model, images, labels, epsilon=0.05, num_steps=3)
        assert adv.min() >= 0.0
        assert adv.max() <= 1.0


class TestJPEGCompress:
    def test_jpeg_compress_preserves_shape(self, sample_batch):
        from core.adversarial import jpeg_compress
        images, _ = sample_batch
        compressed = jpeg_compress(images, quality=50)
        assert compressed.shape == images.shape

    def test_jpeg_compress_changes_images(self, sample_batch):
        from core.adversarial import jpeg_compress
        images, _ = sample_batch
        compressed = jpeg_compress(images, quality=50)
        # JPEG should change pixel values
        assert not torch.allclose(compressed, images, atol=0.01)

    def test_jpeg_high_quality_minimal_change(self, sample_batch):
        from core.adversarial import jpeg_compress
        images, _ = sample_batch
        # Compare quality=100 vs quality=10 — high quality should produce less distortion
        compressed_high = jpeg_compress(images, quality=100)
        compressed_low = jpeg_compress(images, quality=10)
        diff_high = (compressed_high - images).abs().mean()
        diff_low = (compressed_low - images).abs().mean()
        # High quality JPEG should distort less than low quality
        assert diff_high < diff_low
