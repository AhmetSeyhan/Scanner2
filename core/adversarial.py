"""
Scanner Prime - Adversarial Robustness Module
Implements adversarial attacks for model hardening and evaluation.

Supported Attacks:
1. FGSM (Fast Gradient Sign Method) - Goodfellow et al. 2014
2. PGD (Projected Gradient Descent) - Madry et al. 2018
3. Gaussian Noise - Simple noise perturbation baseline
4. JPEG Compression - Simulates social media compression

Usage:
    from core.adversarial import AdversarialTrainer, fgsm_attack
    trainer = AdversarialTrainer(model, device)
    adv_images = trainer.generate_fgsm(images, labels, epsilon=0.03)

Copyright (c) 2026 Scanner Technologies. All rights reserved.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Args:
        model: Target model.
        images: Input images (B, C, H, W), requires_grad must be settable.
        labels: Ground truth labels (B,).
        epsilon: Perturbation magnitude (default 0.03 ~ 8/255).
        criterion: Loss function (default BCEWithLogitsLoss).

    Returns:
        Adversarial images tensor.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    images_adv = images.clone().detach().requires_grad_(True)
    outputs = model(images_adv)
    loss = criterion(outputs.squeeze(), labels.float())
    loss.backward()

    # FGSM: perturb in the direction of the gradient sign
    perturbation = epsilon * images_adv.grad.sign()
    adv_images = images + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.007,
    num_steps: int = 10,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack - stronger iterative version of FGSM.

    Args:
        model: Target model.
        images: Input images (B, C, H, W).
        labels: Ground truth labels (B,).
        epsilon: Max perturbation magnitude.
        alpha: Step size per iteration.
        num_steps: Number of PGD iterations.
        criterion: Loss function.

    Returns:
        Adversarial images tensor.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    adv_images = images.clone().detach()
    # Random start within epsilon ball
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(num_steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            # Project back to epsilon-ball around original
            delta = torch.clamp(adv_images - images, -epsilon, epsilon)
            adv_images = torch.clamp(images + delta, 0, 1)

    return adv_images.detach()


def jpeg_compress(
    images: torch.Tensor,
    quality: int = 70,
) -> torch.Tensor:
    """
    Simulate JPEG compression artifacts (social media pipeline).

    Args:
        images: Input images (B, C, H, W) in [0, 1] range.
        quality: JPEG quality (lower = more compression artifacts).

    Returns:
        Compressed images tensor.
    """
    B = images.shape[0]
    result = torch.zeros_like(images)

    for i in range(B):
        img = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", img, encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        result[i] = torch.from_numpy(dec.astype(np.float32) / 255.0).permute(2, 0, 1)

    return result.to(images.device)


class AdversarialTrainer:
    """
    Adversarial training wrapper for the DeepfakeDetector model.

    Augments each training batch with adversarial examples to improve
    model robustness against evasion attacks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        epsilon: float = 0.03,
        pgd_steps: int = 7,
        mix_ratio: float = 0.5,
    ):
        """
        Args:
            model: DeepfakeDetector model.
            device: Torch device.
            epsilon: Max perturbation magnitude.
            pgd_steps: PGD iteration count.
            mix_ratio: Fraction of batch to replace with adversarial examples.
        """
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.pgd_steps = pgd_steps
        self.mix_ratio = mix_ratio
        self.criterion = nn.BCEWithLogitsLoss()

    def adversarial_training_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        """
        Single adversarial training step.

        Mixes clean and adversarial examples for robust gradient updates.

        Returns:
            Tuple of (clean_loss, adversarial_loss).
        """
        self.model.train()
        batch_size = images.shape[0]
        n_adv = int(batch_size * self.mix_ratio)

        # Split batch
        clean_images = images[n_adv:]
        clean_labels = labels[n_adv:]
        adv_source = images[:n_adv]
        adv_labels = labels[:n_adv]

        # Generate adversarial examples
        self.model.eval()
        adv_images = pgd_attack(
            self.model, adv_source, adv_labels,
            epsilon=self.epsilon, num_steps=self.pgd_steps,
            criterion=self.criterion,
        )
        self.model.train()

        # Combine and forward
        combined_images = torch.cat([adv_images, clean_images], dim=0)
        combined_labels = torch.cat([adv_labels, clean_labels], dim=0)

        optimizer.zero_grad()
        outputs = self.model(combined_images)
        loss = self.criterion(outputs.squeeze(), combined_labels.float())
        loss.backward()
        optimizer.step()

        # Compute separate losses for logging
        with torch.no_grad():
            clean_out = self.model(clean_images)
            clean_loss = self.criterion(clean_out.squeeze(), clean_labels.float()).item()
            adv_out = self.model(adv_images)
            adv_loss = self.criterion(adv_out.squeeze(), adv_labels.float()).item()

        return clean_loss, adv_loss

    def evaluate_robustness(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Evaluate model robustness against multiple attack types.

        Returns:
            Dict with accuracy under each attack.
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            clean_out = self.model.predict_proba(images).squeeze()
            clean_pred = (clean_out > 0.5).long()
            results["clean_accuracy"] = (clean_pred == labels).float().mean().item()

        # FGSM attack
        for eps in [0.01, 0.03, 0.05]:
            adv = fgsm_attack(self.model, images, labels, epsilon=eps)
            with torch.no_grad():
                adv_out = self.model.predict_proba(adv).squeeze()
                adv_pred = (adv_out > 0.5).long()
                results[f"fgsm_eps{eps}_accuracy"] = (adv_pred == labels).float().mean().item()

        # PGD attack
        adv_pgd = pgd_attack(self.model, images, labels, epsilon=0.03, num_steps=10)
        with torch.no_grad():
            pgd_out = self.model.predict_proba(adv_pgd).squeeze()
            pgd_pred = (pgd_out > 0.5).long()
            results["pgd_accuracy"] = (pgd_pred == labels).float().mean().item()

        # JPEG compression
        for q in [50, 70, 90]:
            jpeg_imgs = jpeg_compress(images, quality=q)
            with torch.no_grad():
                jpeg_out = self.model.predict_proba(jpeg_imgs).squeeze()
                jpeg_pred = (jpeg_out > 0.5).long()
                results[f"jpeg_q{q}_accuracy"] = (jpeg_pred == labels).float().mean().item()

        return results
