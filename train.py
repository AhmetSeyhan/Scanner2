"""
Scanner - Model Training Script
Train custom deepfake detection model on FF++, Celeb-DF, or custom datasets.

Usage:
    python train.py --dataset faceforensics --data_dir /path/to/data --epochs 50
    python train.py --dataset celeb_df --data_dir /path/to/celeb_df --epochs 30
    python train.py --dataset custom --data_dir /path/to/custom --epochs 100

Dataset Structure (expected):
    data_dir/
    ├── train/
    │   ├── real/
    │   │   ├── video1/
    │   │   │   ├── frame_001.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── fake/
    │       └── ...
    └── val/
        ├── real/
        └── fake/
"""

import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm

from model import DeepfakeDetector


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection training."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        max_frames_per_video: int = 10
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing train/val splits
            split: 'train' or 'val'
            transform: Optional transforms to apply
            max_frames_per_video: Maximum frames to sample per video
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.max_frames = max_frames_per_video

        self.samples = []  # List of (image_path, label)

        # Load real samples (label=0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            self._load_samples(real_dir, label=0)

        # Load fake samples (label=1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            self._load_samples(fake_dir, label=1)

        print(f"[{split}] Loaded {len(self.samples)} samples")

    def _load_samples(self, directory: Path, label: int):
        """Load image samples from directory."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        for item in directory.iterdir():
            if item.is_dir():
                # Video directory with frames
                frames = [f for f in item.iterdir() if f.suffix.lower() in extensions]
                # Sample frames
                if len(frames) > self.max_frames:
                    indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
                    frames = [frames[i] for i in indices]
                for frame in frames:
                    self.samples.append((str(frame), label))
            elif item.suffix.lower() in extensions:
                # Direct image file
                self.samples.append((str(item), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(split: str = "train"):
    """Get data transforms for training/validation."""
    if split == "train":
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

    return total_loss / len(dataloader), correct / total


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train Scanner deepfake detection model")
    parser.add_argument("--dataset", type=str, default="custom",
                        choices=["faceforensics", "celeb_df", "dfdc", "custom"],
                        help="Dataset type")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./weights",
                        help="Directory to save trained weights")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use ImageNet pretrained backbone")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets
    print(f"\nLoading dataset from: {args.data_dir}")
    train_dataset = DeepfakeDataset(
        args.data_dir, split="train",
        transform=get_transforms("train")
    )
    val_dataset = DeepfakeDataset(
        args.data_dir, split="val",
        transform=get_transforms("val")
    )

    if len(train_dataset) == 0:
        print("ERROR: No training samples found!")
        print("Expected structure:")
        print("  data_dir/train/real/... and data_dir/train/fake/...")
        return

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Create model
    print("\nInitializing model...")
    model = DeepfakeDetector(
        pretrained=args.pretrained,
        dropout_rate=0.3,
        auto_load_deepfake_weights=False
    )
    model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = output_dir / f"efficientnet_b0_{args.dataset}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'dataset': args.dataset,
                'timestamp': datetime.now().isoformat()
            }, save_path)
            print(f"Saved best model to: {save_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'scheduler_state_dict': scheduler.state_dict()
            }, checkpoint_path)

    print("\n" + "=" * 60)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
