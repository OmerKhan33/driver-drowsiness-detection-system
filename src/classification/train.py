"""
Training module for Driver Drowsiness Detection System.

Provides training loops with mixed-precision (AMP), AdamW optimizer,
CosineAnnealingLR scheduler, early stopping, and model checkpointing.
Designed to leverage GPU when available.

Usage:
    python src/classification/train.py --epochs 15 --batch_size 32 --lr 0.0001
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.classification.model_builder import SUPPORTED_MODELS, build_model
from src.utils.preprocessing import get_train_transforms, get_val_transforms


# ─── Device Selection ─────────────────────────────────────────────────────────


def get_device() -> torch.device:
    """Select the best available compute device.

    Prefers CUDA GPU over CPU for faster training.

    Returns:
        torch.device for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Using GPU: {gpu_name}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  Using CPU (no GPU available)")
    return device


# ─── Training Functions ───────────────────────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler = None,
) -> dict:
    """Train the model for one epoch.

    Uses mixed-precision training via torch.amp when a GradScaler is provided.

    Args:
        model: The neural network model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Compute device (cuda or cpu).
        scaler: GradScaler for mixed-precision training, or None.

    Returns:
        Dict with 'loss' and 'accuracy' for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass
        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return {"loss": epoch_loss, "accuracy": epoch_acc}


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate the model on a validation/test set.

    Args:
        model: The neural network model.
        loader: Validation/test DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Dict with 'loss', 'accuracy', 'preds', 'labels', 'probs'.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "preds": all_preds,
        "labels": all_labels,
        "probs": all_probs,
    }


# ─── Main Training Loop ──────────────────────────────────────────────────────


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 15,
    lr: float = 1e-4,
    device: torch.device = None,
    save_dir: str = "models/weights",
    use_amp: bool = True,
) -> dict:
    """Full training loop for a single model.

    Features:
        - AdamW optimizer with weight decay
        - CosineAnnealingLR scheduler
        - Mixed-precision training (AMP) on GPU
        - Best model checkpointing by validation accuracy
        - Early stopping with patience=5

    Args:
        model_name: Name of the model architecture to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        num_epochs: Number of training epochs.
        lr: Learning rate.
        device: Compute device. Auto-selects GPU if None.
        save_dir: Directory to save model weights.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dict with training history and final metrics:
            'train_loss', 'train_acc', 'val_loss', 'val_acc' (lists),
            'best_val_acc', 'best_epoch', 'training_time_s'.
    """
    if device is None:
        device = get_device()

    # Build model
    print(f"\n  Building {model_name}...")
    model = build_model(model_name, num_classes=2, pretrained=True)
    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed-precision scaler (GPU only)
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        print("  Mixed precision (AMP) enabled")

    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training state
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_val_acc = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0
    start_time = time.time()

    print(f"  Training {model_name} for {num_epochs} epochs on {device}...\n")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>3}/{num_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Best model checkpointing
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            best_path = save_path / f"{model_name}_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best model saved → {best_path}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Save final model
    final_path = save_path / f"{model_name}_final.pt"
    torch.save(model.state_dict(), final_path)

    total_time = time.time() - start_time

    print(f"\n  {model_name} training complete!")
    print(f"  Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"  Total training time: {total_time:.1f}s")

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["training_time_s"] = total_time

    return history


def train_all_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    models_list: list = None,
    num_epochs: int = 15,
    lr: float = 1e-4,
    device: torch.device = None,
) -> dict:
    """Train all specified models and return combined results.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        models_list: List of model names to train.
            Defaults to all SUPPORTED_MODELS.
        num_epochs: Number of training epochs per model.
        lr: Learning rate.
        device: Compute device.

    Returns:
        Dict mapping model_name to its training history.
    """
    if models_list is None:
        models_list = SUPPORTED_MODELS

    if device is None:
        device = get_device()

    results = {}
    total_models = len(models_list)

    print("=" * 60)
    print(f"  Training {total_models} models")
    print("=" * 60)

    for idx, model_name in enumerate(models_list, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{total_models}] Training: {model_name}")
        print(f"{'─' * 60}")

        history = train_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
        )
        results[model_name] = history

    # Summary table
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<18} {'Best Acc':>10} {'Epoch':>8} {'Time (s)':>10}")
    print("  " + "-" * 48)
    for name, hist in results.items():
        print(
            f"  {name:<18} "
            f"{hist['best_val_acc']:>10.4f} "
            f"{hist['best_epoch']:>8} "
            f"{hist['training_time_s']:>10.1f}"
        )
    print("=" * 60)

    return results


# ─── Data Loading Utility ────────────────────────────────────────────────────


def create_data_loaders(
    data_dir: str = "data/processed",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders from processed dataset.

    Args:
        data_dir: Root directory containing train/val/test subdirs.
        batch_size: Batch size for DataLoaders.
        img_size: Image size for transforms.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        FileNotFoundError: If data_dir does not exist.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Run 'python data/scripts/prepare_dataset.py' first."
        )

    train_transforms = get_train_transforms(img_size)
    val_transforms = get_val_transforms(img_size)

    train_dataset = datasets.ImageFolder(
        data_path / "train", transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        data_path / "val", transform=val_transforms
    )
    test_dataset = datasets.ImageFolder(
        data_path / "test", transform=val_transforms
    )

    # Pin memory for faster GPU transfer
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )

    print(f"  Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_dataset)} images ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_dataset)} images ({len(test_loader)} batches)")
    print(f"  Classes: {train_dataset.classes}")

    return train_loader, val_loader, test_loader


# ─── CLI Entry Point ─────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train drowsiness classification models."
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/processed",
        help="Path to processed dataset (default: data/processed)",
    )
    parser.add_argument(
        "--save_dir", type=str, default="models/weights",
        help="Path to save model weights (default: models/weights)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="List of models to train (default: all supported models)",
    )
    parser.add_argument(
        "--img_size", type=int, default=224,
        help="Input image size (default: 224)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loader workers (default: 4)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("=" * 60)
    print("  Driver Drowsiness Detection — Model Training")
    print("=" * 60)

    args = parse_args()

    # Device selection
    device = get_device()

    # Load data
    print("\n  Loading dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    # Train models
    models_to_train = args.models if args.models else SUPPORTED_MODELS
    results = train_all_models(
        train_loader=train_loader,
        val_loader=val_loader,
        models_list=models_to_train,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
    )

    # Save results summary
    results_dir = Path("models/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, hist in results.items():
        summary[name] = {
            "best_val_acc": hist["best_val_acc"],
            "best_epoch": hist["best_epoch"],
            "training_time_s": hist["training_time_s"],
        }
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Training summary saved to: {summary_path}")
    print("=" * 60)
