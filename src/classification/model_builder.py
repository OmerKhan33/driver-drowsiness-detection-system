"""
Model Builder module for Driver Drowsiness Detection System.

Provides factory functions to build, load, and compare CNN classifier
architectures for drowsiness classification (ALERT vs DROWSY).

Supported models:
    - CustomCNN (from scratch)
    - ResNet18 (pretrained)
    - ResNet50 (pretrained)
    - VGG16 (pretrained)
    - EfficientNet-B0 (pretrained)
    - MobileNetV2 (pretrained)
"""

import torch
import torch.nn as nn
import torchvision.models as models


# ─── Custom CNN Architecture ─────────────────────────────────────────────────


class CustomCNN(nn.Module):
    """Custom CNN with 3 convolutional blocks + GAP + classifier.

    Architecture:
        Conv(3->32) -> Conv(32->32) -> MaxPool ->
        Conv(32->64) -> Conv(64->64) -> MaxPool ->
        Conv(64->128) -> Conv(128->128) -> MaxPool ->
        GAP -> FC(128->256) -> FC(256->num_classes)

    Each conv has BatchNorm + ReLU + Dropout2d(0.25).

    Args:
        num_classes: Number of output classes.
        dropout: Dropout rate for the classifier head.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        """Initialize the CustomCNN.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout rate for the classifier head.
        """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.25),
            nn.MaxPool2d(2, 2),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output logits of shape (B, num_classes).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ─── Model Builder Factory ───────────────────────────────────────────────────


SUPPORTED_MODELS = [
    "CustomCNN",
    "ResNet18",
    "ResNet50",
    "VGG16",
    "EfficientNet-B0",
    "MobileNetV2",
]


def _build_custom_cnn(num_classes: int, dropout: float) -> nn.Module:
    """Build a CustomCNN model from scratch.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout rate for the classifier.

    Returns:
        CustomCNN model instance.
    """
    return CustomCNN(num_classes=num_classes, dropout=dropout)


def _build_resnet18(
    num_classes: int,
    freeze_backbone: bool,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    """Build a ResNet18 model with custom head.

    Unfreezes layer4 + fc only when freeze_backbone is True.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze early layers.
        dropout: Dropout rate for the classifier.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Modified ResNet18 model.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze layer4 and fc
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_resnet50(
    num_classes: int,
    freeze_backbone: bool,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    """Build a ResNet50 model with custom head.

    Unfreezes layer3 + layer4 + fc when freeze_backbone is True.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze early layers.
        dropout: Dropout rate for the classifier.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Modified ResNet50 model.
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, num_classes),
    )
    return model


def _build_vgg16(
    num_classes: int,
    freeze_backbone: bool,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    """Build a VGG16 model with custom classifier head.

    Freezes all feature layers when freeze_backbone is True.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze feature layers.
        dropout: Dropout rate for the classifier.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Modified VGG16 model.
    """
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )
    return model


def _build_efficientnet_b0(
    num_classes: int,
    freeze_backbone: bool,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    """Build an EfficientNet-B0 model with custom head.

    Unfreezes the last 2 feature blocks when freeze_backbone is True.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze early layers.
        dropout: Dropout rate for the classifier.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Modified EfficientNet-B0 model.
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last 2 feature blocks
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_mobilenetv2(
    num_classes: int,
    freeze_backbone: bool,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    """Build a MobileNetV2 model with custom head.

    Unfreezes the last 3 feature blocks when freeze_backbone is True.

    Args:
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze early layers.
        dropout: Dropout rate for the classifier.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        Modified MobileNetV2 model.
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze last 3 feature blocks
        feature_blocks = list(model.features.children())
        for block in feature_blocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─── Public API ───────────────────────────────────────────────────────────────


def build_model(
    model_name: str,
    num_classes: int = 2,
    freeze_backbone: bool = True,
    dropout: float = 0.4,
    pretrained: bool = True,
) -> nn.Module:
    """Build a classification model by name.

    Args:
        model_name: One of 'CustomCNN', 'ResNet18', 'ResNet50',
            'VGG16', 'EfficientNet-B0', 'MobileNetV2'.
        num_classes: Number of output classes.
        freeze_backbone: Whether to freeze pretrained backbone layers.
        dropout: Dropout rate for classifier head.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        A PyTorch nn.Module ready for training.

    Raises:
        ValueError: If model_name is not supported.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from: {SUPPORTED_MODELS}"
        )

    builders = {
        "CustomCNN": lambda: _build_custom_cnn(num_classes, dropout),
        "ResNet18": lambda: _build_resnet18(
            num_classes, freeze_backbone, dropout, pretrained
        ),
        "ResNet50": lambda: _build_resnet50(
            num_classes, freeze_backbone, dropout, pretrained
        ),
        "VGG16": lambda: _build_vgg16(
            num_classes, freeze_backbone, dropout, pretrained
        ),
        "EfficientNet-B0": lambda: _build_efficientnet_b0(
            num_classes, freeze_backbone, dropout, pretrained
        ),
        "MobileNetV2": lambda: _build_mobilenetv2(
            num_classes, freeze_backbone, dropout, pretrained
        ),
    }

    return builders[model_name]()


def load_model(
    model_name: str,
    weights_path: str,
    num_classes: int = 2,
    device: torch.device = None,
) -> nn.Module:
    """Load a model with saved weights.

    Args:
        model_name: Name of the model architecture.
        weights_path: Path to the .pt weights file.
        num_classes: Number of output classes.
        device: Device to load the model onto.

    Returns:
        Model with loaded weights in eval mode.

    Raises:
        FileNotFoundError: If weights_path does not exist.
    """
    from pathlib import Path

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}"
        )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model = build_model(
        model_name,
        num_classes=num_classes,
        freeze_backbone=False,
        pretrained=False,
    )
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def model_summary(model_name: str) -> dict:
    """Get parameter counts for a model architecture.

    Args:
        model_name: Name of the model architecture.

    Returns:
        Dict with 'model_name', 'total_params', 'trainable_params',
        'frozen_params', 'size_mb'.
    """
    model = build_model(model_name, pretrained=False)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    size_mb = total * 4 / (1024 * 1024)  # float32 = 4 bytes

    return {
        "model_name": model_name,
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "size_mb": round(size_mb, 2),
    }


def compare_model_sizes() -> None:
    """Print a formatted comparison table of all supported models.

    Displays total parameters, trainable parameters, frozen parameters,
    and estimated model size in MB.
    """
    print("\n" + "=" * 75)
    print(f"  {'Model':<18} {'Total Params':>14} {'Trainable':>14} {'Frozen':>14} {'Size MB':>10}")
    print("  " + "-" * 71)

    for name in SUPPORTED_MODELS:
        info = model_summary(name)
        print(
            f"  {info['model_name']:<18} "
            f"{info['total_params']:>14,} "
            f"{info['trainable_params']:>14,} "
            f"{info['frozen_params']:>14,} "
            f"{info['size_mb']:>10.2f}"
        )

    print("=" * 75 + "\n")


# ─── Smoke Test ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("  Model Builder Module — Smoke Test")
    print("=" * 50)

    device = torch.device("cpu")
    dummy_input = torch.randn(2, 3, 224, 224, device=device)

    for name in SUPPORTED_MODELS:
        model = build_model(name, num_classes=2, pretrained=False)
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (2, 2), (
            f"{name}: expected (2, 2), got {out.shape}"
        )
        assert not torch.isnan(out).any(), f"{name}: output contains NaN"
        info = model_summary(name)
        print(
            f"  ✓ {name:<18} "
            f"output={tuple(out.shape)}  "
            f"params={info['total_params']:,}"
        )

    print("\n  Model size comparison:")
    compare_model_sizes()

    print("  All model builder tests passed! ✓")
    print("=" * 50)
