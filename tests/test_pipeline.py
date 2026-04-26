"""
Comprehensive pytest test suite for Driver Drowsiness Detection System.

Tests preprocessing, EAR/MAR calculations, model architectures,
and the alert system.
"""

import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest
import torch
from PIL import Image

from src.alert.alert_system import DrowsinessAlertSystem
from src.classification.model_builder import build_model
from src.utils.drowsiness_utils import (
    compute_avg_ear,
    compute_drowsiness_score,
    compute_ear,
    drowsiness_level,
)
from src.utils.preprocessing import (
    bgr_to_rgb,
    get_train_transforms,
    get_val_transforms,
    is_blurry,
    normalize_image,
    preprocess_frame,
    resize_image,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TestPreprocessing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPreprocessing:
    """Tests for image preprocessing functions."""

    def test_image_resize(self):
        """Resized image should have shape (224, 224, 3)."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        resized = resize_image(img, (224, 224))
        assert resized.shape == (224, 224, 3)

    def test_resize_raises_on_none(self):
        """resize_image should raise ValueError when image is None."""
        with pytest.raises(ValueError):
            resize_image(None)

    def test_normalize_range(self):
        """Normalized image should have values in [0, 1]."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        normed = normalize_image(img)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0
        assert normed.dtype == np.float32

    def test_bgr_to_rgb_channels(self):
        """BGR to RGB should swap first and last channels."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        rgb = bgr_to_rgb(img)
        assert np.array_equal(rgb[:, :, 0], img[:, :, 2])
        assert np.array_equal(rgb[:, :, 2], img[:, :, 0])

    def test_train_transforms_output_shape(self):
        """Train transforms should produce (3, 224, 224) tensor."""
        pil_img = Image.fromarray(
            np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        )
        transform = get_train_transforms(224)
        tensor = transform(pil_img)
        assert tensor.shape == (3, 224, 224)

    def test_val_transforms_output_shape(self):
        """Val transforms should produce (3, 224, 224) tensor."""
        pil_img = Image.fromarray(
            np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        )
        transform = get_val_transforms(224)
        tensor = transform(pil_img)
        assert tensor.shape == (3, 224, 224)

    def test_preprocess_frame_output_shape(self):
        """preprocess_frame should produce (1, 3, 224, 224) tensor."""
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_frame(frame, 224)
        assert tensor.shape == (1, 3, 224, 224)

    def test_is_blurry_on_uniform_image(self):
        """A flat uniform image should be detected as blurry."""
        uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
        assert is_blurry(uniform) is True


# ═══════════════════════════════════════════════════════════════════════════════
# TestEAR
# ═══════════════════════════════════════════════════════════════════════════════


class TestEAR:
    """Tests for EAR/MAR and drowsiness scoring."""

    def _open_eye(self):
        return np.array([
            [10, 50], [20, 60], [30, 60], [40, 50], [30, 40], [20, 40]
        ], dtype=np.float32)

    def _closed_eye(self):
        return np.array([
            [10, 50], [20, 52], [30, 52], [40, 50], [30, 48], [20, 48]
        ], dtype=np.float32)

    def test_ear_open_eye(self):
        """Open eye EAR should be > 0.2."""
        ear = compute_ear(self._open_eye())
        assert ear > 0.2

    def test_ear_closed_eye(self):
        """Closed eye EAR should be < 0.15."""
        ear = compute_ear(self._closed_eye())
        assert ear < 0.15

    def test_ear_returns_float(self):
        """compute_ear should return a float."""
        ear = compute_ear(self._open_eye())
        assert isinstance(ear, float)

    def test_ear_wrong_shape_raises(self):
        """compute_ear should raise ValueError for shape (4, 2)."""
        with pytest.raises(ValueError):
            compute_ear(np.zeros((4, 2)))

    def test_avg_ear(self):
        """Average EAR should be between open and closed values."""
        avg = compute_avg_ear(self._open_eye(), self._closed_eye())
        ear_open = compute_ear(self._open_eye())
        ear_closed = compute_ear(self._closed_eye())
        assert ear_closed < avg < ear_open

    def test_drowsiness_score_range(self):
        """Drowsiness score should be in [0, 1]."""
        score = compute_drowsiness_score(
            ear=0.25, mar=0.5, cnn_confidence=0.6
        )
        assert 0.0 <= score <= 1.0

    def test_drowsiness_level_alert(self):
        """Score 0.2 should map to 'ALERT'."""
        assert drowsiness_level(0.2) == "ALERT"

    def test_drowsiness_level_drowsy(self):
        """Score 0.8 should map to 'DROWSY'."""
        assert drowsiness_level(0.8) == "DROWSY"


# ═══════════════════════════════════════════════════════════════════════════════
# TestModelArchitecture
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelArchitecture:
    """Tests for CNN model architectures."""

    def _forward_pass(self, model_name):
        model = build_model(model_name, num_classes=2, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        return out

    def test_custom_cnn_output_shape(self):
        """CustomCNN should output (2, 2) for batch=2."""
        out = self._forward_pass("CustomCNN")
        assert out.shape == (2, 2)

    def test_resnet18_output_shape(self):
        """ResNet18 should output (2, 2) for batch=2."""
        out = self._forward_pass("ResNet18")
        assert out.shape == (2, 2)

    def test_resnet50_output_shape(self):
        """ResNet50 should output (2, 2) for batch=2."""
        out = self._forward_pass("ResNet50")
        assert out.shape == (2, 2)

    def test_vgg16_output_shape(self):
        """VGG16 should output (2, 2) for batch=2."""
        out = self._forward_pass("VGG16")
        assert out.shape == (2, 2)

    def test_efficientnet_b0_output_shape(self):
        """EfficientNet-B0 should output (2, 2) for batch=2."""
        out = self._forward_pass("EfficientNet-B0")
        assert out.shape == (2, 2)

    def test_mobilenetv2_output_shape(self):
        """MobileNetV2 should output (2, 2) for batch=2."""
        out = self._forward_pass("MobileNetV2")
        assert out.shape == (2, 2)

    def test_all_models_no_nan(self):
        """No model output should contain NaN values."""
        for name in ["CustomCNN", "ResNet18", "ResNet50", "VGG16", "EfficientNet-B0", "MobileNetV2"]:
            out = self._forward_pass(name)
            assert not torch.isnan(out).any(), f"{name} output contains NaN"

    def test_model_on_cpu(self):
        """Model should load on CPU without error."""
        model = build_model("CustomCNN", num_classes=2, pretrained=False)
        model.to(torch.device("cpu"))
        assert next(model.parameters()).device.type == "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# TestAlertSystem
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlertSystem:
    """Tests for the drowsiness alert system."""

    def _make_system(self, consec=3):
        return DrowsinessAlertSystem(
            ear_threshold=0.25,
            mar_threshold=0.60,
            consec_frames=consec,
            cnn_threshold=0.65,
            alert_sound=False,
        )

    def test_drowsy_triggers_after_consecutive_frames(self):
        """System should flag drowsy after enough consecutive drowsy frames."""
        system = self._make_system(consec=3)
        for _ in range(5):
            system.update(ear=0.15, mar=0.3, cnn_confidence=0.8)
        assert system.is_drowsy()

    def test_not_drowsy_with_high_ear(self):
        """High EAR (eyes open) should not trigger drowsiness."""
        system = self._make_system(consec=3)
        system.update(ear=0.35, mar=0.3, cnn_confidence=0.1)
        assert not system.is_drowsy()

    def test_reset_clears_counter(self):
        """reset() should set consecutive_frames back to 0."""
        system = self._make_system(consec=3)
        for _ in range(5):
            system.update(ear=0.15, mar=0.3, cnn_confidence=0.8)
        system.reset()
        assert system.consecutive_frames == 0
        assert not system.is_drowsy()

    def test_yawn_detection_with_high_mar(self):
        """High MAR should trigger yawning detection."""
        system = self._make_system()
        status = system.update(ear=0.30, mar=0.75, cnn_confidence=0.2)
        assert status["is_yawning"] is True

    def test_update_returns_dict(self):
        """update() should return dict with required keys."""
        system = self._make_system()
        status = system.update(ear=0.30, mar=0.40, cnn_confidence=0.3)
        expected_keys = {
            "is_drowsy", "is_yawning", "alert_level",
            "ear", "mar", "consecutive_frames",
        }
        assert set(status.keys()) == expected_keys
