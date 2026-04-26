"""
Prediction module for Driver Drowsiness Detection System.

Provides inference functionality for single images, batches,
and live video frame face crops.
"""

import numpy as np
import torch
from PIL import Image

from src.classification.model_builder import build_model
from src.utils.preprocessing import get_inference_transforms, preprocess_face_crop


# ─── Class Labels ─────────────────────────────────────────────────────────────

CLASS_NAMES = ["alert", "drowsy"]


# ─── Predictor ────────────────────────────────────────────────────────────────


class DrowsinessPredictor:
    """Inference predictor for drowsiness classification.

    Loads a trained model and provides methods to predict on single images,
    batches, and face crops from video frames.

    Args:
        model_name: Name of the model architecture.
        weights_path: Path to the saved .pt weights file.
        device: Compute device. Auto-selects GPU if None.
    """

    def __init__(
        self,
        model_name: str,
        weights_path: str,
        device: torch.device = None,
    ):
        """Initialize the predictor with a trained model.

        Args:
            model_name: Name of the model architecture.
            weights_path: Path to saved weights (.pt file).
            device: Compute device. Auto-selects GPU if None.

        Raises:
            FileNotFoundError: If weights_path does not exist.
        """
        from pathlib import Path

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model_name = model_name
        self.transform = get_inference_transforms(img_size=224)

        # Build model architecture
        self.model = build_model(
            model_name,
            num_classes=2,
            freeze_backbone=False,
            pretrained=False,
        )

        # Load weights
        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}"
            )

        state_dict = torch.load(
            weights_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _prepare_input(self, image) -> torch.Tensor:
        """Convert various image formats to a model-ready tensor.

        Accepts PIL Image, numpy BGR array, or torch tensor.

        Args:
            image: Input image in PIL, numpy BGR (H, W, 3), or
                torch tensor (C, H, W) format.

        Returns:
            Tensor of shape (1, 3, 224, 224) on the correct device.

        Raises:
            ValueError: If image format is not recognized.
        """
        if isinstance(image, Image.Image):
            tensor = self.transform(image)
        elif isinstance(image, np.ndarray):
            # Assume BGR from OpenCV — convert to RGB PIL
            import cv2

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = self.transform(pil_img)
        elif isinstance(image, torch.Tensor):
            if image.dim() == 4:
                # Already batched
                return image.to(self.device)
            tensor = image
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                f"Expected PIL Image, numpy array, or torch Tensor."
            )

        return tensor.unsqueeze(0).to(self.device)

    def predict(self, image) -> dict:
        """Predict drowsiness for a single image.

        Args:
            image: Input image as PIL Image, numpy BGR array,
                or torch tensor.

        Returns:
            Dict with:
                - 'class': str ('alert' or 'drowsy')
                - 'confidence': float (confidence of predicted class)
                - 'probabilities': dict {'alert': float, 'drowsy': float}
        """
        tensor = self._prepare_input(image)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

        return {
            "class": pred_class,
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(len(CLASS_NAMES))
            },
        }

    def predict_batch(self, images: list) -> list[dict]:
        """Predict drowsiness for a batch of images.

        Args:
            images: List of images (PIL, numpy BGR, or torch tensors).

        Returns:
            List of prediction dicts, one per image.
        """
        results = []
        # Process individually to handle mixed input types
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results

    def predict_from_frame(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple,
    ) -> dict:
        """Predict drowsiness from a face crop within a video frame.

        Crops the face region using the bounding box, applies padding,
        and runs prediction.

        Args:
            frame_bgr: Full video frame in BGR format (H, W, 3).
            bbox: Face bounding box as (x1, y1, x2, y2).

        Returns:
            Dict with 'class', 'confidence', and 'probabilities'.

        Raises:
            ValueError: If frame is None or bbox is invalid.
        """
        if frame_bgr is None:
            raise ValueError(
                "Input frame is None. Expected a BGR numpy array."
            )
        if len(bbox) != 4:
            raise ValueError(
                f"Expected bbox with 4 values (x1, y1, x2, y2), "
                f"got {len(bbox)} values."
            )

        # Use preprocessing utility to crop with padding
        tensor = preprocess_face_crop(
            frame_bgr, bbox, img_size=224, padding=0.15
        )
        tensor = tensor.to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = self.model(tensor)
            else:
                logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = probs[pred_idx].item()

        return {
            "class": pred_class,
            "confidence": confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(len(CLASS_NAMES))
            },
        }


# ─── Smoke Test ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("  Predict Module — Smoke Test")
    print("=" * 50)

    # Test that the class can be instantiated concept
    # (without weights file, just verify the code structure)
    device = torch.device("cpu")

    # Build a model and save temp weights to test loading
    from pathlib import Path
    import tempfile

    from src.classification.model_builder import build_model as bm

    model = bm("CustomCNN", num_classes=2, pretrained=False)
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_weights = tmp_dir / "test_weights.pt"
    torch.save(model.state_dict(), tmp_weights)

    # Create predictor
    predictor = DrowsinessPredictor(
        model_name="CustomCNN",
        weights_path=str(tmp_weights),
        device=device,
    )
    print("  ✓ DrowsinessPredictor created")

    # Test predict with numpy array
    dummy_bgr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    result = predictor.predict(dummy_bgr)
    assert "class" in result, "Missing 'class' key"
    assert "confidence" in result, "Missing 'confidence' key"
    assert "probabilities" in result, "Missing 'probabilities' key"
    assert result["class"] in CLASS_NAMES, f"Unknown class: {result['class']}"
    assert 0.0 <= result["confidence"] <= 1.0, "Confidence out of range"
    print(f"  ✓ predict (numpy): {result['class']} ({result['confidence']:.3f})")

    # Test predict with PIL Image
    pil_img = Image.fromarray(dummy_bgr[:, :, ::-1])  # BGR to RGB
    result_pil = predictor.predict(pil_img)
    assert result_pil["class"] in CLASS_NAMES
    print(f"  ✓ predict (PIL): {result_pil['class']} ({result_pil['confidence']:.3f})")

    # Test predict_batch
    batch_results = predictor.predict_batch([dummy_bgr, dummy_bgr])
    assert len(batch_results) == 2, "Batch should return 2 results"
    print(f"  ✓ predict_batch: {len(batch_results)} results")

    # Test predict_from_frame
    big_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    frame_result = predictor.predict_from_frame(big_frame, (100, 100, 300, 300))
    assert frame_result["class"] in CLASS_NAMES
    print(f"  ✓ predict_from_frame: {frame_result['class']} ({frame_result['confidence']:.3f})")

    # Cleanup
    tmp_weights.unlink()
    tmp_dir.rmdir()

    print("\n  All predict tests passed! ✓")
    print("=" * 50)
