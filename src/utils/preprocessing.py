"""
Image Preprocessing Module for Driver Drowsiness Detection.

Provides image transformation utilities, data augmentation pipelines,
and frame preprocessing functions for the detection pipeline.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ─── ImageNet Normalization Constants ─────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ─── Basic Image Operations ──────────────────────────────────────────────────


def resize_image(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resize an image to the specified dimensions.

    Args:
        image: Input image as numpy array (H, W, C) or (H, W).
        size: Target size as (width, height).

    Returns:
        Resized image as numpy array.

    Raises:
        ValueError: If image is None.
    """
    if image is None:
        raise ValueError("Input image is None. Expected a numpy array with shape (H, W, C) or (H, W).")
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range.

    Args:
        image: Input image as numpy array with uint8 values [0, 255].

    Returns:
        Normalized image as float32 array in [0, 1].

    Raises:
        ValueError: If image is None.
    """
    if image is None:
        raise ValueError("Input image is None. Cannot normalize a None image.")
    return image.astype(np.float32) / 255.0


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR (OpenCV format) to RGB.

    Args:
        image: Input BGR image as numpy array (H, W, 3).

    Returns:
        RGB image as numpy array.

    Raises:
        ValueError: If image is None or doesn't have 3 channels.
    """
    if image is None:
        raise ValueError("Input image is None. Expected a BGR image with shape (H, W, 3).")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with 3 channels (H, W, 3), got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert image from RGB to BGR (OpenCV format).

    Args:
        image: Input RGB image as numpy array (H, W, 3).

    Returns:
        BGR image as numpy array.

    Raises:
        ValueError: If image is None or doesn't have 3 channels.
    """
    if image is None:
        raise ValueError("Input image is None. Expected an RGB image with shape (H, W, 3).")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with 3 channels (H, W, 3), got shape {image.shape}")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert image from BGR to grayscale.

    Args:
        image: Input BGR image as numpy array (H, W, 3).

    Returns:
        Grayscale image as numpy array (H, W).

    Raises:
        ValueError: If image is None.
    """
    if image is None:
        raise ValueError("Input image is None. Expected a BGR image with shape (H, W, 3).")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ─── Torchvision Transform Pipelines ─────────────────────────────────────────


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Get training data augmentation pipeline.

    Includes random horizontal flip, rotation, color jitter, random grayscale,
    and ImageNet normalization.

    Args:
        img_size: Target image size (square).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Get validation/evaluation transform pipeline.

    Only resize, convert to tensor, and normalize — no augmentation.

    Args:
        img_size: Target image size (square).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(img_size: int = 224) -> transforms.Compose:
    """Get inference transform pipeline (same as validation).

    Args:
        img_size: Target image size (square).

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Tensor Utilities ────────────────────────────────────────────────────────


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization for visualization.

    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
        Denormalized tensor with pixel values in [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device)

    if tensor.dim() == 4:
        # Batch dimension: (B, C, H, W)
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
    elif tensor.dim() == 3:
        # Single image: (C, H, W)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
    else:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

    denorm = tensor * std + mean
    return torch.clamp(denorm, 0.0, 1.0)


# ─── Frame Preprocessing ─────────────────────────────────────────────────────


def preprocess_frame(frame: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Preprocess a BGR video frame to a model-ready tensor.

    Converts BGR numpy array → RGB PIL Image → normalized tensor → batched.

    Args:
        frame: Input BGR frame from OpenCV (H, W, 3).
        img_size: Target image size.

    Returns:
        Tensor of shape (1, 3, img_size, img_size).

    Raises:
        ValueError: If frame is None.
    """
    if frame is None:
        raise ValueError("Input frame is None. Expected a BGR numpy array with shape (H, W, 3).")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    transform = get_inference_transforms(img_size)
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)


def preprocess_face_crop(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    img_size: int = 224,
    padding: float = 0.15,
) -> torch.Tensor:
    """Crop a face region from a frame with padding and preprocess it.

    Args:
        frame: Input BGR frame (H, W, 3).
        bbox: Bounding box as (x1, y1, x2, y2).
        img_size: Target image size for the model.
        padding: Fractional padding around the bounding box (0.15 = 15%).

    Returns:
        Tensor of shape (1, 3, img_size, img_size).

    Raises:
        ValueError: If frame is None or bbox is invalid.
    """
    if frame is None:
        raise ValueError("Input frame is None. Expected a BGR numpy array with shape (H, W, 3).")
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values (x1, y1, x2, y2), got {len(bbox)} values.")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Add padding
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    # Crop and preprocess
    face_crop = frame[y1:y2, x1:x2]
    return preprocess_frame(face_crop, img_size)


# ─── Image Quality Checks ────────────────────────────────────────────────────


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """Check if an image is blurry using Laplacian variance.

    A lower variance indicates more blur. Default threshold of 100.0
    works well for 224x224 face crops.

    Args:
        image: Input image as numpy array (can be BGR or grayscale).
        threshold: Variance threshold below which image is considered blurry.

    Returns:
        True if the image is blurry, False otherwise.

    Raises:
        ValueError: If image is None.
    """
    if image is None:
        raise ValueError("Input image is None. Cannot compute blur metric on None image.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def adjust_brightness(
    image: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0,
) -> np.ndarray:
    """Adjust image brightness and contrast.

    Applies: output = alpha * image + beta

    Args:
        image: Input image as numpy array.
        alpha: Contrast multiplier (1.0 = no change, >1.0 = more contrast).
        beta: Brightness offset (0 = no change, >0 = brighter).

    Returns:
        Adjusted image as numpy array.

    Raises:
        ValueError: If image is None.
    """
    if image is None:
        raise ValueError("Input image is None. Cannot adjust brightness of None image.")
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# ─── Smoke Test ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("  Preprocessing Module — Smoke Test")
    print("=" * 50)

    # Create a dummy BGR image (480x640x3)
    dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Test resize_image
    resized = resize_image(dummy_bgr, (224, 224))
    assert resized.shape == (224, 224, 3), f"resize failed: {resized.shape}"
    print("  ✓ resize_image")

    # Test resize raises on None
    try:
        resize_image(None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ✓ resize_image raises ValueError on None")

    # Test normalize_image
    normed = normalize_image(dummy_bgr)
    assert normed.dtype == np.float32, f"normalize dtype: {normed.dtype}"
    assert 0.0 <= normed.min() and normed.max() <= 1.0, "normalize range failed"
    print("  ✓ normalize_image")

    # Test bgr_to_rgb
    rgb = bgr_to_rgb(dummy_bgr)
    assert rgb.shape == dummy_bgr.shape, "bgr_to_rgb shape mismatch"
    assert np.array_equal(rgb[:, :, 0], dummy_bgr[:, :, 2]), "channel swap failed"
    print("  ✓ bgr_to_rgb")

    # Test rgb_to_bgr
    bgr_back = rgb_to_bgr(rgb)
    assert np.array_equal(bgr_back, dummy_bgr), "rgb_to_bgr roundtrip failed"
    print("  ✓ rgb_to_bgr")

    # Test bgr_to_gray
    gray = bgr_to_gray(dummy_bgr)
    assert gray.ndim == 2, f"gray should be 2D, got {gray.ndim}D"
    print("  ✓ bgr_to_gray")

    # Test train transforms
    pil_img = Image.fromarray(bgr_to_rgb(dummy_bgr))
    train_tf = get_train_transforms(224)
    train_out = train_tf(pil_img)
    assert train_out.shape == (3, 224, 224), f"train transform: {train_out.shape}"
    print("  ✓ get_train_transforms")

    # Test val transforms
    val_tf = get_val_transforms(224)
    val_out = val_tf(pil_img)
    assert val_out.shape == (3, 224, 224), f"val transform: {val_out.shape}"
    print("  ✓ get_val_transforms")

    # Test inference transforms
    inf_tf = get_inference_transforms(224)
    inf_out = inf_tf(pil_img)
    assert inf_out.shape == (3, 224, 224), f"inference transform: {inf_out.shape}"
    print("  ✓ get_inference_transforms")

    # Test denormalize
    denormed = denormalize(val_out)
    assert denormed.min() >= 0.0 and denormed.max() <= 1.0, "denormalize range failed"
    print("  ✓ denormalize")

    # Test preprocess_frame
    frame_tensor = preprocess_frame(dummy_bgr, 224)
    assert frame_tensor.shape == (1, 3, 224, 224), f"preprocess_frame: {frame_tensor.shape}"
    print("  ✓ preprocess_frame")

    # Test preprocess_face_crop
    bbox = (100, 100, 300, 300)
    crop_tensor = preprocess_face_crop(dummy_bgr, bbox, 224, 0.15)
    assert crop_tensor.shape == (1, 3, 224, 224), f"preprocess_face_crop: {crop_tensor.shape}"
    print("  ✓ preprocess_face_crop")

    # Test is_blurry
    uniform = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert is_blurry(uniform) is True, "uniform image should be blurry"
    print("  ✓ is_blurry")

    # Test adjust_brightness
    bright = adjust_brightness(dummy_bgr, alpha=1.5, beta=30)
    assert bright.shape == dummy_bgr.shape, "adjust_brightness shape mismatch"
    print("  ✓ adjust_brightness")

    print("\n  All preprocessing tests passed! ✓")
    print("=" * 50)
