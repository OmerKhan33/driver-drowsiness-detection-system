"""
Face Detection module for Driver Drowsiness Detection System.

Provides YOLO-based and Haar Cascade-based face detectors
for locating driver faces in video frames.
"""

import time

import cv2
import numpy as np


class FaceDetector:
    """YOLO-based face detector supporting multiple YOLO versions.

    Args:
        model_name: Name of the YOLO model to use.
            Supported: 'yolov8n', 'yolov9c', 'yolo11n', 'yolo12n'.
        conf_threshold: Minimum confidence threshold for detections.
        device: Device to run inference on ('cpu', 'cuda', or 'auto').
    """

    def __init__(
        self,
        model_name: str = "yolo11n",
        conf_threshold: float = 0.4,
        device: str = "auto",
    ):
        """Initialize the FaceDetector.

        Args:
            model_name: Name of the YOLO model variant to load.
            conf_threshold: Confidence threshold for face detections.
            device: Compute device. 'auto' selects GPU if available.

        Raises:
            ValueError: If model_name is not in the list of available models.
        """
        from ultralytics import YOLO

        if model_name not in self.available_models():
            raise ValueError(
                f"Model '{model_name}' not supported. "
                f"Choose from: {self.available_models()}"
            )

        self.model_name = model_name
        self.conf_threshold = conf_threshold

        if device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load YOLO model — Ultralytics auto-downloads weights
        model_file = f"{model_name}.pt"
        self.model = YOLO(model_file)
        self.model.to(self.device)

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect faces in a BGR frame.

        Args:
            frame_bgr: Input image in BGR format (H, W, 3).

        Returns:
            List of detection dicts, each containing:
                - 'bbox': tuple (x1, y1, x2, y2)
                - 'confidence': float
                - 'crop': np.ndarray of the cropped face region
        """
        if frame_bgr is None:
            raise ValueError(
                "Input frame is None. Expected a BGR numpy array."
            )

        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            verbose=False,
            classes=[0],  # class 0 = person in COCO
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                h, w = frame_bgr.shape[:2]
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(w, x2)
                y2c = min(h, y2)
                crop = frame_bgr[y1c:y2c, x1c:x2c].copy()

                detections.append({
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": conf,
                    "crop": crop,
                })

        return detections

    def detect_largest(self, frame_bgr: np.ndarray) -> dict | None:
        """Detect the largest face in the frame (closest driver).

        Args:
            frame_bgr: Input image in BGR format (H, W, 3).

        Returns:
            Detection dict for the largest face, or None if no face found.
        """
        detections = self.detect(frame_bgr)
        if not detections:
            return None

        def bbox_area(det):
            x1, y1, x2, y2 = det["bbox"]
            return (x2 - x1) * (y2 - y1)

        return max(detections, key=bbox_area)

    def draw_detections(
        self, frame_bgr: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """Draw bounding boxes on the frame for each detection.

        Args:
            frame_bgr: Input image in BGR format.
            detections: List of detection dicts from detect().

        Returns:
            Frame with bounding boxes drawn on it.
        """
        frame_out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face {conf:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )[0]
            cv2.rectangle(
                frame_out,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                frame_out,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                1,
            )
        return frame_out

    def benchmark(
        self, frames: list, n_warmup: int = 3
    ) -> dict:
        """Benchmark inference speed on a list of frames.

        Args:
            frames: List of BGR numpy arrays.
            n_warmup: Number of warmup iterations before timing.

        Returns:
            Dict with 'avg_ms', 'std_ms', 'fps', 'min_ms', 'max_ms'.
        """
        # Warmup runs
        for i in range(min(n_warmup, len(frames))):
            self.detect(frames[i % len(frames)])

        times = []
        for frame in frames:
            start = time.perf_counter()
            self.detect(frame)
            elapsed = (time.perf_counter() - start) * 1000.0
            times.append(elapsed)

        times_arr = np.array(times)
        avg_ms = float(np.mean(times_arr))
        return {
            "avg_ms": avg_ms,
            "std_ms": float(np.std(times_arr)),
            "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
            "min_ms": float(np.min(times_arr)),
            "max_ms": float(np.max(times_arr)),
        }

    @staticmethod
    def available_models() -> list[str]:
        """Return list of supported YOLO model names.

        Returns:
            List of model name strings.
        """
        return ["yolov8n", "yolov9c", "yolo11n", "yolo12n"]


class HaarFaceDetector:
    """OpenCV Haar Cascade face detector (baseline).

    Uses the pre-trained frontal face cascade from OpenCV.
    """

    def __init__(self):
        """Initialize the Haar Cascade face detector.

        Raises:
            RuntimeError: If the cascade file cannot be loaded.
        """
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar Cascade from: {cascade_path}"
            )

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """Detect faces in a BGR frame using Haar Cascade.

        Args:
            frame_bgr: Input image in BGR format (H, W, 3).

        Returns:
            List of detection dicts, each containing:
                - 'bbox': tuple (x1, y1, x2, y2)
                - 'confidence': float (always 1.0 for Haar)
                - 'crop': np.ndarray of the cropped face region
        """
        if frame_bgr is None:
            raise ValueError(
                "Input frame is None. Expected a BGR numpy array."
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        detections = []
        h, w = frame_bgr.shape[:2]
        for (x, y, fw, fh) in faces:
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)
            crop = frame_bgr[y1:y2, x1:x2].copy()
            detections.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": 1.0,
                "crop": crop,
            })

        return detections

    def benchmark(self, frames: list, n_warmup: int = 3) -> dict:
        """Benchmark inference speed on a list of frames.

        Args:
            frames: List of BGR numpy arrays.
            n_warmup: Number of warmup iterations before timing.

        Returns:
            Dict with 'avg_ms', 'std_ms', 'fps', 'min_ms', 'max_ms'.
        """
        for i in range(min(n_warmup, len(frames))):
            self.detect(frames[i % len(frames)])

        times = []
        for frame in frames:
            start = time.perf_counter()
            self.detect(frame)
            elapsed = (time.perf_counter() - start) * 1000.0
            times.append(elapsed)

        times_arr = np.array(times)
        avg_ms = float(np.mean(times_arr))
        return {
            "avg_ms": avg_ms,
            "std_ms": float(np.std(times_arr)),
            "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
            "min_ms": float(np.min(times_arr)),
            "max_ms": float(np.max(times_arr)),
        }


if __name__ == "__main__":
    print("=" * 50)
    print("  Face Detector Module — Smoke Test")
    print("=" * 50)

    # Test Haar Cascade detector (no model download needed)
    haar = HaarFaceDetector()
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    dets = haar.detect(dummy_frame)
    assert isinstance(dets, list), "detect() should return a list"
    print("  ✓ HaarFaceDetector.detect")

    bench = haar.benchmark([dummy_frame] * 5, n_warmup=1)
    assert "avg_ms" in bench, "benchmark missing avg_ms"
    assert "fps" in bench, "benchmark missing fps"
    print(f"  ✓ HaarFaceDetector.benchmark (avg: {bench['avg_ms']:.1f}ms)")

    # Test available_models
    models = FaceDetector.available_models()
    assert len(models) == 4, f"Expected 4 models, got {len(models)}"
    assert "yolo11n" in models, "yolo11n should be available"
    print(f"  ✓ FaceDetector.available_models: {models}")

    print("\n  All face detector tests passed! ✓")
    print("=" * 50)
