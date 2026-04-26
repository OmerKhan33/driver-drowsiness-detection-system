"""
Drowsiness utilities module for Driver Drowsiness Detection System.

Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and extracts
MediaPipe landmarks for drowsiness scoring.
"""

import math

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────

EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.60
EAR_CONSEC_FRAMES = 20
MAR_CONSEC_FRAMES = 15

# MediaPipe face mesh indices
# (Using 6 landmarks per eye, following Soukupova & Cech model)
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Using 8 landmarks for the mouth
MOUTH_IDX = [61, 291, 39, 181, 0, 17, 269, 405]

# ─── EAR / MAR Calculations ───────────────────────────────────────────────────

def compute_ear(eye: np.ndarray) -> float:
    """Compute the Eye Aspect Ratio (EAR) given eye landmarks.

    Uses the formula by Soukupová & Čech (2016):
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    Args:
        eye: A numpy array of shape (6, 2) representing the 6 eye landmarks (x, y).

    Returns:
        The computed EAR value.

    Raises:
        ValueError: If the eye array does not have shape (6, 2).
    """
    if not isinstance(eye, np.ndarray) or eye.shape != (6, 2):
        raise ValueError(f"Expected eye array of shape (6, 2), got {getattr(eye, 'shape', type(eye))}")

    # Vertical distances
    v1 = np.linalg.norm(eye[1] - eye[5])
    v2 = np.linalg.norm(eye[2] - eye[4])
    # Horizontal distance
    h = np.linalg.norm(eye[0] - eye[3])

    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return float(ear)

def compute_avg_ear(left_eye: np.ndarray, right_eye: np.ndarray) -> float:
    """Compute the average EAR of both eyes.

    Args:
        left_eye: A numpy array of shape (6, 2) for the left eye.
        right_eye: A numpy array of shape (6, 2) for the right eye.

    Returns:
        The average EAR value.
    """
    left_ear = compute_ear(left_eye)
    right_ear = compute_ear(right_eye)
    return (left_ear + right_ear) / 2.0

def is_eye_closed(ear: float, threshold: float = EAR_THRESHOLD) -> bool:
    """Check if the eye is considered closed based on EAR.

    Args:
        ear: The computed EAR value.
        threshold: The threshold below which the eye is closed.

    Returns:
        True if the eye is closed, False otherwise.
    """
    return ear < threshold

def compute_mar(mouth: np.ndarray) -> float:
    """Compute the Mouth Aspect Ratio (MAR) given mouth landmarks.

    Formula roughly analogous to EAR, comparing vertical distances
    to horizontal distance.

    Args:
        mouth: A numpy array of shape (8, 2) representing the mouth landmarks.

    Returns:
        The computed MAR value.

    Raises:
        ValueError: If the mouth array does not have shape (8, 2).
    """
    if not isinstance(mouth, np.ndarray) or mouth.shape != (8, 2):
        raise ValueError(f"Expected mouth array of shape (8, 2), got {getattr(mouth, 'shape', type(mouth))}")

    # Vertical distances
    v1 = np.linalg.norm(mouth[2] - mouth[6])
    v2 = np.linalg.norm(mouth[3] - mouth[5])
    v3 = np.linalg.norm(mouth[4] - mouth[5]) # using additional point for robust vertical
    
    # Actually let's use standard MAR based on the 8 points provided:
    # 61 (left corner), 291 (right corner)
    # top: 39, 0, 269
    # bottom: 181, 17, 405
    # Distances between pairs: (39, 181), (0, 17), (269, 405)
    # Let's map indices for clarity:
    # 0: 61 (Left), 1: 291 (Right)
    # 2: 39 (TopL), 3: 181 (BotL)
    # 4: 0 (TopC), 5: 17 (BotC)
    # 6: 269 (TopR), 7: 405 (BotR)
    
    # If the points are literally passed in the order MOUTH_IDX:
    # [61, 291, 39, 181, 0, 17, 269, 405]
    
    v1 = np.linalg.norm(mouth[2] - mouth[3]) # 39 to 181
    v2 = np.linalg.norm(mouth[4] - mouth[5]) # 0 to 17
    v3 = np.linalg.norm(mouth[6] - mouth[7]) # 269 to 405
    
    h = np.linalg.norm(mouth[0] - mouth[1])  # 61 to 291

    if h == 0:
        return 0.0

    mar = (v1 + v2 + v3) / (3.0 * h)
    return float(mar)

def is_yawning(mar: float, threshold: float = MAR_THRESHOLD) -> bool:
    """Check if the subject is yawning based on MAR.

    Args:
        mar: The computed MAR value.
        threshold: The threshold above which yawning is detected.

    Returns:
        True if yawning, False otherwise.
    """
    return mar > threshold

# ─── Landmark Extraction ──────────────────────────────────────────────────────

def extract_eye_landmarks(face_landmarks, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract left and right eye landmarks from MediaPipe face mesh results.

    Args:
        face_landmarks: The normalized landmark list from MediaPipe.
        img_w: Image width to scale the normalized x coordinates.
        img_h: Image height to scale the normalized y coordinates.

    Returns:
        A tuple of (left_eye, right_eye) numpy arrays, each of shape (6, 2).
    """
    left_eye = np.zeros((6, 2), dtype=np.float32)
    right_eye = np.zeros((6, 2), dtype=np.float32)

    for i, idx in enumerate(LEFT_EYE_IDX):
        lm = face_landmarks.landmark[idx]
        left_eye[i] = [lm.x * img_w, lm.y * img_h]

    for i, idx in enumerate(RIGHT_EYE_IDX):
        lm = face_landmarks.landmark[idx]
        right_eye[i] = [lm.x * img_w, lm.y * img_h]

    return left_eye, right_eye

def extract_mouth_landmarks(face_landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Extract mouth landmarks from MediaPipe face mesh results.

    Args:
        face_landmarks: The normalized landmark list from MediaPipe.
        img_w: Image width to scale the normalized x coordinates.
        img_h: Image height to scale the normalized y coordinates.

    Returns:
        A numpy array of shape (8, 2).
    """
    mouth = np.zeros((len(MOUTH_IDX), 2), dtype=np.float32)
    for i, idx in enumerate(MOUTH_IDX):
        lm = face_landmarks.landmark[idx]
        mouth[i] = [lm.x * img_w, lm.y * img_h]
    return mouth

def estimate_head_tilt(face_landmarks, img_w: int, img_h: int) -> float:
    """Estimate head tilt angle (roll) using eye centers.

    Args:
        face_landmarks: The normalized landmark list from MediaPipe.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Tilt angle in degrees.
    """
    left_eye, right_eye = extract_eye_landmarks(face_landmarks, img_w, img_h)
    
    # Compute centroids
    left_center = np.mean(left_eye, axis=0)
    right_center = np.mean(right_eye, axis=0)
    
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    return float(angle)

def is_head_drooping(tilt_angle: float, threshold: float = 15.0) -> bool:
    """Check if the head is drooping based on the tilt angle.

    Args:
        tilt_angle: Head tilt angle in degrees.
        threshold: Absolute angle threshold.

    Returns:
        True if the absolute tilt is greater than the threshold.
    """
    return abs(tilt_angle) > threshold

# ─── Scoring System ───────────────────────────────────────────────────────────

def compute_drowsiness_score(
    ear: float, 
    mar: float, 
    cnn_confidence: float, 
    head_tilt: float = 0.0, 
    weights: dict = None
) -> float:
    """Compute an overall drowsiness score by combining signals.

    Normalizes components and uses a weighted sum. 
    Score is returned in the range [0.0, 1.0].

    Args:
        ear: Computed EAR value.
        mar: Computed MAR value.
        cnn_confidence: CNN probability of being "drowsy" [0.0, 1.0].
        head_tilt: Head tilt angle in degrees.
        weights: Dictionary of weights for 'ear', 'mar', 'cnn', 'tilt'.

    Returns:
        A combined drowsiness score in [0.0, 1.0].
    """
    if weights is None:
        weights = {
            'ear': 0.35,
            'mar': 0.15,
            'cnn': 0.40,
            'tilt': 0.10
        }
    
    # Normalize EAR: mapped so that lower EAR means higher drowsiness contribution
    # Assume normal EAR is ~0.35, drowsy is <= 0.20
    ear_norm = max(0.0, min(1.0, (0.35 - ear) / (0.35 - 0.20)))
    
    # Normalize MAR: mapped so that higher MAR means higher drowsiness (yawning)
    # Assume normal MAR is ~0.40, yawning is >= 0.60
    mar_norm = max(0.0, min(1.0, (mar - 0.40) / (0.60 - 0.40)))
    
    # Normalize tilt
    tilt_norm = max(0.0, min(1.0, abs(head_tilt) / 25.0)) # 25 degrees max tilt logic

    score = (
        weights.get('ear', 0.0) * ear_norm +
        weights.get('mar', 0.0) * mar_norm +
        weights.get('cnn', 0.0) * cnn_confidence +
        weights.get('tilt', 0.0) * tilt_norm
    )
    
    # Ensure weights sum logic doesn't overflow 1.0
    total_weight = sum(weights.values())
    if total_weight > 0:
        score /= total_weight
        
    return float(max(0.0, min(1.0, score)))

def drowsiness_level(score: float) -> str:
    """Map a numerical drowsiness score to a categorical level.

    Args:
        score: The drowsiness score [0.0, 1.0].

    Returns:
        String representing the level: 'ALERT', 'MILD', or 'DROWSY'.
    """
    if score < 0.4:
        return 'ALERT'
    elif score < 0.7:
        return 'MILD'
    else:
        return 'DROWSY'

# ─── Smoke Test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("  Drowsiness Utils Module — Smoke Test")
    print("=" * 50)

    # Test EAR
    dummy_eye_open = np.array([
        [10, 50], [20, 60], [30, 60], [40, 50], [30, 40], [20, 40]
    ], dtype=np.float32)
    dummy_eye_closed = np.array([
        [10, 50], [20, 52], [30, 52], [40, 50], [30, 48], [20, 48]
    ], dtype=np.float32)
    
    ear_open = compute_ear(dummy_eye_open)
    ear_closed = compute_ear(dummy_eye_closed)
    
    assert ear_open > 0.2, f"Expected open EAR > 0.2, got {ear_open}"
    assert ear_closed < 0.15, f"Expected closed EAR < 0.15, got {ear_closed}"
    print("  ✓ compute_ear")
    
    try:
        compute_ear(np.zeros((4, 2)))
        assert False, "Should have raised ValueError for wrong shape"
    except ValueError:
        print("  ✓ compute_ear raises ValueError for wrong shape")

    avg_ear = compute_avg_ear(dummy_eye_open, dummy_eye_closed)
    assert isinstance(avg_ear, float), "Average EAR should be a float"
    print("  ✓ compute_avg_ear")

    assert not is_eye_closed(ear_open), "Open eye detected as closed"
    assert is_eye_closed(ear_closed), "Closed eye not detected"
    print("  ✓ is_eye_closed")

    # Test MAR
    dummy_mouth = np.array([
        [10, 50], [90, 50], [30, 60], [30, 40], [50, 65], [50, 35], [70, 60], [70, 40]
    ], dtype=np.float32)
    mar_val = compute_mar(dummy_mouth)
    assert isinstance(mar_val, float), "MAR should be a float"
    print("  ✓ compute_mar")

    try:
        compute_mar(np.zeros((6, 2)))
        assert False, "Should have raised ValueError for wrong shape"
    except ValueError:
        print("  ✓ compute_mar raises ValueError for wrong shape")
        
    assert is_yawning(0.7), "0.7 MAR should be yawning"
    assert not is_yawning(0.3), "0.3 MAR should not be yawning"
    print("  ✓ is_yawning")
    
    # Test Scoring
    score_alert = compute_drowsiness_score(ear=0.35, mar=0.3, cnn_confidence=0.1, head_tilt=0.0)
    score_drowsy = compute_drowsiness_score(ear=0.15, mar=0.7, cnn_confidence=0.9, head_tilt=20.0)
    
    assert 0.0 <= score_alert <= 1.0, f"Score out of bounds: {score_alert}"
    assert 0.0 <= score_drowsy <= 1.0, f"Score out of bounds: {score_drowsy}"
    assert score_drowsy > score_alert, "Drowsy score should be higher than alert score"
    print("  ✓ compute_drowsiness_score")
    
    assert drowsiness_level(0.2) == 'ALERT'
    assert drowsiness_level(0.5) == 'MILD'
    assert drowsiness_level(0.8) == 'DROWSY'
    print("  ✓ drowsiness_level")
    
    print("\n  All drowsiness_utils tests passed! ✓")
    print("=" * 50)
