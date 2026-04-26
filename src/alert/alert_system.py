"""
Alert System module for Driver Drowsiness Detection System.

Provides a stateful alert system that combines EAR, MAR, and CNN
confidence signals to detect drowsiness and trigger audio/visual alerts.
"""

import logging
import time

import numpy as np

from src.utils.drowsiness_utils import (
    EAR_CONSEC_FRAMES,
    EAR_THRESHOLD,
    MAR_THRESHOLD,
    compute_drowsiness_score,
    drowsiness_level,
    is_eye_closed,
    is_yawning,
)

logger = logging.getLogger(__name__)


# ─── Alert System ─────────────────────────────────────────────────────────────


class DrowsinessAlertSystem:
    """Stateful drowsiness alert system combining multiple signals.

    Tracks consecutive drowsy frames and triggers alerts when
    the drowsiness threshold is exceeded.

    Args:
        ear_threshold: EAR value below which eyes are considered closed.
        mar_threshold: MAR value above which yawning is detected.
        consec_frames: Number of consecutive drowsy frames to trigger alert.
        cnn_threshold: CNN drowsy probability threshold.
        alert_sound: Whether to play alert sounds.
    """

    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        mar_threshold: float = MAR_THRESHOLD,
        consec_frames: int = EAR_CONSEC_FRAMES,
        cnn_threshold: float = 0.65,
        alert_sound: bool = True,
    ):
        """Initialize the alert system.

        Args:
            ear_threshold: EAR threshold for eye closure detection.
            mar_threshold: MAR threshold for yawn detection.
            consec_frames: Consecutive drowsy frames before alerting.
            cnn_threshold: CNN confidence threshold for drowsy class.
            alert_sound: Whether to enable audio alerts.
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consec_frames_threshold = consec_frames
        self.cnn_threshold = cnn_threshold
        self.alert_sound_enabled = alert_sound

        # Public state attributes
        self.consecutive_frames = 0
        self._is_drowsy = False
        self._is_yawning = False
        self._current_ear = 0.0
        self._current_mar = 0.0
        self._current_score = 0.0
        self._alert_level = "ALERT"

        # Session statistics
        self._total_alerts = 0
        self._total_yawns = 0
        self._ear_history = []
        self._mar_history = []
        self._alert_log = []
        self._session_start = time.time()

        # Sound system initialization
        self._sound_initialized = False
        if self.alert_sound_enabled:
            self._init_sound()

    def _init_sound(self) -> None:
        """Initialize the pygame mixer for alert sounds.

        Fails silently if pygame is not available or mixer init fails.
        """
        try:
            import pygame

            pygame.mixer.init(frequency=22050, size=-16, channels=1)
            self._sound_initialized = True
            logger.info("Audio alert system initialized (pygame)")
        except (ImportError, Exception) as e:
            self._sound_initialized = False
            logger.warning(f"Audio alerts disabled: {e}")

    def update(
        self,
        ear: float = None,
        mar: float = None,
        cnn_confidence: float = None,
    ) -> dict:
        """Update the alert system state with new signal values.

        Args:
            ear: Current Eye Aspect Ratio value, or None.
            mar: Current Mouth Aspect Ratio value, or None.
            cnn_confidence: CNN probability of drowsy class, or None.

        Returns:
            Status dict with:
                - 'is_drowsy': bool
                - 'is_yawning': bool
                - 'alert_level': str ('ALERT', 'MILD', or 'DROWSY')
                - 'ear': float
                - 'mar': float
                - 'consecutive_frames': int
        """
        # Update current values
        if ear is not None:
            self._current_ear = ear
            self._ear_history.append(ear)
        if mar is not None:
            self._current_mar = mar
            self._mar_history.append(mar)

        # Determine eye closure
        eye_closed = False
        if ear is not None:
            eye_closed = is_eye_closed(ear, self.ear_threshold)

        # Determine yawning
        yawn_detected = False
        if mar is not None:
            yawn_detected = is_yawning(mar, self.mar_threshold)
            if yawn_detected and not self._is_yawning:
                self._total_yawns += 1

        self._is_yawning = yawn_detected

        # Determine CNN drowsiness
        cnn_drowsy = False
        cnn_conf = 0.0
        if cnn_confidence is not None:
            cnn_conf = cnn_confidence
            cnn_drowsy = cnn_confidence >= self.cnn_threshold

        # Combined drowsiness check
        is_frame_drowsy = eye_closed or cnn_drowsy

        if is_frame_drowsy:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = max(0, self.consecutive_frames - 1)

        # Update drowsy state
        self._is_drowsy = self.consecutive_frames >= self.consec_frames_threshold

        # Compute drowsiness score
        safe_ear = ear if ear is not None else 0.30
        safe_mar = mar if mar is not None else 0.40
        self._current_score = compute_drowsiness_score(
            ear=safe_ear,
            mar=safe_mar,
            cnn_confidence=cnn_conf,
        )
        self._alert_level = drowsiness_level(self._current_score)

        # Trigger alert if drowsy
        if self._is_drowsy:
            self.trigger_alert("drowsy")
        elif self._is_yawning:
            self.trigger_alert("yawn")

        return {
            "is_drowsy": self._is_drowsy,
            "is_yawning": self._is_yawning,
            "alert_level": self._alert_level,
            "ear": self._current_ear,
            "mar": self._current_mar,
            "consecutive_frames": self.consecutive_frames,
        }

    def is_drowsy(self) -> bool:
        """Check if the driver is currently drowsy.

        Returns:
            True if drowsy, False otherwise.
        """
        return self._is_drowsy

    def is_yawning(self) -> bool:
        """Check if the driver is currently yawning.

        Returns:
            True if yawning, False otherwise.
        """
        return self._is_yawning

    def trigger_alert(self, alert_type: str = "drowsy") -> None:
        """Trigger an alert event.

        Logs the event and plays a sound if audio is enabled.

        Args:
            alert_type: Type of alert ('drowsy' or 'yawn').
        """
        timestamp = time.time()
        event = {
            "type": alert_type,
            "timestamp": timestamp,
            "ear": self._current_ear,
            "mar": self._current_mar,
            "consecutive_frames": self.consecutive_frames,
        }
        self._alert_log.append(event)
        self._total_alerts += 1

        logger.warning(
            f"ALERT [{alert_type.upper()}] — "
            f"EAR: {self._current_ear:.3f}, "
            f"MAR: {self._current_mar:.3f}, "
            f"Consec: {self.consecutive_frames}"
        )

        # Play sound
        if self._sound_initialized:
            self._play_beep(alert_type)

    def _play_beep(self, alert_type: str = "drowsy") -> None:
        """Generate and play a beep sound via pygame.

        Args:
            alert_type: Type of alert determines beep frequency.
        """
        try:
            import pygame

            freq = 880 if alert_type == "drowsy" else 660
            duration_ms = 300
            sample_rate = 22050
            n_samples = int(sample_rate * duration_ms / 1000)

            t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
            wave = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)
            stereo_wave = np.column_stack([wave, wave])

            sound = pygame.sndarray.make_sound(stereo_wave)
            sound.play()
        except Exception as e:
            logger.debug(f"Could not play beep: {e}")

    def reset(self) -> None:
        """Reset the consecutive frames counter and drowsy state."""
        self.consecutive_frames = 0
        self._is_drowsy = False
        self._is_yawning = False
        self._current_score = 0.0
        self._alert_level = "ALERT"

    def get_stats(self) -> dict:
        """Get session statistics.

        Returns:
            Dict with session stats including total alerts,
            average EAR/MAR, session duration, etc.
        """
        session_duration = time.time() - self._session_start
        avg_ear = float(np.mean(self._ear_history)) if self._ear_history else 0.0
        avg_mar = float(np.mean(self._mar_history)) if self._mar_history else 0.0

        return {
            "total_alerts": self._total_alerts,
            "total_yawns": self._total_yawns,
            "average_ear": avg_ear,
            "average_mar": avg_mar,
            "session_duration_s": session_duration,
            "alert_log_count": len(self._alert_log),
            "recent_alerts": self._alert_log[-10:],
        }


# ─── Smoke Test ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 50)
    print("  Alert System Module — Smoke Test")
    print("=" * 50)

    # Create system with sound disabled for testing
    system = DrowsinessAlertSystem(
        ear_threshold=0.25,
        mar_threshold=0.60,
        consec_frames=3,
        cnn_threshold=0.65,
        alert_sound=False,
    )
    print("  ✓ DrowsinessAlertSystem created")

    # Simulate alert frames (low EAR = eyes closed)
    for i in range(5):
        status = system.update(ear=0.15, mar=0.3, cnn_confidence=0.8)

    assert system.is_drowsy(), "Should be drowsy after 5 consecutive frames (threshold=3)"
    assert status["is_drowsy"] is True
    print("  ✓ Drowsy detection after consecutive frames")

    # Test yawn detection
    status = system.update(ear=0.30, mar=0.75, cnn_confidence=0.2)
    assert status["is_yawning"] is True
    print("  ✓ Yawn detection with high MAR")

    # Test reset
    system.reset()
    assert system.consecutive_frames == 0, "consecutive_frames should be 0 after reset"
    assert not system.is_drowsy(), "Should not be drowsy after reset"
    print("  ✓ Reset clears state")

    # Test not drowsy with high EAR
    status = system.update(ear=0.35, mar=0.3, cnn_confidence=0.1)
    assert not status["is_drowsy"]
    print("  ✓ Not drowsy with high EAR")

    # Test update returns correct dict keys
    expected_keys = {"is_drowsy", "is_yawning", "alert_level", "ear", "mar", "consecutive_frames"}
    assert set(status.keys()) == expected_keys, f"Missing keys: {expected_keys - set(status.keys())}"
    print("  ✓ update() returns correct dict keys")

    # Test get_stats
    stats = system.get_stats()
    assert "total_alerts" in stats
    assert "average_ear" in stats
    print(f"  ✓ get_stats (total_alerts={stats['total_alerts']})")

    print("\n  All alert system tests passed! ✓")
    print("=" * 50)
