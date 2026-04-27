"""
Driver Drowsiness Detection — Streamlit Application.

Professional real-time drowsiness monitoring dashboard with
webcam feed, EAR/MAR tracking, alert system, and session analytics.

Run locally:  streamlit run app/streamlit_app.py
Run via ngrok: python app/run_app.py --ngrok-token YOUR_TOKEN
"""

import sys
import time
from collections import deque
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2

import streamlit as st

from src.alert.alert_system import DrowsinessAlertSystem
from src.utils.drowsiness_utils import (
    EAR_THRESHOLD,
    MAR_THRESHOLD,
    compute_avg_ear,
    compute_drowsiness_score,
    compute_mar,
    drowsiness_level,
    extract_eye_landmarks,
    extract_mouth_landmarks,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Driver Drowsiness Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .app-header h1 {
        color: #fff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: rgba(255,255,255,0.6);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-label {
        color: rgba(255,255,255,0.5);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .metric-value.green { color: #00d97e; }
    .metric-value.yellow { color: #f6c343; }
    .metric-value.red { color: #e63757; }

    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.8rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .status-alert {
        background: linear-gradient(135deg, #00d97e, #00b368);
        color: #fff;
    }
    .status-mild {
        background: linear-gradient(135deg, #f6c343, #e6a800);
        color: #1a1a2e;
    }
    .status-drowsy {
        background: linear-gradient(135deg, #e63757, #c41230);
        color: #fff;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(230,55,87,0.5); }
        50% { box-shadow: 0 0 20px 10px rgba(230,55,87,0.2); }
    }

    /* Stats panel */
    .stats-panel {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 14px;
        padding: 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stats-panel h4 {
        color: rgba(255,255,255,0.5);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    .stats-panel .value {
        color: #fff;
        font-size: 1.4rem;
        font-weight: 600;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }

    /* Video frame border */
    .video-container {
        border-radius: 14px;
        overflow: hidden;
        border: 2px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>🚗 Driver Drowsiness Detection System</h1>
    <p>Real-time monitoring using YOLO + CNN • EAR/MAR Analysis • Intelligent Alerts</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown("---")
    st.markdown("**Detection Parameters**")
    ear_thresh = st.slider(
        "EAR Threshold", 0.15, 0.35, EAR_THRESHOLD, 0.01,
        help="Eyes considered closed below this value"
    )
    mar_thresh = st.slider(
        "MAR Threshold", 0.40, 0.80, MAR_THRESHOLD, 0.05,
        help="Yawning detected above this value"
    )
    consec_frames = st.slider(
        "Consecutive Frames", 5, 30, 15, 1,
        help="Drowsy frames before alert triggers"
    )
    cnn_thresh = st.slider(
        "CNN Confidence", 0.40, 0.90, 0.65, 0.05,
        help="Minimum CNN drowsy probability"
    )

    st.markdown("---")
    st.markdown("**Display Options**")
    show_landmarks = st.checkbox("Show Face Landmarks", value=True)
    show_metrics_overlay = st.checkbox("Show Metrics on Frame", value=True)

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "Built with PyTorch, Ultralytics YOLO, MediaPipe, "
        "and Streamlit. Designed for real-time driver safety monitoring."
    )


# ─── Session State Init ──────────────────────────────────────────────────────

if "alert_system" not in st.session_state:
    st.session_state.alert_system = DrowsinessAlertSystem(
        ear_threshold=ear_thresh,
        mar_threshold=mar_thresh,
        consec_frames=consec_frames,
        cnn_threshold=cnn_thresh,
        alert_sound=False,
    )
if "ear_history" not in st.session_state:
    st.session_state.ear_history = deque(maxlen=100)
if "mar_history" not in st.session_state:
    st.session_state.mar_history = deque(maxlen=100)
if "score_history" not in st.session_state:
    st.session_state.score_history = deque(maxlen=100)
if "running" not in st.session_state:
    st.session_state.running = False
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "drowsy_frames" not in st.session_state:
    st.session_state.drowsy_frames = 0
if "fps" not in st.session_state:
    st.session_state.fps = 0.0


# ─── Helper Functions ────────────────────────────────────────────────────────


def get_status_html(level: str) -> str:
    """Generate status badge HTML."""
    css_class = {
        "ALERT": "status-alert",
        "MILD": "status-mild",
        "DROWSY": "status-drowsy",
    }.get(level, "status-alert")
    return f'<div style="text-align:center"><span class="status-badge {css_class}">{level}</span></div>'


def get_metric_html(label: str, value: str, color: str = "green") -> str:
    """Generate metric card HTML."""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {color}">{value}</div>
    </div>"""


def draw_overlay(frame, ear, mar, score, level, fps):
    """Draw metrics overlay on the video frame."""
    h, w = frame.shape[:2]
    # Semi-transparent overlay bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = {
        "ALERT": (126, 217, 0),
        "MILD": (67, 195, 246),
        "DROWSY": (87, 55, 230),
    }.get(level, (126, 217, 0))

    cv2.putText(frame, f"EAR: {ear:.3f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR: {mar:.3f}", (170, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Score: {score:.2f}", (330, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, level, (500, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
    return frame


# ─── Main Layout ──────────────────────────────────────────────────────────────

# Control buttons
col_start, col_stop, col_reset = st.columns([1, 1, 1])
with col_start:
    start_btn = st.button("▶️  Start Detection", use_container_width=True, type="primary")
with col_stop:
    stop_btn = st.button("⏹  Stop", use_container_width=True)
with col_reset:
    reset_btn = st.button("🔄  Reset Stats", use_container_width=True)

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False
if reset_btn:
    st.session_state.alert_system.reset()
    st.session_state.ear_history.clear()
    st.session_state.mar_history.clear()
    st.session_state.score_history.clear()
    st.session_state.total_frames = 0
    st.session_state.drowsy_frames = 0

# Main content
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.markdown("#### 📹 Live Feed")
    video_placeholder = st.empty()

with col_stats:
    st.markdown("#### 📊 Live Metrics")
    status_placeholder = st.empty()
    ear_metric = st.empty()
    mar_metric = st.empty()
    score_metric = st.empty()
    fps_metric = st.empty()
    consec_metric = st.empty()

# Charts row
st.markdown("---")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.markdown("#### 👁 EAR History")
    ear_chart = st.empty()
with chart_col2:
    st.markdown("#### 👄 MAR History")
    mar_chart = st.empty()

# Session stats
st.markdown("---")
st.markdown("#### 📈 Session Statistics")
stats_cols = st.columns(5)
stat_frames = stats_cols[0].empty()
stat_drowsy = stats_cols[1].empty()
stat_pct = stats_cols[2].empty()
stat_alerts = stats_cols[3].empty()
stat_yawns = stats_cols[4].empty()


# ─── Detection Loop ──────────────────────────────────────────────────────────

if st.session_state.running:
    # Initialize MediaPipe Face Mesh
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    except ImportError:
        st.error("MediaPipe not installed. Run: pip install mediapipe")
        st.session_state.running = False
        st.stop()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Cannot access webcam. Please check your camera connection.")
        st.session_state.running = False
        st.stop()

    # Update alert system thresholds
    alert_sys = st.session_state.alert_system
    alert_sys.ear_threshold = ear_thresh
    alert_sys.mar_threshold = mar_thresh
    alert_sys.consec_frames_threshold = consec_frames
    alert_sys.cnn_threshold = cnn_thresh

    while st.session_state.running:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Lost webcam feed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # MediaPipe face mesh
        results = face_mesh.process(rgb_frame)

        ear_val = 0.30
        mar_val = 0.40
        cnn_conf = 0.0

        if results.multi_face_landmarks:
            face_lm = results.multi_face_landmarks[0]

            # Extract landmarks
            left_eye, right_eye = extract_eye_landmarks(face_lm, w, h)
            mouth = extract_mouth_landmarks(face_lm, w, h)

            # Compute metrics
            ear_val = compute_avg_ear(left_eye, right_eye)
            mar_val = compute_mar(mouth)

            # Draw landmarks if enabled
            if show_landmarks:
                for eye_pts in [left_eye, right_eye]:
                    pts = eye_pts.astype(int)
                    for i in range(len(pts)):
                        cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 128), 1)
                    for pt in pts:
                        cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1)
                mouth_pts = mouth.astype(int)
                for pt in mouth_pts:
                    cv2.circle(frame, tuple(pt), 2, (255, 128, 0), -1)

        # Update alert system
        status = alert_sys.update(ear=ear_val, mar=mar_val, cnn_confidence=cnn_conf)
        score = compute_drowsiness_score(ear=ear_val, mar=mar_val, cnn_confidence=cnn_conf)
        level = drowsiness_level(score)

        # FPS
        elapsed = time.perf_counter() - t0
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        st.session_state.fps = current_fps

        # Update histories
        st.session_state.ear_history.append(ear_val)
        st.session_state.mar_history.append(mar_val)
        st.session_state.score_history.append(score)
        st.session_state.total_frames += 1
        if status["is_drowsy"]:
            st.session_state.drowsy_frames += 1

        # Draw overlay
        if show_metrics_overlay:
            frame = draw_overlay(frame, ear_val, mar_val, score, level, current_fps)

        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update UI
        video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

        status_placeholder.markdown(get_status_html(level), unsafe_allow_html=True)

        ear_color = "green" if ear_val > ear_thresh else "red"
        mar_color = "green" if mar_val < mar_thresh else "yellow"
        score_color = "green" if score < 0.4 else ("yellow" if score < 0.7 else "red")

        ear_metric.markdown(get_metric_html("EAR", f"{ear_val:.3f}", ear_color), unsafe_allow_html=True)
        mar_metric.markdown(get_metric_html("MAR", f"{mar_val:.3f}", mar_color), unsafe_allow_html=True)
        score_metric.markdown(get_metric_html("Score", f"{score:.2f}", score_color), unsafe_allow_html=True)
        fps_metric.markdown(get_metric_html("FPS", f"{current_fps:.0f}", "green"), unsafe_allow_html=True)
        consec_metric.markdown(get_metric_html("Consec", str(status["consecutive_frames"]),
                                               "red" if status["is_drowsy"] else "green"), unsafe_allow_html=True)

        # Charts
        if len(st.session_state.ear_history) > 2:
            ear_chart.line_chart(list(st.session_state.ear_history), height=150)
            mar_chart.line_chart(list(st.session_state.mar_history), height=150)

        # Session stats
        total = st.session_state.total_frames
        drowsy = st.session_state.drowsy_frames
        pct = (drowsy / total * 100) if total > 0 else 0
        session_stats = alert_sys.get_stats()

        stat_frames.metric("Total Frames", f"{total:,}")
        stat_drowsy.metric("Drowsy Frames", f"{drowsy:,}")
        stat_pct.metric("Drowsy %", f"{pct:.1f}%")
        stat_alerts.metric("Alerts", f"{session_stats['total_alerts']}")
        stat_yawns.metric("Yawns", f"{session_stats['total_yawns']}")

        time.sleep(0.01)

    cap.release()

else:
    # Idle state
    video_placeholder.markdown("""
    <div style="
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 14px; padding: 4rem 2rem; text-align: center;
        border: 2px dashed rgba(255,255,255,0.1);
    ">
        <p style="font-size: 3rem; margin: 0;">📹</p>
        <p style="color: rgba(255,255,255,0.5); font-size: 1.1rem; margin-top: 1rem;">
            Click <strong>Start Detection</strong> to begin monitoring
        </p>
        <p style="color: rgba(255,255,255,0.3); font-size: 0.85rem;">
            Ensure your webcam is connected and accessible
        </p>
    </div>
    """, unsafe_allow_html=True)

    status_placeholder.markdown(get_status_html("ALERT"), unsafe_allow_html=True)
    ear_metric.markdown(get_metric_html("EAR", "—", "green"), unsafe_allow_html=True)
    mar_metric.markdown(get_metric_html("MAR", "—", "green"), unsafe_allow_html=True)
    score_metric.markdown(get_metric_html("Score", "—", "green"), unsafe_allow_html=True)
    fps_metric.markdown(get_metric_html("FPS", "—", "green"), unsafe_allow_html=True)
    consec_metric.markdown(get_metric_html("Consec", "0", "green"), unsafe_allow_html=True)
