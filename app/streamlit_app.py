"""
Driver Drowsiness Detection — Streamlit Application.

Features: Login/Signup, Driver webcam dashboard, Admin history dashboard.
Run: streamlit run app/streamlit_app.py
"""

import sys
import time
from collections import deque
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import pandas as pd
import streamlit as st

from app.database import (
    authenticate_user,
    create_user,
    end_session,
    get_all_drivers,
    get_all_sessions,
    get_dashboard_stats,
    get_driver_events,
    log_event,
    start_session,
)
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
.stApp { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1rem; max-width: 1400px; }
.app-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.app-header h1 { color: #fff; font-size: 1.8rem; font-weight: 700; margin: 0; }
.app-header p { color: rgba(255,255,255,0.6); font-size: 0.95rem; margin: 0.3rem 0 0 0; }
.metric-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-radius: 14px; padding: 1.2rem 1.5rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.metric-label {
    color: rgba(255,255,255,0.5); font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600;
}
.metric-value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
.metric-value.green { color: #00d97e; }
.metric-value.yellow { color: #f6c343; }
.metric-value.red { color: #e63757; }
.status-badge {
    display: inline-block; padding: 0.5rem 1.8rem; border-radius: 50px;
    font-weight: 700; font-size: 1.1rem; letter-spacing: 1.5px; text-transform: uppercase;
}
.status-alert { background: linear-gradient(135deg, #00d97e, #00b368); color: #fff; }
.status-mild { background: linear-gradient(135deg, #f6c343, #e6a800); color: #1a1a2e; }
.status-drowsy {
    background: linear-gradient(135deg, #e63757, #c41230); color: #fff;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(230,55,87,0.5); }
    50% { box-shadow: 0 0 20px 10px rgba(230,55,87,0.2); }
}
.login-box {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-radius: 20px; padding: 2.5rem; max-width: 420px; margin: 3rem auto;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 40px rgba(0,0,0,0.4);
}
.admin-stat {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border-radius: 14px; padding: 1.5rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.06);
}
.admin-stat h2 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.admin-stat p { color: rgba(255,255,255,0.5); font-size: 0.8rem; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def metric_html(label, value, color="green"):
    """Generate metric card HTML."""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value {color}">{value}</div></div>'
    )


def status_html(level):
    """Generate status badge HTML."""
    css = {"ALERT": "status-alert", "MILD": "status-mild", "DROWSY": "status-drowsy"}
    return f'<div style="text-align:center"><span class="status-badge {css.get(level, "status-alert")}">{level}</span></div>'


def draw_overlay(frame, ear, mar, score, level, fps):
    """Draw metrics overlay on the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    color = {"ALERT": (126, 217, 0), "MILD": (67, 195, 246), "DROWSY": (87, 55, 230)}.get(level, (126, 217, 0))
    cv2.putText(frame, f"EAR:{ear:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR:{mar:.3f}", (160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Score:{score:.2f}", (310, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, level, (480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
    return frame


# ─── Auth Page ────────────────────────────────────────────────────────────────


def render_auth_page():
    """Render the login/signup page."""
    st.markdown("""
    <div class="app-header" style="text-align:center">
        <h1>🚗 Driver Drowsiness Detection System</h1>
        <p>Secure Login Portal • Real-time Safety Monitoring</p>
    </div>""", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["🔐 Login", "📝 Sign Up"])

    with tab_login:
        with st.form("login_form"):
            st.markdown("### Welcome Back")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
            if submitted and username and password:
                result = authenticate_user(username, password)
                if result["success"]:
                    st.session_state.user = result
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")

    with tab_signup:
        with st.form("signup_form"):
            st.markdown("### Create Driver Account")
            full_name = st.text_input("Full Name", key="signup_name")
            new_user = st.text_input("Username", key="signup_user")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
            confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if not all([full_name, new_user, new_pass, confirm]):
                    st.warning("Please fill all fields")
                elif new_pass != confirm:
                    st.error("Passwords do not match")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters")
                else:
                    result = create_user(new_user, new_pass, full_name)
                    if result["success"]:
                        st.success("✅ Account created! Please login.")
                    else:
                        st.error(f"❌ {result['message']}")

    st.markdown("---")
    st.caption("Default admin: username `admin` / password `admin123`")


# ─── Driver Dashboard ────────────────────────────────────────────────────────


def render_driver_dashboard():
    """Render the driver's real-time monitoring dashboard."""
    user = st.session_state.user

    st.markdown(f"""
    <div class="app-header">
        <h1>🚗 Driver Dashboard</h1>
        <p>Welcome, {user['full_name']} • Real-time Drowsiness Monitoring</p>
    </div>""", unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.markdown(f"### 👤 {user['full_name']}")
        st.caption(f"Role: Driver | @{user['username']}")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.markdown("---")
        ear_thresh = st.slider("EAR Threshold", 0.15, 0.35, EAR_THRESHOLD, 0.01)
        mar_thresh = st.slider("MAR Threshold", 0.40, 0.80, MAR_THRESHOLD, 0.05)
        consec_frames = st.slider("Consecutive Frames", 5, 30, 15, 1)
        show_landmarks = st.checkbox("Show Landmarks", True)
        show_overlay = st.checkbox("Show Metrics Overlay", True)

    # Init session state
    if "alert_system" not in st.session_state:
        st.session_state.alert_system = DrowsinessAlertSystem(
            ear_threshold=ear_thresh, mar_threshold=mar_thresh,
            consec_frames=consec_frames, alert_sound=False,
        )
    if "ear_history" not in st.session_state:
        st.session_state.ear_history = deque(maxlen=100)
    if "mar_history" not in st.session_state:
        st.session_state.mar_history = deque(maxlen=100)
    if "total_frames" not in st.session_state:
        st.session_state.total_frames = 0
    if "drowsy_frames" not in st.session_state:
        st.session_state.drowsy_frames = 0
    if "running" not in st.session_state:
        st.session_state.running = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "last_event_time" not in st.session_state:
        st.session_state.last_event_time = 0

    # Control buttons
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.session_state.session_id = start_session(user["user_id"])
            st.session_state.total_frames = 0
            st.session_state.drowsy_frames = 0
            st.session_state.alert_system.reset()
            st.session_state.ear_history.clear()
            st.session_state.mar_history.clear()
    with c2:
        if st.button("⏹ Stop", use_container_width=True):
            if st.session_state.running and st.session_state.session_id:
                stats = st.session_state.alert_system.get_stats()
                end_session(
                    st.session_state.session_id,
                    st.session_state.total_frames,
                    st.session_state.drowsy_frames,
                    stats["total_alerts"], stats["total_yawns"],
                    stats["average_ear"], stats["average_mar"],
                )
            st.session_state.running = False
    with c3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.alert_system.reset()
            st.session_state.ear_history.clear()
            st.session_state.mar_history.clear()

    # Layout
    col_video, col_stats = st.columns([3, 1])
    with col_video:
        video_ph = st.empty()
    with col_stats:
        status_ph = st.empty()
        ear_ph = st.empty()
        mar_ph = st.empty()
        score_ph = st.empty()
        fps_ph = st.empty()
        consec_ph = st.empty()

    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.markdown("#### 👁 EAR History")
        ear_chart = st.empty()
    with chart_c2:
        st.markdown("#### 👄 MAR History")
        mar_chart = st.empty()

    # Detection loop
    if st.session_state.running:
        try:
            import mediapipe as mp
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
        except ImportError:
            st.error("MediaPipe not installed")
            st.stop()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            st.session_state.running = False
            st.stop()

        alert_sys = st.session_state.alert_system
        alert_sys.ear_threshold = ear_thresh
        alert_sys.mar_threshold = mar_thresh
        alert_sys.consec_frames_threshold = consec_frames

        while st.session_state.running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            results = face_mesh.process(rgb)

            ear_val, mar_val = 0.30, 0.40

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0]
                left_eye, right_eye = extract_eye_landmarks(lm, w, h)
                mouth = extract_mouth_landmarks(lm, w, h)
                ear_val = compute_avg_ear(left_eye, right_eye)
                mar_val = compute_mar(mouth)

                if show_landmarks:
                    for eye_pts in [left_eye, right_eye]:
                        pts = eye_pts.astype(int)
                        for i in range(len(pts)):
                            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % len(pts)]), (0, 255, 128), 1)
                        for pt in pts:
                            cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1)

            status = alert_sys.update(ear=ear_val, mar=mar_val, cnn_confidence=0.0)
            score = compute_drowsiness_score(ear=ear_val, mar=mar_val, cnn_confidence=0.0)
            level = drowsiness_level(score)
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0

            st.session_state.ear_history.append(ear_val)
            st.session_state.mar_history.append(mar_val)
            st.session_state.total_frames += 1
            if status["is_drowsy"]:
                st.session_state.drowsy_frames += 1

            # Log events to DB (throttle: max once per 2 seconds)
            now = time.time()
            if now - st.session_state.last_event_time > 2 and st.session_state.session_id:
                if status["is_drowsy"]:
                    log_event(st.session_state.session_id, user["user_id"],
                              "drowsy", ear_val, mar_val, score, status["consecutive_frames"])
                    st.session_state.last_event_time = now
                elif status["is_yawning"]:
                    log_event(st.session_state.session_id, user["user_id"],
                              "yawn", ear_val, mar_val, score, status["consecutive_frames"])
                    st.session_state.last_event_time = now

            if show_overlay:
                frame = draw_overlay(frame, ear_val, mar_val, score, level, fps)

            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            status_ph.markdown(status_html(level), unsafe_allow_html=True)

            ec = "green" if ear_val > ear_thresh else "red"
            mc = "green" if mar_val < mar_thresh else "yellow"
            sc = "green" if score < 0.4 else ("yellow" if score < 0.7 else "red")

            ear_ph.markdown(metric_html("EAR", f"{ear_val:.3f}", ec), unsafe_allow_html=True)
            mar_ph.markdown(metric_html("MAR", f"{mar_val:.3f}", mc), unsafe_allow_html=True)
            score_ph.markdown(metric_html("Score", f"{score:.2f}", sc), unsafe_allow_html=True)
            fps_ph.markdown(metric_html("FPS", f"{fps:.0f}", "green"), unsafe_allow_html=True)
            consec_ph.markdown(metric_html("Consec", str(status["consecutive_frames"]),
                                           "red" if status["is_drowsy"] else "green"), unsafe_allow_html=True)

            if len(st.session_state.ear_history) > 2:
                ear_chart.line_chart(list(st.session_state.ear_history), height=150)
                mar_chart.line_chart(list(st.session_state.mar_history), height=150)

            time.sleep(0.01)

        cap.release()
    else:
        video_ph.markdown("""
        <div style="background:linear-gradient(145deg,#1a1a2e,#16213e);border-radius:14px;
        padding:4rem 2rem;text-align:center;border:2px dashed rgba(255,255,255,0.1);">
        <p style="font-size:3rem;margin:0;">📹</p>
        <p style="color:rgba(255,255,255,0.5);font-size:1.1rem;margin-top:1rem;">
        Click <strong>Start</strong> to begin monitoring</p></div>
        """, unsafe_allow_html=True)
        status_ph.markdown(status_html("ALERT"), unsafe_allow_html=True)
        for ph in [ear_ph, mar_ph, score_ph, fps_ph]:
            ph.markdown(metric_html("—", "—", "green"), unsafe_allow_html=True)
        consec_ph.markdown(metric_html("Consec", "0", "green"), unsafe_allow_html=True)


# ─── Admin Dashboard ─────────────────────────────────────────────────────────


def render_admin_dashboard():
    """Render the admin analytics dashboard."""
    user = st.session_state.user

    st.markdown("""
    <div class="app-header">
        <h1>🛡️ Admin Dashboard</h1>
        <p>System Administrator • Driver Monitoring Analytics</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### 🛡️ {user['full_name']}")
        st.caption("Role: Administrator")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Overview stats
    stats = get_dashboard_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="admin-stat"><h2 style="color:#4ECDC4">{stats["total_drivers"]}</h2>'
                f'<p>Registered Drivers</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="admin-stat"><h2 style="color:#45B7D1">{stats["total_sessions"]}</h2>'
                f'<p>Total Sessions</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="admin-stat"><h2 style="color:#e63757">{stats["total_alerts"]}</h2>'
                f'<p>Drowsy Alerts</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="admin-stat"><h2 style="color:#f6c343">{stats["total_yawns"]}</h2>'
                f'<p>Yawns Detected</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    tab_events, tab_sessions, tab_drivers, tab_db = st.tabs([
        "🚨 Events Log", "📋 Sessions", "👥 Drivers", "🗄️ Raw Database"
    ])

    # Driver filter
    drivers = get_all_drivers()
    driver_options = {"All Drivers": None}
    for d in drivers:
        driver_options[f"{d['full_name']} (@{d['username']})"] = d["id"]

    with tab_events:
        st.markdown("### 🚨 Drowsiness Events Log")
        sel = st.selectbox("Filter by Driver", list(driver_options.keys()), key="ev_filter")
        did = driver_options[sel]
        events = get_driver_events(driver_id=did, limit=500)
        if events:
            df = pd.DataFrame(events)
            display_cols = ["timestamp", "full_name", "event_type", "ear_value", "mar_value", "score", "consec"]
            available = [c for c in display_cols if c in df.columns]
            df_show = df[available].copy()
            df_show.columns = ["Timestamp", "Driver", "Event", "EAR", "MAR", "Score", "Consec"][:len(available)]

            # Color-code event types
            st.dataframe(df_show, use_container_width=True, height=400)

            drowsy_count = len(df[df["event_type"] == "drowsy"])
            yawn_count = len(df[df["event_type"] == "yawn"])
            st.markdown(f"**Total:** {len(events)} events | "
                        f"🔴 {drowsy_count} drowsy | 🟡 {yawn_count} yawns")
        else:
            st.info("No events recorded yet.")

    with tab_sessions:
        st.markdown("### 📋 Session History")
        sel2 = st.selectbox("Filter by Driver", list(driver_options.keys()), key="sess_filter")
        did2 = driver_options[sel2]
        sessions = get_all_sessions(driver_id=did2)
        if sessions:
            df_s = pd.DataFrame(sessions)
            display_cols = ["start_time", "end_time", "full_name", "total_frames",
                            "drowsy_frames", "total_alerts", "total_yawns", "avg_ear", "status"]
            available = [c for c in display_cols if c in df_s.columns]
            df_show = df_s[available]
            st.dataframe(df_show, use_container_width=True, height=400)
        else:
            st.info("No sessions recorded yet.")

    with tab_drivers:
        st.markdown("### 👥 Registered Drivers")
        if drivers:
            df_d = pd.DataFrame(drivers)
            st.dataframe(df_d, use_container_width=True)
        else:
            st.info("No drivers registered yet.")

    with tab_db:
        st.markdown("### 🗄️ Raw Database View")
        st.caption("Direct SQL query on the database")
        query = st.text_input("SQL Query", "SELECT * FROM events ORDER BY timestamp DESC LIMIT 50")
        if st.button("Run Query"):
            try:
                import sqlite3
                from app.database import DB_PATH
                conn = sqlite3.connect(str(DB_PATH))
                df_q = pd.read_sql_query(query, conn)
                conn.close()
                st.dataframe(df_q, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")


# ─── Main Router ─────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_auth_page()
elif st.session_state.user["role"] == "admin":
    render_admin_dashboard()
else:
    render_driver_dashboard()
