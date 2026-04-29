"""
Driver Drowsiness Detection - Streamlit Application.

Features: Role-based login, Driver webcam dashboard, Admin control panel.
Run: python -m streamlit run app/streamlit_app.py
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
    compute_avg_ear,
    compute_drowsiness_score,
    compute_mar,
    drowsiness_level,
    extract_eye_landmarks,
    extract_mouth_landmarks,
)

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DrowsiGuard - Driver Safety System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 0.8rem; max-width: 1440px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a1a 0%, #111827 100%); }

/* Header */
.hdr {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 40%, #0f172a 100%);
    padding: 1.6rem 2.2rem; border-radius: 18px; margin-bottom: 1.2rem;
    border: 1px solid rgba(59,130,246,0.15);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    position: relative; overflow: hidden;
}
.hdr::before {
    content: ''; position: absolute; top: -50%; right: -30%; width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 70%);
}
.hdr h1 { color: #f1f5f9; font-size: 1.75rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hdr p { color: rgba(148,163,184,0.8); font-size: 0.9rem; margin: 0.25rem 0 0 0; }

/* Metric cards */
.mc {
    background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,58,95,0.6));
    border-radius: 16px; padding: 1rem 1.2rem; text-align: center;
    border: 1px solid rgba(59,130,246,0.1);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px); transition: all 0.3s ease;
}
.mc:hover { border-color: rgba(59,130,246,0.3); transform: translateY(-1px); }
.mc .lbl { color: #94a3b8; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; }
.mc .val { font-size: 1.6rem; font-weight: 800; margin: 0.2rem 0; letter-spacing: -0.5px; }
.mc .val.g { color: #22c55e; } .mc .val.y { color: #eab308; } .mc .val.r { color: #ef4444; }

/* Status pill */
.sp {
    display: inline-block; padding: 0.45rem 2rem; border-radius: 50px;
    font-weight: 800; font-size: 0.95rem; letter-spacing: 2px; text-transform: uppercase;
}
.sp.ok { background: linear-gradient(135deg, #22c55e, #16a34a); color: #fff; box-shadow: 0 4px 15px rgba(34,197,94,0.3); }
.sp.wn { background: linear-gradient(135deg, #eab308, #ca8a04); color: #0f172a; box-shadow: 0 4px 15px rgba(234,179,8,0.3); }
.sp.dg { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; animation: pls 1.2s infinite; box-shadow: 0 4px 20px rgba(239,68,68,0.4); }
@keyframes pls { 0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); } 50% { box-shadow: 0 0 25px 8px rgba(239,68,68,0.15); } }

/* Login */
.login-wrap {
    max-width: 440px; margin: 2rem auto;
    background: linear-gradient(145deg, rgba(15,23,42,0.95), rgba(30,58,95,0.7));
    border-radius: 24px; padding: 2.5rem; border: 1px solid rgba(59,130,246,0.12);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5); backdrop-filter: blur(20px);
}
.login-wrap h2 { color: #f1f5f9; text-align: center; font-weight: 800; font-size: 1.4rem; margin-bottom: 0.3rem; }
.login-wrap .sub { color: #64748b; text-align: center; font-size: 0.85rem; margin-bottom: 1.5rem; }

/* Admin stat card */
.asc {
    background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,58,95,0.6));
    border-radius: 16px; padding: 1.5rem; text-align: center;
    border: 1px solid rgba(59,130,246,0.1); box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.asc h2 { font-size: 2.4rem; font-weight: 800; margin: 0; }
.asc p { color: #94a3b8; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; margin: 0.3rem 0 0 0; }

/* Idle cam */
.idle-cam {
    background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(30,58,95,0.4));
    border-radius: 18px; padding: 5rem 2rem; text-align: center;
    border: 2px dashed rgba(59,130,246,0.15);
}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mc(label, value, color="g"):
    return f'<div class="mc"><div class="lbl">{label}</div><div class="val {color}">{value}</div></div>'


def sp(level):
    c = {"ALERT": "ok", "MILD": "wn", "DROWSY": "dg"}.get(level, "ok")
    return f'<div style="text-align:center"><span class="sp {c}">{level}</span></div>'


def overlay(frame, ear, mar, score, level, fps):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 50), (10, 10, 25), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    col = {"ALERT": (94, 234, 34), "MILD": (8, 179, 234), "DROWSY": (68, 68, 239)}.get(level, (94, 234, 34))
    cv2.putText(frame, f"EAR:{ear:.3f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR:{mar:.3f}", (155, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Score:{score:.2f}", (298, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, level, (460, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w - 95, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 3)
    return frame


# Default thresholds (admin can change via DB later)
DEFAULT_EAR = 0.25
DEFAULT_MAR = 0.60
DEFAULT_CONSEC = 15
DEFAULT_CNN = 0.65


# ── Auth Page ─────────────────────────────────────────────────────────────────

def render_auth_page():
    st.markdown("""
    <div class="hdr" style="text-align:center">
        <h1>🛡️ DrowsiGuard</h1>
        <p>Intelligent Driver Safety Monitoring System</p>
    </div>""", unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["Login", "Create Account"])

    with tab_login:
        st.markdown("""<div class="login-wrap">
            <h2>Welcome Back</h2>
            <p class="sub">Sign in to your account</p>
        </div>""", unsafe_allow_html=True)
        role = st.selectbox("Login as", ["Driver", "Admin"], key="login_role")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")
            if submitted and username and password:
                result = authenticate_user(username, password)
                if not result["success"]:
                    st.error(result["message"])
                elif result["role"] != role.lower():
                    st.error(f"This account is not registered as {role}.")
                else:
                    st.session_state.user = result
                    st.session_state.logged_in = True
                    st.rerun()

    with tab_signup:
        st.markdown("""<div class="login-wrap">
            <h2>Driver Registration</h2>
            <p class="sub">Create a new driver account</p>
        </div>""", unsafe_allow_html=True)
        with st.form("signup_form"):
            full_name = st.text_input("Full Name")
            new_user = st.text_input("Choose Username")
            new_pass = st.text_input("Password", type="password", key="s_pass")
            confirm = st.text_input("Confirm Password", type="password", key="s_conf")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if not all([full_name, new_user, new_pass, confirm]):
                    st.warning("Please fill all fields.")
                elif new_pass != confirm:
                    st.error("Passwords do not match.")
                elif len(new_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                else:
                    result = create_user(new_user, new_pass, full_name)
                    if result["success"]:
                        st.success("Account created! Go to Login tab.")
                    else:
                        st.error(result["message"])


# ── Driver Dashboard ──────────────────────────────────────────────────────────

def render_driver_dashboard():
    user = st.session_state.user

    st.markdown(f"""<div class="hdr">
        <h1>🚗 Driver Dashboard</h1>
        <p>Welcome, {user['full_name']} &bull; Real-time Monitoring Active</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### 👤 {user['full_name']}")
        st.caption(f"@{user['username']} | Driver")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("---")
        show_lm = st.checkbox("Show Landmarks", True)
        show_ov = st.checkbox("Show Overlay", True)

    # Init state
    for key, default in [
        ("alert_system", DrowsinessAlertSystem(
            ear_threshold=DEFAULT_EAR, mar_threshold=DEFAULT_MAR,
            consec_frames=DEFAULT_CONSEC, alert_sound=False)),
        ("ear_hist", deque(maxlen=100)),
        ("mar_hist", deque(maxlen=100)),
        ("total_frames", 0), ("drowsy_frames", 0),
        ("running", False), ("session_id", None), ("last_evt", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Controls
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶  Start Monitoring", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.session_state.session_id = start_session(user["user_id"])
            st.session_state.total_frames = 0
            st.session_state.drowsy_frames = 0
            st.session_state.alert_system.reset()
            st.session_state.ear_hist.clear()
            st.session_state.mar_hist.clear()
    with c2:
        if st.button("⏹  Stop", use_container_width=True):
            if st.session_state.running and st.session_state.session_id:
                stats = st.session_state.alert_system.get_stats()
                end_session(st.session_state.session_id,
                            st.session_state.total_frames,
                            st.session_state.drowsy_frames,
                            stats["total_alerts"], stats["total_yawns"],
                            stats["average_ear"], stats["average_mar"])
            st.session_state.running = False

    # Layout
    col_v, col_s = st.columns([3, 1])
    with col_v:
        vid = st.empty()
    with col_s:
        stat_ph = st.empty()
        ear_ph = st.empty()
        mar_ph = st.empty()
        scr_ph = st.empty()
        fps_ph = st.empty()
        con_ph = st.empty()

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("#### 👁 EAR History")
        ear_ch = st.empty()
    with cc2:
        st.markdown("#### 👄 MAR History")
        mar_ch = st.empty()

    if st.session_state.running:
        try:
            import mediapipe as mp_mod
            face_mesh = mp_mod.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5)
        except ImportError:
            st.error("MediaPipe not installed.")
            st.stop()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam.")
            st.session_state.running = False
            st.stop()

        asys = st.session_state.alert_system

        while st.session_state.running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            res = face_mesh.process(rgb)
            ear_v, mar_v = 0.30, 0.40

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                le, re = extract_eye_landmarks(lm, w, h)
                mo = extract_mouth_landmarks(lm, w, h)
                ear_v = compute_avg_ear(le, re)
                mar_v = compute_mar(mo)
                if show_lm:
                    for pts in [le, re]:
                        pts_i = pts.astype(int)
                        for i in range(len(pts_i)):
                            cv2.line(frame, tuple(pts_i[i]),
                                     tuple(pts_i[(i + 1) % len(pts_i)]), (0, 255, 128), 1)

            status = asys.update(ear=ear_v, mar=mar_v, cnn_confidence=0.0)
            score = compute_drowsiness_score(ear=ear_v, mar=mar_v, cnn_confidence=0.0)
            level = drowsiness_level(score)
            elapsed = time.perf_counter() - t0
            fps = 1.0 / elapsed if elapsed > 0 else 0

            st.session_state.ear_hist.append(ear_v)
            st.session_state.mar_hist.append(mar_v)
            st.session_state.total_frames += 1
            if status["is_drowsy"]:
                st.session_state.drowsy_frames += 1

            now = time.time()
            if now - st.session_state.last_evt > 2 and st.session_state.session_id:
                if status["is_drowsy"]:
                    log_event(st.session_state.session_id, user["user_id"],
                              "drowsy", ear_v, mar_v, score, status["consecutive_frames"])
                    st.session_state.last_evt = now
                elif status["is_yawning"]:
                    log_event(st.session_state.session_id, user["user_id"],
                              "yawn", ear_v, mar_v, score, status["consecutive_frames"])
                    st.session_state.last_evt = now

            if show_ov:
                frame = overlay(frame, ear_v, mar_v, score, level, fps)

            vid.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            stat_ph.markdown(sp(level), unsafe_allow_html=True)
            ec = "g" if ear_v > DEFAULT_EAR else "r"
            mco = "g" if mar_v < DEFAULT_MAR else "y"
            sco = "g" if score < 0.4 else ("y" if score < 0.7 else "r")
            ear_ph.markdown(mc("EAR", f"{ear_v:.3f}", ec), unsafe_allow_html=True)
            mar_ph.markdown(mc("MAR", f"{mar_v:.3f}", mco), unsafe_allow_html=True)
            scr_ph.markdown(mc("Score", f"{score:.2f}", sco), unsafe_allow_html=True)
            fps_ph.markdown(mc("FPS", f"{fps:.0f}", "g"), unsafe_allow_html=True)
            con_ph.markdown(mc("Consec", str(status["consecutive_frames"]),
                               "r" if status["is_drowsy"] else "g"), unsafe_allow_html=True)
            if len(st.session_state.ear_hist) > 2:
                ear_ch.line_chart(list(st.session_state.ear_hist), height=140)
                mar_ch.line_chart(list(st.session_state.mar_hist), height=140)
            time.sleep(0.01)
        cap.release()
    else:
        vid.markdown("""<div class="idle-cam">
            <p style="font-size:3.5rem;margin:0;">📹</p>
            <p style="color:#94a3b8;font-size:1.1rem;margin-top:1rem;">
            Click <strong>Start Monitoring</strong> to begin</p>
        </div>""", unsafe_allow_html=True)
        stat_ph.markdown(sp("ALERT"), unsafe_allow_html=True)
        for ph in [ear_ph, mar_ph, scr_ph, fps_ph]:
            ph.markdown(mc("--", "--", "g"), unsafe_allow_html=True)
        con_ph.markdown(mc("Consec", "0", "g"), unsafe_allow_html=True)


# ── Admin Dashboard ───────────────────────────────────────────────────────────

def render_admin_dashboard():
    user = st.session_state.user

    st.markdown("""<div class="hdr">
        <h1>🛡️ Admin Control Panel</h1>
        <p>System Administrator &bull; Monitor &bull; Configure &bull; Analyze</p>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"### 🛡️ {user['full_name']}")
        st.caption("Administrator")
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("---")
        st.markdown("**Detection Thresholds**")
        st.slider("EAR Threshold", 0.15, 0.35, DEFAULT_EAR, 0.01,
                  help="Eyes closed below this", key="adm_ear")
        st.slider("MAR Threshold", 0.40, 0.80, DEFAULT_MAR, 0.05,
                  help="Yawning above this", key="adm_mar")
        st.slider("Consecutive Frames", 5, 30, DEFAULT_CONSEC, 1, key="adm_consec")
        st.slider("CNN Confidence", 0.40, 0.90, DEFAULT_CNN, 0.05, key="adm_cnn")

    # Stats
    stats = get_dashboard_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="asc"><h2 style="color:#22c55e">{stats["total_drivers"]}</h2>'
                '<p>Registered Drivers</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="asc"><h2 style="color:#3b82f6">{stats["total_sessions"]}</h2>'
                '<p>Total Sessions</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="asc"><h2 style="color:#ef4444">{stats["total_alerts"]}</h2>'
                '<p>Drowsy Alerts</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="asc"><h2 style="color:#eab308">{stats["total_yawns"]}</h2>'
                '<p>Yawns Detected</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    drivers = get_all_drivers()
    d_opts = {"All Drivers": None}
    for d in drivers:
        d_opts[f"{d['full_name']} (@{d['username']})"] = d["id"]

    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Events Log", "📋 Sessions", "👥 Drivers", "🗄️ Raw SQL"])

    with tab1:
        st.markdown("### 🚨 Drowsiness Events")
        sel = st.selectbox("Filter by Driver", list(d_opts.keys()), key="ev_f")
        events = get_driver_events(driver_id=d_opts[sel], limit=500)
        if events:
            df = pd.DataFrame(events)
            cols = ["timestamp", "full_name", "event_type", "ear_value", "mar_value", "score", "consec"]
            avail = [c for c in cols if c in df.columns]
            df_show = df[avail].copy()
            df_show.columns = ["Timestamp", "Driver", "Event", "EAR", "MAR", "Score", "Consec"][:len(avail)]
            st.dataframe(df_show, use_container_width=True, height=400)
            dc = len(df[df["event_type"] == "drowsy"])
            yc = len(df[df["event_type"] == "yawn"])
            st.markdown(f"**Total:** {len(events)} events | "
                        f"🔴 {dc} drowsy | 🟡 {yc} yawns")
        else:
            st.info("No events recorded yet.")

    with tab2:
        st.markdown("### 📋 Session History")
        sel2 = st.selectbox("Filter by Driver", list(d_opts.keys()), key="ss_f")
        sessions = get_all_sessions(driver_id=d_opts[sel2])
        if sessions:
            df_s = pd.DataFrame(sessions)
            cols = ["start_time", "end_time", "full_name", "total_frames",
                    "drowsy_frames", "total_alerts", "total_yawns", "avg_ear", "status"]
            avail = [c for c in cols if c in df_s.columns]
            st.dataframe(df_s[avail], use_container_width=True, height=400)
        else:
            st.info("No sessions yet.")

    with tab3:
        st.markdown("### 👥 Registered Drivers")
        if drivers:
            st.dataframe(pd.DataFrame(drivers), use_container_width=True)
        else:
            st.info("No drivers registered.")

    with tab4:
        st.markdown("### 🗄️ Raw Database Query")
        query = st.text_input("SQL", "SELECT * FROM events ORDER BY timestamp DESC LIMIT 50")
        if st.button("Run Query"):
            try:
                import sqlite3
                from app.database import DB_PATH
                conn = sqlite3.connect(str(DB_PATH))
                df_q = pd.read_sql_query(query, conn)
                conn.close()
                st.dataframe(df_q, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ── Router ────────────────────────────────────────────────────────────────────

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    render_auth_page()
elif st.session_state.user["role"] == "admin":
    render_admin_dashboard()
else:
    render_driver_dashboard()
