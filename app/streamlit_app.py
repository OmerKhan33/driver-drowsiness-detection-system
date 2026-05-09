"""
Driver Drowsiness Detection - Streamlit Application.

Features: Role-based login, Driver webcam dashboard, Admin control panel.
Run: python -m streamlit run app/streamlit_app.py
"""

import sys
import time
import threading

from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import cv2
import av
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

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
from src.classification.predict import DrowsinessPredictor

# ── Thread-safe drowsy flag (from cached module, survives Streamlit re-runs) ──
# Streamlit re-executes this script top-to-bottom on every refresh, so Events
# defined HERE would be recreated each time (breaking cross-thread refs).
# By importing from a separate module, Python's import cache keeps them alive.
from app._shared import drowsy_event as _drowsy_event, yawn_event as _yawn_event

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DrowsiGuard — Driver Safety System",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;600&display=swap');

:root{
    --bg:#0b1220;
    --bg-elev:#111a2e;
    --bg-card:#13203a;
    --border:rgba(148,163,184,0.10);
    --border-strong:rgba(148,163,184,0.18);
    --text:#e6edf7;
    --text-dim:#94a3b8;
    --text-mute:#64748b;
    --accent:#3b82f6;
    --accent-2:#60a5fa;
    --ok:#22c55e;
    --warn:#f59e0b;
    --danger:#ef4444;
}

*{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;}
.main .block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1440px;}
[data-testid="stHeader"]{background:transparent;}

[data-testid="stSidebar"]{
    background:#0a1020!important;
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] *{color:var(--text)!important;}
[data-testid="stSidebar"] .stMarkdown small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:var(--text-dim)!important;}

/* ─ Header ─ */
.hdr{
    background:var(--bg-elev);
    padding:1.4rem 1.8rem;border-radius:14px;margin-bottom:1.4rem;
    border:1px solid var(--border);
    display:flex;align-items:center;justify-content:space-between;gap:1rem;
}
.hdr .hdr-title{display:flex;align-items:center;gap:0.9rem;}
.hdr .hdr-mark{
    width:42px;height:42px;border-radius:10px;
    background:linear-gradient(135deg,var(--accent) 0%,#1d4ed8 100%);
    display:flex;align-items:center;justify-content:center;
    color:#fff;font-weight:700;font-size:1.05rem;letter-spacing:0.5px;
    box-shadow:0 4px 16px rgba(59,130,246,0.25);
}
.hdr h1{
    color:var(--text);font-size:1.35rem;font-weight:700;margin:0;
    letter-spacing:-0.2px;line-height:1.15;
}
.hdr p{color:var(--text-dim);font-size:0.82rem;margin:0.15rem 0 0;font-weight:400;}
.hdr-meta{
    color:var(--text-mute);font-size:0.72rem;font-weight:500;
    text-transform:uppercase;letter-spacing:1.5px;
}

/* ─ Metric Cards ─ */
.mc{
    background:var(--bg-card);
    border-radius:12px;padding:1rem 1.15rem;
    border:1px solid var(--border);
    transition:border-color 0.2s ease;
}
.mc:hover{border-color:var(--border-strong);}
.mc .lbl{
    color:var(--text-mute);font-size:0.68rem;text-transform:uppercase;
    letter-spacing:1.6px;font-weight:600;margin-bottom:0.45rem;
}
.mc .val{
    font-family:'JetBrains Mono',monospace;
    font-size:1.5rem;font-weight:600;letter-spacing:-0.5px;line-height:1;
}
.mc .val.g{color:var(--ok);}
.mc .val.y{color:var(--warn);}
.mc .val.r{color:var(--danger);}

/* ─ Status Pill ─ */
@keyframes pulse-danger{
    0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.5);}
    50%{box-shadow:0 0 0 8px rgba(239,68,68,0);}
}
.sp{
    display:inline-flex;align-items:center;gap:0.55rem;
    padding:0.5rem 1.4rem;border-radius:8px;
    font-weight:600;font-size:0.78rem;letter-spacing:1.6px;text-transform:uppercase;
    border:1px solid transparent;
}
.sp::before{
    content:'';width:8px;height:8px;border-radius:50%;
}
.sp.ok{background:rgba(34,197,94,0.10);color:#4ade80;border-color:rgba(34,197,94,0.25);}
.sp.ok::before{background:#22c55e;box-shadow:0 0 8px rgba(34,197,94,0.6);}
.sp.wn{background:rgba(245,158,11,0.10);color:#fbbf24;border-color:rgba(245,158,11,0.25);}
.sp.wn::before{background:#f59e0b;box-shadow:0 0 8px rgba(245,158,11,0.6);}
.sp.dg{background:rgba(239,68,68,0.12);color:#fca5a5;border-color:rgba(239,68,68,0.35);animation:pulse-danger 1.2s infinite;}
.sp.dg::before{background:#ef4444;box-shadow:0 0 10px rgba(239,68,68,0.8);}

/* ─ Login ─ */
.login-wrap{
    max-width:420px;margin:2rem auto 1rem;
    background:var(--bg-elev);
    border-radius:16px;padding:2rem 2.2rem 1rem;
    border:1px solid var(--border);
}
.login-wrap h2{
    color:var(--text);text-align:left;font-weight:700;font-size:1.35rem;
    margin:0 0 0.25rem;letter-spacing:-0.3px;
}
.login-wrap .sub{
    color:var(--text-dim);text-align:left;font-size:0.85rem;
    margin-bottom:0.5rem;font-weight:400;
}

/* ─ Admin Stat Card ─ */
.asc{
    background:var(--bg-card);
    border-radius:12px;padding:1.3rem 1.4rem;
    border:1px solid var(--border);
    transition:border-color 0.2s ease;
}
.asc:hover{border-color:var(--border-strong);}
.asc h2{
    font-family:'JetBrains Mono',monospace;
    font-size:2rem;font-weight:600;margin:0;letter-spacing:-0.8px;line-height:1;
}
.asc p{
    color:var(--text-mute);font-size:0.7rem;text-transform:uppercase;
    letter-spacing:1.6px;margin:0.55rem 0 0;font-weight:600;
}

/* ─ Section Headings ─ */
h1,h2,h3,h4{color:var(--text)!important;letter-spacing:-0.3px;}
h3{font-size:1.05rem!important;font-weight:600!important;margin-top:0.5rem!important;}

/* ─ Inputs ─ */
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"]>div{
    background:var(--bg-card)!important;
    color:var(--text)!important;
    border:1px solid var(--border)!important;
    border-radius:8px!important;
}
.stTextInput input:focus,.stTextArea textarea:focus{
    border-color:var(--accent)!important;
    box-shadow:0 0 0 3px rgba(59,130,246,0.15)!important;
}
label,.stSelectbox label,.stSlider label{
    color:var(--text-dim)!important;font-weight:500!important;font-size:0.82rem!important;
}

/* ─ Buttons ─ */
.stButton>button{
    background:var(--bg-card)!important;color:var(--text)!important;
    border:1px solid var(--border-strong)!important;border-radius:8px!important;
    font-weight:500!important;letter-spacing:0.2px!important;
    transition:all 0.18s ease!important;padding:0.55rem 1.1rem!important;
}
.stButton>button:hover{
    background:var(--bg-elev)!important;border-color:var(--accent)!important;
    color:#fff!important;
}
.stButton>button[kind="primary"]{
    background:var(--accent)!important;border-color:var(--accent)!important;color:#fff!important;
    font-weight:600!important;
}
.stButton>button[kind="primary"]:hover{
    background:#2563eb!important;border-color:#2563eb!important;
    box-shadow:0 4px 14px rgba(59,130,246,0.3)!important;
}

/* ─ Form submit buttons (st.form_submit_button) ─ */
[data-testid="stFormSubmitButton"]>button{
    background:var(--accent)!important;color:#fff!important;
    border:1px solid var(--accent)!important;border-radius:8px!important;
    font-weight:600!important;
}
[data-testid="stFormSubmitButton"]>button:hover{
    background:#2563eb!important;border-color:#2563eb!important;
}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{
    background:transparent!important;border-radius:6px 6px 0 0!important;
    border:none!important;color:var(--text-dim)!important;
    font-weight:500!important;font-size:0.88rem!important;
    padding:0.65rem 1.1rem!important;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--text)!important;background:rgba(148,163,184,0.05)!important;}
.stTabs [aria-selected="true"]{
    color:var(--accent-2)!important;
    border-bottom:2px solid var(--accent)!important;
    background:transparent!important;
}

/* ─ DataFrame ─ */
div[data-testid="stDataFrame"]{
    border-radius:10px;overflow:hidden;
    border:1px solid var(--border);
}

/* ─ Alerts (st.info / st.success / st.error / st.warning) ─ */
[data-testid="stAlert"]{
    background:var(--bg-card)!important;border-radius:8px!important;
    border-left-width:3px!important;
}

/* ─ Sliders ─ */
.stSlider [data-baseweb="slider"]>div>div{background:var(--accent)!important;}

hr{border-color:var(--border)!important;margin:0.8rem 0!important;}

/* ─ Hide Streamlit branding for cleaner look ─ */
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mc(label, value, color="g"):
    return f'<div class="mc"><div class="lbl">{label}</div><div class="val {color}">{value}</div></div>'


def sp(level):
    c = {"ALERT": "ok", "MILD": "wn", "DROWSY": "dg"}.get(level, "ok")
    return (
        f'<div class="mc" style="display:flex;flex-direction:column;justify-content:center;'
        f'align-items:flex-start;">'
        f'<div class="lbl">Status</div>'
        f'<div style="margin-top:0.4rem;"><span class="sp {c}">{level}</span></div>'
        f'</div>'
    )


def overlay(frame, ear, mar, score, level, fps, consec=0):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 50), (10, 10, 25), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    col = {"ALERT": (94, 234, 34), "MILD": (8, 179, 234), "DROWSY": (68, 68, 239)}.get(level, (94, 234, 34))
    cv2.putText(frame, f"EAR:{ear:.3f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"MAR:{mar:.3f}", (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"S:{score:.2f}", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"C:{consec}", (360, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
    cv2.putText(frame, level, (430, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w - 95, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), col, 3)
    return frame


# Default thresholds (admin can change via DB later)
# 0.23 is forgiving for users whose closed-eye EAR sits at 0.20–0.22 on MediaPipe.
DEFAULT_EAR = 0.23
DEFAULT_MAR = 0.60
DEFAULT_CONSEC = 4
DEFAULT_CNN = 0.65


# ── Auth Page ─────────────────────────────────────────────────────────────────

def render_auth_page():
    st.markdown("""
    <div class="hdr" style="justify-content:center;text-align:center;">
        <div class="hdr-title" style="flex-direction:column;align-items:center;gap:0.6rem;">
            <div class="hdr-mark">DG</div>
            <div>
                <h1>DrowsiGuard</h1>
                <p>Intelligent Driver Safety Monitoring</p>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    _, center, _ = st.columns([1, 2, 1])
    with center:
        tab_login, tab_signup = st.tabs(["Sign In", "Create Account"])

        with tab_login:
            st.markdown("""<div class="login-wrap">
                <h2>Welcome back</h2>
                <p class="sub">Sign in to access your dashboard.</p>
            </div>""", unsafe_allow_html=True)
            role = st.selectbox("Account type", ["Driver", "Admin"], key="login_role")
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
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
                <h2>Create driver account</h2>
                <p class="sub">Register to start monitoring your driving sessions.</p>
            </div>""", unsafe_allow_html=True)
            with st.form("signup_form"):
                full_name = st.text_input("Full name")
                new_user = st.text_input("Username")
                new_pass = st.text_input("Password", type="password", key="s_pass")
                confirm = st.text_input("Confirm password", type="password", key="s_conf")
                submitted = st.form_submit_button("Create account", use_container_width=True, type="primary")
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
                            st.success("Account created. Switch to Sign In to continue.")
                        else:
                            st.error(result["message"])


# ── Driver Dashboard ──────────────────────────────────────────────────────────

def render_driver_dashboard():
    user = st.session_state.user

    st.markdown(f"""<div class="hdr">
        <div class="hdr-title">
            <div class="hdr-mark">DG</div>
            <div>
                <h1>Driver Dashboard</h1>
                <p>{user['full_name']} &middot; Real-time monitoring</p>
            </div>
        </div>
        <div class="hdr-meta">Live</div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"""<div style="padding:0.4rem 0 0.8rem;">
            <div style="font-weight:600;font-size:1rem;color:#e6edf7;">{user['full_name']}</div>
            <div style="color:#94a3b8;font-size:0.78rem;margin-top:0.15rem;">@{user['username']} &middot; Driver</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

        st.markdown("---")
        st.markdown("**Display options**")
        show_lm = st.checkbox("Show landmarks", True)
        show_ov = st.checkbox("Show overlay", True)

        st.markdown("---")
        st.markdown("**Classification model**")
        model_choice = st.selectbox("Backbone", [
            "MobileNetV2", "CustomCNN", "ResNet18", "ResNet50", "VGG16", "EfficientNet-B0"
        ], label_visibility="collapsed")

    # Init state
    for key, default in [
        ("alert_system", DrowsinessAlertSystem(
            ear_threshold=DEFAULT_EAR, mar_threshold=DEFAULT_MAR,
            consec_frames=DEFAULT_CONSEC, alert_sound=False)),
        ("session_id", None),
        ("last_evt", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.session_id is None:
        if st.button("Start monitoring session", use_container_width=True, type="primary"):
            st.session_state.session_id = start_session(user["user_id"])
            st.rerun()
    else:
        if st.button("End session", use_container_width=True):
            if st.session_state.session_id:
                stats = st.session_state.alert_system.get_stats()
                end_session(st.session_state.session_id,
                            0, 0, stats["total_alerts"], stats["total_yawns"],
                            stats["average_ear"], stats["average_mar"])
                st.session_state.session_id = None
                st.session_state.alert_system.reset()
                st.rerun()

    st.markdown("### Live camera feed")
    st.info("Grant camera permissions when prompted. Video is processed in real time on the server.")

    # ── Setup MediaPipe (modern Tasks API — works with mediapipe ≥0.10.30) ──
    # The legacy `mp.solutions.face_mesh` namespace was removed in 0.10.21+,
    # so we use FaceLandmarker via `mp.tasks.vision`. Same 478-point landmark
    # set, same indices, just a different entry point.
    mp_available = False
    face_landmarker = None
    mp_module = None
    try:
        import mediapipe as mp_module
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision

        _model_path = Path(_PROJECT_ROOT) / "models" / "face_landmarker.task"
        if _model_path.exists():
            _options = _mp_vision.FaceLandmarkerOptions(
                base_options=_mp_python.BaseOptions(model_asset_path=str(_model_path)),
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=_mp_vision.RunningMode.IMAGE,
            )
            face_landmarker = _mp_vision.FaceLandmarker.create_from_options(_options)
            mp_available = True
            print(f"[INIT] FaceLandmarker loaded from {_model_path.name}")
        else:
            print(f"[INIT] Model file not found: {_model_path} — falling back to Haar")
    except Exception as e:
        print(f"[INIT] MediaPipe init failed: {e!r} — falling back to Haar")

    # Adapter so extract_*_landmarks() (which expects the legacy
    # `face_landmarks.landmark[idx].x` interface) still works on the new
    # `List[NormalizedLandmark]` returned by FaceLandmarker.
    class _LMAdapter:
        __slots__ = ("landmark",)

        def __init__(self, lm_list):
            self.landmark = lm_list

    try:
        weights_path = Path(_PROJECT_ROOT) / "models" / "weights" / f"{model_choice}_best.pt"
        predictor = DrowsinessPredictor(model_name=model_choice, weights_path=str(weights_path))
    except Exception:
        predictor = None

    face_cascade = None
    if not mp_available:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    asys = st.session_state.alert_system
    # Force-apply latest defaults each render so a stale alert_system from a
    # previous session_state still picks up new thresholds.
    asys.ear_threshold = DEFAULT_EAR
    asys.mar_threshold = DEFAULT_MAR
    asys.consec_frames_threshold = DEFAULT_CONSEC

    session_id = st.session_state.session_id
    user_id = user["user_id"]
    last_evt = [st.session_state.last_evt]
    lock = threading.Lock()
    cached_result = [{"ear": 0.30, "mar": 0.40, "cnn": 0.0,
                      "score": 0.0, "level": "ALERT", "consec": 0}]
    # EMA state for EAR / MAR — None means "no smoothed value yet"
    ema_state = {"ear": None, "mar": None}
    EMA_ALPHA = 0.55  # higher = more responsive, lower = smoother

    # CLAHE for improving eye region contrast (helps in low-light)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        t0 = time.perf_counter()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        # Process EVERY frame so consec_frames accumulates at full FPS;
        # detection used to skip every 2nd frame which doubled the time
        # needed before the alarm could fire.
        proc_scale = 0.75 if w > 480 else 1.0
        if proc_scale < 1.0:
            proc_img = cv2.resize(img, None, fx=proc_scale, fy=proc_scale)
        else:
            proc_img = img
        ph, pw = proc_img.shape[:2]

        # Apply CLAHE to improve contrast for eye detection
        lab = cv2.cvtColor(proc_img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        proc_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(proc_enhanced, cv2.COLOR_BGR2RGB)

        raw_ear, raw_mar = None, None
        cnn_conf = 0.0
        face_detected = False

        if mp_available and face_landmarker is not None:
            try:
                mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)
                res = face_landmarker.detect(mp_image)
            except Exception:
                res = None
            if res is not None and res.face_landmarks:
                face_detected = True
                lm = _LMAdapter(res.face_landmarks[0])
                le, re = extract_eye_landmarks(lm, pw, ph)
                mo = extract_mouth_landmarks(lm, pw, ph)
                raw_ear = compute_avg_ear(le, re)
                raw_mar = compute_mar(mo)

                if show_lm:
                    scale_x = w / pw
                    scale_y = h / ph
                    for pts in [le, re]:
                        pts_orig = (pts * [scale_x, scale_y]).astype(int)
                        for i in range(len(pts_orig)):
                            cv2.line(img, tuple(pts_orig[i]),
                                     tuple(pts_orig[(i + 1) % len(pts_orig)]),
                                     (0, 255, 128), 1)
                            cv2.circle(img, tuple(pts_orig[i]), 2,
                                       (100, 255, 200), -1)

                x_min = int(min(p.x for p in lm.landmark) * pw)
                y_min = int(min(p.y for p in lm.landmark) * ph)
                x_max = int(max(p.x for p in lm.landmark) * pw)
                y_max = int(max(p.y for p in lm.landmark) * ph)

                if predictor is not None:
                    try:
                        pad_x = int((x_max - x_min) * 0.15)
                        pad_y = int((y_max - y_min) * 0.15)
                        x1 = max(0, x_min - pad_x)
                        y1 = max(0, y_min - pad_y)
                        x2 = min(pw, x_max + pad_x)
                        y2 = min(ph, y_max + pad_y)
                        pred_res = predictor.predict_from_frame(
                            proc_img, (x1, y1, x2, y2))
                        cnn_conf = pred_res["probabilities"]["drowsy"]
                    except Exception:
                        pass
        else:
            gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                face_detected = True
                x, y, w_f, h_f = faces[0]
                if predictor is not None:
                    try:
                        pred_res = predictor.predict_from_frame(
                            proc_img, (x, y, x + w_f, y + h_f))
                        cnn_conf = pred_res["probabilities"]["drowsy"]
                    except Exception:
                        pass
                # Without MediaPipe we can't get EAR/MAR — synthesize from CNN
                if cnn_conf > 0.6:
                    raw_ear = 0.10
                    raw_mar = 0.40
                else:
                    raw_ear = 0.30
                    raw_mar = 0.40
                sx, sy = w / pw, h / ph
                cv2.rectangle(img,
                              (int(x * sx), int(y * sy)),
                              (int((x + w_f) * sx), int((y + h_f) * sy)),
                              (255, 0, 0), 2)

        # ── EMA smoothing — only update when a face is actually detected ──
        if face_detected and raw_ear is not None and raw_mar is not None:
            prev_ear = ema_state["ear"]
            prev_mar = ema_state["mar"]
            ema_state["ear"] = raw_ear if prev_ear is None else (EMA_ALPHA * raw_ear + (1 - EMA_ALPHA) * prev_ear)
            ema_state["mar"] = raw_mar if prev_mar is None else (EMA_ALPHA * raw_mar + (1 - EMA_ALPHA) * prev_mar)
            ear_v = ema_state["ear"]
            mar_v = ema_state["mar"]
        else:
            # No face — reset smoothing so a brief glance away doesn't poison the EMA
            ema_state["ear"] = None
            ema_state["mar"] = None
            ear_v = 0.30
            mar_v = 0.40

        # Thread-safe update
        with lock:
            status = asys.update(ear=ear_v, mar=mar_v,
                                 cnn_confidence=cnn_conf)
            score = compute_drowsiness_score(
                ear=ear_v, mar=mar_v, cnn_confidence=cnn_conf)
            score_level = drowsiness_level(score)

            # The displayed level must follow the alert system's authoritative
            # flags — without a CNN signal the score caps around 0.50 (MILD)
            # even with eyes fully shut, but `is_drowsy` (consec frames over
            # threshold) is the truth source for whether to alarm.
            if status["is_drowsy"]:
                level = "DROWSY"
            elif status["is_yawning"] and score_level == "ALERT":
                level = "MILD"
            else:
                level = score_level

            cached_result[0] = {"ear": ear_v, "mar": mar_v,
                                "cnn": cnn_conf, "score": score,
                                "level": level,
                                "consec": status["consecutive_frames"]}

            now = time.time()
            if session_id and (now - last_evt[0]) > 2:
                logged = False
                if status["is_drowsy"]:
                    log_event(session_id, user_id, "drowsy",
                              ear_v, mar_v, score,
                              status["consecutive_frames"])
                    logged = True
                if status["is_yawning"]:
                    log_event(session_id, user_id, "yawn",
                              ear_v, mar_v, score,
                              status["consecutive_frames"])
                    logged = True
                if logged:
                    last_evt[0] = now

            # Set shared flag for alarm (thread-safe Event)
            if status["is_drowsy"]:
                _drowsy_event.set()
            else:
                _drowsy_event.clear()
            if status["is_yawning"]:
                _yawn_event.set()
            else:
                _yawn_event.clear()

        cr = cached_result[0]
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        if show_ov:
            img = overlay(img, cr["ear"], cr["mar"],
                          cr["score"], cr["level"], fps,
                          cr.get("consec", 0))

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="drowsiness-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                      "frameRate": {"ideal": 15}},
            "audio": False},
        async_processing=True,
    )

    # ── Metrics display (updated via auto-refresh) ──
    if webrtc_ctx.state.playing:
        # Read thread-safe flags (NOT session_state — that's not thread-safe)
        is_drowsy_now = _drowsy_event.is_set()
        is_yawning_now = _yawn_event.is_set()

        # Auto-refresh so the main thread polls the drowsy flag.
        # 1000 ms gives a tighter alarm cadence than the previous 1500 ms.
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=1000, limit=0, key="alarm_refresh")

        cr = cached_result[0]
        ear_c = "r" if cr["ear"] < DEFAULT_EAR else "g"
        mar_c = "r" if cr["mar"] > DEFAULT_MAR else "g"
        level = cr["level"]

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(mc("Eye aspect ratio", f"{cr['ear']:.3f}", ear_c), unsafe_allow_html=True)
        m2.markdown(mc("Mouth aspect ratio", f"{cr['mar']:.3f}", mar_c), unsafe_allow_html=True)
        m3.markdown(mc("Drowsiness score", f"{cr['score']:.2f}", "r" if cr['score'] > 0.6 else "g"),
                    unsafe_allow_html=True)
        m4.markdown(sp(level), unsafe_allow_html=True)

        # ── Browser-side alarm (Web Audio API) ──
        # Beeps are scheduled directly on the AudioContext clock, NOT via
        # setInterval / setTimeout. The streamlit auto-refresh destroys this
        # iframe every ~1s, which would kill any JS timers; but the
        # AudioContext lives on `window.parent` and survives, so already-
        # scheduled audio events fire correctly. Each iframe load schedules
        # ~1s worth of beeps if alarm is active — yielding a continuous tone.
        # Three statuses: DROWSY (urgent — high beeps), YAWN (lower beeps),
        # OK (silent).
        if is_drowsy_now:
            alarm_status = "DROWSY"
        elif is_yawning_now:
            alarm_status = "YAWN"
        else:
            alarm_status = "OK"
        import streamlit.components.v1 as components
        components.html(f"""
        <div id="drowsi-status" style="display:none">{alarm_status}</div>
        <div id="alarm-btn-wrap" style="text-align:center;margin:6px 0;">
            <button id="enable-sound-btn" onclick="initAudio()" style="
                background:#3b82f6;color:#fff;
                border:none;border-radius:8px;padding:9px 22px;font-size:13px;
                font-weight:600;cursor:pointer;letter-spacing:0.3px;
                font-family:'Inter',-apple-system,sans-serif;
                box-shadow:0 4px 14px rgba(59,130,246,0.3);
            ">Enable alarm sound</button>
        </div>
        <script>
        var win = window.parent || window;

        // ── Piercing siren — frequency sweep from 800 Hz to 1600 Hz ──
        // A wail/yelp pattern is FAR more attention-grabbing than steady
        // beeps, mirroring the way emergency siren tones are designed to
        // disrupt drowsy/inattentive states. Sawtooth is harsher than square
        // and harmonically rich, hitting the brainstem like a vehicle horn.
        function _scheduleSiren(ctx, whenSec, fStart, fEnd, duration) {{
            try {{
                var t = ctx.currentTime + whenSec;

                // Two detuned oscillators stacked = phaser-like roughness.
                // Plus a third high oscillator for piercing harmonics.
                var osc1 = ctx.createOscillator();
                var osc2 = ctx.createOscillator();
                var osc3 = ctx.createOscillator();
                var gain = ctx.createGain();

                osc1.type = 'sawtooth';
                osc2.type = 'square';
                osc3.type = 'triangle';

                osc1.frequency.setValueAtTime(fStart, t);
                osc1.frequency.exponentialRampToValueAtTime(fEnd, t + duration);
                osc2.frequency.setValueAtTime(fStart * 1.005, t);  // slight detune
                osc2.frequency.exponentialRampToValueAtTime(fEnd * 1.005, t + duration);
                osc3.frequency.setValueAtTime(fStart * 2, t);  // octave up — piercing
                osc3.frequency.exponentialRampToValueAtTime(fEnd * 2, t + duration);

                osc1.connect(gain);
                osc2.connect(gain);
                osc3.connect(gain);
                gain.connect(ctx.destination);

                // Loud, fast attack — no gentle ramp. Sustain near full volume.
                gain.gain.setValueAtTime(0.0001, t);
                gain.gain.linearRampToValueAtTime(0.85, t + 0.005);
                gain.gain.setValueAtTime(0.85, t + duration - 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);

                osc1.start(t); osc2.start(t); osc3.start(t);
                osc1.stop(t + duration + 0.02);
                osc2.stop(t + duration + 0.02);
                osc3.stop(t + duration + 0.02);
            }} catch(e) {{ console.error('Siren schedule error:', e); }}
        }}

        function _scheduleTone(ctx, whenSec, freq, duration, volume) {{
            try {{
                var t = ctx.currentTime + whenSec;
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain); gain.connect(ctx.destination);
                osc.type = 'sawtooth';
                osc.frequency.value = freq;
                gain.gain.setValueAtTime(0.0001, t);
                gain.gain.linearRampToValueAtTime(volume, t + 0.01);
                gain.gain.setValueAtTime(volume, t + duration - 0.02);
                gain.gain.exponentialRampToValueAtTime(0.0001, t + duration);
                osc.start(t);
                osc.stop(t + duration + 0.02);
            }} catch(e) {{ console.error('Tone error:', e); }}
        }}

        function _resumeThen(fn) {{
            var ctx = win._drowsiAudioCtx;
            if (!ctx) return;
            if (ctx.state === 'suspended') {{
                ctx.resume().then(fn).catch(function(e){{ console.error('Resume failed:', e); }});
            }} else {{
                fn();
            }}
        }}

        function initAudio() {{
            if (!win._drowsiAudioCtx) {{
                win._drowsiAudioCtx = new (win.AudioContext || win.webkitAudioContext)();
            }}
            if (win._drowsiAudioCtx.state === 'suspended') {{
                win._drowsiAudioCtx.resume();
            }}
            win._drowsiSoundEnabled = true;
            // Confirmation: short ascending chirp at lower volume
            var ctx = win._drowsiAudioCtx;
            _scheduleTone(ctx, 0.00, 660, 0.10, 0.25);
            _scheduleTone(ctx, 0.12, 990, 0.14, 0.25);
            var wrap = document.getElementById('alarm-btn-wrap');
            if (wrap) wrap.innerHTML = '<span style="color:#22c55e;font-size:12px;font-weight:600;letter-spacing:0.3px;">Alarm sound enabled</span>';
        }}

        if (win._drowsiSoundEnabled) {{
            var wrap = document.getElementById('alarm-btn-wrap');
            if (wrap) wrap.innerHTML = '<span style="color:#22c55e;font-size:12px;font-weight:600;letter-spacing:0.3px;">Alarm sound enabled</span>';
        }}

        var el = document.getElementById('drowsi-status');
        var status = el ? el.textContent.trim() : 'OK';
        if (win._drowsiSoundEnabled) {{
            _resumeThen(function() {{
                var ctx = win._drowsiAudioCtx;
                if (status === 'DROWSY') {{
                    // Yelp siren — fast frequency sweeps, 4 wails per second.
                    // 800→1600 Hz sweep is in the most piercing band of human
                    // hearing and matches the emergency-vehicle yelp pattern.
                    _scheduleSiren(ctx, 0.00, 800, 1600, 0.22);
                    _scheduleSiren(ctx, 0.25, 800, 1600, 0.22);
                    _scheduleSiren(ctx, 0.50, 800, 1600, 0.22);
                    _scheduleSiren(ctx, 0.75, 800, 1600, 0.22);
                }} else if (status === 'YAWN') {{
                    // Warning chirp — slower sweep, lower band, less aggressive
                    _scheduleSiren(ctx, 0.00, 500, 800, 0.30);
                    _scheduleSiren(ctx, 0.55, 500, 800, 0.30);
                }}
            }});
        }}
        </script>
        """, height=50)


# ── Admin Dashboard ───────────────────────────────────────────────────────────

def render_admin_dashboard():
    user = st.session_state.user

    st.markdown("""<div class="hdr">
        <div class="hdr-title">
            <div class="hdr-mark">DG</div>
            <div>
                <h1>Admin Control Panel</h1>
                <p>Monitor &middot; Configure &middot; Analyze</p>
            </div>
        </div>
        <div class="hdr-meta">Administrator</div>
    </div>""", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"""<div style="padding:0.4rem 0 0.8rem;">
            <div style="font-weight:600;font-size:1rem;color:#e6edf7;">{user['full_name']}</div>
            <div style="color:#94a3b8;font-size:0.78rem;margin-top:0.15rem;">Administrator</div>
        </div>""", unsafe_allow_html=True)
        if st.button("Sign out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        st.markdown("---")
        st.markdown("**Detection thresholds**")
        st.slider("EAR threshold", 0.15, 0.35, DEFAULT_EAR, 0.01,
                  help="Eyes considered closed when below this value", key="adm_ear")
        st.slider("MAR threshold", 0.40, 0.80, DEFAULT_MAR, 0.05,
                  help="Yawn detected when above this value", key="adm_mar")
        st.slider("Consecutive frames", 5, 30, DEFAULT_CONSEC, 1, key="adm_consec")
        st.slider("CNN confidence", 0.40, 0.90, DEFAULT_CNN, 0.05, key="adm_cnn")

    # Stats
    stats = get_dashboard_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="asc"><h2 style="color:#22c55e">{stats["total_drivers"]}</h2>'
                '<p>Registered drivers</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="asc"><h2 style="color:#3b82f6">{stats["total_sessions"]}</h2>'
                '<p>Total sessions</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="asc"><h2 style="color:#ef4444">{stats["total_alerts"]}</h2>'
                '<p>Drowsy alerts</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="asc"><h2 style="color:#eab308">{stats["total_yawns"]}</h2>'
                '<p>Yawns detected</p></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    drivers = get_all_drivers()
    d_opts = {"All drivers": None}
    for d in drivers:
        d_opts[f"{d['full_name']} (@{d['username']})"] = d["id"]

    tab1, tab2, tab3, tab4 = st.tabs(["Events", "Sessions", "Drivers", "Query"])

    with tab1:
        st.markdown("### Drowsiness events")
        sel = st.selectbox("Filter by driver", list(d_opts.keys()), key="ev_f")
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
            st.caption(f"{len(events)} events total &nbsp;·&nbsp; {dc} drowsy &nbsp;·&nbsp; {yc} yawns")
        else:
            st.info("No events recorded yet.")

    with tab2:
        st.markdown("### Session history")
        sel2 = st.selectbox("Filter by driver", list(d_opts.keys()), key="ss_f")
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
        st.markdown("### Registered drivers")
        if drivers:
            st.dataframe(pd.DataFrame(drivers), use_container_width=True)
        else:
            st.info("No drivers registered.")

    with tab4:
        st.markdown("### Raw database query")
        query = st.text_input("SQL", "SELECT * FROM events ORDER BY timestamp DESC LIMIT 50")
        if st.button("Run query"):
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
