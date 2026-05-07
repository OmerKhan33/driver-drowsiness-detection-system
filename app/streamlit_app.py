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
    page_title="DrowsiGuard - Driver Safety System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif;}
html,body,[data-testid="stAppViewContainer"]{background:#060918!important;}
.main .block-container{padding-top:0.6rem;max-width:1480px;}
[data-testid="stSidebar"]{
    background:linear-gradient(195deg,#070b1e 0%,#0d1330 40%,#111c47 100%)!important;
    border-right:1px solid rgba(99,102,241,0.08);
}
[data-testid="stSidebar"] *{color:#c7d2e8!important;}

/* ─ Animated Header ─ */
@keyframes shimmer{0%{background-position:200% 50%;}100%{background-position:-200% 50%;}}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-6px);}}
.hdr{
    background:linear-gradient(135deg,#0a1128 0%,#1a2755 35%,#162040 65%,#0a1128 100%);
    padding:1.8rem 2.4rem;border-radius:20px;margin-bottom:1.4rem;
    border:1px solid rgba(99,102,241,0.12);
    box-shadow:0 12px 48px rgba(0,0,0,0.5),0 0 0 1px rgba(99,102,241,0.05) inset;
    position:relative;overflow:hidden;
}
.hdr::before{
    content:'';position:absolute;top:-60%;right:-20%;width:500px;height:500px;
    background:radial-gradient(circle,rgba(99,102,241,0.07) 0%,transparent 65%);
    animation:float 6s ease-in-out infinite;
}
.hdr::after{
    content:'';position:absolute;bottom:-40%;left:-10%;width:350px;height:350px;
    background:radial-gradient(circle,rgba(56,189,248,0.05) 0%,transparent 65%);
    animation:float 8s ease-in-out infinite reverse;
}
.hdr h1{
    color:#f1f5f9;font-size:1.85rem;font-weight:900;margin:0;
    letter-spacing:-0.5px;position:relative;z-index:1;
    background:linear-gradient(135deg,#e2e8f0,#93c5fd,#e2e8f0);
    background-size:400% 100%;-webkit-background-clip:text;-webkit-text-fill-color:transparent;
    animation:shimmer 6s ease-in-out infinite;
}
.hdr p{color:rgba(148,163,184,0.7);font-size:0.85rem;margin:0.3rem 0 0;position:relative;z-index:1;}

/* ─ Metric Cards ─ */
.mc{
    background:linear-gradient(145deg,rgba(13,19,48,0.95),rgba(26,39,85,0.7));
    border-radius:16px;padding:1.1rem 1.3rem;text-align:center;
    border:1px solid rgba(99,102,241,0.08);
    box-shadow:0 4px 24px rgba(0,0,0,0.3),0 0 0 1px rgba(99,102,241,0.03) inset;
    backdrop-filter:blur(12px);transition:all 0.35s cubic-bezier(0.4,0,0.2,1);
}
.mc:hover{border-color:rgba(99,102,241,0.25);transform:translateY(-3px);box-shadow:0 8px 32px rgba(99,102,241,0.1);}
.mc .lbl{color:#7c8db5;font-size:0.6rem;text-transform:uppercase;letter-spacing:2px;font-weight:700;}
.mc .val{font-size:1.65rem;font-weight:900;margin:0.2rem 0;letter-spacing:-0.5px;}
.mc .val.g{color:#34d399;text-shadow:0 0 20px rgba(52,211,153,0.2);}
.mc .val.y{color:#fbbf24;text-shadow:0 0 20px rgba(251,191,36,0.2);}
.mc .val.r{color:#f87171;text-shadow:0 0 20px rgba(248,113,113,0.2);}

/* ─ Status Pill ─ */
@keyframes pls{0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4);}50%{box-shadow:0 0 30px 10px rgba(239,68,68,0.12);}}
@keyframes pulse_ok{0%,100%{box-shadow:0 4px 15px rgba(52,211,153,0.3);}50%{box-shadow:0 4px 25px rgba(52,211,153,0.15);}}
.sp{
    display:inline-block;padding:0.5rem 2.2rem;border-radius:50px;
    font-weight:900;font-size:0.9rem;letter-spacing:2.5px;text-transform:uppercase;
    transition:all 0.3s ease;
}
.sp.ok{background:linear-gradient(135deg,#059669,#10b981);color:#fff;animation:pulse_ok 3s infinite;}
.sp.wn{background:linear-gradient(135deg,#d97706,#f59e0b);color:#0f172a;box-shadow:0 4px 18px rgba(245,158,11,0.25);}
.sp.dg{background:linear-gradient(135deg,#dc2626,#ef4444);color:#fff;animation:pls 1s infinite;}

/* ─ Login ─ */
.login-wrap{
    max-width:460px;margin:2.5rem auto;
    background:linear-gradient(145deg,rgba(13,19,48,0.97),rgba(26,39,85,0.75));
    border-radius:28px;padding:2.8rem;border:1px solid rgba(99,102,241,0.1);
    box-shadow:0 24px 72px rgba(0,0,0,0.5),0 0 0 1px rgba(99,102,241,0.05) inset;
    backdrop-filter:blur(24px);
}
.login-wrap h2{
    color:#f1f5f9;text-align:center;font-weight:900;font-size:1.5rem;margin-bottom:0.3rem;
    background:linear-gradient(135deg,#e2e8f0,#818cf8);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.login-wrap .sub{color:#64748b;text-align:center;font-size:0.82rem;margin-bottom:1.5rem;}

/* ─ Admin Stat Card ─ */
.asc{
    background:linear-gradient(145deg,rgba(13,19,48,0.95),rgba(26,39,85,0.7));
    border-radius:18px;padding:1.6rem;text-align:center;
    border:1px solid rgba(99,102,241,0.08);
    box-shadow:0 6px 28px rgba(0,0,0,0.3);
    transition:all 0.35s cubic-bezier(0.4,0,0.2,1);
}
.asc:hover{transform:translateY(-2px);border-color:rgba(99,102,241,0.2);}
.asc h2{font-size:2.6rem;font-weight:900;margin:0;letter-spacing:-1px;}
.asc p{color:#7c8db5;font-size:0.65rem;text-transform:uppercase;letter-spacing:2px;margin:0.4rem 0 0;font-weight:600;}

/* ─ Cam Area ─ */
.idle-cam{
    background:linear-gradient(145deg,rgba(13,19,48,0.9),rgba(26,39,85,0.4));
    border-radius:20px;padding:5rem 2rem;text-align:center;
    border:2px dashed rgba(99,102,241,0.12);
}

/* ─ Global Tweaks ─ */
.stButton>button{
    background:linear-gradient(135deg,#4f46e5,#6366f1)!important;color:#fff!important;
    border:none!important;border-radius:12px!important;font-weight:700!important;
    transition:all 0.3s ease!important;letter-spacing:0.3px!important;
}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 20px rgba(99,102,241,0.3)!important;}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;}
div[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden;}
.stTabs [data-baseweb="tab-list"]{gap:8px;}
.stTabs [data-baseweb="tab"]{
    background:rgba(13,19,48,0.6)!important;border-radius:10px!important;
    border:1px solid rgba(99,102,241,0.06)!important;color:#94a3b8!important;
    font-weight:600!important;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,rgba(79,70,229,0.2),rgba(99,102,241,0.1))!important;
    border-color:rgba(99,102,241,0.3)!important;color:#e2e8f0!important;
}
</style>""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def mc(label, value, color="g"):
    return f'<div class="mc"><div class="lbl">{label}</div><div class="val {color}">{value}</div></div>'


def sp(level):
    c = {"ALERT": "ok", "MILD": "wn", "DROWSY": "dg"}.get(level, "ok")
    return f'<div style="text-align:center"><span class="sp {c}">{level}</span></div>'


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

        st.markdown("---")
        st.markdown("**CNN Classification Model**")
        model_choice = st.selectbox("Select Model", [
            "MobileNetV2", "CustomCNN", "ResNet18", "ResNet50", "VGG16", "EfficientNet-B0"
        ])

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
        if st.button("▶ Start New Session (For Logging)", use_container_width=True, type="primary"):
            st.session_state.session_id = start_session(user["user_id"])
            st.rerun()
    else:
        if st.button("⏹ Stop Session", use_container_width=True):
            if st.session_state.session_id:
                stats = st.session_state.alert_system.get_stats()
                end_session(st.session_state.session_id,
                            0, 0, stats["total_alerts"], stats["total_yawns"],
                            stats["average_ear"], stats["average_mar"])
                st.session_state.session_id = None
                st.session_state.alert_system.reset()
                st.rerun()

    st.markdown("### 📷 Live Camera Feed")
    st.info("💡 Grant camera permissions. The video runs entirely in your browser "
            "and sends frames to the server for processing.")

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
        m1.markdown(mc("EAR", f"{cr['ear']:.3f}", ear_c), unsafe_allow_html=True)
        m2.markdown(mc("MAR", f"{cr['mar']:.3f}", mar_c), unsafe_allow_html=True)
        m3.markdown(mc("Score", f"{cr['score']:.2f}", "r" if cr['score'] > 0.6 else "g"),
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
        <div id="alarm-btn-wrap" style="text-align:center;margin:8px 0;">
            <button id="enable-sound-btn" onclick="initAudio()" style="
                background:linear-gradient(135deg,#4f46e5,#6366f1);color:#fff;
                border:none;border-radius:12px;padding:10px 28px;font-size:14px;
                font-weight:700;cursor:pointer;letter-spacing:0.5px;
                box-shadow:0 4px 15px rgba(99,102,241,0.3);
            ">🔊 Click to Enable Alarm Sound</button>
        </div>
        <script>
        var win = window.parent || window;

        function _scheduleBeep(ctx, whenSec, freq, duration) {{
            try {{
                var startT = ctx.currentTime + whenSec;
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.frequency.value = freq;
                osc.type = 'square';
                gain.gain.setValueAtTime(0.0001, startT);
                gain.gain.linearRampToValueAtTime(0.45, startT + 0.01);
                gain.gain.exponentialRampToValueAtTime(0.0001, startT + duration);
                osc.start(startT);
                osc.stop(startT + duration + 0.05);
            }} catch(e) {{ console.error('Beep schedule error:', e); }}
        }}

        function _beepAt(whenSec, freq, duration) {{
            var ctx = win._drowsiAudioCtx;
            if (!ctx) return;
            // Browsers (esp. Chrome) auto-suspend AudioContext after the tab
            // is backgrounded. Resume each call so the alarm survives.
            if (ctx.state === 'suspended') {{
                ctx.resume().then(function() {{
                    _scheduleBeep(ctx, whenSec, freq, duration);
                }}).catch(function(e) {{ console.error('Resume failed:', e); }});
            }} else {{
                _scheduleBeep(ctx, whenSec, freq, duration);
            }}
        }}

        function initAudio() {{
            // Runs on user click — satisfies browser autoplay policy.
            // Create the AudioContext on the parent window so it survives
            // iframe destruction during auto-refresh.
            if (!win._drowsiAudioCtx) {{
                win._drowsiAudioCtx = new (win.AudioContext || win.webkitAudioContext)();
            }}
            if (win._drowsiAudioCtx.state === 'suspended') {{
                win._drowsiAudioCtx.resume();
            }}
            win._drowsiSoundEnabled = true;
            // Confirmation chirp so the user knows audio is working
            _beepAt(0.0, 440, 0.12);
            _beepAt(0.15, 880, 0.18);
            var wrap = document.getElementById('alarm-btn-wrap');
            if (wrap) wrap.innerHTML = '<span style="color:#34d399;font-size:13px;font-weight:600;">✓ Alarm Sound Enabled</span>';
        }}

        // If sound was already enabled in a previous render, hide button
        if (win._drowsiSoundEnabled) {{
            var wrap = document.getElementById('alarm-btn-wrap');
            if (wrap) wrap.innerHTML = '<span style="color:#34d399;font-size:13px;font-weight:600;">✓ Alarm Sound Enabled</span>';
        }}

        // ── Schedule alarm beeps based on status ──
        // Beeps are queued on the AudioContext clock — they play even if
        // this iframe is destroyed by the next auto-refresh.
        var el = document.getElementById('drowsi-status');
        var status = el ? el.textContent.trim() : 'OK';
        if (win._drowsiSoundEnabled) {{
            if (status === 'DROWSY') {{
                // Urgent two-tone alarm — 4 beeps over 1s
                _beepAt(0.00, 880,  0.15);
                _beepAt(0.25, 1100, 0.15);
                _beepAt(0.50, 880,  0.15);
                _beepAt(0.75, 1100, 0.15);
            }} else if (status === 'YAWN') {{
                // Lower-pitched warning — 2 beeps over 1s
                _beepAt(0.00, 660, 0.20);
                _beepAt(0.50, 660, 0.20);
            }}
        }}
        </script>
        """, height=55)


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
