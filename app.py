import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wildlife Poaching Detection System",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
        background-color: #0a0f0a;
        color: #a8ff78;
    }

    .main {
        background-color: #0a0f0a;
    }

    h1, h2, h3 {
        font-family: 'Orbitron', monospace !important;
        color: #a8ff78 !important;
        letter-spacing: 2px;
    }

    .alert-box {
        padding: 16px;
        border-radius: 8px;
        margin: 10px 0;
        font-weight: bold;
        font-size: 1rem;
        border-left: 6px solid;
    }

    .alert-normal {
        background: #102010;
        color: #a8ff78;
        border-color: #a8ff78;
    }

    .alert-warn {
        background: #2a2100;
        color: #ffd700;
        border-color: #ffd700;
    }

    .alert-danger {
        background: #2a0000;
        color: #ff4444;
        border-color: #ff4444;
        animation: pulse 1s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 5px #ff4444; }
        50% { box-shadow: 0 0 20px #ff4444; }
        100% { box-shadow: 0 0 5px #ff4444; }
    }

    .metric-card {
        background: #111a11;
        border: 1px solid #2a4a2a;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: bold;
        color: #a8ff78;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #4a7a4a;
        letter-spacing: 2px;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    wildlife_model = YOLO("best01.pt")

    weapon_model = YOLO("weapon_detection_best.pt")

    return wildlife_model, weapon_model


wildlife_model, weapon_model = load_models()

# =========================================================
# SESSION STATE
# =========================================================
if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []

if "total_detections" not in st.session_state:
    st.session_state.total_detections = 0

if "threat_count" not in st.session_state:
    st.session_state.threat_count = 0

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("⚙ CONTROL PANEL")

    mode = st.selectbox(
        "Detection Mode",
        ["📷 Image Upload", "🎥 Webcam Live"]
    )

    st.markdown("---")

    conf_thresh = st.slider(
        "Confidence Threshold",
        0.10,
        0.90,
        0.40,
        0.05
    )

    img_size = st.selectbox(
        "Image Size",
        [640, 960, 1280],
        index=1
    )

    brightness = st.slider(
        "Brightness",
        1.0,
        2.5,
        1.3,
        0.1
    )

    contrast = st.slider(
        "Contrast",
        0,
        80,
        30,
        5
    )

    st.markdown("---")

    st.subheader("📊 SESSION STATS")

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{st.session_state.total_detections}</div>
        <div class="metric-label">TOTAL SCANS</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color:#ff4444;">
            {st.session_state.threat_count}
        </div>
        <div class="metric-label">THREATS</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("🐾 AI Wildlife Poaching Detection System")

st.markdown("""
Real-Time Intelligent Wildlife Surveillance using:
- YOLOv8
- OpenCV
- Streamlit
- Multi-Model AI Detection
""")

st.markdown("---")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def enhance_image(frame, alpha=1.3, beta=30):

    return cv2.convertScaleAbs(
        frame,
        alpha=alpha,
        beta=beta
    )

# =========================================================
# DETECTION ANALYSIS
# =========================================================
def analyze_detections(
    wildlife_results,
    weapon_results,
    wildlife_model,
    weapon_model,
    threshold=0.4
):

    animal = False
    human = False
    weapon = False

    labels = []

    # =========================
    # WILDLIFE MODEL
    # =========================
    for box in wildlife_results[0].boxes:

        conf = float(box.conf[0])

        if conf < threshold:
            continue

        cls_id = int(box.cls[0])

        label = wildlife_model.names[cls_id].lower()

        labels.append(f"{label} {conf:.2f}")

        if "animal" in label:
            animal = True

        if "person" in label or "human" in label:
            human = True

    # =========================
    # WEAPON MODEL
    # =========================
    for box in weapon_results[0].boxes:

        conf = float(box.conf[0])

        if conf < 0.20:
            continue

        cls_id = int(box.cls[0])

        label = weapon_model.names[cls_id].lower()

        labels.append(f"{label} {conf:.2f}")

        if (
            "weapon" in label
            or "gun" in label
            or "rifle" in label
            or "knife" in label
        ):
            weapon = True

    return animal, human, weapon, labels

# =========================================================
# THREAT LOGIC
# =========================================================
def get_alert(animal, human, weapon):

    if human and animal and weapon:
        return (
            "danger",
            "🚨 POACHING THREAT DETECTED"
        )

    elif human and weapon:
        return (
            "danger",
            "🚨 ARMED HUMAN DETECTED"
        )

    elif human and animal:
        return (
            "warn",
            "⚠ HUMAN + ANIMAL DETECTED"
        )

    elif weapon:
        return (
            "warn",
            "⚠ WEAPON DETECTED"
        )

    elif animal:
        return (
            "normal",
            "✅ ANIMAL DETECTED"
        )

    elif human:
        return (
            "normal",
            "✅ HUMAN DETECTED"
        )

    else:
        return (
            "normal",
            "✅ NO THREAT DETECTED"
        )

# =========================================================
# ALERT DISPLAY
# =========================================================
def render_alert(level, message):

    cls_map = {
        "danger": "alert-danger",
        "warn": "alert-warn",
        "normal": "alert-normal"
    }

    st.markdown(
        f'<div class="alert-box {cls_map[level]}">{message}</div>',
        unsafe_allow_html=True
    )

# =========================================================
# ALERT LOGGING
# =========================================================
def log_alert(level, message):

    timestamp = datetime.now().strftime("%H:%M:%S")

    st.session_state.alert_log.insert(
        0,
        {
            "time": timestamp,
            "level": level,
            "message": message
        }
    )

    st.session_state.alert_log = st.session_state.alert_log[:30]

    st.session_state.total_detections += 1

    if level == "danger":
        st.session_state.threat_count += 1

# =========================================================
# IMAGE UPLOAD MODE
# =========================================================
if mode == "📷 Image Upload":

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is not None:

        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(file_bytes, 1)

        frame = enhance_image(
            frame,
            alpha=brightness,
            beta=contrast
        )

        with st.spinner("🔍 Running AI Detection..."):

            wildlife_results = wildlife_model(
                frame,
                conf=conf_thresh,
                imgsz=img_size
            )

            weapon_results = weapon_model(
                frame,
                conf=0.20,
                imgsz=1280
            )

        animal, human, weapon, labels = analyze_detections(
            wildlife_results,
            weapon_results,
            wildlife_model,
            weapon_model,
            conf_thresh
        )

        level, message = get_alert(
            animal,
            human,
            weapon
        )

        log_alert(level, message)

        # =========================
        # DRAW RESULTS
        # =========================
        annotated1 = wildlife_results[0].plot()

        annotated2 = weapon_results[0].plot()

        final_frame = cv2.addWeighted(
            annotated1,
            0.7,
            annotated2,
            0.3,
            0
        )

        # =========================
        # DISPLAY
        # =========================
        col1, col2 = st.columns([2, 1])

        with col1:

            st.image(
                final_frame,
                channels="BGR",
                use_column_width=True
            )

        with col2:

            st.subheader("🚨 Threat Analysis")

            render_alert(level, message)

            st.markdown("### Detections")

            if labels:
                for label in labels:
                    st.write(f"✔ {label}")

            else:
                st.write("No detections")

# =========================================================
# WEBCAM MODE
# =========================================================
elif mode == "🎥 Webcam Live":

    st.warning(
        "Webcam works only in LOCAL system execution."
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ START WEBCAM"):
            st.session_state.webcam_running = True

    with col2:
        if st.button("⏹ STOP WEBCAM"):
            st.session_state.webcam_running = False

    frame_placeholder = st.empty()

    if st.session_state.webcam_running:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():

            st.error("Cannot open webcam")

        else:

            while st.session_state.webcam_running:

                ret, frame = cap.read()

                if not ret:
                    break

                frame = enhance_image(
                    frame,
                    alpha=brightness,
                    beta=contrast
                )

                # =========================
                # RUN BOTH MODELS
                # =========================
                wildlife_results = wildlife_model(
                    frame,
                    conf=conf_thresh,
                    imgsz=img_size,
                    verbose=False
                )

                weapon_results = weapon_model(
                    frame,
                    conf=0.20,
                    imgsz=1280,
                    verbose=False
                )

                # =========================
                # ANALYZE
                # =========================
                animal, human, weapon, labels = analyze_detections(
                    wildlife_results,
                    weapon_results,
                    wildlife_model,
                    weapon_model,
                    conf_thresh
                )

                level, message = get_alert(
                    animal,
                    human,
                    weapon
                )

                log_alert(level, message)

                # =========================
                # DRAW RESULTS
                # =========================
                annotated1 = wildlife_results[0].plot()

                annotated2 = weapon_results[0].plot()

                final_frame = cv2.addWeighted(
                    annotated1,
                    0.7,
                    annotated2,
                    0.3,
                    0
                )

                # =========================
                # ALERT TEXT
                # =========================
                color_map = {
                    "danger": (0, 0, 255),
                    "warn": (0, 255, 255),
                    "normal": (0, 255, 0)
                }

                color = color_map[level]

                cv2.putText(
                    final_frame,
                    message,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                # =========================
                # DISPLAY FRAME
                # =========================
                frame_placeholder.image(
                    final_frame,
                    channels="BGR",
                    use_column_width=True
                )

                time.sleep(0.03)

            cap.release()

# =========================================================
# ALERT LOG SECTION
# =========================================================
if st.session_state.alert_log:

    st.markdown("---")

    st.subheader("📋 Alert Log")

    for log in st.session_state.alert_log:

        st.write(
            f"[{log['time']}] {log['message']}"
        )