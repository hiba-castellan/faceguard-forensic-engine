import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import random
from collections import deque
from tensorflow.keras.models import load_model

# ================= CONFIGURATION =================
MODEL_PATH = "models/faceguard_phase2_finetuned.h5"
IMG_SIZE = (224, 224)
FRAME_INTERVAL = 30
FAKE_THRESHOLD = 0.40

# ================= PAGE SETUP =================
st.set_page_config(
    page_title="FaceGuard System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= 🎨 SCI-FI THEME & CRT SCANLINES =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&family=Roboto+Mono&display=swap');

.stApp {
    background: radial-gradient(circle at top, #2b004d, #050012 90%);
    color: #00ffff;
    font-family: 'Roboto Mono', monospace;
}

[data-testid="stSidebar"] {
    background-color: #0d001a !important;
    border-right: 1px solid rgba(0, 255, 255, 0.1);
    box-shadow: 2px 0 15px rgba(0, 255, 255, 0.15);
}

h2, h3, h4, h5 {
    font-family: 'Orbitron', sans-serif;
    color: #ff00ff;
    text-shadow: 0 0 10px rgba(255, 0, 255, 0.7);
}

.stButton>button {
    width: 100%;
    border-radius: 5px;
    background: transparent;
    border: 2px solid #00ffff;
    font-family: 'Orbitron', sans-serif;
    font-weight: bold;
    color: #00ffff;
    transition: 0.3s;
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.2);
}
.stButton>button:hover {
    box-shadow: 0 0 20px #ff00ff;
    border: 2px solid #ff00ff;
    color: #ff00ff;
    transform: scale(1.02);
}

.scanlines {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), 
                linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
    background-size: 100% 4px, 6px 100%;
    z-index: 9999;
    pointer-events: none;
    opacity: 0.4;
    animation: flicker 0.15s infinite;
}
@keyframes flicker {
    0% { opacity: 0.3; }
    50% { opacity: 0.4; }
    100% { opacity: 0.3; }
}
</style>
<div class="scanlines"></div>
""", unsafe_allow_html=True)

# ================= 🧠 MEDIAPIPE SETUP =================
VISUALS_ENABLED = False
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    sci_fi_connections_style = mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=1, circle_radius=0) 
    sci_fi_landmark_style = mp_drawing.DrawingSpec(color=(0, 255, 200), thickness=0, circle_radius=0) 
    
    VISUALS_ENABLED = True
except Exception as e:
    print(f"MediaPipe Warning: {e}")
    VISUALS_ENABLED = False

# ================= HELPER FUNCTIONS =================

def apply_glitch(image):
    """Creates a chromatic aberration effect for fake detection."""
    glitched = np.zeros_like(image)
    shift = 15
    # Shift color channels to create a glitch
    glitched[:, shift:, 0] = image[:, :-shift, 0]  # Red shift
    glitched[:, :, 1] = image[:, :, 1]             # Green normal
    glitched[:, :-shift, 2] = image[:, shift:, 2]  # Blue shift
    return glitched

def draw_digital_barcode(predictions, sensitivity):
    """Generates a sleek, transparent loading-bar style barcode."""
    h, w = 60, 800
    # Use 4 channels (RGBA) and initialize to 0 for full transparency
    canvas = np.zeros((h, w, 4), dtype=np.uint8)
    if not predictions: return canvas
    
    border_color = (0, 255, 255, 255) # Cyan, fully opaque
    pad_x, pad_y = 20, 15
    track_w = w - 2 * pad_x
    track_h = h - 2 * pad_y
    
    # Outer frame outline
    cv2.rectangle(canvas, (pad_x, pad_y), (w - pad_x, h - pad_y), border_color, 1)
    
    # Tiny tech accents
    cv2.line(canvas, (pad_x, pad_y - 4), (pad_x + 60, pad_y - 4), border_color, 2)
    cv2.line(canvas, (w - pad_x - 60, h - pad_y + 4), (w - pad_x, h - pad_y + 4), border_color, 2)

    # Tiny segmented boxes
    box_gap = 4
    usable_w = track_w - 10
    
    num_preds = len(predictions)
    max_boxes = usable_w // (2 + box_gap)
    
    display_preds = predictions[-max_boxes:] if num_preds > max_boxes else predictions
    num_display = len(display_preds)
    
    available_w = max(1, usable_w - (num_display * box_gap))
    box_w = max(2, available_w // max(num_display, 1))
    
    start_x = pad_x + 5
    start_y = pad_y + 4
    box_h = track_h - 8
    
    for i, pred in enumerate(display_preds):
        # Colors use RGBA to ensure they show up solid on the transparent background
        color = (255, 0, 0, 255) if pred > sensitivity else (0, 255, 255, 255) 
        x_pos = start_x + i * (box_w + box_gap)
        cv2.rectangle(canvas, (x_pos, start_y), (x_pos + box_w, start_y + box_h), color, -1)
        
    return canvas

def draw_cyber_hud(image, face_landmarks, score=None, sensitivity=0.40):
    """Draws a high-tech targeting box and biometric data around the face."""
    h, w, c = image.shape
    
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in face_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y

    pad = 20
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(w, x_max + pad)
    y_max = min(h, y_max + pad)

    # Change HUD color based on threat level
    if score is None:
        color = (0, 255, 255) # Scanning (Cyan)
        status_text = "SCANNING..."
    elif score > sensitivity:
        color = (255, 0, 0) # Fake (Red)
        status_text = f"THREAT LVL: {score*100:.1f}%"
    else:
        color = (0, 255, 255) # Authentic (Cyan)
        status_text = f"SECURE: {(1-score)*100:.1f}%"

    thickness = 2
    line_len = 30

    cv2.line(image, (x_min, y_min), (x_min + line_len, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + line_len), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max - line_len, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + line_len), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + line_len, y_max), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min, y_max - line_len), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - line_len, y_max), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max, y_max - line_len), color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    
    # Print Terminator Data
    depth = round(random.uniform(1.2, 3.5), 2)
    cv2.putText(image, f"TRK: {x_min},{y_min}", (x_max + 10, y_min + 20), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(image, f"DPTH: {depth}m", (x_max + 10, y_min + 40), font, font_scale, color, 1, cv2.LINE_AA)
    cv2.putText(image, status_text, (x_max + 10, y_min + 60), font, font_scale, color, 1, cv2.LINE_AA)

    return image

def draw_visuals(image, score=None, sensitivity=0.40):
    """Draws triangular mesh and Cyber HUD."""
    if not VISUALS_ENABLED: return image
    try:
        output = image.copy()
        results = face_mesh.process(output)
        
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    output, 
                    lm, 
                    mp_face_mesh.FACEMESH_TESSELATION, 
                    landmark_drawing_spec=sci_fi_landmark_style,
                    connection_drawing_spec=sci_fi_connections_style
                )
                output = draw_cyber_hud(output, lm, score, sensitivity)
                
        return output
    except: return image

def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def simulate_terminal_logging():
    terminal = st.empty()
    steps = [
        "Initiating Neural Protocols...",
        "Scanning Biometric Vectors...",
        "Analyzing Pixel Variances...",
        "Isolating Anomalies...",
        "Compiling Forensic Verdict..."
    ]
    
    for i, step in enumerate(steps):
        progress = (i + 1) * 20
        bar = "█" * (progress // 5) + "░" * (20 - (progress // 5))
        
        terminal.markdown(f"""
        ```shell
        > {step}
        > [{bar}] {progress}%
        ```
        """)
        time.sleep(0.3)
        
    time.sleep(0.2)
    terminal.empty()

# ================= MAIN APP =================
@st.cache_resource
def load_faceguard_model():
    return load_model(MODEL_PATH)

try:
    model = load_faceguard_model()
except:
    st.error("❌ Model not found! Check path.")
    st.stop()

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("⚙️ CONTROL PANEL")
mode = st.sidebar.radio("SELECT MODE", ["📸 Image Analysis", "🎥 Video Forensics", "🛑 Live Webcam Scan"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ SETTINGS")
sensitivity = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 RESET SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ SYSTEM SPECS"):
    st.markdown("**Core:** MobileNetV2 (Full Frame Analysis)")
    st.markdown("**Dataset:** FaceForensics++")
    st.markdown("**Backend:** TensorFlow + OpenCV")
    st.markdown(f"**Visuals:** {'🟢 Online' if VISUALS_ENABLED else '🔴 Offline'}")

# --- MAIN HEADER ---
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 5px;">
    <span style="font-size: 3rem; margin-right: 15px; text-shadow: 0 0 10px #00d4ff;">🛡️</span>
    <h1 style="margin: 0; padding: 0; font-family: 'Orbitron', sans-serif; font-size: 3.5rem; letter-spacing: 2px;">
        <span style="color: #00d4ff; text-shadow: 0 0 15px rgba(0, 212, 255, 0.8);">Face</span><span style="color: #ff00ff; text-shadow: 0 0 15px rgba(255, 0, 255, 0.8);">Guard</span> 
        <span style="color: #008b8b; font-size: 0.35em; margin-left: 10px; vertical-align: middle; text-shadow: none;">OS v3.0</span>
    </h1>
</div>
<div style="color: #00d4ff; font-family: 'Roboto Mono', monospace; margin-bottom: 30px;">> Advanced Deepfake Detection & Forensic Analysis</div>
""", unsafe_allow_html=True)

# ================= MODE: LIVE WEBCAM =================
if mode == "🛑 Live Webcam Scan":
    st.markdown("### 🔴 REAL-TIME SURVEILLANCE")
    col1, col2 = st.columns([2, 1])
    with col2:
        run = st.checkbox('🔴 START CAMERA FEED')
    
    if run:
        frame_placeholder = col1.empty()
        verdict_placeholder = st.empty()
        st.markdown("#### DIGITAL FINGERPRINT RADAR")
        barcode_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        score_buffer = deque(maxlen=10)
        history = []
        
        while run:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pred = model.predict(preprocess_image(rgb), verbose=0)[0][0]
            score_buffer.append(pred)
            avg_score = sum(score_buffer) / len(score_buffer)
            history.append(avg_score)
            
            # Apply visuals, HUD, and Glitch if fake
            visual_frame = draw_visuals(rgb, avg_score, sensitivity)
            if avg_score > sensitivity:
                visual_frame = apply_glitch(visual_frame)
                
            frame_placeholder.image(visual_frame, caption="LIVE BIOMETRIC FEED", use_container_width=True)
            
            conf = avg_score * 100
            if avg_score > sensitivity:
                verdict_placeholder.error(f"🚨 THREAT DETECTED: FAKE ({conf:.1f}%)")
            else:
                verdict_placeholder.success(f"✅ STATUS: AUTHENTIC ({100-conf:.1f}%)")
            
            # Update Barcode and Line Chart
            barcode_img = draw_digital_barcode(history[-50:], sensitivity)
            barcode_placeholder.image(barcode_img, use_container_width=True)
            chart_placeholder.line_chart(history[-50:], height=150)
            
        cap.release()

# ================= MODE: VIDEO FORENSICS =================
elif mode == "🎥 Video Forensics":
    uploaded_video = st.file_uploader("UPLOAD VIDEO EVIDENCE", type=["mp4"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.markdown("#### 📼 ORIGINAL EVIDENCE")
            st.video(tfile.name)
        
        with col_proc:
            st.markdown("#### 🕵️‍♂️ FORENSIC SCAN")
            st_frame = st.empty()
            bar = st.progress(0)

        if st.button("INITIATE FORENSIC SCAN"):
            simulate_terminal_logging()
            
            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            score_buffer = deque(maxlen=5)
            smoothed_predictions = []
            suspicious_frames = []
            
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_id % FRAME_INTERVAL == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    raw_pred = model.predict(preprocess_image(rgb), verbose=0)[0][0]
                    score_buffer.append(raw_pred)
                    smoothed_score = sum(score_buffer) / len(score_buffer)
                    smoothed_predictions.append(smoothed_score)
                    
                    # Keep a clean copy of the original frame for evidence
                    clean_evidence_frame = rgb.copy()
                    
                    # Apply visuals, HUD, and Glitch for the active scanner
                    vis = draw_visuals(rgb, raw_pred, sensitivity)
                    if raw_pred > sensitivity:
                        vis = apply_glitch(vis)
                        # Save the CLEAN frame to the gallery, not the glitched one
                        suspicious_frames.append((clean_evidence_frame, raw_pred))
                    
                    st_frame.image(vis, caption=f"Processing Frame {frame_id}", use_container_width=True)
                    bar.progress(min(frame_id/total_frames, 1.0))
                    
                frame_id += 1
            cap.release()
            
            st.markdown("---")
            st.markdown("### 📋 FORENSIC REPORT")
            
            avg_score = np.mean(smoothed_predictions)
            c1, c2 = st.columns(2)
            if avg_score > sensitivity:
                c1.error("🚨 VERDICT: FAKE DETECTED")
                c2.metric("CONFIDENCE", f"{avg_score*100:.2f}%")
            else:
                c1.success("✅ VERDICT: AUTHENTIC")
                c2.metric("REALITY SCORE", f"{(100 - avg_score*100):.2f}%")
            
            st.markdown("#### DIGITAL FINGERPRINT")
            barcode_img = draw_digital_barcode(smoothed_predictions, sensitivity)
            st.image(barcode_img, use_container_width=True)
            st.line_chart(smoothed_predictions, height=200)

            if suspicious_frames:
                st.markdown("#### 🚩 SUSPICIOUS ARTIFACTS")
                cols = st.columns(4) 
                for i, (f, s) in enumerate(suspicious_frames[:4]):
                    # Now shows the clean original frame
                    cols[i].image(f, caption=f"Risk: {s*100:.0f}%", use_container_width=True)

        try: os.remove(tfile.name)
        except: pass

# ================= MODE: IMAGE ANALYSIS =================
elif mode == "📸 Image Analysis":
    uploaded = st.file_uploader("UPLOAD IMAGE EVIDENCE", type=["jpg", "png"])
    if uploaded:
        bytes_data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(bytes_data, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 🖼️ ORIGINAL")
            st.image(img_rgb, use_container_width=True)
        with col_r:
            st.markdown("#### 🤖 BIOMETRIC SCAN")
            scan_placeholder = st.empty()
            # Initial scan without score
            scan_placeholder.image(draw_visuals(img_rgb), use_container_width=True)
        
        if st.button("RUN DIAGNOSTIC"):
            simulate_terminal_logging()
            
            pred = model.predict(preprocess_image(img_rgb))[0][0]
            conf = pred * 100
            
            # Update scan with score and possible glitch
            final_visual = draw_visuals(img_rgb, pred, sensitivity)
            if pred > sensitivity:
                final_visual = apply_glitch(final_visual)
                
            scan_placeholder.image(final_visual, use_container_width=True)
            
            c1, c2 = st.columns(2)
            if pred > sensitivity:
                c1.error("🚨 VERDICT: FAKE DETECTED")
                c2.metric("FAKE PROBABILITY", f"{conf:.2f}%", delta="CRITICAL")
            else:
                c1.success("✅ VERDICT: AUTHENTIC")
                c2.metric("AUTHENTICITY SCORE", f"{100-conf:.2f}%", delta="SAFE")