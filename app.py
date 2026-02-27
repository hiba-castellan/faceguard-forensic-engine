import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

def apply_glitch(image, score):
    """Adds a red transparent overlay and a small neon alert in the bottom corner."""
    red_layer = np.full_like(image, (255, 0, 0)) 
    overlay = cv2.addWeighted(image, 0.75, red_layer, 0.25, 0)
    
    glitched = np.copy(overlay)
    shift = 8
    glitched[:, shift:, 0] = overlay[:, :-shift, 0] 
    glitched[:, :-shift, 2] = overlay[:, shift:, 2] 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"!!! MANIPULATION ALERT ({score*100:.0f}%) !!!"
    text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
    
    x_pos = image.shape[1] - text_size[0] - 20
    y_pos = image.shape[0] - 20
    
    cv2.putText(glitched, text, (x_pos+1, y_pos+1), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(glitched, text, (x_pos, y_pos), font, 0.6, (255, 0, 100), 2, cv2.LINE_AA)
    
    return glitched

def draw_cyber_hud(image, face_landmarks, score=None, sensitivity=0.40):
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
    x_min, y_min, x_max, y_max = max(0, x_min - pad), max(0, y_min - pad), min(w, x_max + pad), min(h, y_max + pad)

    if score is None:
        color, status_text = (0, 255, 255), "SCANNING..."
    elif score > sensitivity:
        color, status_text = (255, 0, 0), f"THREAT LVL: {score*100:.1f}%"
    else:
        color, status_text = (0, 255, 255), f"SECURE: {(1-score)*100:.1f}%"

    thickness, line_len = 2, 30
    cv2.line(image, (x_min, y_min), (x_min + line_len, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + line_len), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - line_len, y_max), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max, y_max - line_len), color, thickness)
    
    cv2.putText(image, status_text, (x_max + 10, y_min + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image

def draw_visuals(image, score=None, sensitivity=0.40):
    if not VISUALS_ENABLED: return image
    try:
        output = image.copy()
        results = face_mesh.process(output)
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(output, lm, mp_face_mesh.FACEMESH_TESSELATION, 
                                          landmark_drawing_spec=sci_fi_landmark_style,
                                          connection_drawing_spec=sci_fi_connections_style)
                output = draw_cyber_hud(output, lm, score, sensitivity)
        return output
    except: return image

def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def simulate_terminal_logging():
    terminal = st.empty()
    steps = ["Initiating Neural Protocols...", "Scanning Biometric Vectors...", 
             "Analyzing Pixel Variances...", "Isolating Anomalies...", "Compiling Forensic Verdict..."]
    for i, step in enumerate(steps):
        progress = (i + 1) * 20
        bar = "█" * (progress // 5) + "░" * (20 - (progress // 5))
        terminal.markdown(f"```shell\n> {step}\n> [{bar}] {progress}%\n```")
        time.sleep(0.3)
    terminal.empty()

# ================= MAIN APP =================

st.write("Current directory files:", os.listdir())
if os.path.exists("models"):
    st.write("Files in models folder:", os.listdir("models"))
else:
    st.write("Models folder does NOT exist")

@st.cache_resource
def load_faceguard_model():
    return load_model(MODEL_PATH)

try:
    model = load_faceguard_model()
except Exception as e:
    st.error("❌ Model not found!")
    st.write("Actual error:", e)
    st.stop()

st.sidebar.title("⚙️ CONTROL PANEL")
mode = st.sidebar.radio("SELECT MODE", ["📸 Image Analysis", "🎥 Video Forensics", "🛑 Live Webcam Scan"])
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎚️ SETTINGS")
sensitivity = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40, 0.05)
if st.sidebar.button("🔄 RESET SYSTEM"):
    st.cache_resource.clear()
    st.rerun()

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
# ================= MODE: LIVE WEBCAM =================
if mode == "🛑 Live Webcam Scan":
    st.markdown("### 🔴 REAL-TIME SURVEILLANCE")
    col1, col2 = st.columns([2, 1])
    with col2:
        run = st.checkbox('🔴 START CAMERA FEED')
    
    if run:
        frame_placeholder = col1.empty()
        verdict_placeholder = st.empty()
        st.markdown("#### 📡 LIVE RISK MONITOR")
        st.caption("The gauge shows real-time risk. The red line marks the 'Danger Threshold'; bars above this indicate active manipulation.")
        chart_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pred = model.predict(preprocess_image(rgb), verbose=0)[0][0]
            
            # Apply visuals and red overlay
            visual_frame = draw_visuals(rgb, pred, sensitivity)
            if pred > sensitivity: 
                visual_frame = apply_glitch(visual_frame, pred)
            frame_placeholder.image(visual_frame, use_container_width=True)
            
            # Verdict Text
            if pred > sensitivity:
                verdict_placeholder.error(f"🚨 THREAT DETECTED: FAKE ({pred*100:.1f}%)")
            else:
                verdict_placeholder.success(f"✅ STATUS: AUTHENTIC ({(1-pred)*100:.1f}%)")
            
            # --- UPDATED VERTICAL INDICATOR GAUGE ---
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Level (%)"},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': "white"},
                    'bar': {'color': "#ff4b4b" if pred > sensitivity else "#00f2ff"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': sensitivity * 100
                    }
                }
            ))
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#00ffff", family="Roboto Mono"),
                height=300, margin=dict(l=20, r=20, t=40, b=20)
            )
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            
        cap.release()
# ================= MODE: VIDEO FORENSICS =================
elif mode == "🎥 Video Forensics":
    uploaded_video = st.file_uploader("UPLOAD VIDEO EVIDENCE", type=["mp4"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        col_orig, col_proc = st.columns(2)
        with col_orig: st.video(tfile.name)
        with col_proc: st_frame, bar = st.empty(), st.progress(0)
        if c2.button("INITIATE FORENSIC SCAN") if 'c2' in locals() else st.button("INITIATE FORENSIC SCAN"):
            simulate_terminal_logging()
            cap = cv2.VideoCapture(tfile.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            preds, suspicious = [], []
            for i in range(total):
                ret, frame = cap.read()
                if not ret: break
                if i % FRAME_INTERVAL == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    p = model.predict(preprocess_image(rgb), verbose=0)[0][0]
                    preds.append(p)
                    vis = draw_visuals(rgb, p, sensitivity)
                    if p > sensitivity:
                        vis = apply_glitch(vis, p)
                        suspicious.append((rgb, p))
                    st_frame.image(vis, use_container_width=True)
                    bar.progress(min(i/total, 1.0))
            cap.release()
            st.markdown("### 📋 FORENSIC ANALYSIS REPORT")
            avg = np.mean(preds)
            if avg > sensitivity: st.error(f"🚨 VERDICT: MANIPULATION DETECTED ({avg*100:.1f}%)")
            else: st.success("✅ CONTENT AUTHENTIC")
            
            # Dashboard Graph
            t_sec = [round(i * (FRAME_INTERVAL / 30), 2) for i in range(len(preds))]
            scores = [p * 100 for p in preds]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=t_sec, y=scores, marker_color=['#ff4b4b' if s > sensitivity*100 else '#00f2ff' for s in scores]))
            fig.add_trace(go.Scatter(x=t_sec, y=scores, mode='lines+markers', line=dict(color='white', width=2), marker=dict(size=8, color='#ff00ff')))
            fig.add_hline(y=sensitivity*100, line_dash="dash", line_color="#ff4b4b")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#00ffff"), height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            if suspicious:
                cols = st.columns(4)
                for i, (f, s) in enumerate(suspicious[:4]): cols[i].image(f, caption=f"Risk: {s*100:.0f}%", use_container_width=True)
        try: os.remove(tfile.name)
        except: pass

# ================= MODE: IMAGE ANALYSIS =================
elif mode == "📸 Image Analysis":
    uploaded = st.file_uploader("UPLOAD IMAGE EVIDENCE", type=["jpg", "png"])
    if uploaded:
        img_rgb = cv2.cvtColor(cv2.imdecode(np.asarray(bytearray(uploaded.read()), dtype=np.uint8), 1), cv2.COLOR_BGR2RGB)
        c1, c2 = st.columns(2)
        c1.image(img_rgb)
        scan = c2.empty()
        scan.image(draw_visuals(img_rgb))
        if c2.button("RUN DIAGNOSTIC"):
            simulate_terminal_logging()
            pred = model.predict(preprocess_image(img_rgb))[0][0]
            final_vis = apply_glitch(draw_visuals(img_rgb, pred, sensitivity), pred) if pred > sensitivity else draw_visuals(img_rgb, pred, sensitivity)
            scan.image(final_vis, use_container_width=True)
            if pred > sensitivity: st.error(f"🚨 MANIPULATION DETECTED: {pred*100:.2f}%")
            else: st.success("✅ AUTHENTIC")



