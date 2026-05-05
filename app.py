import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import uuid
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

from database import init_db, save_plate
from detector import PlateDetector

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ANPR System — YOLOv8 + EasyOCR",
    page_icon="🚗",
    layout="wide"
)

# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_detector():
    return PlateDetector()

# ── WebRTC config: Google public STUN server so HuggingFace proxy is bypassed ─
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# ── Live-camera video processor ───────────────────────────────────────────────
class ANPRVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = load_detector()
        self.last_plate = None
        self.last_accuracy = 0.0
        self.last_crop = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        _, plate_text, accuracy = self.detector.detect_and_read_plate(img)

        if plate_text:
            self.last_plate = plate_text
            self.last_accuracy = accuracy

            # Draw overlay on live frame
            h, w = img.shape[:2]
            label = f"{plate_text}  {accuracy*100:.1f}%"
            cv2.rectangle(img, (10, h - 60), (10 + len(label) * 14, h - 10), (0, 200, 0), -1)
            cv2.putText(img, label, (14, h - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── Shared helper ──────────────────────────────────────────────────────────────
def display_result(cropped_img, text, accuracy, original_frame=None):
    st.success(f"### ✅ Detected Plate: **{text}**   *(Accuracy: {accuracy*100:.1f}%)*")
    col1, col2 = st.columns(2)
    with col1:
        display = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB) if len(cropped_img.shape) == 3 else cropped_img
        st.image(display, caption="Cropped Plate Region", use_container_width=True)
    with col2:
        img_to_save = original_frame if original_frame is not None else cropped_img
        img_filename = f"captures/{text}_{uuid.uuid4().hex[:6]}.jpg"
        os.makedirs("captures", exist_ok=True)
        cv2.imwrite(img_filename, img_to_save)
        save_plate(text, img_filename)
        st.info("✅ Logged into SQLite database!")
        st.metric("Detection Confidence", f"{accuracy*100:.1f}%")


# ── Main UI ───────────────────────────────────────────────────────────────────
def main():
    st.title("🚗 Automatic Number Plate Recognition")
    st.markdown(
        "**Powered by:** YOLOv8 Object Detection + EasyOCR Deep Learning  |  "
        "Supports images, videos, and **live camera** 📷"
    )
    st.markdown("---")

    init_db()

    tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🎬 Upload Video", "📷 Live Camera"])

    # ── TAB 1: Image Upload ───────────────────────────────────────────────────
    with tab1:
        st.subheader("Upload a Car Image")
        uploaded_img = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            key="img_uploader"
        )
        if uploaded_img:
            detector = load_detector()
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

            with st.spinner("🔍 Running YOLOv8 detection + EasyOCR..."):
                crop, text, accuracy = detector.detect_and_read_plate(frame)

            if text:
                display_result(crop, text, accuracy, frame)
            else:
                st.error("❌ No license plate detected. Try a clearer image.")

    # ── TAB 2: Video Upload ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Upload a Car Video")
        uploaded_vid = st.file_uploader(
            "Choose a video...",
            type=["mp4", "mov", "avi"],
            key="vid_uploader"
        )
        if uploaded_vid:
            detector = load_detector()
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.close()

            st.video(tfile.name)

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            frame_skip = 10
            count = 0
            plates_found = set()

            st.info("🔍 Scanning video frames with YOLOv8...")
            progress = st.progress(0)
            status = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                count += 1
                if count % frame_skip != 0:
                    continue

                progress.progress(min(count / total_frames, 1.0))
                status.text(f"Scanning frame {count}/{total_frames}...")

                crop, text, accuracy = detector.detect_and_read_plate(frame)
                if text and text not in plates_found:
                    plates_found.add(text)
                    display_result(crop, text, accuracy, frame)

            cap.release()
            progress.progress(1.0)
            status.text("✅ Video scan complete!")

            if not plates_found:
                st.warning("No plates detected in this video.")

    # ── TAB 3: Live Camera ────────────────────────────────────────────────────
    with tab3:
        st.subheader("Live Camera Detection")
        st.markdown(
            "Click **START** to activate your webcam. "
            "The AI will detect license plates in real-time and overlay the result directly on the video feed!"
        )
        st.warning(
            "⚠️ Your browser will ask for camera permission — please click **Allow**. "
            "If the stream doesn't start, try Chrome or Edge."
        )

        ctx = webrtc_streamer(
            key="anpr-live",
            video_processor_factory=ANPRVideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        if ctx.video_processor:
            st.markdown("---")
            st.subheader("📋 Last Detected Plate")
            if ctx.video_processor.last_plate:
                st.success(
                    f"### **{ctx.video_processor.last_plate}**  "
                    f"*(Accuracy: {ctx.video_processor.last_accuracy*100:.1f}%)*"
                )
            else:
                st.info("Waiting for a clear plate in the camera frame...")

    # ── Database Viewer ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🗄️ Detection History")
    if st.checkbox("Show database logs"):
        import sqlite3
        import pandas as pd
        try:
            conn = sqlite3.connect("plates.db")
            df = pd.read_sql_query(
                "SELECT id, plate_text, timestamp FROM plates ORDER BY timestamp DESC LIMIT 50",
                conn
            )
            conn.close()
            if df.empty:
                st.info("No plates logged yet.")
            else:
                st.dataframe(df, use_container_width=True)
        except Exception:
            st.warning("Database is empty or could not be queried.")


if __name__ == "__main__":
    main()
