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

# ── Shared result display ─────────────────────────────────────────────────────
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

# ── Main App ──────────────────────────────────────────────────────────────────
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

            with st.spinner("🔍 Running YOLOv8 + EasyOCR..."):
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

    # ── TAB 3: Live Camera (via st.camera_input) ──────────────────────────────
    with tab3:
        st.subheader("📷 Live Camera Capture")
        st.markdown(
            "Point your camera at a license plate, then click **📷 Take Photo** below. "
            "The AI will instantly detect and read the plate from your photo!"
        )
        st.info("💡 **Tip:** Get close enough so the plate fills at least 1/4 of the frame for best accuracy.")

        camera_photo = st.camera_input("📷 Take Photo", key="camera_snap", facing_mode="environment")

        if camera_photo:
            detector = load_detector()
            file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("🔍 Running YOLOv8 + EasyOCR on captured photo..."):
                crop, text, accuracy = detector.detect_and_read_plate(frame)

            if text:
                display_result(crop, text, accuracy, frame)
            else:
                st.error("❌ No plate detected. Try taking the photo closer to the plate.")

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
