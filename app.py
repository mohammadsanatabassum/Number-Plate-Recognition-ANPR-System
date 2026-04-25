import os
# FATAL CLOUD CRASH FIX: Strictly lock CPU threading to 1 before PyTorch/OpenCV boot to prevent Streamlit Free Tier (1GB RAM limit) from instantly hitting Linux OOM-Kills!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import uuid
import os

from database import init_db, save_plate
from detector import PlateDetector

st.set_page_config(page_title="ANPR System", page_icon="🚗", layout="centered")

@st.cache_resource
def load_detector():
    return PlateDetector()

def process_and_display_frame(frame, detector):
    cropped_img, text, accuracy = detector.detect_and_read_plate(frame)
    if text:
        st.success(f"### Detected Plate: **{text}** *(Accuracy: {accuracy*100:.1f}%)*")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cropped_img, caption="Cropped Plate", clamp=True)
        with col2:
            img_filename = f"captures/{text}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(img_filename, frame)
            save_plate(text, img_filename)
            st.info("✅ Logged securely into the SQLite database!")
        return True
    return False

def main():
    st.title("🚗 Automatic Number Plate Recognition")
    st.write("Upload an image or video of a car to detect and extract its license plate.")

    init_db()
    if not os.path.exists("captures"):
        os.makedirs("captures")

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

    if uploaded_file is not None:
        # SUPER LAZY LOADING: ONLY load massive 1.5GB Neural Network AFTER health-checks pass and user interacts!
        detector = load_detector()
        
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext in ['jpg', 'jpeg', 'png']:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image")
            with st.spinner('Scanning for license plates...'):
                found = process_and_display_frame(frame, detector)
            if not found:
                st.error("No license plate detected. Try an image with a clearer view of the plate.")
                
        elif file_ext in ['mp4', 'mov', 'avi']:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
            tfile.write(uploaded_file.read())
            tfile.close()
            
            st.video(tfile.name)
            
            cap = cv2.VideoCapture(tfile.name)
            frame_skip = 10  # process every 10th frame to run faster
            count = 0
            
            st.info("Processing frames from the video...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 100 # fallback
                
            plates_found = set()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                count += 1
                if count % frame_skip == 0:
                    status_text.text(f"Scanning frame {count}/{total_frames}...")
                    progress_bar.progress(min(count / total_frames, 1.0))
                    
                    cropped_img, text, accuracy = detector.detect_and_read_plate(frame)
                    if text and text not in plates_found:
                        plates_found.add(text)
                        
                        st.success(f"### Detected Plate in Video: **{text}** *(Accuracy: {accuracy*100:.1f}%)*")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cropped_img, caption="Cropped Snapshot", clamp=True)
                        with col2:
                            img_filename = f"captures/{text}_{uuid.uuid4().hex[:6]}.jpg"
                            cv2.imwrite(img_filename, frame)
                            save_plate(text, img_filename)
                            st.info("✅ Logged securely into the SQLite database!")
            
            cap.release()
            progress_bar.progress(1.0)
            status_text.text("Video processing complete!")
            if len(plates_found) == 0:
                st.warning("No plates detected in this video.")
                
    st.markdown("---")
    st.subheader("🗄️ Database Logs")
    if st.checkbox("Show detection history"):
        import sqlite3
        import pandas as pd
        
        try:
            conn = sqlite3.connect("plates.db")
            df = pd.read_sql_query("SELECT id, plate_text, timestamp FROM plates ORDER BY timestamp DESC", conn)
            if df.empty:
                st.info("No plates logged yet.")
            else:
                st.dataframe(df)
            conn.close()
        except Exception as e:
            st.warning("Database is empty or could not be queried.")

if __name__ == '__main__':
    main()
