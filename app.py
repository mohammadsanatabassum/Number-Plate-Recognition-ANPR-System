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
    cropped_img, text = detector.detect_and_read_plate(frame)
    if text:
        st.success(f"### Detected Plate: **{text}**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cropped_img, caption="Cropped Plate", use_container_width=True, clamp=True)
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

    detector = load_detector()

    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext in ['jpg', 'jpeg', 'png']:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
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
                    
                    cropped_img, text = detector.detect_and_read_plate(frame)
                    if text and text not in plates_found:
                        plates_found.add(text)
                        
                        st.success(f"### Detected Plate in Video: **{text}**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(cropped_img, caption="Cropped Snapshot", use_container_width=True, clamp=True)
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
                
if __name__ == '__main__':
    main()
