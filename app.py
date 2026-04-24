import streamlit as st
import cv2
import numpy as np
from PIL import Image
import uuid
import os

from database import init_db, save_plate
from detector import PlateDetector

st.set_page_config(page_title="ANPR System", page_icon="🚗", layout="centered")

@st.cache_resource
def load_detector():
    return PlateDetector()

def main():
    st.title("🚗 Automatic Number Plate Recognition")
    st.write("Upload an image of a car to detect and extract its license plate.")

    # Init database and directories
    init_db()
    if not os.path.exists("captures"):
        os.makedirs("captures")

    detector = load_detector()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert to OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        # Show original image
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        with st.spinner('Scanning for license plates...'):
            cropped_img, text = detector.detect_and_read_plate(frame)

        if text:
            st.success(f"### Detected Plate: **{text}**")
            
            col1, col2 = st.columns(2)
            with col1:
                # Display cropped plate (ensure we correctly render grayscale or bgr)
                st.image(cropped_img, caption="Cropped Plate", use_container_width=True, clamp=True)
            
            with col2:
                # Save to database and show log info
                img_filename = f"captures/{text}_{uuid.uuid4().hex[:6]}.jpg"
                cv2.imwrite(img_filename, frame)
                save_plate(text, img_filename)
                st.info("✅ Logged securely into the SQLite database!")
        else:
            st.error("No license plate detected. Try an image with a clearer view of the plate.")

if __name__ == '__main__':
    main()
