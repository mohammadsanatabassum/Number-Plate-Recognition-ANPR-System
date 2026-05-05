FROM python:3.10-slim

# System libs required by OpenCV headless + WebRTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Install all Python dependencies
# (ultralytics will pull in opencv-python as a side effect)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 2: ALWAYS install opencv-python-headless LAST and FORCE it.
# This overwrites whatever opencv variant ultralytics installed,
# ensuring cv2 works without any display/X11 requirement.
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless

# Step 3: Copy application files
COPY . .

# HuggingFace requires port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.enableXsrfProtection", "false"]
