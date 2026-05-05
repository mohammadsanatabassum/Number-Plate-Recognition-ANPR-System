FROM python:3.10-slim

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force opencv-python-headless LAST so ultralytics cannot overwrite it
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless

# Pre-download YOLOv8n weights during BUILD (not runtime) so first user request is instant
# yolov8n.pt is the standard public Ultralytics model (~6MB), no authentication required
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('YOLOv8n weights cached.')"

# Copy application files
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.enableXsrfProtection", "false"]
