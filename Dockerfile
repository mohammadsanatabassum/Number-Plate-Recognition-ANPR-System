FROM python:3.10-slim

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install all Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force opencv-python-headless LAST so ultralytics cannot overwrite it
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless

# Copy application files with user ownership
COPY --chown=user . $HOME/app

# Switch to non-root user
USER user

# Pre-download YOLOv8n weights during BUILD (not runtime) under user ownership
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('YOLOv8n weights cached.')"

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.enableXsrfProtection", "false"]
