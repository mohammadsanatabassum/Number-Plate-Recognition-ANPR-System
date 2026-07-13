FROM python:3.10-slim

# System libs required by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libx11-6 \
    libsrtp2-1 \
    && rm -rf /var/lib/apt/lists/*

# Set up user 1000
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR $HOME/app

# Install all Python dependencies (as root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force opencv-python-headless LAST so ultralytics cannot overwrite it
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless

# Pre-download YOLOv8n weights as root (root can write to WORKDIR)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); print('YOLOv8n weights cached.')"

# Copy application files then give full ownership of the entire app dir to user
COPY . $HOME/app
RUN chown -R user:user $HOME/app && chown -R user:user $HOME/.cache 2>/dev/null || true

# Switch to non-root user
USER user

EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.enableXsrfProtection", "false"]
