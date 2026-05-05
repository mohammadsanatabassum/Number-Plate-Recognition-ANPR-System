FROM python:3.10-slim

# Install minimal system libs needed by libGL (ultralytics pulls it in)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    libxext6 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Force-remove GUI OpenCV that ultralytics silently installs,
# then reinstall the headless variant so no display is needed.
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir opencv-python-headless

# Copy application files
COPY . .

# HuggingFace requires port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.enableXsrfProtection", "false"]
