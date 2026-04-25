FROM python:3.10-slim

# System dependencies are bypassed via headless packages

WORKDIR /app

# Copy requirement files first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the file contents
COPY . .

# HuggingFace requires applications to run on exactly port 7860
EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.enableXsrfProtection", "false"]
