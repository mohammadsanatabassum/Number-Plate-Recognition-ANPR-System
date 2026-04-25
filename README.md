---
title: Anpr System
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---
# 🚗 Automatic Number Plate Recognition (ANPR) System

A complete, end-to-end Machine Learning pipeline utilizing a Dual-Stage Architecture (OpenCV Object Detection + PyTorch Deep Learning) to universally detect, exact-crop, and extract vehicle license plates from both arbitrary images and local video streams.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/mohammadsana02/anpr)

## ✨ Features
- **Dual-Stage OCR Detection:** Utilizes an ultra-fast OpenCV `haarcascade` to physically isolate plate rectangles first, scaling them up before feeding ONLY the pure plate to the `easyocr` Neural Network. This bypasses structural noise and guarantees universal accuracy.
- **Video & Image Uploads:** Full support for processing `.mp4`, `.avi`, `.jpg`, and `.png` files via the intuitive Streamlit Web UI.
- **Historical SQLite Database:** Automatically logs all successfully detected plates, timestamps, and physical file-paths into a permanent `plates.db` local SQLite database.
- **Headless Cloud Support:** Architected with a custom `Dockerfile` targeting Hugging Face Spaces (allocating 16GB RAM) utilizing `opencv-python-headless` for flawless remote rendering.

## ⚙️ Technologies Used
- **Language:** Python 3.10+
- **Computer Vision:** OpenCV (`haarcascade_russian_plate_number.xml`)
- **Machine Learning / Deep OCR:** EasyOCR + PyTorch CPU
- **Web Framework:** Streamlit
- **Containerization:** Docker
- **Database:** SQLite3 + Pandas

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mohammadsanatabassum/Number-Plate-Recognition-ANPR-System.git
   cd Number-Plate-Recognition-ANPR-System
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

To launch the full interactive web application locally:

```bash
streamlit run app.py
```

This will spin up a local server at `http://localhost:8501`. 
1. Navigate to the page.
2. Upload a clear picture (or video) of a car.
3. The HaarCascade will physically cut out the license plate, drastically upscale it, and the AI will extract the characters exactly as they appear!

## ☁️ Cloud Deployment
Due to the heavy RAM requirements of the PyTorch Deep Learning model (`>1GB`), this application is architected to be deployed on **Hugging Face Spaces** via the embedded `Dockerfile`.
Streamlit Community Cloud is **not** supported, as the 1GB RAM constraint will forcibly crash the underlying Linux kernel with an OOM Kill.
To deploy, simply link this Github Repository directly to a new Hugging Face Docker Space!
