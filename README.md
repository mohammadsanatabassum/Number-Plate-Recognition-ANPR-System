---
title: Anpr System 🚗
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---
# 🚗 Automatic Number Plate Recognition (ANPR) System

A complete, end-to-end Machine Learning pipeline utilizing PyTorch and Optical Character Recognition (OCR) to automatically detect, extract, and log vehicle license plates from both images and localized video streams.

## ✨ Features
- **AI-Powered OCR Detection:** Utilizes `easyocr` (backed by PyTorch) to systematically scan frames, pinpointing alphanumeric structures regardless of high-angle warping or visual noise.
- **Video & Image Uploads:** Full support for processing `.mp4`, `.avi`, `.jpg`, and `.png` files via the intuitive Streamlit Web UI.
- **Historical SQLite Database:** Automatically logs all successfully detected plates, timestamps, and physical file-paths into a permanent `plates.db` local SQLite database.
- **Headless Cloud Support:** Fully architected with `opencv-python-headless` to seamlessly support native deployment onto cloud architectures like Streamlit Community Cloud without `libGL` environment crashes.
- **Interactive UI:** View the real-time processing logs and interrogate the database directly from the front-end interface!

## ⚙️ Technologies Used
- **Language:** Python 3.10+
- **Computer Vision:** OpenCV (`opencv-python-headless`)
- **Machine Learning / OCR:** EasyOCR + PyTorch
- **Web Framework:** Streamlit
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
   *Note: EasyOCR will automatically download its pre-trained Neural Network weights (approx 100MB) directly into memory upon the very first run.*

## 🚀 Usage

To launch the full interactive web application locally:

```bash
streamlit run app.py
```

This will spin up a local server at `http://localhost:8501`. 
1. Navigate to the page.
2. Upload a clear picture (or video) of a car.
3. The AI will instantly isolate the number plate, cross-check the characters, heavily crop the bounding box, display the result, and log it securely into `plates.db`.

## ☁️ Cloud Deployment Notes
This application is strictly optimized for **Streamlit Community Cloud**. 
Simply link this repository to `share.streamlit.io` for an instant 1-click deployment! 
*(Warning: Cloud providers utilize Ephemeral Storage. The local `plates.db` and the corresponding `/captures/` folder will ultimately reset or purge whenever the specific cloud container spins down to sleep).*
