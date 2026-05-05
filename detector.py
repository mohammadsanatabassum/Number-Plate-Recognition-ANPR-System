import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

# Confidence threshold - discard any reading below this certainty
ACCURACY_THRESHOLD = 0.60

def post_process_plate(text):
    """Clean up raw OCR output. Strip noise but keep what the AI actually sees."""
    text = re.sub(r'[^A-Z0-9\- ]', '', text.upper())
    text = re.sub(r'^[^A-Z0-9]+', '', text)
    text = re.sub(r'[^A-Z0-9]+$', '', text)
    return text.strip()


class PlateDetector:
    def __init__(self):
        print("Initializing Triple-Stage Architecture: YOLOv8 + Upscale + EasyOCR...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, 'model_weights')
        os.makedirs(weights_dir, exist_ok=True)

        # ── Stage A: YOLOv8 License Plate Detector ──────────────────────────────
        # keremberke/yolov8n-license-plate-detection is a fine-tuned YOLOv8 Nano
        # model trained on thousands of global license plates. It is only ~6MB.
        # On first boot the ultralytics hub will automatically fetch the weights.
        yolo_weights = os.path.join(base_dir, 'yolo_plate.pt')
        if not os.path.exists(yolo_weights):
            from huggingface_hub import hf_hub_download
            yolo_weights = hf_hub_download(
                repo_id="keremberke/yolov8n-license-plate-detection",
                filename="best.pt",
                local_dir=base_dir,
                local_dir_use_symlinks=False,
            )
            # rename for clarity
            os.rename(yolo_weights, os.path.join(base_dir, 'yolo_plate.pt'))
            yolo_weights = os.path.join(base_dir, 'yolo_plate.pt')

        self.yolo = YOLO(yolo_weights)

        # ── Stage B: EasyOCR Deep Learning Text Reader ──────────────────────────
        self.reader = easyocr.Reader(
            ['en'], gpu=False,
            model_storage_directory=weights_dir,
            download_enabled=True
        )

        print("All models loaded successfully.")

    # ── Public API ────────────────────────────────────────────────────────────
    def detect_and_read_plate(self, img_array):
        """
        Returns (cropped_plate_img, plate_text, accuracy_float).
        If nothing is found, returns (img_array, None, 0.0).
        """
        results = self.yolo.predict(
            source=img_array,
            conf=0.25,      # minimum YOLO detection confidence
            verbose=False,
            device='cpu',
        )

        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return self._fallback_ocr(img_array)

        # Pick the box with the highest YOLO confidence
        best_idx = int(boxes.conf.argmax())
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
        yolo_conf = float(boxes.conf[best_idx])

        h, w = img_array.shape[:2]
        x1, y1 = max(0, x1 - 6), max(0, y1 - 6)
        x2, y2 = min(w, x2 + 6), min(h, y2 + 6)

        plate_crop = img_array[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return self._fallback_ocr(img_array)

        # ── Stage B: Upscale the crop for better OCR accuracy ─────────────────
        ph, pw = plate_crop.shape[:2]
        scale = max(1.0, 800 / max(pw, ph))
        plate_crop_up = cv2.resize(
            plate_crop,
            (int(pw * scale), int(ph * scale)),
            interpolation=cv2.INTER_CUBIC
        )

        # ── Stage C: EasyOCR reads the isolated, upscaled plate ───────────────
        ocr_results = self.reader.readtext(plate_crop_up)
        ocr_results = sorted(ocr_results, key=lambda r: r[0][0][1])  # top-to-bottom

        raw_text, conf_scores = "", []
        for (_, text, prob) in ocr_results:
            if prob > 0.15:
                raw_text += text
                conf_scores.append(prob)

        plate_text = post_process_plate(raw_text)
        ocr_accuracy = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        # Combined score: average of YOLO detection confidence and OCR read confidence
        combined_accuracy = (yolo_conf + ocr_accuracy) / 2.0

        if len(plate_text.replace(" ", "").replace("-", "")) >= 4 and combined_accuracy >= ACCURACY_THRESHOLD:
            display_crop = img_array[y1:y2, x1:x2]
            return display_crop, plate_text, combined_accuracy

        return self._fallback_ocr(img_array)

    # ── Internal fallback ─────────────────────────────────────────────────────
    def _fallback_ocr(self, img_array):
        """
        If YOLO finds nothing (e.g. severely angled frame), fall back to raw
        full-image EasyOCR. This ensures the system never returns nothing without
        at least trying.
        """
        h, w = img_array.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            proc = cv2.resize(img_array, (int(w * scale), int(h * scale)))
        else:
            proc = img_array

        ocr_results = self.reader.readtext(proc)
        ocr_results = sorted(ocr_results, key=lambda r: r[0][0][1])

        raw_text, conf_scores, best_bbox, best_prob = "", [], None, 0.0
        for (bbox, text, prob) in ocr_results:
            if prob > 0.15:
                raw_text += text
                conf_scores.append(prob)
                if prob > best_prob:
                    best_prob = prob
                    best_bbox = bbox

        plate_text = post_process_plate(raw_text)
        accuracy = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        if len(plate_text.replace(" ", "").replace("-", "")) >= 4 and accuracy >= ACCURACY_THRESHOLD and best_bbox:
            scale_mult = 1.0 if proc.shape == img_array.shape else (max(h, w) / max_dim)
            xmin = int(min(p[0] for p in best_bbox) * scale_mult)
            xmax = int(max(p[0] for p in best_bbox) * scale_mult)
            ymin = int(min(p[1] for p in best_bbox) * scale_mult)
            ymax = int(max(p[1] for p in best_bbox) * scale_mult)
            crop = img_array[max(0, ymin-10):min(h, ymax+10), max(0, xmin-10):min(w, xmax+10)]
            return crop, plate_text, accuracy

        return img_array, None, 0.0
