import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

# ── Tuning Constants ──────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.55      # minimum combined confidence to accept a result
MIN_PLATE_CHARS    = 4         # ignore reads shorter than this
OCR_ALLOWLIST      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


# ── Preprocessing Helpers ─────────────────────────────────────────────────────
def _enhance_plate(img):
    """
    Apply a sequence of OpenCV enhancement passes to make characters
    as legible as possible for the OCR engine.

    Pipeline:
      1. Grayscale
      2. CLAHE  — normalises uneven lighting across the plate
      3. Gaussian denoise
      4. Unsharp mask sharpening — makes character edges crisp
      5. Return both the enhanced grey and the original colour crop
         so we have two OCR candidates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE: Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    eq = clahe.apply(gray)

    # Gaussian denoise
    denoised = cv2.GaussianBlur(eq, (3, 3), 0)

    # Unsharp mask — amount=1.5
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 2.0, blurred, -1.0, 0)

    # Convert back to BGR so EasyOCR (expects colour or grey) is consistent
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def _upscale(img, target=900):
    """Upscale so the shorter plate dimension is at least `target` pixels."""
    h, w = img.shape[:2]
    scale = max(1.0, target / max(h, w))
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)


# ── Character-level corrections ───────────────────────────────────────────────
# Classic OCR confusion pairs — we decide which conversion to apply based on
# whether the character sits in a numeric position or letter position.
_NUM_SUBS  = str.maketrans("OIQSB",  "01058")   # in clearly-numeric zones
_CHAR_SUBS = str.maketrans("01",     "OI")       # in clearly-alpha zones

def _correct_ocr_chars(text):
    """
    Heuristic: for tokens that look purely alphabetic, convert digits that
    are well-known OCR confusions; for tokens that look purely numeric,
    convert look-alike letters.
    """
    tokens = re.split(r'(\s|-)', text)
    corrected = []
    for token in tokens:
        if token in (' ', '-'):
            corrected.append(token)
            continue
        alpha = sum(c.isalpha() for c in token)
        digit = sum(c.isdigit() for c in token)
        if digit > alpha:
            corrected.append(token.translate(_NUM_SUBS))
        elif alpha > digit:
            corrected.append(token.translate(_CHAR_SUBS))
        else:
            corrected.append(token)
    return ''.join(corrected)


def post_process_plate(text):
    """Strip OCR noise, apply character corrections, return cleaned text."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9\- ]', '', text)   # allowlist
    text = re.sub(r'^[^A-Z0-9]+', '', text)     # strip leading junk
    text = re.sub(r'[^A-Z0-9]+$', '', text)     # strip trailing junk
    text = _correct_ocr_chars(text)
    return text.strip()


# ── Main Detector ─────────────────────────────────────────────────────────────
class PlateDetector:
    def __init__(self):
        print("Initialising YOLOv8 + CLAHE + EasyOCR pipeline...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, 'model_weights')
        os.makedirs(weights_dir, exist_ok=True)

        # ── YOLOv8 license-plate detector ─────────────────────────────────────
        yolo_weights = os.path.join(base_dir, 'yolo_plate.pt')
        if not os.path.exists(yolo_weights):
            from huggingface_hub import hf_hub_download
            tmp = hf_hub_download(
                repo_id="keremberke/yolov8n-license-plate-detection",
                filename="best.pt",
                local_dir=base_dir,
                local_dir_use_symlinks=False,
            )
            os.rename(tmp, yolo_weights)

        self.yolo = YOLO(yolo_weights)

        # ── EasyOCR with strict alphanumeric allowlist ─────────────────────────
        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=weights_dir,
            download_enabled=True,
        )

        print("All models loaded.")

    # ── OCR helper: run two preprocessing variants, keep highest confidence ───
    def _ocr_best(self, crop_bgr):
        """
        Run EasyOCR twice:
          Pass 1: colour crop (as-is, upscaled)
          Pass 2: CLAHE-enhanced, sharpened, upscaled
        Return (text, confidence) for whichever pass scores higher.
        """
        candidates = []
        for variant in [crop_bgr, _enhance_plate(crop_bgr)]:
            img = _upscale(variant)
            results = self.reader.readtext(
                img,
                allowlist=OCR_ALLOWLIST,
                detail=1,
                paragraph=False,
            )
            results = sorted(results, key=lambda r: r[0][0][1])  # top→bottom

            raw, scores = "", []
            for (_, text, prob) in results:
                if prob > 0.10:
                    raw += text
                    scores.append(prob)

            cleaned = post_process_plate(raw)
            avg_conf = sum(scores) / len(scores) if scores else 0.0
            candidates.append((cleaned, avg_conf))

        # pick best by confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0]

    # ── Public API ────────────────────────────────────────────────────────────
    def detect_and_read_plate(self, img_array):
        """
        Returns (cropped_plate_img, plate_text, accuracy_float).
        Falls back to full-image OCR if YOLO finds nothing.
        """
        results = self.yolo.predict(
            source=img_array,
            conf=0.20,
            verbose=False,
            device='cpu',
        )

        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            return self._fallback_ocr(img_array)

        # Sort boxes by YOLO confidence descending; try each until OCR succeeds
        h, w = img_array.shape[:2]
        sorted_idx = boxes.conf.argsort(descending=True).tolist()

        for idx in sorted_idx:
            x1, y1, x2, y2 = map(int, boxes.xyxy[idx].tolist())
            yolo_conf = float(boxes.conf[idx])

            # Pad slightly
            x1, y1 = max(0, x1 - 8), max(0, y1 - 8)
            x2, y2 = min(w, x2 + 8), min(h, y2 + 8)

            plate_crop = img_array[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue

            plate_text, ocr_conf = self._ocr_best(plate_crop)

            # Combined = weighted mean (YOLO locates, OCR reads → OCR weighted higher)
            combined = 0.35 * yolo_conf + 0.65 * ocr_conf

            char_count = len(plate_text.replace(" ", "").replace("-", ""))
            if char_count >= MIN_PLATE_CHARS and combined >= ACCURACY_THRESHOLD:
                display_crop = img_array[y1:y2, x1:x2]
                return display_crop, plate_text, combined

        return self._fallback_ocr(img_array)

    # ── Fallback: full-image EasyOCR ─────────────────────────────────────────
    def _fallback_ocr(self, img_array):
        h, w = img_array.shape[:2]
        max_dim = 1200
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            proc = cv2.resize(img_array, (int(w * scale), int(h * scale)))
        else:
            proc = img_array

        # Run enhanced OCR on the full image
        enhanced = _enhance_plate(proc)
        best_text, best_conf, best_bbox, best_prob = "", 0.0, None, 0.0

        for variant in [proc, enhanced]:
            results = self.reader.readtext(
                variant,
                allowlist=OCR_ALLOWLIST,
                detail=1,
            )
            results = sorted(results, key=lambda r: r[0][0][1])
            raw, scores, bbox = "", [], None
            for (box, text, prob) in results:
                if prob > 0.10:
                    raw += text
                    scores.append(prob)
                    if prob > best_prob:
                        best_prob = prob
                        bbox = box

            cleaned = post_process_plate(raw)
            conf = sum(scores) / len(scores) if scores else 0.0
            if conf > best_conf:
                best_conf = conf
                best_text = cleaned
                best_bbox = bbox

        char_count = len(best_text.replace(" ", "").replace("-", ""))
        if char_count >= MIN_PLATE_CHARS and best_conf >= ACCURACY_THRESHOLD and best_bbox:
            scale_mult = 1.0 if proc.shape == img_array.shape else (max(h, w) / max_dim)
            xmin = int(min(p[0] for p in best_bbox) * scale_mult)
            xmax = int(max(p[0] for p in best_bbox) * scale_mult)
            ymin = int(min(p[1] for p in best_bbox) * scale_mult)
            ymax = int(max(p[1] for p in best_bbox) * scale_mult)
            crop = img_array[max(0, ymin-10):min(h, ymax+10),
                             max(0, xmin-10):min(w, xmax+10)]
            return crop, best_text, best_conf

        return img_array, None, 0.0
