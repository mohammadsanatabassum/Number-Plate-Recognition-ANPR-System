import cv2
import numpy as np
import easyocr
import re
import os
from ultralytics import YOLO

# ── Tuning Constants ──────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.50
MIN_PLATE_CHARS    = 4
# Expanded allowlist — preserves ALL characters that physically appear on real plates globally:
# letters, digits, dash (-), dot (.), slash (/), space, and state/country codes
OCR_ALLOWLIST      = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-./ "

# COCO class IDs that represent vehicles — YOLO will crop these zones for plate search
VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck


# ── Preprocessing ─────────────────────────────────────────────────────────────
def _enhance(img):
    """CLAHE contrast boost + unsharp-mask sharpening."""
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    eq    = clahe.apply(gray)
    blur  = cv2.GaussianBlur(eq, (3, 3), 0)
    sharp = cv2.addWeighted(eq, 2.0, cv2.GaussianBlur(eq, (0, 0), 3), -1.0, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _upscale(img, target=900):
    h, w  = img.shape[:2]
    scale = max(1.0, target / max(h, w, 1))
    if scale == 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)


# ── Character correction ──────────────────────────────────────────────────────
_NUM_MAP  = str.maketrans("OIQSB", "01058")
_CHAR_MAP = str.maketrans("01",    "OI")

def _fix_chars(text):
    parts, out = re.split(r'([\s\-])', text), []
    for p in parts:
        if p in (' ', '-'):
            out.append(p); continue
        a = sum(c.isalpha() for c in p)
        d = sum(c.isdigit() for c in p)
        out.append(p.translate(_NUM_MAP) if d > a else
                   p.translate(_CHAR_MAP) if a > d else p)
    return ''.join(out)


def _clean(text):
    text = text.upper()
    # Keep all real plate characters: letters, digits, dash, dot, slash, space
    text = re.sub(r'[^A-Z0-9\-./ ]', '', text)
    # Strip leading/trailing non-alphanumeric noise (bolts, shadows etc.)
    text = re.sub(r'^[^A-Z0-9]+', '', text)
    text = re.sub(r'[^A-Z0-9]+$', '', text)
    # Collapse multiple consecutive spaces
    text = re.sub(r' {2,}', ' ', text)
    return _fix_chars(text).strip()


def _alnum_count(text):
    """Count only letters and digits — ignore punctuation for minimum length check."""
    return sum(c.isalnum() for c in text)


# Known car body words that are NOT license plates
_NON_PLATE_WORDS = {
    "BOYACA", "HONDA", "TOYOTA", "SUZUKI", "YAMAHA", "KAWASAKI", "BAJAJ",
    "NISSAN", "HYUNDAI", "FORD", "CHEVROLET", "BMW", "AUDI", "MERCEDES",
    "VOLKSWAGEN", "KIA", "MAHINDRA", "TATA", "MARUTI", "RENAULT", "SKODA",
    "JEEP", "MITSUBISHI", "SUBARU", "LEXUS", "VOLVO", "PEUGEOT", "FIAT",
    "DATSUN", "ISUZU", "OPEL", "SEAT", "CITROEN", "ALFA", "ROMEO",
    "POLICE", "AMBULANCE", "FIRE", "TAXI", "BUS", "TRUCK",
}


def _is_valid_plate(text):
    """
    Return True only if the text looks like a real license plate.
    Key rules:
      1. Must contain at least ONE digit (BOYACA, HONDA etc. have zero digits)
      2. Must contain at least ONE letter (pure numbers are rare/invalid)
      3. Must not be a known car brand or body label word
      4. Total alphanumeric length must be between 4 and 12
    """
    alphanums = re.sub(r'[^A-Z0-9]', '', text.upper())
    digits  = sum(c.isdigit() for c in alphanums)
    letters = sum(c.isalpha() for c in alphanums)
    length  = len(alphanums)

    if digits == 0:          # Pure letters → BOYACA, HONDA etc. → reject
        return False
    if letters == 0:         # Pure digits → unlikely to be a real plate → reject
        return False
    if length < 4 or length > 14:
        return False
    # Reject if the text (stripped of punctuation) exactly matches a known non-plate word
    if alphanums in _NON_PLATE_WORDS:
        return False
    return True



# ── Main Detector ─────────────────────────────────────────────────────────────
class PlateDetector:
    def __init__(self):
        print("Loading YOLOv8n + EasyOCR …")
        base_dir    = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, 'model_weights')
        os.makedirs(weights_dir, exist_ok=True)

        # YOLOv8n — standard COCO model, always publicly downloadable from Ultralytics GitHub.
        # Pre-downloaded during Docker build (see Dockerfile); this just loads from cache.
        self.yolo = YOLO("yolov8n.pt")

        # EasyOCR
        self.reader = easyocr.Reader(
            ['en'], gpu=False,
            model_storage_directory=weights_dir,
            download_enabled=True,
        )
        print("Models ready.")

    # ── OCR: run two preprocessing variants and return the better result ──────
    def _ocr(self, crop):
        best_text, best_conf = "", 0.0
        for variant in (crop, _enhance(crop)):
            img     = _upscale(variant)
            results = self.reader.readtext(img, allowlist=OCR_ALLOWLIST, detail=1)
            results = sorted(results, key=lambda r: r[0][0][1])
            raw, scores = "", []
            for (_, t, p) in results:
                if p > 0.10:
                    raw += t; scores.append(p)
            cleaned = _clean(raw)
            conf    = sum(scores) / len(scores) if scores else 0.0
            if conf > best_conf:
                best_conf, best_text = conf, cleaned
        return best_text, best_conf

    # ── Public API ────────────────────────────────────────────────────────────
    def detect_and_read_plate(self, img_array):
        """Returns (crop_img, plate_text, accuracy). Falls back to full-image OCR."""
        h, w  = img_array.shape[:2]
        results = self.yolo.predict(img_array, conf=0.25, verbose=False, device='cpu')
        boxes   = results[0].boxes if results else None

        candidates = []

        if boxes is not None and len(boxes):
            # Collect all vehicle boxes sorted by confidence (highest first)
            cls_ids = boxes.cls.int().tolist()
            confs   = boxes.conf.tolist()
            xyxys   = boxes.xyxy.tolist()

            vehicle_boxes = [
                (c, xyxy, cls)
                for c, xyxy, cls in sorted(
                    zip(confs, xyxys, cls_ids), reverse=True
                )
                if cls in VEHICLE_CLASSES
            ]

            for yolo_conf, (x1, y1, x2, y2), _ in vehicle_boxes:
                x1, y1 = max(0, int(x1) - 8), max(0, int(y1) - 8)
                x2, y2 = min(w, int(x2) + 8), min(h, int(y2) + 8)
                crop   = img_array[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Look specifically in the LOWER THIRD of the vehicle crop (where plates sit)
                ph = crop.shape[0]
                plate_zone = crop[int(ph * 0.55):, :]
                if plate_zone.size == 0:
                    plate_zone = crop

                text, ocr_conf = self._ocr(plate_zone)
                combined = 0.35 * yolo_conf + 0.65 * ocr_conf
                char_n   = _alnum_count(text)

                if char_n >= MIN_PLATE_CHARS and combined >= ACCURACY_THRESHOLD and _is_valid_plate(text):
                    candidates.append((combined, crop[int(ph * 0.55):, :], text))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            acc, crop_img, text = candidates[0]
            return crop_img, text, acc

        # ── Fallback: full-image EasyOCR ──────────────────────────────────────
        return self._full_image_ocr(img_array)

    def _full_image_ocr(self, img_array):
        h, w   = img_array.shape[:2]
        max_d  = 1200
        if max(h, w) > max_d:
            s   = max_d / max(h, w)
            proc = cv2.resize(img_array, (int(w * s), int(h * s)))
        else:
            proc = img_array

        best_text, best_conf, best_bbox = "", 0.0, None

        for variant in (proc, _enhance(proc)):
            results = self.reader.readtext(variant, allowlist=OCR_ALLOWLIST, detail=1)
            results = sorted(results, key=lambda r: r[0][0][1])
            raw, scores, top_bbox, top_p = "", [], None, 0.0
            for (bbox, t, p) in results:
                if p > 0.10:
                    raw += t; scores.append(p)
                    if p > top_p:
                        top_p, top_bbox = p, bbox
            cleaned = _clean(raw)
            conf    = sum(scores) / len(scores) if scores else 0.0
            if conf > best_conf:
                best_conf, best_text, best_bbox = conf, cleaned, top_bbox

        char_n = _alnum_count(best_text)
        if char_n >= MIN_PLATE_CHARS and best_conf >= ACCURACY_THRESHOLD and best_bbox and _is_valid_plate(best_text):
            sm   = 1.0 if proc.shape == img_array.shape else (max(h, w) / max_d)
            xmin = int(min(p[0] for p in best_bbox) * sm)
            xmax = int(max(p[0] for p in best_bbox) * sm)
            ymin = int(min(p[1] for p in best_bbox) * sm)
            ymax = int(max(p[1] for p in best_bbox) * sm)
            crop = img_array[max(0,ymin-10):min(h,ymax+10), max(0,xmin-10):min(w,xmax+10)]
            return crop, best_text, best_conf

        return img_array, None, 0.0
