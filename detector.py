import cv2
import numpy as np
import easyocr
import re
import os

def post_process_plate(text):
    # UNIVERSAL LOGIC: We strip out absolute noise (like semicolons or slashes) but preserve 
    # numbers, letters, spaces, and standard formatting characters. 
    # We DO NOT hardcode country-specific indexes or string layouts. What the AI sees purely on the physical plate is what it outputs globally.
    text = re.sub(r'[^A-Z0-9\- ]', '', text.upper())
    
    # Strip hallucinated leading and trailing letters caused by physical plate boundaries (bolts/edges)
    text = re.sub(r'^[^A-Z0-9]+', '', text)
    text = re.sub(r'[^A-Z0-9]+$', '', text)
    return text.strip()

class PlateDetector:
    def __init__(self):
        print("Initializing Dual-Stage Architecture: HaarCascade + EasyOCR...")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, 'model_weights')
        
        # Load Deep Learning OCR Engine
        self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=weights_dir, download_enabled=True)
        
        # Load the ultra-fast Optical Plate isolator
        cascade_path = os.path.join(base_dir, 'haarcascade_russian_plate_number.xml')
        self.plate_cascade = cv2.CascadeClassifier(cascade_path)
        print("Models successfully loaded into RAM.")

    def detect_and_read_plate(self, img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
        # Stage 1: Blazing fast physical isolation of objects shaped exactly like License Plates
        # (This bypasses Taxi Signs, billboards, and irrelevant noise universally)
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        
        best_box = None
        highest_conf_text = ""
        
        for (x, y, w, h) in plates:
            plate_crop = img_array[y:y+h, x:x+w]
            if plate_crop.size == 0: continue
            
            # Stage 2: Deep Upscale
            # If the plate is tiny in the video feed, we mathematically heavily zoom in to simulate 300DPI text
            scale = 800 / max(w, h)
            if scale > 1.0:
                plate_crop = cv2.resize(plate_crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

            # Stage 3: Pure Artificial Intelligence Neural Reading
            results = self.reader.readtext(plate_crop)
            results = sorted(results, key=lambda r: r[0][0][1])  # Sort by height for line processing
            
            raw_text = ""
            for (bbox, text, prob) in results:
                # We filter out pure garbage noise inside the crop
                if prob > 0.15:
                    raw_text += text
                    
            cleaned_plate = post_process_plate(raw_text)
            
            # Standard license plates globally are virtually never under 4 alphanumeric lengths
            if len(cleaned_plate.replace(" ", "").replace("-", "")) >= 4:
                highest_conf_text = cleaned_plate
                best_box = (x, y, w, h)
                break
                
        # If the HaarCascade successfully identified the isolated rectangle:
        if best_box is not None:
            (x, y, w, h) = best_box
            display_crop_color = img_array[max(0, y-10):min(img_array.shape[0], y+h+10), max(0, x-10):min(img_array.shape[1], x+w+10)]
            return display_crop_color, highest_conf_text
            
        # FALLBACK STAGE: If the plate was so severely angled that HaarCascade missed the shape entirely, 
        # we fallback to reading the entire raw image via the Neural Network (Original Heavy Approach).
        h_orig, w_orig = img_array.shape[:2]
        max_dim = 1200
        if max(h_orig, w_orig) > max_dim:
            scale = max_dim / max(h_orig, w_orig)
            process_img = cv2.resize(img_array, (int(w_orig * scale), int(h_orig * scale)))
        else:
            process_img = img_array

        results = self.reader.readtext(process_img)
        results = sorted(results, key=lambda r: r[0][0][1])  
        
        raw_text = ""
        best_bbox = None
        highest_conf = 0.0
        
        for (bbox, text, prob) in results:
            if prob > 0.15:
                raw_text += text
                if prob > highest_conf:
                    highest_conf = prob
                    best_bbox = bbox

        final_plate = post_process_plate(raw_text)
        
        if len(final_plate.replace(" ", "").replace("-", "")) >= 4 and best_bbox is not None:
            scale_multiplier = 1.0 if process_img.shape == img_array.shape else (max(h_orig, w_orig) / max_dim)
            x_min = int(min(pt[0] for pt in best_bbox) * scale_multiplier)
            x_max = int(max(pt[0] for pt in best_bbox) * scale_multiplier)
            y_min = int(min(pt[1] for pt in best_bbox) * scale_multiplier)
            y_max = int(max(pt[1] for pt in best_bbox) * scale_multiplier)
            
            display_crop_color = img_array[max(0, y_min-10):min(h_orig, y_max+10), max(0, x_min-10):min(w_orig, x_max+10)]
            return display_crop_color, final_plate
            
        return img_array, None
