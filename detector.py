import cv2
import numpy as np
import easyocr

class PlateDetector:
    def __init__(self):
        print("Initializing EasyOCR Model (may take a moment to download weights on first run...)")
        self.reader = easyocr.Reader(['en'])
        
    def detect_and_read_plate(self, frame):
        """
        Extract text using EasyOCR's highly accurate AI text localization, 
        and return a tuple of (cropped_img, text).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Adaptive Thresholding to force the image into pure black & white!
        # This removes shadows and definitively sharpens ambiguous letter legs.
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # OCR across the binary image
        results = self.reader.readtext(
            thresh, 
            mag_ratio=3.0, 
            allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-.'
        )
        
        if len(results) == 0:
            return None, None
            
        best_cleaned_text = None
        best_cropped = None
        best_confidence = -1
        
        for bbox, text, prob in results:
            raw = text.replace(" ", "").upper()
            # Allow common license plate separating symbols
            valid_chars = [c for c in raw if c.isalnum() or c in "-."]
            cleaned = "".join(valid_chars)
            
            # Plates usually have 4+ characters. 
            if len(cleaned) < 4:
                continue
                
            # Calculate mathematical aspect ratio of the text bounding box
            (tl, tr, br, bl) = bbox
            width = np.linalg.norm(np.array(tr) - np.array(tl))
            height = np.linalg.norm(np.array(br) - np.array(tr))
            
            if height == 0: 
                continue
                
            aspect_ratio = width / height
            
            # True license plates are distinctly horizontal rectangles 
            # We filter out stacked text, tall phone numbers, or square signs
            if aspect_ratio < 1.3 or aspect_ratio > 8.0:
                continue
                
            if prob > best_confidence:
                best_confidence = prob
                best_cleaned_text = cleaned
                
                # Extract bounding box to generate the nice crop for UI
                (tl, tr, br, bl) = bbox
                
                # Expand bounding box slightly for padding
                padding = 5
                y1 = max(0, int(tl[1]) - padding)
                y2 = min(frame.shape[0], int(br[1]) + padding)
                x1 = max(0, int(tl[0]) - padding)
                x2 = min(frame.shape[1], int(br[0]) + padding)
                
                best_cropped = frame[y1:y2, x1:x2]
                
        if best_cleaned_text and best_cropped is not None and best_cropped.size > 0:
            return best_cropped, best_cleaned_text
            
        return None, None
