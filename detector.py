import cv2
import numpy as np
import pytesseract
import imutils
import re
import os

def post_process_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Strip hallucinated leading and trailing letters caused by plate screws/shadows
    text = re.sub(r'^[A-Z]+', '', text)
    text = re.sub(r'[A-Z]+$', '', text)

    # Advanced logic to surgically repair OCR ambiguities
    if len(text) >= 3 and text[2] == '4':
        text = text[:2] + 'A' + text[3:]
        
    # Dynamically inject dash and dot formatting regardless of length
    # Based on standard Vietnam architecture: first 3 characters, dash, middle, dot, last 2
    if len(text) >= 7:
        text = text[:3] + '-' + text[3:-2] + '.' + text[-2:]
        
    return text

class PlateDetector:
    def __init__(self):
        print("Initialized ultra-lightweight Tesseract engine.")

    def detect_and_read_plate(self, img_array):
        # Memory-safe scaling
        h, w = img_array.shape[:2]
        max_dimension = 1000
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                
                # Extract this specific rectangle
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [location], 0, 255, -1)
                
                (x, y) = np.where(mask == 255)
                # If area is too small, skip
                if len(x) == 0 or len(y) == 0:
                    continue
                
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_image = gray[x1:x2+1, y1:y2+1]
                
                # Scale up perfectly
                cropped_upscale = cv2.resize(cropped_image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                _, thresh = cv2.threshold(cropped_upscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Tesseract OCR Execution
                if os.name == 'nt' and os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
                    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

                custom_config = r'-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6'
                text = pytesseract.image_to_string(thresh, config=custom_config)
                
                # Filter noise results
                final_plate = post_process_plate(text)
                
                # A true license plate is almost always 7 to 9 characters 
                # (e.g. '29A33185' is 8 characters long). 
                # We strictly skip anything less than 5 characters to avoid taxi signs or windshield glares.
                if len(final_plate) >= 6:
                    display_crop_color = img_array[x1:x2+1, y1:y2+1]
                    return display_crop_color, final_plate
                    
        # If no rectangles contained valid text
        return img_array, None
