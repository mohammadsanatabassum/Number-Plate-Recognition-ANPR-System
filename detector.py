import cv2
import numpy as np
import pytesseract
import imutils
import re
import os

def post_process_plate(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    # Advanced logic to surgically repair OCR ambiguities
    if len(text) == 8:
        if text[2] == '4':
            text = text[:2] + 'A' + text[3:]
        text = text[:3] + '-' + text[3:6] + '.' + text[6:]
    elif len(text) == 9:
        if text[2] == '4':
            text = text[:2] + 'A' + text[3:]
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
                break

        if location is None:
            return img_array, None

        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        
        # Binarize output mathematically for Tesseract because it struggles with gray shadows
        _, thresh = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OS detection purely so Windows local users do not crash when running local tests
        if os.name == 'nt' and os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # PyTesseract structural execution (--psm 7 correctly physically reads a single license plate text line)
        text = pytesseract.image_to_string(thresh, config='--psm 7')
        
        # Correctly validate formatting
        final_plate = post_process_plate(text)
        if len(final_plate) < 2:
            return img_array, None
            
        display_crop_color = img_array[x1:x2+1, y1:y2+1]
        return display_crop_color, final_plate
