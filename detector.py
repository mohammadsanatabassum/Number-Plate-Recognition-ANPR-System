import cv2
import imutils
import numpy as np
import easyocr

class PlateDetector:
    def __init__(self):
        print("Initializing EasyOCR Model (may take a moment to download weights on first run...)")
        # Initialize easyocr reader for english
        self.reader = easyocr.Reader(['en'])
        
    def detect_and_read_plate(self, frame):
        """
        Process the frame, detect the license plate shape, extract text, 
        and return a tuple of (cropped_img, text).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Noise reduction & Edge Detection
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
        edged = cv2.Canny(bfilter, 30, 200)
        
        # Find contours
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        location = None
        for contour in contours:
            # Approximate the contour
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4: # looking for rectangles
                location = approx
                break
                
        if location is None:
            return None, None
            
        # Create mask
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Crop mask to bounding box coordinates
        (x, y) = np.where(mask == 255)
        if len(x) == 0 or len(y) == 0:
            return None, None
            
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        
        # Optical Character Recognition
        result = self.reader.readtext(cropped_image)
        
        if len(result) == 0:
            return cropped_image, None
            
        # result format from easyocr: [[bbox, text, confidence], ...]
        # We take the text element from the highest confidence result (index 0)
        text = result[0][-2]
        
        # Clean text
        text = text.replace(" ", "").upper()
        valid_chars = [c for c in text if c.isalnum()]
        cleaned_text = "".join(valid_chars)
        
        if len(cleaned_text) < 3: # Ignore trivially short texts
            return cropped_image, None
            
        return cropped_image, cleaned_text
