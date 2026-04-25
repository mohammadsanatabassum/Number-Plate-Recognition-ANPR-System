import cv2
import numpy as np
import easyocr
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
        # We enforce gpu=False because Hugging Face free spaces mostly provide pure CPU instances. 
        # But this is totally okay, EasyOCR CPU is very fast for small bounding boxes.
        print("Initializing Heavyweight EasyOCR AI...")
        
        # Determine execution directory to safely find the model_weights folder
        base_dir = os.path.dirname(os.path.abspath(__file__))
        weights_dir = os.path.join(base_dir, 'model_weights')
        
        # Load the pre-trained neural network
        self.reader = easyocr.Reader(['en'], gpu=False, model_storage_directory=weights_dir, download_enabled=False)
        print("EasyOCR successfully loaded into RAM.")

    def detect_and_read_plate(self, img_array):
        # Neural Networks are extremely robust. We don't have to manually draw contours anymore!
        # EasyOCR has a built-in highly advanced CRAFT Text Detector. 
        # We can just give it the image directly.
        
        # To avoid processing heavy backgrounds in huge images, we scale it down safely.
        h, w = img_array.shape[:2]
        max_dimension = 1200
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            process_img = cv2.resize(img_array, (int(w * scale), int(h * scale)))
        else:
            process_img = img_array.copy()

        # The EasyOCR reader returns a list of bounding boxes, text, and confidence scores.
        # Format: [([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], 'Text', confidence), ...]
        results = self.reader.readtext(process_img)
        
        if not results:
            return img_array, None

        # Because a license plate might be split onto two lines (e.g. 29A on top, 33185 on bottom),
        # EasyOCR might detect them as TWO separate text blocks.
        # We will concatenate all highly confident alphanumeric text blocks in the center area.
        
        # Sort results top-to-bottom so multi-line plates read in the correct order
        results = sorted(results, key=lambda r: r[0][0][1])  
        
        raw_text = ""
        best_box = None
        highest_conf = 0.0
        
        for (bbox, text, prob) in results:
            # We filter out pure garbage noise (like tiny blurred signs)
            if prob > 0.15:
                # Merge the text
                raw_text += text
                
                # We'll use the bounding box of the highest confidence text block for the visualization
                if prob > highest_conf:
                    highest_conf = prob
                    best_box = bbox

        final_plate = post_process_plate(raw_text)
        
        # If we failed to build a valid plate length, return None
        if len(final_plate) < 4:
            return img_array, None

        # Draw a beautiful green visualization box directly on the image around where the AI found the text
        if best_box is not None:
            # Convert bounding box coordinates to integers
            # Scale coordinates back up if the original image was larger
            scale_multiplier = 1.0 if process_img.shape == img_array.shape else (max(h, w) / max_dimension)
            
            x_min = int(min(pt[0] for pt in best_box) * scale_multiplier)
            x_max = int(max(pt[0] for pt in best_box) * scale_multiplier)
            y_min = int(min(pt[1] for pt in best_box) * scale_multiplier)
            y_max = int(max(pt[1] for pt in best_box) * scale_multiplier)
            
            # Extract just that sub-region to display as the "plate crop"
            display_crop_color = img_array[max(0, y_min-10):min(h, y_max+10), max(0, x_min-10):min(w, x_max+10)]
            return display_crop_color, final_plate
            
        return img_array, final_plate
