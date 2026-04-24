import cv2
import argparse
import os
import time
import uuid
from database import init_db, save_plate
from detector import PlateDetector

def process_frame(frame, detector, recent_plates, cooldown_seconds=5):
    """
    Given a single frame, process it and log the result.
    """
    display_frame = frame.copy()
    cropped_img, text = detector.detect_and_read_plate(frame)
    
    if text:
        curr_time = time.time()
        # Log to db only if we haven't seen this recently
        if text not in recent_plates or (curr_time - recent_plates[text]) > cooldown_seconds:
            print(f"Detected Plate: {text}")
            
            # Save the original frame snapshot
            img_filename = f"captures/{text}_{uuid.uuid4().hex[:6]}.jpg"
            cv2.imwrite(img_filename, frame)
            
            # Insert into database
            save_plate(text, img_filename)
            recent_plates[text] = curr_time
            
        # Draw on frame
        cv2.putText(display_frame, f"Plate: {text}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
    return display_frame

def main():
    parser = argparse.ArgumentParser(description="ANPR System")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam or file path)")
    parser.add_argument("--image", type=str, help="Process a single image instead of video")
    args = parser.parse_args()
    
    # Setup
    print("Setting up sqlite DB...")
    init_db()
    
    if not os.path.exists("captures"):
        os.makedirs("captures")
        
    detector = PlateDetector()
    recent_plates = {}
    
    if args.image:
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
            
        frame = cv2.imread(args.image)
        if frame is None:
            print("Failed to load image.")
            return
            
        print(f"Processing image: {args.image}")
        res_frame = process_frame(frame, detector, recent_plates)
        cv2.imshow("Detected Plate", res_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Assuming video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Failed to open video source: {source}")
        return
        
    print("Starting video stream. Press 'q' to quit.")
    
    # Main video loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or failed to read frame.")
            break
            
        # Process the frame
        res_frame = process_frame(frame, detector, recent_plates)
        
        cv2.imshow("ANPR Live Stream", res_frame)
        
        # Quit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
