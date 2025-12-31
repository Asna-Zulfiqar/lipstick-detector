from ultralytics import YOLO
import cv2
from pathlib import Path

def predict_lipstick(image_path, model_path="runs/detect/lipstick_detector/weights/best.pt"):
    """Detect lipstick in a single image"""
    
    # Load trained model
    model = YOLO(model_path)
    
    # Run prediction
    results = model(image_path)
    
    # Show results
    for r in results:
        # Plot results on image
        im_array = r.plot()
        
        # Save result
        output_path = f"prediction_{Path(image_path).stem}.jpg"
        cv2.imwrite(output_path, im_array)
        
        print(f"✅ Prediction saved to: {output_path}")
        
        # Print detections
        if len(r.boxes) > 0:
            print(f"Found {len(r.boxes)} lipstick(s)")
            for box in r.boxes:
                conf = box.conf[0].item()
                print(f"  - Confidence: {conf:.2f}")
        else:
            print("No lipstick detected")

if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_lipstick(image_path)