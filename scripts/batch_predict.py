from ultralytics import YOLO
from pathlib import Path
import cv2

def batch_predict(input_folder, model_path="runs/detect/lipstick_detector/weights/best.pt"):
    """Detect lipstick in all images in a folder"""
    
    model = YOLO(model_path)
    input_dir = Path(input_folder)
    output_dir = Path("predictions")
    output_dir.mkdir(exist_ok=True)
    
    # Supported image formats
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    images = [f for f in input_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        print("No images found in folder")
        return
    
    print(f"Processing {len(images)} images...")
    
    for img_path in images:
        try:
            # Predict
            results = model(str(img_path))
            
            # Save result
            for r in results:
                im_array = r.plot()
                output_path = output_dir / f"pred_{img_path.name}"
                cv2.imwrite(str(output_path), im_array)
                
                # Print stats
                detections = len(r.boxes)
                if detections > 0:
                    max_conf = max([box.conf[0].item() for box in r.boxes])
                    print(f"✅ {img_path.name}: {detections} lipstick(s), max conf: {max_conf:.2f}")
                else:
                    print(f"❌ {img_path.name}: No lipstick detected")
                    
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    print(f"\n✅ Results saved to {output_dir}/")

if __name__ == "__main__":
    folder = input("Enter folder path with images: ")
    batch_predict(folder)