from ultralytics import YOLO
from pathlib import Path

def train_lipstick_detector():
    """Train YOLOv8 model on lipstick dataset"""
    
    # Check if dataset exists
    if not Path("dataset.yaml").exists():
        print("❌ dataset.yaml not found. Run setup_roboflow_data.py first")
        return
    
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # nano version for faster training
    
    # Train with fewer epochs and verbose output
    results = model.train(
        data='dataset.yaml',
        epochs=20,  # Reduced from 50
        imgsz=640,
        batch=16,
        name='lipstick_detector',
        save=True,
        plots=True,
        verbose=True  # Show detailed metrics each epoch
    )
    
    print("✅ Training complete!")
    print(f"Best model saved to: runs/detect/lipstick_detector/weights/best.pt")
    
    # Test the model
    model = YOLO('runs/detect/lipstick_detector/weights/best.pt')
    
    # Validate
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")

if __name__ == "__main__":
    train_lipstick_detector()