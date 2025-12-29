from ultralytics import YOLO
from pathlib import Path

def train_lipstick_detector():
    """Train YOLOv8 model on lipstick dataset"""
    
    # Check if dataset exists
    if not Path("dataset.yaml").exists():
        print("❌ dataset.yaml not found. Run setup_roboflow_data.py first")
        return
    
    # Load YOLOv8 nano model
    model = YOLO('yolov8n.pt')  # nano version for faster training
    
    # Callback to print metrics after each epoch
    def print_epoch_metrics(epoch, metrics, **kwargs):
        print(f"Epoch {epoch+1}:")
        print(f"  Precision: {metrics['metrics/precision']*100:.2f}%")
        print(f"  Recall:    {metrics['metrics/recall']*100:.2f}%")
        print(f"  mAP50:     {metrics['metrics/mAP_50']*100:.2f}%")
        print(f"  mAP50-95:  {metrics['metrics/mAP_50_95']*100:.2f}%")
        print("-"*40)

    # Train model
    # In YOLOv8, you can enable augment=True in train()
    results = model.train(
        data='dataset.yaml',
        epochs=20,
        batch=16,
        imgsz=640,
        name='lipstick_detector',
        save=True,
        plots=True,
        verbose=True,
        augment=True  # important for small datasets
    )

    
    print("✅ Training complete!")
    print(f"Best model saved to: runs/detect/lipstick_detector/weights/best.pt")
    
    # Validate
    metrics = model.val()
    print("Final Validation Metrics:")
    print(f"mAP50:     {metrics.box.map50*100:.2f}%")
    print(f"mAP50-95:  {metrics.box.map*100:.2f}%")

if __name__ == "__main__":
    train_lipstick_detector()
