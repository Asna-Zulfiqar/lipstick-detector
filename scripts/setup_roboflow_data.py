import shutil
from pathlib import Path

def setup_roboflow_dataset():
    """Move Roboflow dataset to processed directory"""
    
    source_dir = Path("data/lipstick-detector.v1i.yolov8")
    processed_dir = Path("data/processed")
    
    if not source_dir.exists():
        print("❌ Roboflow dataset not found. Extract the zip first.")
        return
    
    # Remove old processed directory
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    
    # Move dataset
    shutil.move(str(source_dir), str(processed_dir))
    
    # Copy and fix dataset.yaml
    yaml_file = processed_dir / "data.yaml"
    if yaml_file.exists():
        # Read and fix paths
        with open(yaml_file, 'r') as f:
            content = f.read()
        
        # Fix relative paths
        content = content.replace('../train/images', 'data/processed/train/images')
        content = content.replace('../valid/images', 'data/processed/valid/images') 
        content = content.replace('../test/images', 'data/processed/test/images')
        
        # Write to root
        with open('dataset.yaml', 'w') as f:
            f.write(content)
        
        print("✅ Copied and fixed dataset.yaml")
    
    print(f"✅ Dataset ready in {processed_dir}")
    print("Next: python scripts/train.py")

if __name__ == "__main__":
    setup_roboflow_dataset()