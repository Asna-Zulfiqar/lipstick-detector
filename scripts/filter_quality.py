#!/usr/bin/env python3
import cv2
import os
import shutil
from pathlib import Path
import random

def is_high_quality(image_path):
    """Check if image is high quality"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    if h < 300 or w < 300:
        return False
    
    # Check blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return blur_score > 100

def organize_dataset():
    """Filter quality images and split into train/val/test"""
    
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        print("Please add images to data/raw/ first")
        return
    
    # Filter high quality images
    quality_images = []
    for img_file in raw_dir.glob("*.jpg"):
        if is_high_quality(img_file):
            quality_images.append(img_file)
    
    print(f"Found {len(quality_images)} high-quality images")
    
    if len(quality_images) < 50:
        print("Need at least 50 high-quality images")
        return
    
    # Shuffle and split
    random.shuffle(quality_images)
    
    train_split = int(0.7 * len(quality_images))
    val_split = int(0.85 * len(quality_images))
    
    splits = {
        'train': quality_images[:train_split],
        'val': quality_images[train_split:val_split],
        'test': quality_images[val_split:]
    }
    
    # Copy images to splits
    for split, images in splits.items():
        img_dir = Path(f"data/processed/{split}/images")
        img_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img_path in enumerate(images):
            new_name = f"lipstick_{i:04d}.jpg"
            shutil.copy2(img_path, img_dir / new_name)
        
        print(f"{split}: {len(images)} images")

if __name__ == "__main__":
    organize_dataset()