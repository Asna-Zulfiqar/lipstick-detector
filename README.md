# Lipstick Detector

Simple YOLO-based lipstick detection project.

## Structure
```
lipstick-detector/
├── data/
│   ├── raw/                    # Raw downloaded images
│   └── processed/              # Organized dataset
│       ├── train/images/
│       ├── val/images/
│       └── test/images/
├── scripts/
│   ├── collect_data.py        # Download images
│   ├── filter_quality.py     # Filter & organize
│   └── train.py              # Train model
├── models/                    # Trained models
├── dataset.yaml              # YOLO config
└── requirements.txt

```

## Quick Start
1. `pip install -r requirements.txt`
2. `python scripts/collect_data.py` - Download images
3. `python scripts/filter_quality.py` - Organize dataset
4. Annotate images (use Roboflow)
5. `python scripts/train.py` - Train model

## Goal
Detect lipstick objects in images with high accuracy.