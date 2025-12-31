import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(page_title="Lipstick Detector", page_icon="💄", layout="centered")

# Title
st.title("💄 Lipstick Detector")
st.write("Upload an image to detect lipstick objects")

# Load model
@st.cache_resource
def load_model():
    # Try different possible model paths
    possible_paths = [
        "runs/detect/lipstick_detector3/weights/best.pt",
        "runs/detect/lipstick_detector/weights/best.pt",
        "runs/detect/lipstick_detector2/weights/best.pt",
        "runs/detect/lipstick_detector/weights/last.pt"
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            return YOLO(model_path)
    
    st.error("❌ Model not found. Please train the model first.")
    return None

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file and model:
    # Load original image
    image = Image.open(uploaded_file)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Run prediction
    with st.spinner("Detecting lipstick..."):
        results = model(tmp_path)
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_img, use_column_width=True)
        
        # Show detection stats
        detections = len(results[0].boxes)
        if detections > 0:
            st.success(f"✅ Found {detections} lipstick object(s)")
            
            # Show confidence scores
            confidences = [box.conf[0].item() for box in results[0].boxes]
            avg_conf = np.mean(confidences)
            max_conf = max(confidences)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Max Confidence", f"{max_conf:.2%}")
            with col2:
                st.metric("Avg Confidence", f"{avg_conf:.2%}")
                
        else:
            st.warning("❌ No lipstick detected")
    
    # Cleanup
    os.unlink(tmp_path)

# Instructions
if not uploaded_file:
    st.info("👆 Upload an image to get started")
    
    # Show example
    st.subheader("How it works:")
    st.write("1. Upload an image containing lipstick")
    st.write("2. The AI model will detect and highlight lipstick objects")
    st.write("3. View confidence scores and detection results")