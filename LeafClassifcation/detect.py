
import cv2
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =======================
# CONFIGURATION
# =======================
MODEL_PATH = "plant_disease_model_final.keras"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.65  # Increased threshold to reduce false positives

def load_resources():
    print("⏳ Loading model and resources...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found.")
        print("   Please run 'train.py' first to train the model.")
        return None, None

    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"❌ Error: Class indices file '{CLASS_INDICES_PATH}' not found.")
        print("   Please run 'train.py' first.")
        return None, None

    # Load Model
    model = load_model(MODEL_PATH)
    
    # Load Class Indices
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)
    
    # Invert the dictionary to get {index: class_name}
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    print("✅ Model loaded successfully.")
    return model, idx_to_class

def detect_realtime():
    model, idx_to_class = load_resources()
    if model is None:
        return

    # Start Video Capture
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("\n🎥 Starting Real-time Detection...")
    print("   Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame.")
            break

        # Get Frame Dimensions
        h, w, _ = frame.shape
        
        # Define Region of Interest (ROI) - Center Square
        # Box size: 300x300 (or larger/smaller depending on needs)
        box_size = 300
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        # 1. Preprocess ROI for Model
        if roi.size != 0:
            roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(roi_resized, axis=0)
            img_array = preprocess_input(img_array)

            # 2. Prediction
            preds = model.predict(img_array, verbose=0)
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx]
            
            label = idx_to_class[class_idx]
        else:
            confidence = 0.0
            label = "None"

        # 3. Visualization
        # Draw ROI Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, "Place Leaf Here", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Logic for Output
        if confidence > CONFIDENCE_THRESHOLD:
            color = (0, 255, 0) # Green
            text = f"{label} ({confidence:.1%})"
            cv2.putText(frame, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            color = (0, 0, 255) # Red
            text = "Analyzing..." if confidence > 0.3 else "No Plant Detected"
            cv2.putText(frame, text, (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Instructions
        cv2.putText(frame, "Press 'q' to exit", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("Real-time Plant Disease Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_realtime()
