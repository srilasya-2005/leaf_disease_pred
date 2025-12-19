import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable OneDNN to prevent MKL crashes
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =======================
# CONFIGURATION
# =======================
IMG_SIZE = 224
BATCH_SIZE = 32

# Paths
BASE_DIR = "plant_dataset"
VAL_DIR = os.path.join(BASE_DIR, "final_dataset", "val")
MODEL_PATHS = ["plant_disease_model_final copy 2.keras", "plant_disease_model_final.keras"]

def evaluate():
    # 1. Find Model
    model_path = None
    for path in MODEL_PATHS:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("Error: No trained model found!")
        print(f"   Checked: {MODEL_PATHS}")
        return


    print(f"Loading model from: {model_path}...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup Data Generator (Validation Set)
    if not os.path.exists(VAL_DIR):
        print(f"Error: Validation directory not found at {VAL_DIR}")
        return

    print("Setting up data generator...")
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False  # CRITICAL: Shuffle must be False for correct metrics
    )

    # 3. Predict with Test Time Augmentation (TTA)
    print("\nGenerating predictions with TTA (5x ensemble)...")
    
    # Standard generator (No Augmentation) - for Ground Truth
    val_gen_no_aug = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    
    # TTA Generator (With Augmentation)
    # We want mild augmentations to test robustness
    tta_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    
    tta_generator = tta_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False 
    )
    
    # TTA Rounds
    preds_tta = []
    TTA_STEPS = 5
    
    # 1. Base Prediction (Clean image)
    print("   Round 1/5 (Base)...")
    preds_base = model.predict(val_gen_no_aug, verbose=1)
    preds_tta.append(preds_base)
    
    # 2. Augmented Rounds
    for i in range(TTA_STEPS - 1):
        print(f"   Round {i+2}/5 (Augmented)...")
        # Important: Reset generator to ensure alignment
        tta_generator.reset()
        preds = model.predict(tta_generator, verbose=1)
        preds_tta.append(preds)
        
    print("   Averaging predictions...")
    # Average all predictions
    final_preds = np.mean(preds_tta, axis=0)
    
    y_pred = np.argmax(final_preds, axis=1)
    y_true = val_gen_no_aug.classes
    class_labels = list(val_gen_no_aug.class_indices.keys())

    # 4. Metrics
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Calculate Accuracy Manually from TTA
    correct_count = np.sum(y_pred == y_true)
    total_count = len(y_true)
    acc = correct_count / total_count
    
    # Calculate Loss (Categorical Crossentropy)
    # Clip predictions to avoid log(0)
    epsilon = 1e-7
    y_pred_clipped = np.clip(final_preds, epsilon, 1 - epsilon)
    # Create one-hot encoded y_true
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))
    loss = -np.sum(y_true_onehot * np.log(y_pred_clipped)) / total_count
    
    print(f"TTA Test Loss:     {loss:.4f}")
    print(f"TTA Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
    print(report)

    # 5. Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # --- PLOTTING ---
    print("\nGenerating Graphs...")
    
    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix (TTA)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("   Saved: confusion_matrix.png")
    
    # 2. Classification Report Heatmap
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    # Remove 'support' column and 'accuracy' row for heatmap clarity
    if 'accuracy' in df_report.index:
        df_report = df_report.drop(index='accuracy')
    
    df_plot = df_report.drop(columns=['support'], errors='ignore')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_plot, annot=True, cmap='RdYlGn', vmin=0, vmax=1.0)
    plt.title('Classification Report Metrics')
    plt.tight_layout()
    plt.savefig('classification_report_heatmap.png')
    print("   Saved: classification_report_heatmap.png")

    
    # Save Everything to a comprehensive file
    output_filename = "complete_model_metrics.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("       PLANT DISEASE MODEL - PERFORMANCE REPORT\n")
        f.write("       (With Test Time Augmentation - 5 Rounds)\n")
        f.write("==================================================\n\n")
        f.write(f"Model Path:    {model_path}\n")
        f.write(f"Test Images:   {val_gen_no_aug.samples}\n")
        f.write(f"TTA Accuracy:  {acc:.6f} ({acc*100:.2f}%)\n")
        f.write(f"TTA Loss:      {loss:.6f}\n\n")
        
        f.write("--------------------------------------------------\n")
        f.write("              CLASSIFICATION REPORT\n")
        f.write("--------------------------------------------------\n")
        f.write(report)
        f.write("\n\n")
        
        f.write("--------------------------------------------------\n")
        f.write("                 CONFUSION MATRIX\n")
        f.write("--------------------------------------------------\n")
        f.write("Row: True Class, Column: Predicted Class\n")
        f.write("Class Indices:\n")
        for k, v in val_gen_no_aug.class_indices.items():
            f.write(f" {v}: {k}\n")
        f.write("\nMatrix:\n")
        f.write(np.array2string(cm, separator=', ', threshold=np.inf))
        
    print(f"\n✅ All metrics saved to '{output_filename}'")

if __name__ == "__main__":
    evaluate()
