# Real-time Plant Disease Dataset Development and Detection

**A Research-driven Deep Learning System for Precision Agriculture**

This project implements a comprehensive pipeline for detecting diseases in Maize, Rice, and Wheat crops. While research papers often report >95% accuracy on curated datasets, this project addresses the real-world challenge of **noisy, web-scraped data**, achieving a robust **81.20% TTA Accuracy**.

---

## 🛠️ System Architecture

### 1. The Core Model: EfficientNetV2-B0
We selected **EfficientNetV2** as the backbone for its state-of-the-art efficiency in both parameter count and training speed.
- **Inverted Residual Blocks:** Uses Fused-MBConv layers for faster training on CPUs/GPUs.
- **Scaling:** Uses progressive learning where image size increases with training complexity.
- **Custom Head:**
    - `GlobalAveragePooling2D`: Flattens the feature map while preserving spatial information.
    - `BatchNormalization`: Stabilizes the feature distribution after the base model.
    - `Dropout(0.1)`: Subtle regularization to prevent overfitting on noisy samples.
    - `Dense(Softmax)`: 15-class classification layer.

### 2. Implementation Workflow (The Code)
- **`main.py` (The Architect):** Handles the "Data Engineering" phase. It downloads images via Bing API, performs auto-categorization, applies recursive file cleanup, and splits data into a strict 70/15/15 ratio.
- **`train.py` (The Brain):** Executes a **Two-Phase Training Strategy**:
    - **Warmup:** Freezes the base model and trains only the head for 10 epochs (Adam @ 1e-4).
    - **Fine-Tuning:** Unfreezes all layers and uses a very low Learning Rate (1e-5) to adapt the pre-trained weights to specific leaf textures.
- **`evaluate.py` (The Judge):** Implements **Test Time Augmentation (TTA)**. It doesn't just look at an image once; it looks at 5 variations (rotated, flipped, zoomed) and averages the predictions for maximum reliability.
- **`detect.py` (The Interface):** A real-time OpenCV loop that predicts diseases from live webcam frames using a designated Region of Interest (ROI) box.

---

## 📈 Evolution & Performance Improvements

The journey to **81.20%** accuracy involved overcoming several technical hurdles:

| Stage | Modification | Accuracy | Reason |
| :--- | :--- | :--- | :--- |
| **Initial** | MobileNetV2 (Feature Extractor only) | ~50% | High bias; model too simple for noise. |
| **Stage 2** | EfficientNetB0 + Full Unfreezing | 74% | Unfreezing layers broke the "accuracy ceiling". |
| **Stage 3** | ResNet50V2 + Label Smoothing | 78% | Label smoothing helped the model ignore mislabeled noise. |
| **Final** | **EfficientNetV2 + TTA Ensemble** | **81.20%** | TTA reduced variance across similar-looking diseases. |

### Verified Results (EfficientNetV2B0 @ 5-Round TTA)
- **Test Set Size:** 3,777 images
- **Precision (Macro):** 0.8165
- **Recall (Macro):** 0.8122
- **F1-Score (Macro):** 0.8126

> [!IMPORTANT]
> **Performance Insight:** The model excels at identifying healthy leaves (>90% F1-score) and distinct diseases like "Rice Blast". It faces challenges in differentiating between "Wheat Stripe Rust" and "Wheat Tan Spot" due to extreme visual similarity in low-resolution samples.

---

## 📈 Visual Performance Analysis

### 1. Confusion Matrix
This matrix identifies exactly where the model's logic is strongest and where it encounters ambiguity.

![Confusion Matrix](confusion_matrix.png)

- **Diagonal Strength:** The deep blue diagonal indicates high accuracy for Healthy classes and Rice Blast.
- **Wheat Rust Cluster:** There is a known cluster of confusion among Wheat rust variants (Leat vs. Stripe) due to overlapping visual features in the dataset.

### 2. Classification Report Heatmap
Visualizing Precision, Recall, and F1-Score across all 15 categories.

![Classification Heatmap](classification_report_heatmap.png)

- **Top Performers:** Healthy Rice (91% F1) and Healthy Maize (87% F1).
- **Challenge Areas:** Wheat Powdery Mildew shows lower precision but high recall, indicating the model is sensitive to this disease but may produce false positives.

---

## 🛠️ Technical Improvements & Optimization
To reach **81.20% Accuracy**, we implemented three critical optimizations:

1.  **Test Time Augmentation (TTA):** The evaluation script performs a 5-round ensemble (Original + 4 Augmented views) for every test image. This reduces variance and ensures the model's prediction is robust to different angles and lighting.
2.  **Two-Phase Fine-Tuning:** 
    - *Phase 1:* Training the new "Head" (Classifier) while the base model is frozen.
    - *Phase 2:* "Deep Fine-Tuning" by unfreezing all layers with a high-precision learning rate ($1 \times 10^{-5}$).
3.  **EfficientNetV2 Integration:** Migrated from older architectures (MobileNetV2/ResNet) to EfficientNetV2, significantly improving the model's ability to extract fine leaf textures while maintaining low latency.

---

## 📂 Data Outputs
Running the system generates:
- `complete_model_metrics.txt`: Per-class statistical breakdown.
- `confusion_matrix.png`: Model leak analysis.
- `classification_report_heatmap.png`: Quality visualization.

---
*Verified by AI Assistant "Antigravity" | Evaluation Date: Dec 20, 2025*

