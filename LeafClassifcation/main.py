"""
END-TO-END REAL-TIME PLANT DISEASE DATASET CREATION
--------------------------------------------------
✔ Web image download (Bing)
✔ Preprocessing
✔ Augmentation
✔ Train / Val / Test split
✔ No Kaggle used
"""

# =======================
# IMPORTS
# =======================
import os
import cv2
import shutil
import random
import numpy as np

from bing_image_downloader import downloader
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.model_selection import train_test_split


# =======================
# CONFIGURATION
# =======================
BASE_DIR = "plant_dataset"
RAW_DIR = os.path.join(BASE_DIR, "raw_images")
PROC_DIR = os.path.join(BASE_DIR, "processed_images")
AUG_DIR = os.path.join(BASE_DIR, "augmented_images")
FINAL_DIR = os.path.join(BASE_DIR, "final_dataset")

IMG_SIZE = 224
IMAGES_PER_CLASS = 120
AUG_PER_IMAGE = 8

DATASET_CONFIG = {
    "rice": [
        "rice bacterial blight leaf",
        "rice blast disease leaf",
        "rice brown spot leaf",
        "rice bacterial leaf streak",
        "healthy rice leaf"
    ],
    "wheat": [
        "wheat leaf rust",
        "wheat powdery mildew leaf",
        "wheat stripe rust",
        "wheat tan spot leaf",
        "healthy wheat leaf"
    ],
    "maize": [
        "maize common rust leaf",
        "maize grey leaf spot",
        "maize northern leaf blight",
        "maize southern rust",
        "healthy maize leaf"
    ]
}


# =======================
# STEP 1: DOWNLOAD IMAGES
# =======================
def download_images():
    print("\n📥 Downloading images from web...")
    for crop, queries in DATASET_CONFIG.items():
        for query in queries:
            print(f"  → {query}")
            downloader.download(
                query,
                limit=IMAGES_PER_CLASS,
                output_dir=RAW_DIR,
                adult_filter_off=True,
                force_replace=False,
                timeout=60
            )
    print("✅ Download complete\n")


# =======================
# STEP 2: PREPROCESSING
# =======================
def preprocess_images():
    print("🛠 Preprocessing images (Resizing Only)...")
    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, f)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Just resize. No CLAHE, No Sharpening.
                # MobileNetV2 handles its own preprocessing effectively.
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                rel = os.path.relpath(root, RAW_DIR)
                out_dir = os.path.join(PROC_DIR, rel)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, f), img)
    print("✅ Preprocessing complete (Clean Images)\n")


# =======================
# STEP 3: AUGMENTATION
# =======================
def augment_images():
    print("🔁 Augmenting images...")
    datagen = ImageDataGenerator(
        rotation_range=90,
        zoom_range=0.2,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.3, 0.9],
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode="nearest"
    )

    for root, _, files in os.walk(PROC_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img = load_img(os.path.join(root, f),
                               target_size=(IMG_SIZE, IMG_SIZE))
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                rel = os.path.relpath(root, PROC_DIR)
                save_dir = os.path.join(AUG_DIR, rel)
                os.makedirs(save_dir, exist_ok=True)

                i = 0
                for _ in datagen.flow(
                        x, batch_size=1,
                        save_to_dir=save_dir,
                        save_prefix="aug",
                        save_format="jpg"):
                    i += 1
                    if i >= AUG_PER_IMAGE:
                        break
    print("✅ Augmentation complete\n")


# =======================
# STEP 4: SPLIT DATASET
# =======================
def split_dataset():
    print("📊 Splitting dataset (80/10/10)...")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(FINAL_DIR, split), exist_ok=True)

    for cls in os.listdir(AUG_DIR):
        cls_path = os.path.join(AUG_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        train, temp = train_test_split(images, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)

        for name, group in zip(["train", "val", "test"], [train, val, test]):
            out_dir = os.path.join(FINAL_DIR, name, cls)
            os.makedirs(out_dir, exist_ok=True)
            for img in group:
                shutil.copy(
                    os.path.join(cls_path, img),
                    os.path.join(out_dir, img)
                )
    print("✅ Dataset split complete\n")


# =======================
# MAIN EXECUTION
# =======================
if __name__ == "__main__":
    print("\n🚀 DATASET CREATION STARTED\n")
    # download_images()  # Skipped to use existing raw images
    preprocess_images()
    augment_images()
    split_dataset()
    print("🎉 DATASET READY FOR TRAINING!")
