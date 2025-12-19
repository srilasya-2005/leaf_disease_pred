
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable OneDNN to prevent MKL crashes
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import json

# =======================
# CONFIGURATION
# =======================
IMG_SIZE = 224
BATCH_SIZE = 8 # Reduced batch size for V2M (Larger model = more VRAM)
EPOCHS = 60 # Extended training
LEARNING_RATE = 1e-4

BASE_DIR = "plant_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "final_dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "final_dataset", "val")
MODEL_SAVE_PATH = "plant_disease_model_final.keras"

def build_model(num_classes):
    """
    EfficientNetV2M: High Capacity Model for 90% Target
    """
    # pooling='avg' includes GlobalAveragePooling inside the model 
    # This prevents the shape mismatch error we saw earlier
    base_model = EfficientNetV2M(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling='avg' 
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    x = base_model.output
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x) # Low dropout to allow high accuracy
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

def train():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error: Training directory not found at {TRAIN_DIR}")
        return

    # =======================
    # DATA GENERATORS
    # =======================
    print("🎨 Setting up data generators...")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    print(f"📦 Loading training data from {TRAIN_DIR}...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    print(f"📦 Loading validation data from {VAL_DIR}...")
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_generator.num_classes
    print(f"🔍 Found {num_classes} classes.")
    
    # =======================
    # BUILD MODEL
    # =======================
    model = build_model(num_classes)
    
    # Standard Crossentropy (No Smoothing) for high raw accuracy
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("\n" + "="*50)
    print(f"MODEL ARCHITECTURE: EfficientNetV2M (Medium)")
    print(f"TARGET: 90% Accuracy")
    print("="*50)
    model.summary()

    # =======================
    # CALLBACKS
    # =======================
    callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=10, 
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # =======================
    # PHASE 1: WARMUP
    # =======================
    print("\n" + "="*50)
    print("🚀 PHASE 1: Warmup Head (10 Epochs)")
    print("="*50)
    
    history1 = model.fit(
        train_generator,
        epochs=10, 
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ Phase 1 complete.")
    print(f"   Best val_accuracy: {max(history1.history['val_accuracy']):.4f}")

    # =======================
    # PHASE 2: DEEP FINE-TUNING
    # =======================
    print("\n" + "="*50)
    print("🔓 PHASE 2: Unfreezing EfficientNetV2M")
    print("="*50)
    
    model.trainable = True
    
    # Freeze BN for stability
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            
    # Compile with Adam 1e-5
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Patience increase for fine-tuning
    callbacks[1] = EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
    
    history2 = model.fit(
        train_generator,
        initial_epoch=history1.epoch[-1],
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(MODEL_SAVE_PATH)
    print(f"\n✅ Training complete! Model saved to {MODEL_SAVE_PATH}")

    # Save class indices
    with open("class_indices.json", "w") as f:
        json.dump(train_generator.class_indices, f, indent=2)

    # Summary
    all_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    print(f"\n🏆 Final Validation Accuracy: {max(all_acc):.4f}")

if __name__ == "__main__":
    train()