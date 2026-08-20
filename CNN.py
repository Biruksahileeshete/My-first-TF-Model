# ========================================
# CNN IMAGE CLASSIFIER
# MNIST Digit Classification with CNN
# ========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

print(f"TensorFlow version: {tf.__version__}")

# ========================================
# PART 1: LOAD AND EXPLORE DATA
# ========================================

print("\n" + "="*50)
print("PART 1: LOADING MNIST DATASET")
print("="*50)

# Load MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n📊 Dataset Information:")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Image shape: {X_train[0].shape}")
print(f"   Pixel range: {X_train.min()} to {X_train.max()}")

# ========================================
# PART 2: PREPROCESS DATA FOR CNN
# ========================================

print("\n" + "="*50)
print("PART 2: PREPROCESSING DATA")
print("="*50)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape for CNN (add channel dimension)
# CNN expects: (samples, height, width, channels)
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)

print(f"✅ Data prepared for CNN:")
print(f"   Training shape: {X_train_cnn.shape}")
print(f"   Test shape: {X_test_cnn.shape}")

# One-hot encode labels
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

print(f"✅ Labels one-hot encoded: {y_train_onehot[0]}")
# ========================================
# PART 3: BUILD CNN MODEL
# ========================================

print("\n" + "="*50)
print("PART 3: BUILDING CNN MODEL")
print("="*50)

def create_cnn_model():
    """Create a CNN model for image classification"""
    
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    return model

# Create the model
model = create_cnn_model()

print("✅ CNN Model created successfully!")
model.summary()

# ========================================
# PART 4: COMPILE MODEL
# ========================================

print("\n" + "="*50)
print("PART 4: COMPILING MODEL")
print("="*50)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model compiled with:")
print("   Optimizer: Adam")
print("   Loss: Categorical Crossentropy")
print("   Metrics: Accuracy")

# ========================================
# PART 5: TRAIN MODEL
# ========================================

print("\n" + "="*50)
print("PART 5: TRAINING CNN MODEL")
print("="*50)

# Callbacks for better training
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

start_time = time.time()

print("🎯 Training started...")
history = model.fit(
    X_train_cnn, y_train_onehot,
    epochs=20,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

training_time = time.time() - start_time
print(f"\n✅ Training complete in {training_time:.2f} seconds!")
# ========================================
# PART 6: EVALUATE MODEL
# ========================================

print("\n" + "="*50)
print("PART 6: EVALUATING MODEL")
print("="*50)

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test_cnn, y_test_onehot, verbose=0)

print(f"\n📊 Test Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Make predictions
y_pred = model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)