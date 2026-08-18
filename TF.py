# ========================================
# BUILD My FIRST TENSORFLOW MODEL
# Complete Beginner Guide
# ========================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"TensorFlow version: {tf.__version__}")

# ========================================
# PART 1: TENSOR BASICS
# ========================================

print("\n" + "="*50)
print("PART 1: TENSOR BASICS")
print("="*50)

# 1. Create tensors
print("\n📊 Creating tensors:")

# Scalar (0D tensor)
scalar = tf.constant(5)
print(f"Scalar: {scalar}")

# Vector (1D tensor)
vector = tf.constant([1, 2, 3, 4, 5])
print(f"Vector: {vector}")

# Matrix (2D tensor)
matrix = tf.constant([[1, 2], [3, 4]])
print(f"Matrix:\n{matrix}")

# 3D tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor shape: {tensor_3d.shape}")

# 2. Tensor properties
print("\n📐 Tensor Properties:")
print(f"Shape: {matrix.shape}")
print(f"Data type: {matrix.dtype}")
print(f"Size: {tf.size(matrix)}")

# 3. Tensor operations
print("\n🔢 Tensor Operations:")
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")
print(f"Dot product: {tf.tensordot(a, b, axes=1)}")

# 4. Creating special tensors
print("\n🎯 Special Tensors:")
print(f"Zeros: {tf.zeros([2, 3])}")
print(f"Ones: {tf.ones([2, 3])}")
print(f"Identity: {tf.eye(3)}")
print(f"Random: {tf.random.normal([2, 3])}")
# ========================================
# PART 2: BUILD A SIMPLE NEURAL NETWORK
# ========================================

print("\n" + "="*50)
print("PART 2: BUILD A SIMPLE NEURAL NETWORK")
print("="*50)

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

print("✅ Model created!")
print(model.summary())
# ========================================
# PART 3: LINEAR REGRESSION WITH TF
# ========================================

print("\n" + "="*50)
print("PART 3: LINEAR REGRESSION WITH TF")
print("="*50)

# Create data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + 3 + np.random.normal(0, 0.5, 100)

print("📊 Data created: 100 samples")

# Build model
model_lr = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model_lr.compile(optimizer='adam', loss='mse')
print(f"Model: {model_lr.summary()}")

# Train model
print("\n🎯 Training Linear Regression...")
history_lr = model_lr.fit(
    X, y, 
    epochs=100, 
    verbose=0
)

print("✅ Training complete!")
print(f"Loss: {history_lr.history['loss'][-1]:.4f}")

# Make predictions
predictions = model_lr.predict(X)

# Visualize
plt.figure(figsize=(10, 5))
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X, predictions, color='red', linewidth=2, label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with TensorFlow')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# ========================================
# PART 4: CLASSIFICATION WITH TF
# ========================================

print("\n" + "="*50)
print("PART 4: CLASSIFICATION WITH TF")
print("="*50)

# Create classification data
from sklearn.datasets import make_classification

X_cls, y_cls = make_classification(
    n_samples=500,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

print(f"📊 Classification data: {len(X_cls)} samples, {X_cls.shape[1]} features")

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Build classification model
model_cls = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_cls.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🎯 Training Classification Model...")
history_cls = model_cls.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    verbose=0
)

print("✅ Training complete!")
print(f"Train Accuracy: {history_cls.history['accuracy'][-1]:.4f}")
print(f"Test Accuracy: {history_cls.history['val_accuracy'][-1]:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss
axes[0].plot(history_cls.history['loss'], label='Training Loss')
axes[0].plot(history_cls.history['val_loss'], label='Validation Loss')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history_cls.history['accuracy'], label='Training Accuracy')
axes[1].plot(history_cls.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

