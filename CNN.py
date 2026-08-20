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