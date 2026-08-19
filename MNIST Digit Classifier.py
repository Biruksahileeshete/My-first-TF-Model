# ========================================
# MNIST DIGIT CLASSIFIER
# Simple Neural Network with TensorFlow
# ========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print(f"TensorFlow version: {tf.__version__}")

# ========================================
# PART 1: LOAD AND EXPLORE DATA
# ========================================

print("\n" + "="*50)
print("PART 1: LOADING MNIST DATASET")
print("="*50)

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n📊 Dataset Information:")
print(f"   Training images: {X_train.shape[0]}")
print(f"   Training labels: {y_train.shape[0]}")
print(f"   Test images: {X_test.shape[0]}")
print(f"   Test labels: {y_test.shape[0]}")
print(f"   Image shape: {X_train[0].shape}")
print(f"   Pixel values: {X_train[0].min()} to {X_train[0].max()}")

# Display class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n📊 Class Distribution:")
for digit, count in zip(unique, counts):
    print(f"   Digit {digit}: {count} images ({count/len(y_train)*100:.1f}%)")

# ========================================
# PART 2: VISUALIZE DATA
# ========================================

print("\n" + "="*50)
print("PART 2: VISUALIZING DATA")
print("="*50)

# Display sample images
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_train))
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f'Label: {y_train[idx]}')
    ax.axis('off')

plt.tight_layout()
plt.show()

# Display digit statistics
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Class distribution
axes[0].bar(unique, counts, color='skyblue', edgecolor='black')
axes[0].set_title('Training Data Distribution')
axes[0].set_xlabel('Digit')
axes[0].set_ylabel('Number of Samples')
axes[0].grid(True, alpha=0.3)

# Average image per digit
avg_images = []
for digit in range(10):
    digit_images = X_train[y_train == digit]
    avg_images.append(np.mean(digit_images, axis=0))

avg_img = np.vstack(avg_images[:5])
axes[1].imshow(avg_img, cmap='gray')
axes[1].set_title('Average Image per Digit')
axes[1].axis('off')

plt.tight_layout()
plt.show()