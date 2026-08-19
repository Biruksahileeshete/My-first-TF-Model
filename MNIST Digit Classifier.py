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
# ========================================
# PART 3: PREPROCESS DATA
# ========================================

print("\n" + "="*50)
print("PART 3: PREPROCESSING DATA")
print("="*50)

# Normalize pixel values (0-255 -> 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

print("✅ Pixel values normalized (0-1)")

# Reshape for neural network
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)

print(f"✅ Data reshaped for neural network:")
print(f"   Training: {X_train_flat.shape}")
print(f"   Testing: {X_test_flat.shape}")

# One-hot encode labels (optional but good practice)
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

print(f"✅ Labels one-hot encoded: {y_train_onehot[0]}")

# ========================================
# PART 4: BUILD NEURAL NETWORK
# ========================================

print("\n" + "="*50)
print("PART 4: BUILDING NEURAL NETWORK")
print("="*50)

# Model 1: Simple 2-layer network
model_simple = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

print("🔧 Simple Model (2 layers):")
model_simple.summary()

# Model 2: Deep network with dropout
model_deep = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

print("\n🔧 Deep Model (with dropout):")
model_deep.summary()

# ========================================
# PART 5: COMPILE AND TRAIN MODELS
# ========================================

print("\n" + "="*50)
print("PART 5: TRAINING MODELS")
print("="*50)

# Compile simple model
model_simple.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train simple model
print("\n🎯 Training Simple Model...")
history_simple = model_simple.fit(
    X_train_flat, y_train_onehot,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\n✅ Simple Model Training Complete!")

# Compile deep model
model_deep.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train deep model
print("\n🎯 Training Deep Model...")
history_deep = model_deep.fit(
    X_train_flat, y_train_onehot,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

print("\n✅ Deep Model Training Complete!")
# ========================================
# PART 6: EVALUATE MODELS
# ========================================

print("\n" + "="*50)
print("PART 6: EVALUATING MODELS")
print("="*50)

# Evaluate simple model
test_loss_simple, test_acc_simple = model_simple.evaluate(
    X_test_flat, y_test_onehot, verbose=0
)

# Evaluate deep model
test_loss_deep, test_acc_deep = model_deep.evaluate(
    X_test_flat, y_test_onehot, verbose=0
)

print("\n📊 Model Performance on Test Data:")
print(f"Simple Model - Loss: {test_loss_simple:.4f}, Accuracy: {test_acc_simple:.4f} ({test_acc_simple*100:.2f}%)")
print(f"Deep Model   - Loss: {test_loss_deep:.4f}, Accuracy: {test_acc_deep:.4f} ({test_acc_deep*100:.2f}%)")

# Make predictions
y_pred_simple = model_simple.predict(X_test_flat)
y_pred_simple_classes = np.argmax(y_pred_simple, axis=1)

y_pred_deep = model_deep.predict(X_test_flat)
y_pred_deep_classes = np.argmax(y_pred_deep, axis=1)

# ========================================
# PART 7: CONFUSION MATRIX
# ========================================

print("\n" + "="*50)
print("PART 7: CONFUSION MATRIX")
print("="*50)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')

# Confusion matrix for simple model
cm_simple = confusion_matrix(y_test, y_pred_simple_classes)
sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Simple Model')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Confusion matrix for deep model
cm_deep = confusion_matrix(y_test, y_pred_deep_classes)
sns.heatmap(cm_deep, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Deep Model')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()