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
# ========================================
# PART 7: VISUALIZE TRAINING HISTORY
# ========================================

print("\n" + "="*50)
print("PART 7: TRAINING HISTORY")
print("="*50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('CNN Training History', fontsize=14, fontweight='bold')

# Loss
axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0].set_title('Loss Over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_title('Accuracy Over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# PART 8: CONFUSION MATRIX
# ========================================

print("\n" + "="*50)
print("PART 8: CONFUSION MATRIX")
print("="*50)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - CNN', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Digit')
plt.ylabel('Actual Digit')
plt.show()

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_classes, digits=3))

# ========================================
# PART 9: VISUALIZE FILTERS (Feature Maps)
# ========================================

print("\n" + "="*50)
print("PART 9: VISUALIZING CNN FILTERS")
print("="*50)

# Get first convolutional layer weights
conv_layer = model.layers[0]
weights, biases = conv_layer.get_weights()

# Normalize weights for visualization
weights_normalized = (weights - weights.min()) / (weights.max() - weights.min())

# Display first 8 filters
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle('CNN Filters (Feature Detectors)', fontsize=14, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < 8:
        filter_img = weights_normalized[:, :, 0, i]
        ax.imshow(filter_img, cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')

plt.tight_layout()
plt.show()

# ========================================
# PART 10: VISUALIZE FEATURE MAPS
# ========================================

print("\n" + "="*50)
print("PART 10: VISUALIZING FEATURE MAPS")
print("="*50)

# Create a model that outputs feature maps
layer_outputs = [layer.output for layer in model.layers[:4]]  # First 4 layers
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

# Pick a sample image
sample_idx = np.random.randint(0, len(X_test))
sample_image = X_test_cnn[sample_idx:sample_idx+1]

# Get activations
activations = activation_model.predict(sample_image)

# Display feature maps
fig, axes = plt.subplots(4, 6, figsize=(12, 8))
fig.suptitle(f'Feature Maps for Digit {y_test[sample_idx]}', fontsize=14, fontweight='bold')

for i, (ax, activation) in enumerate(zip(axes.flat, activations)):
    if i < len(activations):
        layer_activation = activation[0, :, :, i % activation.shape[-1]]
        ax.imshow(layer_activation, cmap='viridis')
        ax.set_title(f'Map {i+1}')
        ax.axis('off')

plt.tight_layout()
plt.show()
# ========================================
# PART 11: SHOW PREDICTIONS
# ========================================

print("\n" + "="*50)
print("PART 11: SAMPLE PREDICTIONS")
print("="*50)

def show_predictions(model, X, y, n_samples=12):
    """Show sample predictions with confidence"""
    predictions = model.predict(X[:n_samples])
    pred_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle('CNN Predictions', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < n_samples:
            # Show image
            ax.imshow(X[i].reshape(28, 28), cmap='gray')
            
            # Color based on correct/incorrect
            color = 'green' if pred_classes[i] == y[i] else 'red'
            
            # Show prediction info
            ax.set_title(f'Pred: {pred_classes[i]}\nConf: {confidences[i]:.2f}\nTrue: {y[i]}', 
                        color=color, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

print("🔍 Sample Predictions:")
show_predictions(model, X_test_cnn, y_test)

# ========================================
# PART 12: MISCLASSIFIED EXAMPLES
# ========================================

print("\n" + "="*50)
print("PART 12: MISCLASSIFIED EXAMPLES")
print("="*50)

# Find misclassified examples
misclassified_idx = np.where(y_test != y_pred_classes)[0]
misclassified_count = len(misclassified_idx)

print(f"📊 Misclassified samples: {misclassified_count}/{len(y_test)} ({misclassified_count/len(y_test)*100:.2f}%)")

if misclassified_count > 0:
    # Show misclassified examples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Misclassified Examples', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < 10 and i < misclassified_count:
            idx = misclassified_idx[i]
            ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
            
            # Show prediction vs actual
            pred_probs = y_pred[idx]
            confidence = np.max(pred_probs)
            
            ax.set_title(f'Pred: {y_pred_classes[idx]}\nActual: {y_test[idx]}\nConf: {confidence:.2f}', 
                        color='red')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()