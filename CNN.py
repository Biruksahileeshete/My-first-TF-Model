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
model.build(input_shape=(None, 28, 28, 1))
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
    # ========================================
# PART 13: COMPARE CNN vs ANN
# ========================================

print("\n" + "="*50)
print("PART 13: CNN vs ANN COMPARISON")
print("="*50)

# Build a simple ANN for comparison
ann_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

ann_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train ANN (fewer epochs)
print("🎯 Training ANN for comparison...")
ann_history = ann_model.fit(
    X_train, y_train_onehot,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=0
)

# Evaluate ANN
ann_loss, ann_accuracy = ann_model.evaluate(X_test, y_test_onehot, verbose=0)

# Compare results
print("\n📊 Model Comparison:")
print("-" * 50)
print(f"{'Metric':<20} {'CNN':<15} {'ANN':<15}")
print("-" * 50)
print(f"{'Test Accuracy':<20} {test_accuracy:.4f}{'':10} {ann_accuracy:.4f}")
print(f"{'Parameters':<20} {model.count_params():,}{'':8} {ann_model.count_params():,}")
print(f"{'Training Time':<20} {training_time:.1f}s{'':10} {time.time() - start_time:.1f}s")

# Calculate improvement
improvement = ((test_accuracy - ann_accuracy) / ann_accuracy) * 100
print(f"\n📈 CNN is {improvement:.1f}% more accurate than ANN!")
print("✅ CNN performs better because it understands spatial patterns!")
# ========================================
# PART 14: SAVE MODEL
# ========================================

print("\n" + "="*50)
print("PART 14: SAVING MODEL")
print("="*50)

# Save models
model.save('cnn_mnist_model.h5')
ann_model.save('ann_mnist_model.h5')

print("✅ Models saved successfully!")
print("   - cnn_mnist_model.h5")
print("   - ann_mnist_model.h5")

print("\n📂 To load the model:")
print("   loaded_model = tf.keras.models.load_model('cnn_mnist_model.h5')")

# ========================================
# PART 15: PREDICT CUSTOM IMAGE
# ========================================

print("\n" + "="*50)
print("PART 15: PREDICT CUSTOM IMAGE")
print("="*50)

def predict_custom_digit(model, image):
    """Predict a single digit from a 28x28 image"""
    # Reshape for model
    image = image.reshape(1, 28, 28, 1)
    
    # Make prediction
    prediction = model.predict(image, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return digit, confidence, prediction[0]

# Test on a random image
test_idx = np.random.randint(0, len(X_test))
digit, confidence, probs = predict_custom_digit(model, X_test[test_idx])

print(f"\n🎯 Test Image {test_idx}:")
print(f"   True Digit: {y_test[test_idx]}")
print(f"   Predicted Digit: {digit}")
print(f"   Confidence: {confidence:.2f}")

# Show prediction probabilities
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

# Show image
ax[0].imshow(X_test[test_idx], cmap='gray')
ax[0].set_title(f'True: {y_test[test_idx]}')
ax[0].axis('off')

# Show probabilities
ax[1].bar(range(10), probs, color='skyblue')
ax[1].axhline(y=0.1, color='red', linestyle='--', alpha=0.5)
ax[1].set_title(f'Predicted: {digit} (Confidence: {confidence:.2f})')
ax[1].set_xlabel('Digit')
ax[1].set_ylabel('Probability')
ax[1].set_xticks(range(10))
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================
# FINAL SUMMARY
# ========================================

print("\n" + "="*50)
print("🎉 CNN IMAGE CLASSIFIER COMPLETE!")
print("="*50)

print("\n📊 Final Results:")
print(f"   Model: CNN for MNIST")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Training Time: {training_time:.1f} seconds")
print(f"   Parameters: {model.count_params():,}")

print("\n📚 What You Learned:")
print("   ✅ CNN architecture (Conv2D, MaxPooling, Flatten)")
print("   ✅ Data preprocessing for CNN")
print("   ✅ Training CNN models")
print("   ✅ Visualizing filters and feature maps")
print("   ✅ Evaluating CNN performance")
print("   ✅ Comparing CNN vs ANN")
print("   ✅ Saving and loading models")
print("   ✅ Making predictions")

print("\n🎯 CNN Architecture Summary:")
print("   Conv2D(32, 3×3) → MaxPool(2×2)")
print("   Conv2D(64, 3×3) → MaxPool(2×2)")
print("   Conv2D(128, 3×3) → MaxPool(2×2)")
print("   Flatten → Dense(128) → Dropout → Dense(10)")

print("\n💡 Why CNN Works Better:")
print("   • Learns spatial features automatically")
print("   • Fewer parameters than ANN")
print("   • Translation invariant")
print("   • Works well with image data")

print("\n🚀 Next Steps:")
print("   • Try CIFAR-10 (color images, 10 classes)")
print("   • Use transfer learning")
print("   • Add data augmentation")
print("   • Build a classifier for your own images")

print("\n✅ Congratulations! You've built a CNN image classifier! 🖼️🚀")