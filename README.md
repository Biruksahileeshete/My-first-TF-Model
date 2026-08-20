# TensorFlow & CNN Image Classification Projects

## 📌 Overview

This repository contains three comprehensive projects demonstrating fundamental machine learning concepts using TensorFlow and Keras. Starting from basic TensorFlow operations to advanced Convolutional Neural Networks (CNNs) for image classification, these projects provide a complete learning path for beginners and intermediate practitioners.

## 📂 Project Descriptions

### 1. **TF.py - TensorFlow Fundamentals**
A beginner-friendly introduction to TensorFlow covering:
- **Tensor operations** (creation, manipulation, properties)
- **Linear regression** implementation
- **Binary classification** with neural networks
- **MNIST digit classification** with a simple neural network
- **Custom model building** using subclassing
- **Model saving/loading** and prediction workflows

**Key Learning Outcomes:**
- Understanding tensor operations and shapes
- Building Sequential models
- Training and evaluating models
- Working with real datasets
- Creating custom neural network architectures

---

### 2. **MNIST Digit Classifier.py - Neural Network Classification**
A deep dive into MNIST digit classification with:
- **Two model architectures**: Simple (2-layer) vs Deep (with dropout)
- **Comprehensive data exploration** and visualization
- **Performance comparison** between architectures
- **Confusion matrices** and classification reports
- **Misclassification analysis**
- **Interactive prediction** capabilities

**Model Architectures:**
```
Simple Model:   784 → 128 → 10
Deep Model:     784 → 256 → 128 → 64 → 10 (with dropout layers)
```

**Key Features:**
- Training history visualization
- Sample predictions with confidence scores
- Visual comparison of model performance
- Saving and loading trained models

---

### 3. **CNN.py - Convolutional Neural Network**
An advanced image classification project featuring:
- **Complete CNN architecture** with three convolutional blocks
- **Filter visualization** to understand what CNNs learn
- **Feature map visualization** for interpretability
- **Comparison with ANN** to demonstrate CNN advantages
- **Custom prediction function** for individual images

**CNN Architecture:**
```
Conv2D(32, 3×3) → MaxPool(2×2)
Conv2D(64, 3×3) → MaxPool(2×2)
Conv2D(128, 3×3) → MaxPool(2×2)
Flatten → Dense(128) → Dropout(0.5) → Dense(10)
```

**Key Features:**
- Callbacks (EarlyStopping, ReduceLROnPlateau)
- Training time tracking
- Filter and feature map visualization
- Misclassified example analysis
- CNN vs ANN comparison
- Save/load functionality

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Required libraries listed in `requirements.txt`

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd tensorflow-projects

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```
tensorflow>=2.0.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
seaborn>=0.11.0
pandas>=1.2.0
```

## 🚀 Running the Projects

### 1. Start with TensorFlow Fundamentals
```bash
python TF.py
```

### 2. Build the Digit Classifier
```bash
python "MNIST Digit Classifier.py"
```

### 3. Run the CNN Classifier
```bash
python CNN.py
```

## 📊 Expected Results

### MNIST Classifier Performance
| Model | Test Accuracy | Parameters |
|-------|--------------|------------|
| Simple ANN | ~97.0% | ~100,000 |
| Deep ANN | ~98.0% | ~200,000 |
| CNN | ~99.2% | ~1.2M |

### Training Times (Approximate)
- Simple ANN: ~30 seconds
- Deep ANN: ~45 seconds
- CNN: ~60-90 seconds

## 📈 Project Structure

```
├── TF.py                    # TensorFlow fundamentals
├── MNIST Digit Classifier.py # Neural network classification
├── CNN.py                   # CNN image classification
├── requirements.txt         # Project dependencies
├── README.md               # This file
├── models/                 # Saved model files
│   ├── cnn_mnist_model.h5
│   ├── mnist_simple_model.h5
│   ├── mnist_deep_model.h5
│   └── linear_regression_model.h5
└── outputs/                # Generated visualizations
    ├── confusion_matrices/
    ├── training_history/
    └── sample_predictions/
```

## 🎯 Learning Path

### Beginner Level (TF.py)
1. Understand tensor operations
2. Build simple models with Sequential API
3. Train regression and classification models
4. Work with MNIST dataset

### Intermediate Level (MNIST Classifier)
1. Explore dataset thoroughly
2. Build deeper architectures with dropout
3. Compare model performances
4. Visualize predictions and mistakes

### Advanced Level (CNN.py)
1. Understand convolutional operations
2. Build CNN architectures
3. Visualize filters and feature maps
4. Apply callbacks for better training
5. Compare CNN vs ANN performance

## 📝 Key Concepts Covered

### TensorFlow Basics
- ✅ Tensor creation and manipulation
- ✅ Model building with Sequential API
- ✅ Training and evaluation
- ✅ Model persistence

### Neural Networks
- ✅ Dense layers and activation functions
- ✅ Dropout regularization
- ✅ Binary and multi-class classification
- ✅ Loss functions and optimizers

### CNN Concepts
- ✅ Convolutional layers (Conv2D)
- ✅ Pooling operations (MaxPooling)
- ✅ Feature extraction
- ✅ Spatial pattern learning
- ✅ Filter visualization

### Model Evaluation
- ✅ Confusion matrices
- ✅ Classification reports
- ✅ Training history analysis
- ✅ Misclassification analysis

## 🎨 Visualizations

Each project generates comprehensive visualizations:
- **Data exploration** - Sample images, class distributions
- **Training history** - Loss and accuracy curves
- **Predictions** - Sample predictions with confidence scores
- **Confusion matrices** - Model performance analysis
- **CNN internals** - Filters and feature maps
- **Model comparison** - Performance metrics

## 💡 Practical Applications

These projects provide skills applicable to:
- Image classification systems
- Document digitization
- Handwriting recognition
- Feature extraction pipelines
- Transfer learning preparation

## 🔧 Customization Options

### Modify Network Architecture
```python
# Add more layers
model.add(tf.keras.layers.Dense(512, activation='relu'))

# Change activation functions
model.add(tf.keras.layers.Dense(128, activation='tanh'))

# Add regularization
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.BatchNormalization())
```

### Experiment with Hyperparameters
```python
# Change optimizer
model.compile(optimizer='sgd', loss='categorical_crossentropy')

# Adjust learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Modify batch size
model.fit(X_train, y_train, batch_size=128)

# Use different loss functions
model.compile(loss='sparse_categorical_crossentropy')
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional dataset examples (CIFAR-10, Fashion-MNIST)
- Transfer learning implementations
- Data augmentation techniques
- Model optimization strategies
- Deployment examples

## 📚 Further Learning

### Next Steps
1. **CIFAR-10**: Color image classification
2. **Transfer Learning**: Use pre-trained models (VGG16, ResNet)
3. **Data Augmentation**: Improve model generalization
4. **Hyperparameter Tuning**: Optimize model performance
5. **Model Deployment**: Export models to production

### Resources
- [TensorFlow Documentation](https://www.tensorflow.org/learn)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [CNN Explained](https://cs231n.github.io/convolutional-networks/)
- [Keras API](https://keras.io/api/)

## 📄 License

This project is open-source and available for educational purposes.

## 👨‍💻 Author

Created for educational purposes to demonstrate TensorFlow and deep learning concepts.

---

## ⚡ Quick Start Commands

```bash
# Run all projects
python TF.py && python "MNIST Digit Classifier.py" && python CNN.py

# Run individual projects
python TF.py                          # Part 1: Fundamentals
python "MNIST Digit Classifier.py"    # Part 2: ANN Classifier
python CNN.py                         # Part 3: CNN Classifier

# Check installed packages
pip list | grep -E "tensorflow|numpy|matplotlib|sklearn|seaborn"
```

## 🎓 Success Metrics

After completing these projects, you should be able to:

- ✅ Build and train neural networks from scratch
- ✅ Implement CNNs for image classification
- ✅ Understand and visualize model internals
- ✅ Evaluate and compare different architectures
- ✅ Save, load, and use trained models
- ✅ Apply these skills to new datasets and problems

---

