# 🤖 My First TensorFlow Model

A comprehensive beginner's guide to building machine learning models using **TensorFlow** and **Keras**. This project covers everything from tensor basics to building and training neural networks.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [What You'll Learn](#what-youll-learn)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Features](#features)
- [Requirements](#requirements)
- [License](#license)

---

## 🎯 Project Overview

This project is a complete beginner-friendly introduction to TensorFlow, demonstrating:
- Core tensor concepts and operations
- Building neural networks with Keras Sequential API
- Training models for regression and classification tasks
- Working with real datasets (MNIST handwritten digits)
- Saving and loading trained models
- Making predictions with trained models

---

## 📚 What You'll Learn

### Part 1: Tensor Basics
- Creating and manipulating tensors (scalars, vectors, matrices, 3D tensors)
- Tensor properties (shape, dtype, size)
- Basic operations (addition, multiplication, dot products)
- Special tensors (zeros, ones, identity, random)

### Part 2: Simple Neural Networks
- Building sequential models with Keras
- Understanding layer architecture
- Model compilation and training basics

### Part 3: Linear Regression
- Creating regression models with TensorFlow
- Training models to predict continuous values
- Visualizing predictions vs actual data

### Part 4: Binary Classification
- Building classifiers for binary problems
- Using appropriate loss functions and metrics
- Evaluating model performance (accuracy, loss curves)

### Part 5: Multi-Class Classification (MNIST)
- Loading and preprocessing the MNIST dataset
- Building a neural network for digit recognition
- Achieving high accuracy on real-world image data

### Part 6: Save & Load Models
- Saving trained models in HDF5 format
- Loading and reusing models
- Making predictions with loaded models

### Part 7: Custom Neural Networks
- Building custom model classes with Keras
- Using functional and subclassing APIs
- Creating complex architectures

### Part 8: Visualization & Analysis
- Plotting training history
- Comparing model performance
- Visualizing predictions and results

---

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Biruksahileeshete/My-first-TF-Model.git
   cd My-first-TF-Model
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install tensorflow numpy pandas matplotlib scikit-learn
   ```

---

## 📁 Project Structure

```
My-first-TF-Model/
├── TF.py                              # Main TensorFlow beginner's guide
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
└── models/                            # Saved trained models (generated when running TF.py)
    ├── linear_regression_model.h5
    ├── classification_model.h5
    └── mnist_model.h5
```

---

## 🚀 Usage

### Run the Complete Project

Execute the main script to run all 9 parts:

```bash
python TF.py
```

This will:
1. Create and explore tensors
2. Build and train 3 different neural networks
3. Train a classifier on synthetic data
4. Train a digit recognizer on MNIST
5. Save all trained models
6. Display visualizations and results

### Expected Output

The script will:
- Display TensorFlow version information
- Print model architectures and training progress
- Generate visualizations (plots will display in windows)
- Save 3 trained models to your working directory
- Show predictions from each model

### Running Individual Models

You can modify `TF.py` to comment out sections and run specific parts:

```python
# Run only Part 3: Linear Regression
# Comment out other parts in the script
```

---

## 🎨 Features

✅ **Comprehensive Coverage** - From basics to advanced concepts  
✅ **Well-Commented Code** - Easy to follow and understand  
✅ **Multiple Model Types** - Regression, binary classification, multi-class classification  
✅ **Real Datasets** - Uses MNIST for practical experience  
✅ **Model Persistence** - Save and load trained models  
✅ **Visualizations** - Training curves and predictions  
✅ **Custom Models** - Learn to build custom Keras models  

---

## 📦 Requirements

- **TensorFlow** >= 2.0
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Scikit-learn** - Data preprocessing and datasets

See `requirements.txt` for exact versions.

---

## 📖 Key Concepts

| Concept | Description |
|---------|-------------|
| **Tensor** | Multi-dimensional array of numbers (TensorFlow's basic data structure) |
| **Model** | Collection of layers trained to solve a specific task |
| **Layer** | Fundamental building block containing neurons/units |
| **Epoch** | One complete pass through the training dataset |
| **Loss** | Metric measuring how far predictions are from actual values |
| **Accuracy** | Percentage of correct predictions (for classification) |
| **Activation** | Mathematical function applied to layer outputs (e.g., ReLU, Sigmoid) |

---

## 💡 Tips for Learning

1. **Run the script line by line** - Use Jupyter notebooks for interactive exploration
2. **Experiment with hyperparameters** - Try different layer sizes, epochs, learning rates
3. **Modify the code** - Change architectures and observe impacts
4. **Study the visualizations** - Understanding plots helps grasp model behavior
5. **Read the comments** - Each section has detailed explanations

---

## 🔗 Additional Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

---

## 📝 License

MIT License - Feel free to use this project for learning and teaching!

---

## 🤝 Contributing

Feel free to fork this repository and make improvements! Some ideas:
- Add more example models (CNNs, RNNs)
- Create Jupyter notebooks for interactive learning
- Add data augmentation examples
- Include GPU optimization tips

---

**Happy Learning! 🎓 Start with Part 1 and progress through each section systematically.**
