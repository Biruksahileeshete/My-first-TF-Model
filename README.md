# 🤖 AI/ML Learning Projects - 30-Day Journey

A collection of 5 machine learning and NLP projects built during the 30-Day AI/ML Engineer challenge.

---

## 📚 Project Overview

| # | Project | Topic | Key Concepts |
|---|---------|-------|--------------|
| 1 | **CNN Image Classifier** | Deep Learning | CNN, MNIST, Image Classification |
| 2 | **Digit Classifier** | Deep Learning | CNN, MNIST, Digit Recognition |
| 3 | **NLP Text Preprocessing** | NLP | Tokenization, Text Cleaning, NLTK |
| 4 | **TensorFlow Basics** | Deep Learning | Tensors, Neural Networks, TF |
| 5 | **Sentiment Analysis App** | NLP | VADER, TextBlob, Streamlit Web App |

---

## 🗂️ Project Structure

```
My-first-TF-Model/
│
├── 📁 CNN Basic/                          # Project 1: CNN Image Classifier
│   └── CNN.py
│
├── 📁 Digit Classifier/                    # Project 2: MNIST Digit Classifier
│   └── MNIST Digit Classifier.py
│
├── 📁 NLP Basic/                           # Project 3 & 5: NLP Projects
│   ├── text preprocessing.py              # NLP Text Preprocessing
│   └── sentiment Analysis.py              # Sentiment Analysis Web App
│
├── 📁 TensorFlow Basics/                   # Project 4: TensorFlow Fundamentals
│   └── tensorflow_basics.py
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 📁 Project Details

### 1. CNN Image Classifier
Recognizes handwritten digits (0-9) using Convolutional Neural Network.

**Test Accuracy: 99.2%**

```bash
cd "CNN Basic"
python CNN.py
```

---

### 2. MNIST Digit Classifier
Classifies handwritten digits using TensorFlow/Keras neural network.

**Test Accuracy: 97.3%**

```bash
cd "Digit Classifier"
python "MNIST Digit Classifier.py"
```

---

### 3. NLP Text Preprocessing
Interactive Streamlit app for text cleaning and preprocessing.

**Features:**
- Lowercase conversion
- Punctuation removal
- Number removal
- Stopwords removal
- Tokenization
- Stemming & Lemmatization
- Part-of-Speech Tagging
- Word Cloud Generation
- Word Frequency Analysis

```bash
cd "NLP Basic"
streamlit run "text preprocessing.py"
```

---

### 4. TensorFlow Basics
Comprehensive introduction to TensorFlow fundamentals.

**Topics Covered:**
- Tensor basics (scalars, vectors, matrices)
- Linear regression with TF
- Binary classification
- Multi-class classification (MNIST)
- Custom models with subclassing
- Model saving and loading

```bash
cd "TensorFlow Basics"
python tensorflow_basics.py
```

---

### 5. Sentiment Analysis Web App
Interactive web app that analyzes text sentiment (Positive, Negative, Neutral).

**Features:**
- Dual sentiment analysis (VADER + TextBlob)
- Real-time text processing
- Sentiment scores visualization
- Keyword extraction
- Word cloud generation
- Sentence-by-sentence analysis
- Interactive UI with Streamlit

```bash
cd "NLP Basic"
streamlit run "sentiment Analysis.py"


