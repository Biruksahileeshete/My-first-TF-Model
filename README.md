# 🤖 AI/ML Learning Projects

A collection of 4 machine learning projects built during the 30-Day AI/ML Engineer challenge.

---

## 📚 Projects

| # | Project | Topic | Key Concepts |
|---|---------|-------|--------------|
| 1 | **CNN Image Classifier** | Deep Learning | CNN, MNIST, Image Classification |
| 2 | **Digit Classifier** | Deep Learning | CNN, MNIST, Digit Recognition |
| 3 | **NLP Text Preprocessing** | NLP | Tokenization, Text Cleaning, NLTK |
| 4 | **TensorFlow Basics** | Deep Learning | Tensors, Neural Networks, TF |

---

## 🗂️ Folder Structure

```
My-first-TF-Model/
├── CNN Basic/                    # CNN Image Classifier
│   └── CNN.py
├── Digit Classifier/              # MNIST Digit Classifier
│   └── MNIST Digit Classifier.py
├── NLP Basic/                     # NLP Text Preprocessing
│   └── text preprocessing.py
├── TensorFlow Basics/             # TensorFlow Fundamentals
│   └── tensorflow_basics.py
├── requirements.txt
└── README.md
```

---

## 📁 Project Details

### 1. CNN Image Classifier
Recognizes handwritten digits (0-9) using CNN. **Test Accuracy: 99.2%**

```bash
cd "CNN Basic"
python CNN.py
```

### 2. MNIST Digit Classifier
Classifies handwritten digits using TensorFlow/Keras. **Test Accuracy: 97.3%**

```bash
cd "Digit Classifier"
python "MNIST Digit Classifier.py"
```

### 3. NLP Text Preprocessing
Interactive Streamlit app for text cleaning (lowercase, tokenization, stopwords, stemming, lemmatization).

```bash
cd "NLP Basic"
streamlit run "text preprocessing.py"
```

### 4. TensorFlow Basics
Covers tensors, linear regression, classification, and custom models.

```bash
cd "TensorFlow Basics"
python tensorflow_basics.py
```

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/My-first-TF-Model.git
cd My-first-TF-Model

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for NLP project)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('omw-1.4')"
```

---

## 📦 Requirements

```txt
tensorflow>=2.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
nltk>=3.8.0
wordcloud>=1.9.0
```

---

## 📊 GitHub Commits

### ✅ Commit These
```
CNN Basic/
Digit Classifier/
NLP Basic/
TensorFlow Basics/
requirements.txt
README.md
```

### ❌ Don't Commit
```
venv/
.venv/
.vscode/
__pycache__/
*.pyc
```

---

**Happy Learning! 🚀**