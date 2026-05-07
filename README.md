# Sentiment Analysis for Emotion Classification

## 📌 Project Overview

This project performs **emotion classification** on textual comments using machine learning techniques. The dataset contains user comments labeled with three emotions: **Anger**, **Fear**, and **Joy**. The goal is to preprocess text data, extract meaningful features, and train classifiers to accurately predict the emotion expressed in a given comment.

## 📊 Dataset

- **Source**: `nlp_dataset.csv`   
- **Size**: 5,937 samples    
- **Distribution**:    
  - Anger: 2,000 samples   
  - Joy: 2,000 samples     
  - Fear: 1,937 samples    
- **Format**: CSV with two columns - `Comment` (text) and `Emotion` (label)    

## 🛠️ Technical Implementation

### 1. Text Preprocessing   
The following preprocessing steps are applied to clean and normalize the text data:    
- Lowercasing   
- URL removal    
- Special characters and punctuation removal   
- Extra whitespace cleanup   
- Tokenization using NLTK   
- Stopword removal (English)   
- Minimum token length filtering (>2 characters)   
   
**Impact**: ~37% average text length reduction, noise removal, improved model focus on emotionally-relevant words.   

### 2. Feature Extraction   
Two vectorization methods are implemented:   
- **CountVectorizer**: Bag-of-words representation with word frequency counts   
- **TfidfVectorizer**: Term Frequency-Inverse Document Frequency weighting     
- **Parameters**: `max_features=5000`, `ngram_range=(1,2)` (includes single words and word pairs)    

### 3. Machine Learning Models    
Two classifiers are trained and compared :   

| Model | Description | Suitability |
|-------|-------------|-------------|
| **Naive Bayes** (MultinomialNB) | Probabilistic classifier based on Bayes' theorem | Fast, works well with high-dimensional sparse text data |
| **SVM** (Linear Kernel) | Margin-maximizing classifier | Robust to overfitting, effective for text classification |

### 4. Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Weighted F1-Score**: Harmonic mean of precision and recall, weighted by class support
- **Classification Report**: Per-class precision, recall, and F1-score
- **Confusion Matrix**: Visual representation of prediction errors

## 📈 Results

| Model | Feature Extractor | Accuracy | Weighted F1-Score |
|-------|------------------|----------|-------------------|
| Naive Bayes | CountVectorizer | 91.33% | 0.9133 |
| Naive Bayes | TfidfVectorizer | 91.92% | 0.9192 |
| SVM | CountVectorizer | 93.86% | 0.9386 |
| **SVM** | **TfidfVectorizer** | **94.02%** | **0.9401** |

### Best Model Performance (SVM + TfidfVectorizer)
