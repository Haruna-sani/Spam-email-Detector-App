# 📧 Spam Email Classifier App

A simple and efficient web application built with **Streamlit** that classifies email content as **Spam** or **Not Spam** using a trained **XGBoost model** and **Bag-of-Words (BoW)** text vectorization.

## 🚀 Features

- Predicts whether an email is spam based on its text content.
- Preprocessing pipeline includes stopword removal and lemmatization using NLTK.
- Bag-of-Words (BoW) vectorization with a max of 5000 features.
- Model trained using **XGBoost Classifier** with SMOTE for handling class imbalance.
- Lightweight and fast Streamlit interface for real-time email classification.

## 🧠 Model Overview

- **Vectorizer**: `CountVectorizer(max_features=5000)`
- **Classifier**: `XGBClassifier()`
- **Imbalance Handling**: `SMOTE` applied to training set
- **Text Cleaning**:
  - Lowercasing
  - Removing non-alphabetic characters
  - Removing stopwords
  - Lemmatization with WordNet

## 🛠️ Installation & Setup

### 🔧 Clone the Repository

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
````

### 📦 Install Dependencies

```bash
import re
import nltk
import joblib
import pandas as pd
import numpy as np
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```

### 🧾 Required NLTK Resources

If not already downloaded, run the following once:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

### 📂 Project Structure

```
spam-email-classifier/
├── app.py
├── models/
│   ├── bow_vectorizer.pkl
│   └── xgb_spam_classifier.pkl
├── utils/
│   └── train.py
├── requirements.txt
└── README.md
```

### ▶️ Run the App

```bash
streamlit run app.py
```

Then open the displayed URL (usually [http://localhost:8501](http://localhost:8501)) in your browser.

## 💡 Example Usage

1. Paste or type an email message in the input box.
2. Click the **"Classify Email"** button.
3. View the prediction result: **Spam** or **Not Spam**.

## 📚 Technologies Used

* Python
* Streamlit
* Scikit-learn
* XGBoost
* NLTK
* Imbalanced-learn (SMOTE)
* Joblib (model persistence)

## 🧠 Author

**\[Haruna Sani]**
📧 \[[harunasulesani@gmail.com](mailto:your.email@example.com)]
🐙 [GitHub](https://github.com/Haruna-sani)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
