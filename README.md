# Email Spam Shield 🛡️

A machine learning application that detects spam emails using hybrid features and multiple classifiers.

![Spam Shield Demo](screenshots/demo.png)

## 📋 Overview

This project implements a spam email classification system using multiple machine learning models and feature extraction techniques. The application provides a user-friendly interface built with Streamlit for analyzing emails and determining whether they are spam or legitimate.

### Features

- **Multiple Feature Extraction Techniques**:
  - TF-IDF Vectorization
  - Word2Vec Embeddings
  - Hybrid feature combination

- **Multiple Classifiers**:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest

- **Interactive UI**:
  - Real-time email analysis
  - Visual representation of results
  - Keyword analysis
  - Performance metrics visualization

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/email-spam-shield.git
   cd email-spam-shield
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. To run the Streamlit application:
   ```bash
   python3 -m streamlit run app.py
   ```

2. To retrain the models using the Jupyter notebook:
   ```bash
   jupyter notebook spamemailclassification.ipynb
   ```

## 🧠 How It Works

The spam classification system works in three main steps:

1. **Text Preprocessing**:
   - Converting to lowercase
   - Removing special characters
   - Tokenizing into words
   - Removing stop words
   - Lemmatizing words

2. **Feature Extraction**:
   - TF-IDF: Converts text into numerical values based on word frequency
   - Word2Vec: Neural network model that learns word associations
   - Hybrid Features: Combines both methods for enhanced performance

3. **Classification**:
   - Multiple machine learning models analyze the features
   - The model with the best F1-score is selected as the final classifier

## 📊 Performance

The system has been tested on the Enron email dataset, with the following performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~92% | ~89% | ~85% | ~87% |
| Naive Bayes | ~88% | ~82% | ~91% | ~86% |
| SVM | ~93% | ~90% | ~87% | ~88% |
| Random Forest | ~91% | ~88% | ~84% | ~86% |

*Note: Actual performance may vary based on the most recent training.*

## 📚 Project Structure

```
email-spam-shield/
├── app.py                       # Streamlit application
├── spamemailclassification.ipynb # Training notebook
├── requirements.txt             # Dependencies
├── best_spam_classifier.pkl     # Best model
├── tfidf_vectorizer.pkl         # TF-IDF model
├── scaler.pkl                   # Feature scaler
├── w2v_model.pkl                # Word2Vec model
└── enrondataset.csv             # Dataset
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/email-spam-shield/issues).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- Enron Email Dataset
- Streamlit for the interactive UI framework
- NLTK for natural language processing tools
- Scikit-learn for machine learning models 