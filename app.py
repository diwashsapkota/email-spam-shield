import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os

import nltk
import os

# Download NLTK resources (with silent option to avoid flooding logs)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.github.com/yourusername/email-spam-classifier',
        'Report a bug': 'https://www.github.com/yourusername/email-spam-classifier/issues',
        'About': "# Email Spam Classifier\nA machine learning application that detects spam emails using hybrid features and multiple classifiers."
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Modern color palette */
    :root {
        --primary-color: #fff5ee;
        --secondary-color: #deb887;
        --accent-color: #4cc9f0;
        --text-color: #2b2d42;
        --light-bg: #f8f9fa;
        --success-color: #2ec4b6;
        --error-color: #e63946;
        --warning-color: #ff9f1c;
    }
    
    /* Typography improvements */
    body {
        color: var(--text-color);
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 1.5rem;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }
    
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #deb887;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
    }
    
    /* Card-like containers */
    .highlight {
        background-color: var(--light-bg);
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
        border-left: 4px solid var(--accent-color);
    }
    
    /* Button styling improvements are handled by Streamlit directly */
    
    /* Info box styling */
    .stAlert {
        border-radius: 8px !important;
    }
    
    /* Footer styling */
    .footer {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
        margin-top: 4rem;
        padding-top: 1rem;
        border-top: 1px solid #f0f0f0;
    }
    
    /* Metrics styling */
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for email text
if 'email_text' not in st.session_state:
    st.session_state.email_text = ""

# Define example emails
example_spam = """URGENT: You've won a $1,000 gift card! Click here to claim your prize now! 
Limited time offer. Forward the OTP you received few minutes ago to confirm your identity."""

example_ham = """Hi Diwash, I hope this email finds you well. I wanted to follow up on our meeting 
last week about the quarterly report. I've attached the latest version with the updates we discussed. 
Let me know if you need anything else before the presentation on Friday."""

# Functions to set examples in session state
def set_spam_example():
    st.session_state.email_text = example_spam

def set_ham_example():
    st.session_state.email_text = example_ham

# Sidebar
with st.sidebar:
    st.markdown("# 🛡️ Spam Shield")
    st.markdown("---")
    
    st.markdown('<p class="sub-header">About</p>', unsafe_allow_html=True)
    st.info(
        "This application uses machine learning to classify emails as spam or non-spam. "
        "It was trained on the Enron email dataset using various models and feature extraction techniques."
    )
    
    st.markdown('<p class="sub-header">Model Information</p>', unsafe_allow_html=True)
    
    # Create expandable sections for each feature technique
    with st.expander("Feature Extraction", expanded=True):
        st.markdown("""
        - **TF-IDF Vectorization**: Converts text into numerical features based on word importance
        - **Word2Vec Embeddings**: Captures semantic meaning of words
        - **Hybrid Features**: Combines both TF-IDF and Word2Vec for enhanced performance
        """)
    
    with st.expander("Classifiers Used"):
        st.markdown("""
        - **Logistic Regression**: Linear model for classification
        - **Naive Bayes**: Probabilistic classifier
        - **SVM**: Support Vector Machine
        - **Random Forest**: Ensemble of decision trees
        """)
    
    st.markdown("---")
    st.markdown('<p class="footer">Made using Streamlit</p>', unsafe_allow_html=True)

# Main content
st.markdown('<p class="main-header">Email Spam Shield</p>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="margin-bottom: 2rem;">
        Detect unwanted spam emails with our advanced machine learning classifier. 
        Simply paste your email content below to analyze whether it's legitimate or spam.
    </div>
    """, 
    unsafe_allow_html=True
)

# Create tabs with icons
tab1, tab2, tab3 = st.tabs([
    "🔍 Classification", 
    "📊 Performance Metrics", 
    "ℹ️ How It Works"
])

with tab1:
    st.markdown('<p class="sub-header">Email Analysis</p>', unsafe_allow_html=True)
    
    # Two-column layout for options and input
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Try Examples")
        st.button("✉️ Normal Email", on_click=set_ham_example, key='ham_button', use_container_width=True)
        st.button("🚫 Spam Email", on_click=set_spam_example, key='spam_button', use_container_width=True)
        
        st.markdown("### Options")
        show_keywords = st.checkbox("Show keyword analysis", value=True)
        show_confidence = st.checkbox("Show confidence score", value=True)
    
    with col2:
        # Text input (using session state)
        st.markdown("### Enter Email Content")
        email_text = st.text_area(
            "Paste email text here",
            height=200, 
            placeholder="Paste email content here to check if it's spam or not...",
            value=st.session_state.email_text,
            label_visibility="collapsed"
        )
        
        # Update session state if text area changes
        if email_text != st.session_state.email_text:
            st.session_state.email_text = email_text
        
        classify_btn = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    
    # Load models (keep this outside the columns)
    @st.cache_resource
    def load_models():
        try:
            model = joblib.load('best_spam_classifier.pkl')
            tfidf = joblib.load('tfidf_vectorizer.pkl')
            scaler = joblib.load('scaler.pkl')
            w2v_model = joblib.load('w2v_model.pkl')
            return model, tfidf, scaler, w2v_model
        except Exception as e:
            st.error(f"Error loading models: {e}")
            return None, None, None, None
    
    model, tfidf, scaler, w2v_model = load_models()
    
    # Preprocessing function
    def preprocess_text(text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    
    # Word2Vec feature extraction
    def get_w2v_features(text):
        tokens = word_tokenize(text)
        vectors = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(50)
    
    # Classification logic
    if classify_btn:
        if not model or not tfidf or not scaler or not w2v_model:
            st.error("Models failed to load. Please check the model files and restart the app.")
        elif email_text:
            with st.spinner("Analyzing email..."):
                start_time = time.time()
                
                # Preprocess
                cleaned_text = preprocess_text(email_text)
                
                # TF-IDF features
                text_vector = tfidf.transform([cleaned_text]).toarray()
                # Ensure non-negative values for MultinomialNB
                text_vector = np.clip(text_vector, a_min=0, a_max=1e6)
                text_vector_scaled = scaler.transform(text_vector)
                
                # Word2Vec features
                w2v_vector = get_w2v_features(cleaned_text)
                
                # Combine features into DataFrame
                tfidf_columns = [f'tfidf_{i}' for i in range(text_vector_scaled.shape[1])]
                w2v_columns = [f'w2v_{i}' for i in range(len(w2v_vector))]
                all_columns = tfidf_columns + w2v_columns
                
                try:
                    combined_vector = np.hstack((text_vector_scaled, w2v_vector.reshape(1, -1)))
                    combined_df = pd.DataFrame(combined_vector, columns=all_columns)
                    
                    # Apply non-negative constraint for MultinomialNB
                    combined_vector_nn = np.clip(combined_vector, a_min=0, a_max=None)
                    combined_df_nn = pd.DataFrame(combined_vector_nn, columns=all_columns)
                    
                    # Select appropriate input for model
                    try:
                        best_model_name = model.named_steps['classifier'].__class__.__name__
                    except:
                        # Direct access to model if it's not in a pipeline
                        best_model_name = model.__class__.__name__
                        
                    if 'MultinomialNB' in best_model_name:
                        # For MultinomialNB, use non-negative values
                        if 'text_vector' in locals():
                            input_vector = text_vector
                        else:
                            input_vector = combined_df_nn
                    elif 'RandomForestClassifier' in best_model_name:
                        # RandomForest usually works better with just TF-IDF
                        input_vector = text_vector
                    else:
                        # For other models (LogisticRegression, SVM), use combined features
                        input_vector = combined_df
                        
                    # Predict
                    prediction = model.predict(input_vector)[0]
                    
                    # Get prediction probability if available
                    confidence = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(input_vector)[0]
                        confidence = proba[1] if prediction == 1 else proba[0]
                        
                    inference_time = time.time() - start_time
                    
                    # Display result in a card-like container
                    st.markdown('<div class="highlight">', unsafe_allow_html=True)
                    
                    # Show result with columns for better layout
                    res_col1, res_col2 = st.columns([3, 1])
                    
                    with res_col1:
                        if prediction == 1:
                            st.error("### ⚠️ SPAM DETECTED")
                            st.markdown("This email has been classified as spam and should be treated with caution.")
                        else:
                            st.success("### ✅ LEGITIMATE EMAIL")
                            st.markdown("This email appears to be legitimate.")
                    
                    with res_col2:
                        if show_confidence and confidence:
                            st.metric(
                                label="Confidence", 
                                value=f"{confidence:.1%}", 
                                delta=None
                            )
                        st.metric(
                            label="Processing Time", 
                            value=f"{inference_time:.3f}s", 
                            delta=None
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show most important keywords that influenced the decision
                    if show_keywords and 'tfidf' in locals() and tfidf is not None:
                        st.markdown('<p class="sub-header">Key Terms Analysis</p>', unsafe_allow_html=True)
                        
                        try:
                            # Get feature names from TF-IDF
                            feature_names = tfidf.get_feature_names_out()
                            
                            # Get top features for this text
                            tfidf_values = text_vector[0]
                            top_indices = tfidf_values.argsort()[-10:][::-1]
                            top_terms = [(feature_names[i], tfidf_values[i]) for i in top_indices if tfidf_values[i] > 0]
                            
                            if top_terms:
                                st.markdown("These terms were most influential in the classification:")
                                top_terms_df = pd.DataFrame(top_terms, columns=['Term', 'Importance'])
                                
                                # Plot top terms
                                fig, ax = plt.subplots(figsize=(10, 4))
                                bars = sns.barplot(x='Importance', y='Term', data=top_terms_df, ax=ax, palette='viridis')
                                ax.set_title('Most Important Terms in Email')
                                ax.set_xlabel('Importance Score')
                                ax.set_ylabel('')
                                
                                # Add value labels to bars
                                for i, p in enumerate(bars.patches):
                                    width = p.get_width()
                                    ax.text(width + 0.01, p.get_y() + p.get_height()/2, 
                                            f'{width:.3f}', ha='left', va='center')
                                            
                                st.pyplot(fig)
                        except Exception as e:
                            st.write("Could not analyze keywords.")
                
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
        else:
            st.warning("Please enter some text to classify.")

with tab2:
    st.markdown('<p class="sub-header">Model Performance Analysis</p>', unsafe_allow_html=True)
    
    # Check if performance files exist
    perf_csv_exists = os.path.exists('/observations/performance_metrics.csv')
    f1_img_exists = os.path.exists('/observations/model_comparison_f1.png')
    cm_img_exists = os.path.exists('/observations/confusion_matrix_logistic_regression.png')
    training_img_exists = os.path.exists('/observations/model_comparison_training_time.png')
    inference_img_exists = os.path.exists('/observations/model_comparison_inference_time.png')
    ablation_img_exists = os.path.exists('/observations/ablation_study.png')
    
    if not perf_csv_exists and not any([f1_img_exists, cm_img_exists, training_img_exists, inference_img_exists]):
        # Info box
        st.info("#### Performance Data Not Available")
        st.markdown("""
        <div style='margin-bottom: 1rem;'>
            The model performance metrics will be generated when you run the training notebook.
            Below is sample data to illustrate what you'll see after training.
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different performance aspects
        perf_tab1, perf_tab2 = st.tabs(["📈 Metrics", "⏱️ Timing"])
        
        with perf_tab1:
            # Sample performance data
            st.markdown("### Sample Model Performance")
            sample_data = {
                'Model': ['Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest'],
                'Accuracy': [0.92, 0.88, 0.93, 0.91],
                'Precision': [0.89, 0.82, 0.90, 0.88],
                'Recall': [0.85, 0.91, 0.87, 0.84],
                'F1-Score': [0.87, 0.86, 0.88, 0.86]
            }
            sample_df = pd.DataFrame(sample_data)
            
            # Display metrics with visual enhancements
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Accuracy", "93%", "SVM")
            with col2:
                st.metric("Best Precision", "90%", "SVM")
            with col3:
                st.metric("Best Recall", "91%", "Naive Bayes")
            with col4:
                st.metric("Best F1-Score", "88%", "SVM")
            
            st.dataframe(sample_df, use_container_width=True)
            
            # Generate and display sample visualization
            st.markdown("### Comparative Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Model', y='F1-Score', data=sample_df, ax=ax, palette='viridis')
            ax.set_title('Model Comparison (Example Data)')
            ax.set_ylim(0, 1)
            for i, v in enumerate(sample_df['F1-Score']):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
            st.pyplot(fig)
            
        with perf_tab2:
            st.markdown("### Training & Inference Time")
            
            # Sample timing data
            timing_data = {
                'Model': ['Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest'],
                'Training Time (s)': [2.5, 0.8, 5.2, 8.1],
                'Inference Time (ms)': [12, 8, 15, 45]
            }
            timing_df = pd.DataFrame(timing_data)
            st.dataframe(timing_df, use_container_width=True)
            
            # Create two columns for training and inference visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Model', y='Training Time (s)', data=timing_df, palette='Blues_d')
                ax.set_title('Training Time Comparison')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x='Model', y='Inference Time (ms)', data=timing_df, palette='Greens_d')
                ax.set_title('Inference Time Comparison')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            
    else:
        # Try to load and display actual performance metrics
        try:
            # Create tabs for different performance aspects
            perf_tab1, perf_tab2 = st.tabs(["📈 Metrics", "⏱️ Timing"])
            
            with perf_tab1:
                if perf_csv_exists:
                    perf_metrics = pd.read_csv('performance_metrics.csv')
                    
                    # Display key metrics at the top
                    st.markdown("### Key Performance Indicators")
                    metrics_to_show = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                    
                    if all(metric in perf_metrics.columns for metric in metrics_to_show):
                        best_model_indices = {metric: perf_metrics[metric].idxmax() for metric in metrics_to_show}
                        
                        cols = st.columns(4)
                        for i, metric in enumerate(metrics_to_show):
                            best_idx = best_model_indices[metric]
                            best_value = perf_metrics.loc[best_idx, metric]
                            best_model = perf_metrics.loc[best_idx, 'Model']
                            cols[i].metric(
                                f"Best {metric}", 
                                f"{best_value:.2f}" if isinstance(best_value, float) else best_value,
                                best_model
                            )
                    
                    st.markdown("### Detailed Performance Metrics")
                    st.dataframe(perf_metrics, use_container_width=True)
                    
                    # Display visualizations
                    st.markdown("### Performance Visualizations")
                    if f1_img_exists:
                        st.image('model_comparison_f1.png', caption='F1-Score Comparison')
                    
                    # Display confusion matrix in a separate section
                    st.markdown("### Confusion Matrix")
                    if cm_img_exists:
                        st.image('confusion_matrix_logistic_regression.png', 
                                caption='Confusion Matrix (Logistic Regression)')
                else:
                    st.info("Performance metrics CSV file is not available. Run the training notebook to generate metrics.")
            
            with perf_tab2:
                st.markdown("### Training & Inference Time Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    if training_img_exists:
                        st.image('model_comparison_training_time.png', caption='Training Time Comparison')
                    else:
                        st.info("Training time visualization is not available.")
                
                with col2:
                    if inference_img_exists:
                        st.image('model_comparison_inference_time.png', caption='Inference Time Comparison')
                    else:
                        st.info("Inference time visualization is not available.")
                
        except Exception as e:
            st.error(f"Error loading performance metrics: {str(e)}")
            st.info("Run the training notebook to generate performance visualizations.")

with tab3:
    st.markdown('<p class="sub-header">Understanding Spam Classification</p>', unsafe_allow_html=True)
    
    # Create process flow diagram for email classification
    st.markdown("### Classification Process Flow")
    
    # Step-by-step process with icons and descriptions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #778899; border-radius: 8px; height: 100%;">
            <h4>1️⃣ Text Preprocessing</h4>
            <p style="text-align: left;">
                • Converting to lowercase<br>
                • Removing special characters<br>
                • Tokenizing into words<br>
                • Removing stop words<br>
                • Lemmatizing words
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #778899; border-radius: 8px; height: 100%;">
            <h4>2️⃣ Feature Extraction</h4>
            <p style="text-align: left;">
                • TF-IDF: Numerical values based on word frequency<br>
                • Word2Vec: Word embeddings capturing semantic meaning<br>
                • Hybrid Features: Combining TF-IDF and Word2Vec for enhanced detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #778899; border-radius: 8px; height: 100%;">
            <h4>3️⃣ Classification</h4>
            <p style="text-align: left;">
                • Machine learning models analyze features<br>
                • Best model selected based on F1-score<br>
                • Prediction made with confidence score<br>
                • Result indicating spam or legitimate
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Feature importance section
    st.markdown("### Model Features Explained")
    
    with st.expander("TF-IDF Vectorization", expanded=False):
        st.markdown("""
        **Term Frequency-Inverse Document Frequency** is a numerical statistic that reflects how important a word is to a document in a collection.
        
        - **Term Frequency (TF)**: How frequently a term appears in a document
        - **Inverse Document Frequency (IDF)**: How rare or common a term is across all documents
        - **TF-IDF Score**: TF × IDF (higher values indicate more relevant terms)
        
        This technique helps identify distinctive words that characterize spam emails, such as "free," "winner," "urgent," etc.
        """)
    
    with st.expander("Word2Vec Embeddings", expanded=False):
        st.markdown("""
        **Word2Vec** is a technique for natural language processing that represents words as vectors in a continuous vector space.
        
        - Words with similar meanings are positioned closer to each other in the vector space
        - Captures semantic relationships between words
        - Each word is represented as a 50-dimensional vector in our model
        - Helps detect spam even when specific trigger words are avoided by spammers
        """)
    
    # Ablation study
    st.markdown('<p class="sub-header">Ablation Study</p>', unsafe_allow_html=True)
    
    if ablation_img_exists:
        st.image('ablation_study.png', caption='Feature Importance by Model Type')
    else:
        st.markdown("""
        An ablation study helps us understand the contribution of each feature type by systematically removing components and measuring performance changes.
        """)
        
        # Show sample ablation study
        st.markdown("### Sample Ablation Study")
        sample_ablation = {
            'Feature Set': ['TF-IDF', 'TF-IDF', 'TF-IDF', 'Word2Vec', 'Word2Vec', 'Hybrid', 'Hybrid'],
            'Model': ['Logistic Regression', 'Naive Bayes', 'SVM', 'Logistic Regression', 'SVM', 'Logistic Regression', 'SVM'],
            'F1-Score': [0.84, 0.82, 0.85, 0.78, 0.80, 0.88, 0.90]
        }
        sample_ablation_df = pd.DataFrame(sample_ablation)
        
        # Create sample ablation plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Model', y='F1-Score', hue='Feature Set', data=sample_ablation_df, palette='viridis')
        ax.set_title('Ablation Study: F1-Score by Feature Set (Example Data)')
        ax.set_ylim(0, 1)
        plt.legend(title='Feature Set', bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Email Spam Shield | Created with Streamlit</p>
    <p>Developed as part of ML research on email classification</p>
</div>
""", unsafe_allow_html=True)

# To Run:
# python3 -m streamlit run app.py
