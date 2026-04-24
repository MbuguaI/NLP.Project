import streamlit as st
import joblib
import re
import string
import os
import numpy as np

# -------------------------------------------------------------------
# Load model and vectorizer
# -------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_paths = [
        "models/best_model.pkl",
        "models/svm_model.pkl",
        "models/logistic_model.pkl",
        "models/naivebayes_model.pkl"
    ]
    model = None
    for path in model_paths:
        if os.path.exists(path):
            model = joblib.load(path)
            break
    if model is None:
        st.error("No trained model found. Please run training scripts first.")
        st.stop()
    
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

# -------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "").replace("-", "")))
    return text

# -------------------------------------------------------------------
# Get confidence scores from any classifier
# -------------------------------------------------------------------
def get_confidence_scores(model, X):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.classes_
    elif hasattr(model, "decision_function"):
        # For SVM and other models with decision_function
        scores = model.decision_function(X)[0]
        # Convert to probabilities using softmax (ensures non-negative summing to 1)
        exp_scores = np.exp(scores - np.max(scores))  # numerical stability
        probs = exp_scores / exp_scores.sum()
        classes = model.classes_
    else:
        return None, None
    return dict(zip(classes, probs)), classes

# -------------------------------------------------------------------
# Custom CSS for lively black theme
# -------------------------------------------------------------------
st.set_page_config(page_title="Luo Language Identifier", layout="centered")

st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #0b0c10 0%, #1f2833 100%);
        color: #e5e7eb;
    }
    /* Text area */
    .stTextArea textarea {
        background-color: #1e1e2a;
        color: #ffffff;
        border-radius: 12px;
        border: 1px solid #3a3f4b;
        font-size: 16px;
        padding: 12px;
    }
    .stTextArea textarea:focus {
        border-color: #6c63ff;
        box-shadow: 0 0 8px rgba(108,99,255,0.5);
    }
    /* Button */
    .stButton button {
        background: linear-gradient(90deg, #6c63ff, #3a2cff);
        color: white;
        border-radius: 30px;
        padding: 0.6em 2em;
        font-weight: bold;
        font-size: 18px;
        border: none;
        transition: transform 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #7c73ff, #4a3cff);
    }
    /* Result card */
    .result-card {
        background: rgba(30,30,46,0.8);
        border-radius: 20px;
        padding: 20px;
        margin: 15px 0;
        backdrop-filter: blur(5px);
        border-left: 6px solid #6c63ff;
        box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
    }
    /* Progress bar container */
    .stProgress > div > div {
        background-color: #6c63ff;
        border-radius: 10px;
    }
    /* Success text */
    .stSuccess {
        background: transparent;
        border: none;
        font-size: 1.2rem;
    }
    hr {
        border-color: #3a3f4b;
    }
    .caption {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        margin-top: 40px;
    }
    .stAlert {
        background: rgba(255,100,100,0.2);
        border-left-color: #ff5c5c;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# UI Header (no emojis)
# -------------------------------------------------------------------
st.title("Natural Language Identification System")
st.markdown("Detects **Swahili, English, Sheng, Luo** from short text (1-2 sentences).")
st.markdown("---")

# Input area with character counter
user_input = st.text_area("Enter your text here:", height=150, placeholder="Example: Agulu kidiedi maonge tach")
col1, col2 = st.columns([6, 1])
with col2:
    st.caption(f"{len(user_input.strip())} characters")

# Predict button with spinner
if st.button("Identify Language", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to identify.")
    else:
        with st.spinner("Analyzing text..."):
            model, vectorizer = load_artifacts()
            cleaned = clean_text(user_input)
            X_input = vectorizer.transform([cleaned])
            
            # Get prediction
            predicted_lang = model.predict(X_input)[0]
            
            # Get confidence scores
            prob_dict, classes = get_confidence_scores(model, X_input)
            
            # Display result in a card
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f"**Predicted Language:**  `{predicted_lang.upper()}`")
            
            if prob_dict:
                st.markdown("**Confidence Scores:**")
                # Sort by probability
                sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                for lang, prob in sorted_items:
                    st.progress(prob, text=f"{lang.upper()}: {prob:.2%}")
            else:
                st.info("Confidence scores not available for this model type.")
            
            # Additional helpful info
            st.caption(f"Processed text length: {len(cleaned.split())} words")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<div class="caption">CSC423 NLP Project – Language Identification for Swahili, English, Sheng, Luo</div>', unsafe_allow_html=True)