
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import os
import scipy.sparse as sp

# Load cleaned data (adjust path if needed)
df = pd.read_csv("data/cleaned_data.csv")

# Use the correct column names:
# - 'clean_text' for the preprocessed sentences
# - 'language' for the language labels (lowercase)
X = df['clean_text']
y = df['language']          # <-- changed from 'Language' to 'language'

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Character n-gram TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=15000,
    sublinear_tf=True
)

print("Fitting vectorizer...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"Train matrix shape: {X_train_vec.shape}")
print(f"Test matrix shape:  {X_test_vec.shape}")

# Save vectorizer and vectorized data
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Save sparse matrices efficiently
sp.save_npz("models/X_train_vec.npz", X_train_vec)
sp.save_npz("models/X_test_vec.npz", X_test_vec)

# Save labels
np.save("models/y_train.npy", y_train.values)
np.save("models/y_test.npy", y_test.values)

print("Feature extraction completed. Saved to models/")