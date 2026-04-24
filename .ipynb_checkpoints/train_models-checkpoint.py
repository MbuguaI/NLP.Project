# train_models.py
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import joblib
import os

# Load vectorized data
X_train = sp.load_npz("models/X_train_vec.npz")
y_train = np.load("models/y_train.npy", allow_pickle=True)

print("Training models...")

# Logistic Regression
print("\n1. Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)
joblib.dump(lr, "models/logistic_model.pkl")

# Naive Bayes
print("2. Training Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train, y_train)
joblib.dump(nb, "models/naivebayes_model.pkl")

# Linear SVM
print("3. Training Linear SVM...")
svm = LinearSVC(C=1.0, max_iter=2000, dual='auto')
svm.fit(X_train, y_train)
joblib.dump(svm, "models/svm_model.pkl")

print(" All models trained and saved to models/")