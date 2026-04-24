# evaluate_models.py
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import os

# Load test data
X_test = sp.load_npz("models/X_test_vec.npz")
y_test = np.load("models/y_test.npy", allow_pickle=True)

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    "Naive Bayes": joblib.load("models/naivebayes_model.pkl"),
    "Linear SVM": joblib.load("models/svm_model.pkl")
}

results = {}
best_f1 = 0
best_name = ""

for name, model in models.items():
    print(f"\n{'='*40}\nEvaluating {name}\n{'='*40}")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    results[name] = {'accuracy': acc, 'f1_macro': f1_macro}
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-score: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f'Confusion Matrix - {name}')
    plt.tight_layout()
    plt.savefig(f"models/cm_{name.replace(' ', '_')}.png")
    plt.close()
    
    if f1_macro > best_f1:
        best_f1 = f1_macro
        best_name = name

print(f"\n Best model: {best_name} with Macro F1 = {best_f1:.4f}")

# Comparison bar chart
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]
f1s = [results[n]['f1_macro'] for n in names]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, accs, width, label='Accuracy')
ax.bar(x + width/2, f1s, width, label='Macro F1')
ax.set_ylabel('Score')
ax.set_title('Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15)
ax.legend()
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig("models/model_comparison.png")
plt.close()

print(" Evaluation plots saved to models/")
print(" All done.")
