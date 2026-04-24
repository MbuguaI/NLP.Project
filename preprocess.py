import pandas as pd
import re
import string
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove punctuation except apostrophe and hyphen
    text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "").replace("-", "")))
    return text

# Load your CSV from the data/ folder
df = pd.read_csv("data/combined.csv")
print("Original shape:", df.shape)
print("Column names:", df.columns.tolist())   # Should show ['sentence', 'language']

# Count languages using the correct column name
print("Language counts:\n", df['language'].value_counts())

# Clean the text from the 'sentence' column
df['clean_text'] = df['sentence'].apply(clean_text)

# Remove very short entries (less than 3 words)
df = df[df['clean_text'].str.split().str.len() >= 3]

# Save cleaned data
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_data.csv", index=False)
print("Cleaned shape:", df.shape)
print("Saved to data/cleaned_data.csv")