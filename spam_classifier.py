"""
Task 1: Email Spam Classification
Arch Technologies - ML/Data Science Internship

Pipeline: load data -> clean/preprocess text -> TF-IDF features ->
train multiple models -> evaluate -> save results & plots.
"""

import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

RANDOM_STATE = 42

# ----------------------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------------------
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "text"])
print("Dataset shape:", df.shape)
print(df["label"].value_counts())

# ----------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # urls
    text = re.sub(r"\d+", " ", text)                       # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(clean_text)
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

# ----------------------------------------------------------------
# 3. Train/test split
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_num"],
    test_size=0.2, random_state=RANDOM_STATE, stratify=df["label_num"]
)

# ----------------------------------------------------------------
# 4. Feature extraction (TF-IDF)
# ----------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ----------------------------------------------------------------
# 5. Train models
# ----------------------------------------------------------------
models = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "preds": preds,
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

# ----------------------------------------------------------------
# 6. Pick best model (highest F1) and plot confusion matrix
# ----------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["f1"])
best_preds = results[best_name]["preds"]
print(f"\nBest model: {best_name}")

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ----------------------------------------------------------------
# 7. Summary comparison chart
# ----------------------------------------------------------------
summary_df = pd.DataFrame({k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1"]}
                            for k, v in results.items()}).T
summary_df.to_csv("model_comparison.csv")
print("\nModel comparison:\n", summary_df)

summary_df.plot(kind="bar", figsize=(8, 5), ylim=(0.8, 1.0))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

print("\nDone. Saved: confusion_matrix.png, model_comparison.png, model_comparison.csv")
