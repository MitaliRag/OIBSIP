
from pathlib import Path
import re
import html
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

DATA_FILE = BASE / "IMDB Dataset.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "IMDB Dataset.csv not found. Put it inside the project folder."
    )

print("Loading IMDb dataset...")
df = pd.read_csv(DATA_FILE)

# Standardize column names
df.columns = [str(c).strip().lower() for c in df.columns]

if not {"review", "sentiment"}.issubset(df.columns):
    raise ValueError(
        f"Expected columns 'review' and 'sentiment'. Found: {df.columns.tolist()}"
    )

print("Original dataset shape:", df.shape)

# Remove missing and duplicate records
df = df.dropna(subset=["review", "sentiment"]).copy()
df["review"] = df["review"].astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

df = df[df["sentiment"].isin(["positive", "negative"])]
df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)

# Text preprocessing
def clean_text(text):
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)       # Remove HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text) # Keep English words
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

df["clean_review"] = df["review"].apply(clean_text)
df = df[df["clean_review"].str.len() > 0].reset_index(drop=True)

# Convert sentiment to numeric labels
df["label"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

print("Dataset shape after cleaning:", df.shape)
print("\nSentiment counts:")
print(df["sentiment"].value_counts())

# Save cleaned dataset
df[["review", "sentiment", "clean_review"]].to_csv(
    OUT / "cleaned_reviews.csv", index=False
)

# 1. Sentiment distribution chart
counts = df["sentiment"].value_counts().reindex(
    ["negative", "positive"], fill_value=0
)

plt.figure(figsize=(7, 5))
counts.plot(kind="bar")
plt.title("IMDb Review Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUT / "sentiment_distribution.png", dpi=150)
plt.close()

# 2. Split data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# 3. TF-IDF feature extraction
vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF training matrix:", X_train_tfidf.shape)

# 4. Train two machine-learning models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

metrics = []
misclassified = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    metrics.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0
    ))

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        OUT / f"{name.lower().replace(' ', '_')}_classification_report.csv"
    )

    # 5. Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Sentiment")
    plt.ylabel("Actual Sentiment")
    plt.tight_layout()
    plt.savefig(
        OUT / f"{name.lower().replace(' ', '_')}_confusion_matrix.png",
        dpi=150
    )
    plt.close()

    # Collect incorrect predictions
    wrong_indices = np.flatnonzero(
        predictions != y_test.to_numpy()
    )

    for idx in wrong_indices[:5]:
        misclassified.append({
            "Model": name,
            "Review": X_test.iloc[idx],
            "Actual": "Positive" if y_test.iloc[idx] == 1 else "Negative",
            "Predicted": "Positive" if predictions[idx] == 1 else "Negative"
        })

# 6. Save model comparison metrics
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(metrics_df.to_string(index=False))

plt.figure(figsize=(9, 5))
metrics_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1_Score"]
].plot(kind="bar", ax=plt.gca())
plt.title("Sentiment Analysis Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT / "model_comparison.png", dpi=150)
plt.close()

# 7. Save five misclassified examples per model, where available
pd.DataFrame(misclassified).to_csv(
    OUT / "misclassified_examples.csv", index=False
)

# 8. Generate positive and negative word clouds
for sentiment in ["positive", "negative"]:
    text = " ".join(
        df.loc[df["sentiment"] == sentiment, "clean_review"].tolist()
    )

    if text.strip():
        cloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            max_words=150,
            collocations=False,
            random_state=42
        ).generate(text)

        plt.figure(figsize=(12, 6))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud - {sentiment.title()} Reviews")
        plt.tight_layout()
        plt.savefig(
            OUT / f"{sentiment}_wordcloud.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

# 9. Save summary
summary = pd.DataFrame({
    "Item": [
        "Original rows",
        "Rows after cleaning",
        "Training rows",
        "Testing rows",
        "Unique sentiments",
        "TF-IDF features"
    ],
    "Value": [
        len(pd.read_csv(DATA_FILE)),
        len(df),
        len(X_train),
        len(X_test),
        ", ".join(sorted(df["sentiment"].unique())),
        X_train_tfidf.shape[1]
    ]
})

summary.to_csv(OUT / "project_summary.csv", index=False)

print("\nTask 4 Sentiment Analysis completed successfully!")
print("All reports and charts saved in:", OUT)
