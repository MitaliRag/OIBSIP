
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
OUTPUT_DIR = BASE_DIR / "screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Download creditcard.csv into the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n--- DATASET OVERVIEW ---")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Clean dataset
df = df.drop_duplicates().dropna().copy()

print("\nCleaned dataset shape:", df.shape)
print("\nTransaction class counts:")
print(df["Class"].value_counts())

fraud_percentage = (df["Class"].sum() / len(df)) * 100
print(f"\nFraudulent transactions: {fraud_percentage:.4f}%")

# 3. Visualize class imbalance
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Class", color="steelblue")
plt.xticks([0, 1], ["Legitimate (0)", "Fraud (1)"])
plt.title("Legitimate vs Fraudulent Transactions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
plt.close()

# 4. Compare transaction amounts
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Class", y="Amount", showfliers=False)
plt.xticks([0, 1], ["Legitimate", "Fraud"])
plt.title("Transaction Amount by Class (outliers hidden)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "amount_comparison.png", dpi=150)
plt.close()

# 5. Explore transaction timing
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,
    x="Time",
    hue="Class",
    bins=50,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Transaction Time Distribution by Class")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "time_analysis.png", dpi=150)
plt.close()

# 6. Prepare features and target
X = df.drop(columns="Class")
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 7. Train two models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
roc_data = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

    roc_data[name] = roc_curve(y_test, probabilities)

    print(f"\n--- {name} ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    # Save confusion matrix for each model
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=[0, 1],
        display_labels=["Legitimate", "Fraud"],
        cmap="Blues",
        values_format="d"
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    filename = name.lower().replace(" ", "_") + "_confusion_matrix.png"
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

    # Feature importance for Random Forest
    if name == "Random Forest":
        importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        importance.to_csv(
            BASE_DIR / "feature_importance.csv",
            index=False
        )

        plt.figure(figsize=(10, 7))
        sns.barplot(
            data=importance.head(15),
            x="Importance",
            y="Feature",
            color="steelblue"
        )
        plt.title("Top 15 Features - Random Forest")
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "feature_importance.png",
            dpi=150
        )
        plt.close()

        # Save predictions
        pd.DataFrame({
            "Actual_Class": y_test,
            "Predicted_Class": predictions,
            "Fraud_Probability": probabilities
        }, index=y_test.index).to_csv(
            BASE_DIR / "prediction_results.csv",
            index_label="Transaction_Index"
        )

# 8. Compare model metrics
results_df = pd.DataFrame(results)
results_df.to_csv(BASE_DIR / "model_comparison.csv", index=False)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))

# 9. ROC curve comparison
plt.figure(figsize=(8, 6))

for name, (fpr, tpr, thresholds) in roc_data.items():
    auc_value = results_df.loc[
        results_df["Model"] == name, "ROC-AUC"
    ].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()

print("\nProject completed successfully!")
print("Charts saved in:", OUTPUT_DIR)
print("Model metrics saved in model_comparison.csv")
print("Predictions saved in prediction_results.csv")
print("Feature importance saved in feature_importance.csv")
