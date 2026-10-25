
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "googleplaystore.csv"
OUTPUT_DIR = BASE_DIR / "screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n--- DATASET OVERVIEW ---")
print(df.head())
print("\nOriginal shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Clean the data
df = df.drop_duplicates().copy()

df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
df["Reviews"] = pd.to_numeric(df["Reviews"], errors="coerce")

df["Installs"] = (
    df["Installs"].astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("+", "", regex=False)
)

df["Installs"] = pd.to_numeric(df["Installs"], errors="coerce")

df["Price"] = (
    df["Price"].astype(str)
    .str.replace("$", "", regex=False)
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

df["Size"] = (
    df["Size"].astype(str)
    .str.replace("M", "", regex=False)
    .str.replace("k", "", regex=False)
)

df["Size"] = pd.to_numeric(df["Size"], errors="coerce")

df = df[
    df["Rating"].between(0, 5)
    & df["Installs"].notna()
    & df["Reviews"].notna()
    & df["Price"].notna()
].copy()

print("\nCleaned shape:", df.shape)
print("\nTop categories:")
print(df["Category"].value_counts().head(10))

# 3. Top 10 app categories
plt.figure(figsize=(11, 6))
category_counts = df["Category"].value_counts().head(10)

sns.barplot(
    x=category_counts.values,
    y=category_counts.index,
    color="steelblue"
)
plt.title("Top 10 Google Play Store Categories")
plt.xlabel("Number of Apps")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_categories.png", dpi=150)
plt.close()

# 4. Rating distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Rating"], bins=20, kde=True)
plt.title("App Rating Distribution")
plt.xlabel("Rating")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rating_distribution.png", dpi=150)
plt.close()

# 5. Free versus paid apps
type_counts = df["Type"].value_counts()

plt.figure(figsize=(7, 5))
plt.bar(type_counts.index.astype(str), type_counts.values)
plt.title("Free vs Paid Apps")
plt.xlabel("App Type")
plt.ylabel("Number of Apps")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "free_vs_paid.png", dpi=150)
plt.close()

# 6. Installs versus rating
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Installs",
    y="Rating",
    hue="Type",
    alpha=0.5
)
plt.xscale("log")
plt.title("App Ratings vs Installs")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ratings_vs_installs.png", dpi=150)
plt.close()

# 7. Average rating by category
average_rating = (
    df.groupby("Category")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(11, 7))
sns.barplot(
    x=average_rating.values,
    y=average_rating.index,
    color="teal"
)
plt.title("Average Rating by Category")
plt.xlabel("Average Rating")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "average_rating_by_category.png", dpi=150)
plt.close()

# 8. Correlation analysis
numeric_columns = ["Rating", "Reviews", "Installs", "Price"]
correlation = df[numeric_columns].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Between App Metrics")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# 9. Save cleaned data and summary
df.to_csv(BASE_DIR / "cleaned_googleplaystore.csv", index=False)

summary = df.groupby("Category").agg(
    Total_Apps=("App", "count"),
    Average_Rating=("Rating", "mean"),
    Average_Reviews=("Reviews", "mean"),
    Average_Installs=("Installs", "mean")
).round(2)

summary.to_csv(BASE_DIR / "category_summary.csv")

print("\n--- KEY FINDINGS ---")
print("Total cleaned app records:", len(df))
print("Average app rating:", round(df["Rating"].mean(), 2))
print("Free apps:", int((df["Type"] == "Free").sum()))
print("Paid apps:", int((df["Type"] == "Paid").sum()))

print("\nHighest-rated categories with at least 20 apps:")
category_ratings = df.groupby("Category").agg(
    Average_Rating=("Rating", "mean"),
    App_Count=("App", "count")
)
category_ratings = category_ratings[
    category_ratings["App_Count"] >= 20
].sort_values("Average_Rating", ascending=False)

print(category_ratings.head(10).round(2))

print("\nProject completed successfully!")
print("Charts saved in:", OUTPUT_DIR)
print("Cleaned dataset: cleaned_googleplaystore.csv")
print("Category summary: category_summary.csv")
