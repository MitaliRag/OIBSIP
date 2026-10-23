
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# Find the input CSV
candidates = [
    BASE / "retail_sales.csv",
    BASE / "retail-sales-dataset.csv"
]
DATA_FILE = next((p for p in candidates if p.exists()), None)

if DATA_FILE is None:
    raise FileNotFoundError(
        "Dataset not found. Copy retail_sales.csv or "
        "retail-sales-dataset.csv into this project folder."
    )

# 1. Load data
df = pd.read_csv(DATA_FILE)
original = df.copy()

print("Original dataset shape:", original.shape)
print("\nOriginal column names:")
print(original.columns.tolist())

# 2. Create the initial quality report
def quality_report(data, stage):
    return {
        "Stage": stage,
        "Rows": len(data),
        "Columns": len(data.columns),
        "Missing_Cells": int(data.isna().sum().sum()),
        "Duplicate_Rows": int(data.duplicated().sum())
    }

reports = [quality_report(original, "Before cleaning")]

# 3. Standardize column names and text formatting
df.columns = [
    str(c).strip().replace("\n", " ")
    for c in df.columns
]

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].apply(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# 4. Remove duplicate rows
before_duplicates = len(df)
df = df.drop_duplicates().copy()
duplicates_removed = before_duplicates - len(df)

# 5. Standardize common column formats
for col in df.columns:
    normalized = col.lower().strip()

    if normalized in ["date", "invoice date", "order date", "transaction date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    elif normalized in [
        "age", "quantity", "price per unit", "total amount",
        "sales", "amount", "unitprice"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Report missing values before filling
missing_before = df.isna().sum().rename("Missing_Values_Before")

# Handle missing values according to column type
for col in df.columns:
    if df[col].isna().sum() == 0:
        continue

    if pd.api.types.is_numeric_dtype(df[col]):
        median = df[col].median()
        if pd.notna(median):
            df[col] = df[col].fillna(median)
    elif pd.api.types.is_datetime64_any_dtype(df[col]):
        # Dates cannot safely be inferred for every row.
        # Keep unknown dates as missing and report them.
        pass
    else:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

# 7. Detect numeric outliers using the IQR method
outlier_rows = []

for col in df.select_dtypes(include=np.number).columns:
    values = df[col].dropna()

    if len(values) < 4:
        continue

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = int(((df[col] < lower) | (df[col] > upper)).sum())

    outlier_rows.append({
        "Column": col,
        "Lower_Bound": lower,
        "Upper_Bound": upper,
        "Outlier_Count": count
    })

# Save outlier report.
# Outliers are flagged, not automatically deleted or capped.
pd.DataFrame(
    outlier_rows,
    columns=["Column", "Lower_Bound", "Upper_Bound", "Outlier_Count"]
).to_csv(OUT / "outlier_report.csv", index=False)

# 8. Save missing-value report
missing_after = df.isna().sum().rename("Missing_Values_After")

missing_report = pd.concat(
    [missing_before, missing_after], axis=1
).fillna(0)

missing_report.to_csv(OUT / "missing_values_report.csv")

# 9. Save cleaned data
df.to_csv(OUT / "cleaned_data.csv", index=False)

reports.append(quality_report(df, "After cleaning"))
pd.DataFrame(reports).to_csv(
    OUT / "before_after_summary.csv", index=False
)

# 10. Create a missing-value comparison chart
comparison = pd.DataFrame({
    "Before": original.isna().sum(),
    "After": df.isna().sum()
})

comparison.plot(kind="bar", figsize=(10, 5))
plt.title("Missing Values Before and After Cleaning")
plt.ylabel("Number of Missing Values")
plt.xlabel("Columns")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT / "missing_values_comparison.png", dpi=150)
plt.close()

# 11. Create a numeric outlier boxplot
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if numeric_cols:
    selected_cols = numeric_cols[:8]
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[selected_cols])
    plt.title("Numeric Features: Outlier Inspection")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "numeric_outlier_boxplot.png", dpi=150)
    plt.close()

# 12. Print summary
print("\nDuplicate rows removed:", duplicates_removed)
print("\nBefore cleaning:")
print(original.isna().sum())

print("\nAfter cleaning:")
print(df.isna().sum())

print("\nBefore/after summary:")
print(pd.DataFrame(reports).to_string(index=False))

print("\nTask 3 cleaning process completed.")
print("Reports and cleaned data saved in:", OUT)
