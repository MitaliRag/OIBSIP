
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "retail_sales.csv"
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

if not file_path.exists() or file_path.stat().st_size == 0:
    raise FileNotFoundError(
        "Please add your actual retail_sales.csv dataset first."
    )

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nFirst 5 records:\n", df.head())
print("\nColumns:\n", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nNull values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

numeric = df.select_dtypes(include="number")

if not numeric.empty:
    stats = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Mode": numeric.mode().iloc[0],
        "Standard Deviation": numeric.std()
    })
    stats.to_csv(output_dir / "descriptive_statistics.csv")
    print("\nStatistics:\n", stats.round(2))

quality = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": [str(df[c].dtype) for c in df.columns],
    "Missing_Values": [int(df[c].isna().sum()) for c in df.columns],
    "Unique_Values": [int(df[c].nunique()) for c in df.columns]
})

quality.to_csv(output_dir / "data_quality_report.csv", index=False)

print("\nReports saved in outputs/")
