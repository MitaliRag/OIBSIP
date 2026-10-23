
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv(BASE / "retail_sales.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert data types
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_cols = [
    "Age", "Quantity", "Price per Unit", "Total Amount"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# 1. Descriptive statistics
stats = df[numeric_cols].agg(["mean", "median", "std"])
stats.loc["mode"] = [
    df[col].mode().iloc[0] for col in numeric_cols
]
stats.to_csv(OUT / "descriptive_statistics.csv")

# 2. Monthly sales trend
monthly = (
    df.dropna(subset=["Date"])
      .set_index("Date")["Total Amount"]
      .resample("MS").sum()
)

monthly.plot(figsize=(10, 5), marker="o")
plt.title("Monthly Retail Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(OUT / "monthly_sales.png", dpi=300)
plt.close()

# 3. Quarterly sales trend
quarterly = (
    df.dropna(subset=["Date"])
      .set_index("Date")["Total Amount"]
      .resample("QS").sum()
)

quarterly.plot(figsize=(9, 5), marker="o")
plt.title("Quarterly Retail Sales")
plt.xlabel("Quarter")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(OUT / "quarterly_sales.png", dpi=300)
plt.close()

# 4. Customer age distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Age", bins=15, kde=True)
plt.title("Customer Age Distribution")
plt.tight_layout()
plt.savefig(OUT / "age_distribution.png", dpi=300)
plt.close()

# 5. Gender breakdown
df["Gender"].value_counts().plot(
    kind="bar", figsize=(7, 5)
)
plt.title("Customer Gender Breakdown")
plt.xlabel("Gender")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(OUT / "gender_breakdown.png", dpi=300)
plt.close()

# 6. Top 10 products by revenue
top_products = (
    df.groupby("Product Category")["Total Amount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

top_products.to_csv(OUT / "top_products.csv")
top_products.sort_values().plot(
    kind="barh", figsize=(9, 5)
)
plt.title("Top Product Categories by Revenue")
plt.xlabel("Total Revenue")
plt.tight_layout()
plt.savefig(OUT / "top_products.png", dpi=300)
plt.close()

# 7. Revenue by product category
category_sales = (
    df.groupby("Product Category")["Total Amount"]
      .sum()
      .sort_values(ascending=False)
)

category_sales.to_csv(OUT / "revenue_by_category.csv")
category_sales.plot(kind="bar", figsize=(9, 5))
plt.title("Revenue by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Revenue")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(OUT / "revenue_by_category.png", dpi=300)
plt.close()

# 8. Correlation heatmap
correlation = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation, annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Between Numerical Variables")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=300)
plt.close()

# 9. Additional insight: sales by weekday
weekday_sales = df.dropna(subset=["Date"]).copy()
weekday_sales["Weekday"] = weekday_sales["Date"].dt.day_name()

weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

weekday_sales = (
    weekday_sales.groupby("Weekday")["Total Amount"]
    .sum().reindex(weekday_order)
)

weekday_sales.to_csv(OUT / "sales_by_weekday.csv")
weekday_sales.plot(kind="bar", figsize=(9, 5))
plt.title("Sales by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(OUT / "sales_by_weekday.png", dpi=300)
plt.close()

# 10. Save useful summary
print("\nTotal revenue:", df["Total Amount"].sum())
print("\nRevenue by category:\n", category_sales)
print("\nSales by weekday:\n", weekday_sales)

print("\nAll charts and reports saved in the outputs folder!")
