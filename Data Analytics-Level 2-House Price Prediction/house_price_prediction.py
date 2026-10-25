
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Housing.csv"
OUTPUT_DIR = BASE_DIR / "screenshots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)

print("\n===== DATASET OVERVIEW =====")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# 3. Clean the data
df = df.drop_duplicates().copy()
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df = df.dropna(subset=["price"])
df = df[df["price"] > 0].copy()

df.to_csv(BASE_DIR / "cleaned_housing.csv", index=False)

# 4. Descriptive statistics
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe(include="all").T)

# 5. Outlier detection using IQR
print("\n===== POTENTIAL OUTLIERS =====")

for column in df.select_dtypes(include=np.number).columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    count = (
        (df[column] < lower) | (df[column] > upper)
    ).sum()

    print(f"{column}: {count} potential outliers")

# Keep potential outliers because they may be genuine houses.

# 6. Exploratory Data Analysis
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="area", y="price")
plt.title("House Area vs Price")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "area_vs_price.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 7))
sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# 7. Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()

# 8. Handle missing values and encode categories
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# 9. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 10. Train Linear Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression()),
])

model.fit(X_train, y_train)

# 11. Predict house prices
y_pred = model.predict(X_test)

# 12. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"MAE  : {mae:,.2f}")
print(f"R²   : {r2:.4f}")

# 13. Actual vs predicted chart
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())

plt.plot([low, high], [low, high], "--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# 14. Residual plot
residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Prediction Error")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=300)
plt.close()

# 15. Analyze model coefficients
feature_names = model.named_steps[
    "preprocessor"
].get_feature_names_out()

coefficients = model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
})

coef_df["Absolute_Coefficient"] = coef_df["Coefficient"].abs()
coef_df = coef_df.sort_values(
    "Absolute_Coefficient", ascending=False
)

print("\n===== TOP 10 COEFFICIENTS =====")
print(coef_df.head(10).to_string(index=False))

coef_df.to_csv(BASE_DIR / "model_coefficients.csv", index=False)

top = coef_df.head(10).sort_values("Coefficient")

plt.figure(figsize=(10, 6))
plt.barh(top["Feature"], top["Coefficient"])
plt.title("Top 10 Linear Regression Coefficients")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_coefficients.png", dpi=300)
plt.close()

# 16. Save prediction results
results = pd.DataFrame({
    "Actual_Price": y_test.to_numpy(),
    "Predicted_Price": y_pred,
    "Prediction_Error": y_test.to_numpy() - y_pred,
})

results.to_csv(BASE_DIR / "prediction_results.csv", index=False)

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(10).round(2).to_string(index=False))

print("\nProject completed successfully!")
print("Charts saved in:", OUTPUT_DIR)
print("Cleaned dataset saved in:", BASE_DIR / "cleaned_housing.csv")
