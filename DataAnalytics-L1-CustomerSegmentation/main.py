
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Load the actual dataset
data_file = BASE / "retail-sales-dataset.csv"
df = pd.read_csv(data_file)

print("Dataset loaded:", df.shape)

# 2. Clean data
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Total Amount"] = pd.to_numeric(df["Total Amount"], errors="coerce")
df["Customer ID"] = df["Customer ID"].astype("string").str.strip()

original_rows = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=["Date", "Customer ID", "Total Amount"])
df = df[df["Total Amount"] > 0].copy()

df.to_csv(OUT / "cleaned_retail_data.csv", index=False)

pd.DataFrame({
    "Metric": [
        "Original rows",
        "Cleaned rows",
        "Rows removed",
        "Customers"
    ],
    "Value": [
        original_rows,
        len(df),
        original_rows - len(df),
        df["Customer ID"].nunique()
    ]
}).to_csv(OUT / "data_quality_report.csv", index=False)

# 3. RFM analysis
reference_date = df["Date"].max().normalize() + pd.Timedelta(days=1)

rfm = df.groupby("Customer ID").agg(
    LastPurchase=("Date", "max"),
    Frequency=("Transaction ID", "nunique"),
    Monetary=("Total Amount", "sum")
)

rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
rfm = rfm[["Recency", "Frequency", "Monetary"]]
rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)].copy()

if len(rfm) < 4:
    raise ValueError("Need at least 4 customers for clustering.")

rfm.to_csv(OUT / "customer_rfm.csv")

print("Customers analyzed:", len(rfm))

# 4. Prepare features and scale them
features = np.log1p(rfm[["Recency", "Frequency", "Monetary"]])
scaler = StandardScaler()
X = scaler.fit_transform(features)

sns.set_theme(style="whitegrid")

# 5. Elbow Method
k_values = list(range(2, min(9, len(rfm))))
inertias = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertias, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(OUT / "elbow_method.png", dpi=150)
plt.close()

# 6. K-Means clustering
k = min(4, len(rfm) - 1)
model = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = model.fit_predict(X)

score = silhouette_score(X, rfm["Cluster"])
rfm.to_csv(OUT / "customer_segments.csv")

# 7. Cluster profiles
profiles = rfm.groupby("Cluster").agg(
    Customers=("Recency", "size"),
    AvgRecency=("Recency", "mean"),
    AvgFrequency=("Frequency", "mean"),
    AvgMonetary=("Monetary", "mean"),
    TotalRevenue=("Monetary", "sum")
).round(2)

profiles.to_csv(OUT / "cluster_profiles.csv")

print("\nCluster profiles:")
print(profiles)
print(f"\nSilhouette score: {score:.3f}")

# 8. Customer count chart
plt.figure(figsize=(8, 5))
sns.countplot(data=rfm.reset_index(), x="Cluster", hue="Cluster", legend=False)
plt.title("Customers in Each Cluster")
plt.tight_layout()
plt.savefig(OUT / "cluster_customer_counts.png", dpi=150)
plt.close()

# 9. Recency versus Monetary
plot_df = rfm.reset_index()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=plot_df, x="Recency", y="Monetary",
    hue="Cluster", palette="tab10"
)
plt.title("Recency vs Monetary")
plt.tight_layout()
plt.savefig(OUT / "recency_vs_monetary.png", dpi=150)
plt.close()

# 10. Frequency versus Monetary
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=plot_df, x="Frequency", y="Monetary",
    hue="Cluster", palette="tab10"
)
plt.title("Frequency vs Monetary")
plt.tight_layout()
plt.savefig(OUT / "frequency_vs_monetary.png", dpi=150)
plt.close()

# 11. PCA visualization
components = PCA(n_components=2).fit_transform(X)

pca_df = pd.DataFrame({
    "PC1": components[:, 0],
    "PC2": components[:, 1],
    "Cluster": rfm["Cluster"].to_numpy()
})

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=pca_df, x="PC1", y="PC2",
    hue="Cluster", palette="tab10"
)
plt.title("Customer Clusters - PCA")
plt.tight_layout()
plt.savefig(OUT / "customer_clusters_pca.png", dpi=150)
plt.close()

# 12. Marketing recommendations based on cluster profiles
recommendations = []

for cluster, row in profiles.iterrows():
    if (
        row["AvgRecency"] <= profiles["AvgRecency"].median()
        and row["AvgMonetary"] >= profiles["AvgMonetary"].median()
    ):
        segment = "Active high-value customers"
        action = "Reward loyalty and provide exclusive offers."
    elif row["AvgRecency"] > profiles["AvgRecency"].median():
        segment = "Potentially inactive customers"
        action = "Test personalized win-back campaigns."
    elif row["AvgFrequency"] >= profiles["AvgFrequency"].median():
        segment = "Frequent customers"
        action = "Offer loyalty rewards and product bundles."
    else:
        segment = "Developing customers"
        action = "Encourage repeat purchases with targeted offers."

    recommendations.append({
        "Cluster": cluster,
        "SuggestedSegment": segment,
        "Recommendation": action
    })

pd.DataFrame(recommendations).to_csv(
    OUT / "marketing_recommendations.csv", index=False
)

print("\nSUCCESS: Task 2 analysis completed.")
print("All real reports and charts saved in:", OUT)
