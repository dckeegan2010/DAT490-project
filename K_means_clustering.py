import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load data
df = pd.read_csv("C:/Users/dckee/OneDrive/Documents/DAT490/price_files/UPenn_pricing_long.csv")

# Keep necessary columns
df = df[["cpt_code", "setting", "billing_class", "negotiated_rate"]]

# Clean negotiated_rate
df["negotiated_rate"] = (
df["negotiated_rate"]
.astype(str)
.str.replace(r"[^0-9.\-]", "", regex=True)
)

df["negotiated_rate"] = pd.to_numeric(df["negotiated_rate"], errors="coerce")
df = df.dropna()
df = df[df["negotiated_rate"] > 0]

# 
# 2. Aggregate to service-level median
svc = (
    df.groupby(["cpt_code", "setting", "billing_class"], as_index=False)
      .agg(median_rate=("negotiated_rate", "median"))
)

# Log transform (important for price data)
svc["log_rate"] = np.log10(svc["median_rate"])

X = svc[["log_rate"]]

# 3. Determine best K using Silhouette Score
best_k = None
best_score = -1

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=67, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    
    print(f"K={k}, Silhouette Score={score:.4f}")
    
    if score > best_score:
        best_k = k
        best_score = score

print("\nBest K:", best_k)
print("Best Silhouette Score:", round(best_score, 4))

# 
# 4. Fit final model with best K
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
svc["cluster"] = kmeans.fit_predict(X)

# Order clusters from low to high price
centers = kmeans.cluster_centers_.flatten()
order = np.argsort(centers)
mapping = {old: new for new, old in enumerate(order, start=1)}
svc["tier"] = svc["cluster"].map(mapping)

# 5. Visualization
plt.figure()
plt.scatter(svc["log_rate"], svc["tier"])
plt.xlabel("log10(Median Negotiated Rate)")
plt.ylabel("Pricing Tier")
plt.title("K-Means Pricing Tiers")
plt.show()

# Save results
svc.to_csv("C:/Users/dckee/OneDrive/Documents/DAT490/price_files/UPenn_RQ1_KMeans_Tiers.csv", index=False)

