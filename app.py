import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Read dataset from GitHub
url = "https://drive.google.com/uc?export=download&id=1UJ8MNsBgexdGNdDwQS-mK46NdmjGTs61"
df = pd.read_csv(url)

df = df.replace("..", np.nan)

year_cols = [col for col in df.columns if col.isdigit()]
df[year_cols] = df[year_cols].apply(pd.to_numeric, errors='coerce')

df["feature_mean"] = df[year_cols].mean(axis=1)

df_clean = df[["Country name", "Series name", "feature_mean"]]

print(df_clean.head())
print(df_clean.shape)

# Pivot table: rows = countries, columns = indicators (Series name)
df_pivot = df_clean.pivot_table(
    index="Country name",
    columns="Series name",
    values="feature_mean"
)

# Reset index to make 'Country name' a column again
df_pivot = df_pivot.reset_index()

print(df_pivot.shape)
print(df_pivot.head())

df_pivot_filled = df_pivot.fillna(df_pivot.mean(numeric_only=True))

print(df_pivot_filled.shape)
print(df_pivot_filled.isna().sum().sum(), "missing values remain")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = df_pivot_filled.drop(columns=["Country name"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  
df_pivot_filled["Cluster"] = kmeans.fit_predict(X_scaled)

print(df_pivot_filled[["Country name", "Cluster"]].head(10))


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df_pivot_filled["Cluster"], cmap="viridis", alpha=0.7)
plt.colorbar(label="Cluster")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Country Clusters on Sustainable Energy & Efficiency")
plt.show()

cluster_counts = df_pivot_filled["Cluster"].value_counts().sort_index()
print("Cluster counts:")
print(cluster_counts)

for c in sorted(df_pivot_filled["Cluster"].unique()):
    print(f"\nCluster {c} sample countries:")
    print(df_pivot_filled[df_pivot_filled["Cluster"] == c]["Country name"].head(5).to_list())

numeric_cols = df_pivot_filled.drop(columns=["Country name", "Cluster"]).columns

cluster_means = df_pivot_filled.groupby("Cluster")[numeric_cols].mean().round(2)
print("\nCluster averages (summary table):")
print(cluster_means)

import plotly.express as px

fig = px.scatter(
    x=X_pca[:,0],
    y=X_pca[:,1],
    color=df_pivot_filled["Cluster"].astype(str),
    hover_name=df_pivot_filled["Country name"],
    title="Country Clusters (Interactive PCA)"
)

fig.show()

# Streamlit UI
st.title("Country Clusters on Sustainable Energy & Efficiency")
st.write("### Sample of clustered data")
st.dataframe(df[["Country name", "Cluster"]].head(10))

# Plot
fig, ax = plt.subplots(figsize=(10,6))
scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster"], cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Cluster")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("Country Clusters")
st.pyplot(fig)
