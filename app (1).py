import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Read dataset from GitHub
url = "https://drive.google.com/uc?export=download&id=1UJ8MNsBgexdGNdDwQS-mK46NdmjGTs61"
df = pd.read_csv(url)

# Preprocessing
X = df.select_dtypes(include=["number"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

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
