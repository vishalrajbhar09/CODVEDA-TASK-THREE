#!/usr/bin/env python
# coding: utf-8

# # NAME :- VISHAL RAMKUMAR RAJBHAR
# ID :- CV/A1/18203 DOMAIN :- Data Science Intern

# Task 3: Clustering (Unsupervised
# Learning).
# • Description: Implement K-Means clustering to group
# data points into clusters without labels (e.g.,
# customer segmentation).

# In[1]:


# Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# In[2]:


# Step 2: Create Custom Cricket Dataset
np.random.seed(42)

# Simulate 4 types of players
bat_avg = np.concatenate([np.random.normal(45, 5, 50), np.random.normal(30, 4, 50),
                          np.random.normal(20, 5, 50), np.random.normal(35, 3, 50)])

strike_rate = np.concatenate([np.random.normal(135, 10, 50), np.random.normal(90, 8, 50),
                              np.random.normal(65, 10, 50), np.random.normal(110, 9, 50)])

wickets = np.concatenate([np.random.normal(20, 5, 50), np.random.normal(5, 2, 50),
                          np.random.normal(35, 6, 50), np.random.normal(15, 4, 50)])

economy = np.concatenate([np.random.normal(6.5, 0.5, 50), np.random.normal(7.2, 0.4, 50),
                          np.random.normal(5.5, 0.6, 50), np.random.normal(6.0, 0.5, 50)])

# Create DataFrame
df = pd.DataFrame({
    'Batting_Avg': bat_avg,
    'Strike_Rate': strike_rate,
    'Wickets': wickets,
    'Economy': economy
})

# Step 3: Elbow Method
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()


# In[3]:


# Step 4: Silhouette Score
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(df)
    score = silhouette_score(df, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='green')
plt.title("Silhouette Scores for Different K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()


# In[4]:


# Step 5: Final Model
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans_final.fit_predict(df)

# Step 6: PCA for 2D Visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df.drop('Cluster', axis=1))

# Step 7: Cluster Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Cluster'], palette='Set2', s=70)
plt.title("K-Means Clustering on Cricket Player Stats")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# Step 8: Summary
print(f"Optimal number of clusters: {optimal_k}")
print(f"Silhouette Score for k={optimal_k}: {silhouette_score(df.drop('Cluster', axis=1), df['Cluster']):.3f}")

