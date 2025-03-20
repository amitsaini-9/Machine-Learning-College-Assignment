import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv('Mall_Customers.csv')

print("Data Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Annual Income (k$)'] = df['Annual Income (k$)'].fillna(df['Annual Income (k$)'].mean())
df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])

df['Genre_encoded'] = df['Genre'].map({'Male': 0, 'Female': 1})

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Genre_encoded']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

df['KMeans_Labels'] = kmeans_labels

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='KMeans_Labels', palette='viridis')
plt.title('K-Means Clustering')
plt.show()

linkage_matrix = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

agg_clustering = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
agg_labels = agg_clustering.fit_predict(X_scaled)

df['Agglomerative_Labels'] = agg_labels

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Agglomerative_Labels', palette='viridis')
plt.title('Agglomerative Clustering')
plt.show()

print("\nClustering Summary:")
print("===================")
print("\nK-Means Cluster Centers:")
print(kmeans.cluster_centers_)

print("\nAgglomerative Clustering Labels Distribution:")
print(df['Agglomerative_Labels'].value_counts())