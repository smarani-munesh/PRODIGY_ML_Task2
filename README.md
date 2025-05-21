# PRODIGY_ML_Task2
This project focuses on unsupervised machine learning to group retail customers based on their purchase behavior. Using the Mall Customers dataset, I applied K-Means clustering after preprocessing and feature engineering to uncover patterns in customer segments.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load and Prepare the Data
data = pd.read_csv("Mall_Customers.csv")

# Step 2: Feature Engineering
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
data['IncomePerAge'] = data['Annual Income (k$)'] / data['Age']
data['CustomerValue'] = data['Annual Income (k$)'] * data['Spending Score (1-100)'] * 0.1

# Step 3: Select Features and Normalize
features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'IncomePerAge', 'CustomerValue']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 5: KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=1)
data['Group'] = kmeans.fit_predict(X_scaled)

# Step 6: Show Cluster Info
print("📊 Customers per group:")
print(data['Group'].value_counts())

print("\n🧾 Sample data with clusters:")
print(data[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Group']].head())

# Step 7: Save the result to a new CSV
data.to_csv("Clustered_Customers.csv", index=False)
print("\n✅ Clustered data saved as 'Clustered_Customers.csv'")

# Step 8: Plot
plt.figure(figsize=(8, 6))
colors = ['orange', 'blue', 'purple', 'green', 'pink']
for i in range(5):
    plt.scatter(X_pca[data['Group'] == i, 0], X_pca[data['Group'] == i, 1], 
                label=f'Group {i}', c=colors[i], s=60)

plt.title("Customer Clusters Based on Behavior")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
