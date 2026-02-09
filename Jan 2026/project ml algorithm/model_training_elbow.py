import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
import os


df = pd.read_csv(r"C:\Users\santo\OneDrive\Desktop\Data science\Jan 2026\project ml algorithm\salary_spending_dataset.csv"
)

print("Dataset Loaded Successfully")
print(df.head())

X = df[['Age', 'Experience', 'Salary', 'Monthly_Spending']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature Scaling Done")

wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("Model Training Completed")
print(df.head())

save_path = r"C:\Users\santo\OneDrive\Desktop\Data science\Jan 2026\project ml algorithm"

model_file = os.path.join(save_path, "kmeans_model.pkl")
scaler_file = os.path.join(save_path, "scaler.pkl")

pickle.dump(kmeans, open(model_file, "wb"))
pickle.dump(scaler, open(scaler_file, "wb"))

print("Model and Scaler Saved Successfully")
print("Saved at:", save_path)

import pandas as pd

sample_data = pd.DataFrame(
    [[30, 5, 50000, 25000]],
    columns=['Age', 'Experience', 'Salary', 'Monthly_Spending']
)

sample_scaled = scaler.transform(sample_data)
predicted_cluster = kmeans.predict(sample_scaled)

print("Predicted Cluster:", predicted_cluster[0])
