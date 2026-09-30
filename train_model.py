import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Load dataset
data = pd.read_csv("smartphone_dataset_1M.csv")

# Features
features = ['ram_gb', 'storage_gb', 'battery_mah', 'display_size_inch']

X = data[features].dropna()
data = data.loc[X.index]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)

# Save model
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model trained successfully!")