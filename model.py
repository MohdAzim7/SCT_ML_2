import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/Mall_Customers.csv")
    return df

def train_model(df, k=5):

    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=42)

    clusters = model.fit_predict(X_scaled)

    df["Cluster"] = clusters

    return df, model, scaler