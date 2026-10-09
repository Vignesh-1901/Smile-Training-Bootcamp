import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

def train():
    # 1. Read music_listeners.csv
    csv_path = 'music_listeners.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('..', 'music_listeners.csv')
        
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully:")
    print(df.head())
    
    # 2. Select listener behaviour features
    feature_cols = ['listening_hours_per_week', 'songs_per_day', 'skip_rate', 'playlist_count']
    X = df[feature_cols]
    
    # 3. Scale the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Create K-Means model with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    
    # 5. Train the model using fit()
    kmeans.fit(X_scaled)
    
    # 6. Find and understand the three listener groups
    print("\n--- Cluster Centers (Original Scale) ---")
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers_original, columns=feature_cols)
    centers_df['Cluster'] = [0, 1, 2]
    print(centers_df)
    
    # Sort cluster indices by listening_hours_per_week to assign meaningful names
    sorted_clusters = centers_df.sort_values(by='listening_hours_per_week')['Cluster'].tolist()
    segment_mapping = {
        sorted_clusters[0]: "Casual Listener",
        sorted_clusters[1]: "Music Explorer",
        sorted_clusters[2]: "Heavy Listener"
    }
    
    print("\n--- Listener Group Definitions ---")
    for cluster_id, label in segment_mapping.items():
        avg_hours = centers_df.loc[centers_df['Cluster'] == cluster_id, 'listening_hours_per_week'].values[0]
        avg_songs = centers_df.loc[centers_df['Cluster'] == cluster_id, 'songs_per_day'].values[0]
        print(f"Cluster {cluster_id} -> {label} (Avg Hours/Week: {avg_hours:.1f}, Avg Songs/Day: {avg_songs:.1f})")
        
    # 7. Save model.pkl and scaler.pkl
    joblib.dump(kmeans, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save segment mapping for the app
    joblib.dump(segment_mapping, 'segment_mapping.pkl')
    print("\n[SUCCESS] Saved model.pkl, scaler.pkl, and segment_mapping.pkl!")

if __name__ == '__main__':
    train()
