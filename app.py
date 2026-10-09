import os
import sys
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
MAPPING_PATH = 'segment_mapping.pkl'

# Global variables for model, scaler, and segment mapping
model = None
scaler = None
segment_mapping = None

def load_artifacts():
    global model, scaler, segment_mapping
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("[ERROR] model.pkl or scaler.pkl not found! Please run train_model.py first.")
        sys.exit(1)
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    if os.path.exists(MAPPING_PATH):
        segment_mapping = joblib.load(MAPPING_PATH)
    else:
        # Fallback default mapping
        segment_mapping = {0: "Music Explorer", 1: "Heavy Listener", 2: "Casual Listener"}
        
    print("[SUCCESS] Loaded model.pkl, scaler.pkl, and segment_mapping.pkl successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 1. Receive input features in exact training order
        features = [
            float(data['listening_hours_per_week']),
            float(data['songs_per_day']),
            float(data['skip_rate']),
            float(data['playlist_count'])
        ]
        
        # 2. Convert into pandas DataFrame with exact feature names
        feature_cols = ['listening_hours_per_week', 'songs_per_day', 'skip_rate', 'playlist_count']
        input_data = pd.DataFrame([features], columns=feature_cols)
        
        # 3. Preprocess using scaler.transform() (Do NOT call fit!)
        scaled_input = scaler.transform(input_data)
        
        # 4. Predict cluster using model.predict()
        cluster_id = int(model.predict(scaled_input)[0])
        
        # 5. Map cluster ID to human readable segment name
        segment_name = segment_mapping.get(cluster_id, f"Cluster {cluster_id}")
        
        return jsonify({
            'status': 'success',
            'cluster': cluster_id,
            'segment': segment_name
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 400

def run_cli_mode():
    load_artifacts()
    print("\n================================================")
    print(" MUSIC LISTENER SEGMENTATION - CLI PREDICTOR ")
    print("================================================\n")
    
    try:
        hours = float(input("1. Enter listening hours per week (e.g. 15): "))
        songs = float(input("2. Enter songs per day (e.g. 40): "))
        skip = float(input("3. Enter skip rate % (e.g. 25): "))
        playlists = float(input("4. Enter playlist count (e.g. 10): "))
        
        feature_cols = ['listening_hours_per_week', 'songs_per_day', 'skip_rate', 'playlist_count']
        input_data = pd.DataFrame([[hours, songs, skip, playlists]], columns=feature_cols)
        scaled_input = scaler.transform(input_data)
        cluster_id = int(model.predict(scaled_input)[0])
        segment_name = segment_mapping.get(cluster_id, f"Cluster {cluster_id}")
        
        print("\n------------------------------------------------")
        print(f" Predicted Cluster ID : {cluster_id}")
        print(f" Listener Segment    : {segment_name}")
        print("------------------------------------------------\n")
    except Exception as e:
        print(f"[ERROR] Invalid input: {e}")

if __name__ == '__main__':
    load_artifacts()
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        run_cli_mode()
    else:
        print("\n🚀 Starting Flask server at http://127.0.0.1:5000 ...")
        app.run(host='0.0.0.0', port=5000, debug=True)
