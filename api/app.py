from flask import Flask, request, jsonify
import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from urllib.parse import quote_plus

# Load API Key from .env file
load_dotenv('.env')
api_key = os.getenv("GOOGLE_API_KEY")

# Replace with your file ID
file_id = "1-2ITUnGFC3p2_YA-M7VCKPfD1iwNIyOe"
download_url = f"https://drive.google.com/uc?id={file_id}"

# Fetch the CSV file from Google Drive
response = requests.get(download_url)
if response.status_code == 200:
    # Read the CSV into a DataFrame
    from io import StringIO
    df = pd.read_csv(StringIO(response.text))
else:
    raise Exception(f"Failed to download file. Status code: {response.status_code}")

# Convert latitude and longitude columns to numeric, handling errors
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=5)
df['Cluster'] = dbscan.fit_predict(df[['latitude', 'longitude']])


def geocode_address_google(user_address):
    """Geocodes the user's address using Google Maps Geocoding API."""
    user_address = quote_plus(user_address)  # URL-encode the address
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={user_address}&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get('results', [])
        if results:
            location = results[0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None


def find_nearest_cluster(user_lat, user_lon):
    """Find the nearest cluster to the user's location."""
    cluster_centers = df.groupby('Cluster')[['latitude', 'longitude']].mean()
    distances = np.sqrt((cluster_centers['latitude'] - user_lat) ** 2 +
                        (cluster_centers['longitude'] - user_lon) ** 2)
    return distances.idxmin()


def recommend_hospitals_cluster_knn(user_lat, user_lon, n_recommendations=5):
    """Recommend hospitals using a hybrid of clustering and KNN."""
    nearest_cluster = find_nearest_cluster(user_lat, user_lon)

    cluster_hospitals = df[df['Cluster'] == nearest_cluster]
    if cluster_hospitals.empty:
        return []  # Return an empty list if no hospitals are found in the cluster

    # KNN within the cluster
    knn = NearestNeighbors(n_neighbors=n_recommendations, algorithm='auto')
    knn.fit(cluster_hospitals[['latitude', 'longitude']])

    # Ensure that user input is passed as a DataFrame with correct feature names
    user_location = pd.DataFrame([[user_lat, user_lon]], columns=['latitude', 'longitude'])

    # Use kneighbors() on the user location
    distances, indices = knn.kneighbors(user_location)
    
    # Get the nearest hospitals based on the indices
    nearest_hospitals = cluster_hospitals.iloc[indices[0]].copy()

    # Sort by ratings
    nearest_hospitals = nearest_hospitals.sort_values(by='ratings', ascending=False)

    return nearest_hospitals[['facility_name', 'address', 'ratings']].to_dict(orient='records')


# Create Flask app
app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend_handler():
    """Handles incoming requests to recommend hospitals."""
    try:
        # Get user input from the request JSON
        user_address = request.json.get('address')
        
        if not user_address:
            return jsonify({"error": "No address provided."}), 400

        # Geocode the address
        user_lat, user_lon = geocode_address_google(user_address)
        if user_lat is None or user_lon is None:
            return jsonify({"error": "Invalid address or geocoding failed."}), 400

        # Get recommendations
        recommendations = recommend_hospitals_cluster_knn(user_lat, user_lon)
        if not recommendations:
            return jsonify({"message": "No hospitals found in the vicinity."}), 404

        return jsonify({"hospitals": recommendations}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
