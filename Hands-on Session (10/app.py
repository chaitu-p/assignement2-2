# Flask API for Stroke Prediction

from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import networkx as nx

# Load the pre-trained models and scaler
rf_model = joblib.load('rf_stroke_model.pkl')
nn_model = load_model('nn_stroke_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define route for stroke prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([data['age'], data['hypertension'], data['heart_disease'],
                         data['avg_glucose_level'], data['bmi'], data['smoking_status']]).reshape(1, -1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Get predictions from both models
    rf_pred = rf_model.predict(features_scaled)
    nn_pred = nn_model.predict(features_scaled)
    
    # Ensemble: Majority voting
    final_pred = np.round((rf_pred + nn_pred.flatten()) / 2)
    
    # Query the knowledge graph for recommendations
    recommendation = get_recommendation(final_pred)
    
    return jsonify({
        'stroke_prediction': 'High Risk' if final_pred == 1 else 'Low Risk',
        'recommendation': recommendation
    })

# Function to query the knowledge graph for recommendations
def get_recommendation(prediction):
    G = nx.Graph()
    
    # Example knowledge graph for stroke prevention recommendations
    G.add_edge('High Risk', 'Exercise Regularly', recommendation='Engage in moderate exercise 30 mins daily')
    G.add_edge('High Risk', 'Diet Control', recommendation='Adopt a low-fat, low-sodium diet')
    G.add_edge('Low Risk', 'Healthy Lifestyle', recommendation='Maintain a balanced diet and stay active')
    
    # Return recommendations based on prediction
    if prediction == 1:
        return {'exercise': G['High Risk']['Exercise Regularly']['recommendation'],
                'diet': G['High Risk']['Diet Control']['recommendation']}
    else:
        return {'recommendation': G['Low Risk']['Healthy Lifestyle']['recommendation']}

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
