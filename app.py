from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
import json
from datetime import datetime
from flask_cors import CORS 


app = Flask(__name__, static_folder='static')
CORS(app)

# Load the trained SVM model
try:
    model = joblib.load('svm_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the StandardScaler (fitted on the 15 selected features)
try:
    scaler = joblib.load('scaler.pkl')
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler = None

try:
    rfe = joblib.load('rfe.pkl')
    print("RFE loaded successfully")
except Exception as e:
    print(f"Error loading RFE: {e}")
    rfe = None
# Use only the selected 15 features (same order as used during training)
selected_feature_names = [
    'anxiety_level', 'self_esteem', 'depression', 'headache', 'blood_pressure',
    'sleep_quality', 'noise_level', 'safety', 'basic_needs', 'academic_performance',
    'teacher_student_relationship', 'future_career_concerns', 'social_support',
    'extracurricular_activities', 'bullying'
]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.endswith('.html'):
        return send_from_directory('.', path)
    return send_from_directory('static', path)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        responses = data.get('responses', [])

        # if len(responses) != len(selected_feature_names):
        #     return jsonify({
        #         'error': f'Expected {len(selected_feature_names)} features, got {len(responses)}'
        #     }), 400

        features = np.array(responses).reshape(1, -1)

        # Apply scaling
        features = scaler.transform(features)
        features = rfe.transform(features)

        # Predict with the model
        if model is not None:
            stress_level = float(model.predict(features)[0])
            normalized_score = max(0, min(2, stress_level))
            recommendations = get_recommendations(normalized_score)

            return jsonify({
                'stress_score': normalized_score,
                'recommendations': recommendations
            })

        else:
            basic_score = calculate_basic_score(responses)
            recommendations = get_recommendations(basic_score)
            return jsonify({
                'stress_score': basic_score,
                'recommendations': recommendations
            })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

def calculate_basic_score(responses):
    avg = sum(responses) / len(responses)
    max_possible = 5
    return min(2, (avg / max_possible) * 2)

def get_recommendations(stress_score):
    if stress_score < 0.5:
        return {
            "level": "Low",
            "color": "green",
            "text": "You have a low stress level. Maintain a balanced lifestyle...",
            "action_plan": [
                "Stay physically active", "Practice mindfulness", "Get enough sleep", "Engage in hobbies"
            ]
        }
    elif stress_score < 1.0:
        return {
            "level": "Moderate", 
            "color": "yellow",
            "text": "Your stress is moderate. Consider strategies to prevent escalation.",
            "action_plan": [
                "Practice deep breathing", "Maintain a routine", "Reduce screen time", "Journal thoughts"
            ]
        }
    elif stress_score < 1.5:
        return {
            "level": "Moderate to High",
            "color": "orange", 
            "text": "You are experiencing moderate to high stress...",
            "action_plan": [
                "Try progressive muscle relaxation", "Manage time effectively", "Seek support"
            ]
        }
    else:
        return {
            "level": "High",
            "color": "red",
            "text": "Your stress level is high. Immediate action is recommended.",
            "action_plan": [
                "Seek professional help", "Engage in physical activity", "Avoid negative coping mechanisms"
            ]
        }

SUBMISSIONS_DIR = 'contact_submissions'

# Ensure the submissions directory exists
if not os.path.exists(SUBMISSIONS_DIR):
    os.makedirs(SUBMISSIONS_DIR)

@app.route('/api/submit-contact', methods=['POST'])
def submit_contact():
    try:
        # Get form data from request
        data = request.json
        
        # Validate required fields
        if not all(key in data for key in ['name', 'email', 'message']):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Add timestamp if not provided
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        
        # Create a unique filename using timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{SUBMISSIONS_DIR}/contact_{timestamp}_{data['name'].replace(' ', '_')}.json"
        
        # Write data to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Contact form submitted successfully'}), 200
    
    except Exception as e:
        print(f"Error processing contact form: {str(e)}")
        return jsonify({'error': 'Server error processing your request'}), 500

if __name__ == '__main__':
    app.run(debug=True)