from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
from flask_cors import CORS 

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
# Load the trained SVM model
try:
    model = joblib.load('svm_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define feature names (these should match those used during training)
feature_names = [
    'anxiety_level',  'mental_health_history', 'depression','headache','blood_pressure',
    'sleep_quality','saftey','basic_needs','academic_performance','teacher_student_relationship',
    'future_career_concerns','social_support','peer_pressure','extracurricular_activities','bullying'
    
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
        # Get data from request
        data = request.json
        responses = data.get('responses', [])
        
        if len(responses) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(responses)}'
            }), 400
        
        # Convert responses to numpy array and reshape for prediction
        features = np.array(responses).reshape(1, -1)
        
        # Make prediction
        if model is not None:
            # Get the raw prediction (classification)
            stress_level = float(model.predict(features)[0])
            
            # Ensure the value is between 0-2
            normalized_score = max(0, min(2, stress_level))
            
            # Get personalized recommendations based on stress level
            recommendations = get_recommendations(normalized_score)
            
            return jsonify({
                'stress_score': normalized_score,
                'recommendations': recommendations
            })
        else:
            # Fallback to basic calculation if model isn't loaded
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
    """Fallback calculation if model fails to load"""
    # Simple formula: average the values and scale to 0-2
    avg = sum(responses) / len(responses)
    max_possible = 5  # Assuming max possible value is 5
    return min(2, (avg / max_possible) * 2)

def get_recommendations(stress_score):
    """Generate personalized recommendations based on stress score"""
    if stress_score < 0.5:
        return {
            "level": "Low",
            "color": "green",
            "text": "You have a low stress level. Maintain a balanced lifestyle by staying physically active, practicing mindfulness, getting enough sleep, and engaging in hobbies.",
            "action_plan": [
                "Stay physically active with walking, yoga, or gym workouts",
                "Practice mindfulness and meditation",
                "Get 7-9 hours of sleep per night",
                "Keep a positive social circle and engage in hobbies"
            ]
        }
    elif stress_score < 1.0:
        return {
            "level": "Moderate", 
            "color": "yellow",
            "text": "Your stress is moderate. Consider implementing strategies to prevent escalation.",
            "action_plan": [
                "Practice deep breathing exercises (box breathing, diaphragmatic breathing)",
                "Maintain a structured daily routine",
                "Reduce screen time and engage in offline activities",
                "Limit caffeine and processed foods, and stay hydrated",
                "Journal your thoughts and practice gratitude"
            ]
        }
    elif stress_score < 1.5:
        return {
            "level": "Moderate to High",
            "color": "orange", 
            "text": "You are experiencing moderate to high stress. Try implementing stress management techniques.",
            "action_plan": [
                "Engage in relaxation techniques like progressive muscle relaxation",
                "Reduce workload and practice effective time management",
                "Seek support from close friends, family, or support groups",
                "Listen to calming music or try aromatherapy",
                "Take short breaks during study or work sessions"
            ]
        }
    else:
        return {
            "level": "High",
            "color": "red",
            "text": "Your stress level is high. It's important to take immediate action.",
            "action_plan": [
                "Consider seeking professional help (counseling, therapy, or support groups)",
                "Engage in high-intensity physical activities to release stress",
                "Avoid negative coping mechanisms such as excessive alcohol or caffeine",
                "Try guided meditation apps or stress relief programs",
                "Prioritize self-care, set boundaries, and focus on self-compassion"
            ]
        }

if __name__ == '__main__':
    app.run(debug=True)