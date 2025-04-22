import joblib
import numpy as np

# Load the trained model and preprocessing components
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
rfe = joblib.load('rfe.pkl')

# Your survey responses
responses = [20, 6, 0, 12, 2, 2, 3, 3, 3, 4, 4, 5, 4, 4, 4, 5, 2, 1, 5, 1]

# Convert responses to the format needed for prediction
features = np.array(responses).reshape(1, -1)

# Apply the same preprocessing steps used during training
features = scaler.transform(features)
features = rfe.transform(features)

# Make prediction
prediction = model.predict(features)
stress_level = float(prediction[0])
normalized_score = max(0, min(2, stress_level))

print(normalized_score)
print(f"Stress Level: {stress_level}")