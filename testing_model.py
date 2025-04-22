import joblib
import numpy as np

model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
rfe = joblib.load('rfe.pkl')

responses = [20, 20, 1, 1, 1,
              1, 1, 5, 1, 1,
              1, 1, 1, 1, 1,
              1, 1, 1, 1, 1]

features = np.array(responses).reshape(1, -1)

features = scaler.transform(features)
features = rfe.transform(features)


# print(features)

prediction = model.predict(features)
stress_level = float(prediction[0])
normalized_score = max(0, min(2, stress_level))

print(normalized_score)
print(f"Stress Level: {stress_level}")