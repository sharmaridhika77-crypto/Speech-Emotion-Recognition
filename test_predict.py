import joblib
from speech_emotion_recognition import extract_features

# Load the trained model
model = joblib.load("emotion_model.pkl")

# Test audio file (chosen from your dataset)
test_file = r"dataset\OAF_angry\OAF_wife_angry.wav"

features = extract_features(test_file).reshape(1, -1)
prediction = model.predict(features)[0]

print(f"Actual emotion (from filename): angry")
print(f"Predicted Emotion: {prediction}")