import os
import glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

FOCUSED_EMOTIONS = ["happy", "sad", "angry", "neutral"]


def get_emotion_label(filename):
    base = os.path.basename(filename).lower()

    parts = base.split("-")
    if len(parts) >= 3 and parts[2] in RAVDESS_EMOTIONS:
        return RAVDESS_EMOTIONS[parts[2]]

    for emo in ["happy", "sad", "angry", "neutral", "fear", "disgust",
                "surprise", "calm", "ps"]:
        if emo in base:
            return "surprised" if emo == "ps" else emo

    return None


def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_path) as sound_file:
        audio = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

        features = np.array([])

        if chroma:
            stft = np.abs(librosa.stft(audio))

        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            features = np.hstack((features, mfccs))

        if chroma:
            chroma_feat = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            features = np.hstack((features, chroma_feat))

        if mel:
            mel_feat = np.mean(
                librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0
            )
            features = np.hstack((features, mel_feat))

        return features


def load_data(dataset_path, test_size=0.2):
    X, y = [], []

    audio_files = glob.glob(os.path.join(dataset_path, "**", "*.wav"), recursive=True)
    print(f"Found {len(audio_files)} audio files.")

    for file in audio_files:
        emotion = get_emotion_label(file)
        if emotion is None or emotion not in FOCUSED_EMOTIONS:
            continue

        try:
            feat = extract_features(file)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            continue

        X.append(feat)
        y.append(emotion)

    print(f"Usable samples after filtering: {len(X)}")
    return train_test_split(np.array(X), np.array(y), test_size=test_size, random_state=42, stratify=y)


def train_model(dataset_path):
    X_train, X_test, y_train, y_test = load_data(dataset_path)

    model = MLPClassifier(
        hidden_layer_sizes=(300,),
        alpha=0.01,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        random_state=42,
    )

    print("Training MLPClassifier...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {acc * 100:.2f}%\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, "emotion_model.pkl")
    print("\nModel saved as emotion_model.pkl")

    return model


def predict_emotion(model, file_path):
    feat = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(feat)[0]
    return prediction


if __name__ == "__main__":
    DATASET_PATH = "dataset"

    trained_model = train_model(DATASET_PATH)