import librosa
import numpy as np
import joblib

# Load the trained model
model = joblib.load("emotion_model.pkl")

def extract_features_from_array(audio, sample_rate, mfcc=True, chroma=True, mel=True):
    """Same feature extraction as before, but works on an audio array directly."""
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


def predict_timeline(file_path, window_sec=2.0, hop_sec=1.0):
    """Split audio into overlapping windows and predict emotion for each window."""
    audio, sr = librosa.load(file_path, sr=None)
    total_duration = librosa.get_duration(y=audio, sr=sr)

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    print(f"Audio duration: {total_duration:.2f} seconds")
    print(f"Analyzing in {window_sec}s windows, every {hop_sec}s...\n")
    print(f"{'Time Range':<20}{'Predicted Emotion'}")
    print("-" * 40)

    start = 0
    while start < len(audio):
        end = start + window_samples
        chunk = audio[start:end]

        # Skip chunks that are too short to analyze
        if len(chunk) < sr * 0.5:
            break

        feat = extract_features_from_array(chunk, sr).reshape(1, -1)
        prediction = model.predict(feat)[0]

        start_time = start / sr
        end_time = min(end, len(audio)) / sr
        print(f"{start_time:5.1f}s - {end_time:5.1f}s     {prediction}")

        start += hop_samples


if __name__ == "__main__":
    # Change this to any longer audio file for a proper timeline demo
    test_file = r"dataset\OAF_angry\OAF_wife_angry.wav"
    predict_timeline(test_file, window_sec=1.0, hop_sec=0.5)