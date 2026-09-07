import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

model = joblib.load("emotion_model.pkl")

def extract_features_from_array(audio, sample_rate, mfcc=True, chroma=True, mel=True):
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


def get_timeline_data(file_path, window_sec=1.0, hop_sec=0.5):
    audio, sr = librosa.load(file_path, sr=None)

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)

    times = []
    emotions = []
    confidences = []

    start = 0
    while start < len(audio):
        end = start + window_samples
        chunk = audio[start:end]

        if len(chunk) < sr * 0.5:
            break

        feat = extract_features_from_array(chunk, sr).reshape(1, -1)
        prediction = model.predict(feat)[0]
        probs = model.predict_proba(feat)[0]
        confidence = max(probs)

        mid_time = (start + end) / 2 / sr
        times.append(mid_time)
        emotions.append(prediction)
        confidences.append(confidence)

        start += hop_samples

    return times, emotions, confidences


def plot_heatmap(file_path):
    times, emotions, confidences = get_timeline_data(file_path)

    unique_emotions = ["angry", "happy", "neutral", "sad"]
    emotion_colors = {
        "angry": "#e74c3c",
        "happy": "#f1c40f",
        "neutral": "#95a5a6",
        "sad": "#3498db",
    }

    fig, ax = plt.subplots(figsize=(10, 3))

    for t, emo, conf in zip(times, emotions, confidences):
        color = emotion_colors.get(emo, "#333333")
        ax.axvspan(t - 0.25, t + 0.25, color=color, alpha=conf)

    # Legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=emotion_colors[e]) for e in unique_emotions]
    ax.legend(handles, unique_emotions, loc="upper right", ncol=4)

    ax.set_xlabel("Time (seconds)")
    ax.set_yticks([])
    ax.set_title("Emotion Heatmap Over Time")

    plt.tight_layout()
    plt.savefig("emotion_heatmap.png", dpi=150)
    print("Heatmap saved as emotion_heatmap.png")
    plt.show()


if __name__ == "__main__":
    test_file = r"dataset\OAF_angry\OAF_wife_angry.wav"
    plot_heatmap(test_file)