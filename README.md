# Speech Emotion Recognition using Librosa & MLPClassifier

This project identifies emotions (**happy, sad, angry, neutral**) from voice recordings using audio feature extraction and a machine learning classifier. It demonstrates how AI can analyze audio data to understand human emotion.

## Tools Used
- **Python**
- **Librosa** — for audio processing and feature extraction (MFCC, Chroma, Mel Spectrogram)
- **Scikit-learn** — for building and training the ML model
- **MLPClassifier** — a neural network model used for emotion classification

## Dataset
- **TESS (Toronto Emotional Speech Set)** — public speech emotion dataset from Kaggle
- Contains labeled `.wav` audio files for different emotions

## How It Works
1. **Feature Extraction**: Each audio file is processed with Librosa to extract MFCC, Chroma, and Mel Spectrogram features — numerical representations of the voice's pitch, tone, and energy.
2. **Model Training**: These features are used to train an `MLPClassifier` (a feedforward neural network) to recognize patterns associated with each emotion.
3. **Prediction**: The trained model can then predict the emotion of any new/unseen audio file.

## Results
- **Accuracy achieved: ~99.7%** on the test set
- Successfully classifies: `angry`, `happy`, `neutral`, `sad`

## Files
| File | Description |
|------|-------------|
| `speech_emotion_recognition.py` | Main script — loads dataset, extracts features, trains the model, and evaluates accuracy |
| `test_predict.py` | Script to test the trained model on a new audio file |
| Saved trained model (ready to use for predictions) |

## How to Run
```bash
pip install librosa scikit-learn numpy soundfile joblib
python speech_emotion_recognition.py
python test_predict.py
```

## Sample Output
Found 5600 audio files.
Usable samples after filtering: 3200
Training MLPClassifier...
Accuracy: 99.69%
## Learning Outcome
This project helped in understanding:
- How raw audio can be converted into meaningful numerical features
- How machine learning models can classify audio-based emotional patterns
- End-to-end workflow of a speech-based AI system: data → features → model → prediction
- ## Addon Features
1. **Emotion Timeline Tracking** — Splits audio into small time windows and predicts emotion for each, showing how emotion changes over the duration of a clip. Useful for interviews, meetings, and podcasts.
2. **Emotion Heatmap Visualization** — Displays emotional intensity across time as an interactive color-coded heatmap.
3. **Multilingual Emotion Recognition** — The model uses acoustic features (MFCC, Chroma, Mel-spectrogram) that capture pitch, tone, and energy patterns rather than words. This makes emotion detection naturally language-independent — it can work on English, Hindi, Gujarati, or any language, since it recognizes *how* something is said rather than *what* is said.
