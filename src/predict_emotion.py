
import os
import librosa
import numpy as np

from tensorflow.keras.models import load_model


MODEL_PATH = "models/cnn_lstm_actor_split.keras"
STATS_PATH = "models/normalization_stats.npz"
CLASSES_PATH = "models/emotion_classes.npy"

MAX_MFCC_LENGTH = 174

print("Loading model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(
        f"Normalization statistics not found: {STATS_PATH}"
    )

stats = np.load(STATS_PATH)

mean = stats["mean"].reshape(40, 1)
std = stats["std"].reshape(40, 1)

print("Normalization statistics loaded!")

if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(
        f"Emotion class file not found: {CLASSES_PATH}. "
        "Run the updated training script first."
    )

EMOTIONS = np.load(CLASSES_PATH, allow_pickle=False).tolist()

print("Emotion classes:", EMOTIONS)


def predict_emotion(audio_path):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("\nLoading audio...")

    audio, sample_rate = librosa.load(
        audio_path,
        sr=22050
    )

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    print("Audio loaded!")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    print("MFCC extracted!")
    print("Original MFCC shape:", mfcc.shape)

    if mfcc.shape[1] < MAX_MFCC_LENGTH:
        padding = MAX_MFCC_LENGTH - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, padding)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_MFCC_LENGTH]

    print("Final MFCC shape:", mfcc.shape)

    # Use the same normalization statistics as training
    mfcc = (mfcc - mean) / (std + 1e-8)

    # Add batch dimension: (40, 174) -> (1, 40, 174)
    mfcc = np.expand_dims(mfcc, axis=0)

    print("Model input shape:", mfcc.shape)

    probabilities = model.predict(
        mfcc,
        verbose=0
    )[0]

    if len(probabilities) != len(EMOTIONS):
        raise ValueError(
            "The model output count does not match the saved emotion classes."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_emotion = EMOTIONS[predicted_index]
    confidence = float(probabilities[predicted_index])

    print("\n==============================")
    print("      EMOTION PREDICTION")
    print("==============================")

    print(f"Emotion: {predicted_emotion}")
    print(f"Model confidence: {confidence * 100:.2f}%")

    print("\nAll probabilities:")

    for emotion, probability in zip(EMOTIONS, probabilities):
        print(f"{emotion:10s}: {probability * 100:.2f}%")

    print("==============================")

    return predicted_emotion, confidence


if __name__ == "__main__":
    audio_path = input("\nEnter path to WAV audio file: ").strip().strip('"')

    try:
        predict_emotion(audio_path)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
