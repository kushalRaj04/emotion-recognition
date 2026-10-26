
from flask import Flask, request, render_template
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

MODEL_PATH = "models/cnn_lstm_actor_split.keras"
STATS_PATH = "models/normalization_stats.npz"

MAX_MFCC_LENGTH = 174

EMOTIONS = [
    "Angry",
    "Calm",
    "Disgust",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
]

print("Loading emotion recognition model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")

stats = np.load(STATS_PATH)

mean = stats["mean"].reshape(40, 1)
std = stats["std"].reshape(40, 1)

print("Normalization statistics loaded!")


def predict_emotion(audio_path):
    audio, sample_rate = librosa.load(
        audio_path,
        sr=22050
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    if mfcc.shape[1] < MAX_MFCC_LENGTH:
        padding = MAX_MFCC_LENGTH - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, padding)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_MFCC_LENGTH]

    mfcc = (mfcc - mean) / (std + 1e-8)

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    probabilities = model.predict(
        mfcc,
        verbose=0
    )[0]

    predicted_index = np.argmax(probabilities)
    predicted_emotion = EMOTIONS[predicted_index]
    confidence = probabilities[predicted_index] * 100

    probability_dict = {}

    for emotion, probability in zip(
        EMOTIONS,
        probabilities
    ):
        probability_dict[emotion] = round(
            float(probability * 100),
            2
        )

    return (
        predicted_emotion,
        round(float(confidence), 2),
        probability_dict
    )


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return "No audio file uploaded."

    file = request.files["audio"]

    if file.filename == "":
        return "No file selected."

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        file.filename
    )

    file.save(file_path)

    try:
        emotion, confidence, probabilities = predict_emotion(
            file_path
        )

        return render_template(
            "result.html",
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(
        debug=False,
        use_reloader=False
    )
