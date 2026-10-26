import os
import librosa
import numpy as np


DATASET_PATH = "dataset/RAVDESS"


emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


X = []
y = []


for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        parts = file.split("-")

        emotion_code = parts[2]

        emotion = emotion_map[emotion_code]

        audio, sample_rate = librosa.load(
            file_path,
            sr=22050
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_mean = np.mean(mfcc.T, axis=0)

        X.append(mfcc_mean)
        y.append(emotion)


X = np.array(X)
y = np.array(y)


print("Dataset preparation completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)