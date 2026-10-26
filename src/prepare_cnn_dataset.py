
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

MAX_MFCC_LENGTH = 174

print("Loading audio files...")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

for actor in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in sorted(os.listdir(actor_path)):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        # Extract the emotion code from the filename
        parts = file.split("-")

        if len(parts) < 3 or parts[2] not in emotion_map:
            print(f"Skipping invalid filename: {file}")
            continue

        emotion_code = parts[2]
        emotion = emotion_map[emotion_code]

        # Load audio at 22050 Hz
        audio, sample_rate = librosa.load(
            file_path,
            sr=22050
        )

        # Extract 40 MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Pad or truncate each sample to 174 frames
        if mfcc.shape[1] < MAX_MFCC_LENGTH:
            padding = MAX_MFCC_LENGTH - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, padding)),
                mode="constant"
            )
        else:
            mfcc = mfcc[:, :MAX_MFCC_LENGTH]

        X.append(mfcc)
        y.append(emotion)

# Convert lists into NumPy arrays
X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

if len(X) == 0:
    raise ValueError(
        "No WAV files were loaded. Check your dataset folder structure."
    )

print("\nDataset preparation completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nEmotion distribution:")
for emotion in sorted(set(y)):
    print(f"{emotion}: {np.sum(y == emotion)}")
