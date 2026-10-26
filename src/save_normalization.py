
import os
import librosa
import numpy as np

DATASET_PATH = "dataset/RAVDESS"
MAX_MFCC_LENGTH = 174

X_train = []

print("Calculating training normalization statistics...")

for actor_number in range(1, 20):
    actor = f"Actor_{actor_number:02d}"
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        print(f"Warning: {actor_path} not found. Skipping.")
        continue

    print(f"Processing {actor}...")

    for file in sorted(os.listdir(actor_path)):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        audio, sample_rate = librosa.load(
            file_path,
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

        X_train.append(mfcc)

if len(X_train) == 0:
    raise ValueError(
        "No audio files found. Check the dataset/RAVDESS folder."
    )

X_train = np.asarray(X_train, dtype=np.float32)

print("\nTraining data shape:")
print(X_train.shape)

mean = np.mean(
    X_train,
    axis=(0, 2),
    keepdims=True
)

std = np.std(
    X_train,
    axis=(0, 2),
    keepdims=True
)

os.makedirs("models", exist_ok=True)

np.savez(
    "models/normalization_stats.npz",
    mean=mean.reshape(40, 1),
    std=std.reshape(40, 1)
)

print("\nNormalization statistics saved!")
print("File: models/normalization_stats.npz")
print("Mean shape:", mean.reshape(40, 1).shape)
print("Standard deviation shape:", std.reshape(40, 1).shape)
