
import os
import librosa
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    BatchNormalization,
    Dropout,
    LSTM,
    Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


DATASET_PATH = "dataset/RAVDESS"
MODEL_PATH = "models/cnn_lstm_actor_split.keras"
STATS_PATH = "models/normalization_stats.npz"
CLASSES_PATH = "models/emotion_classes.npy"

MAX_MFCC_LENGTH = 174

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

X_train = []
y_train = []

X_test = []
y_test = []


print("Loading RAVDESS dataset...")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

for actor in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    # Expected folder names: Actor_01, Actor_02, etc.
    try:
        actor_number = int(actor.split("_")[1])
    except (IndexError, ValueError):
        print(f"Skipping unexpected folder: {actor}")
        continue

    if actor_number < 1 or actor_number > 24:
        print(f"Skipping unexpected actor number: {actor}")
        continue

    print(f"Processing {actor}...")

    for file in sorted(os.listdir(actor_path)):
        if not file.lower().endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)
        parts = file.split("-")

        if len(parts) < 3 or parts[2] not in emotion_map:
            print(f"Skipping invalid filename: {file}")
            continue

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

        if mfcc.shape[1] < MAX_MFCC_LENGTH:
            padding = MAX_MFCC_LENGTH - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, padding)),
                mode="constant"
            )
        else:
            mfcc = mfcc[:, :MAX_MFCC_LENGTH]

        if actor_number <= 19:
            X_train.append(mfcc)
            y_train.append(emotion)
        else:
            X_test.append(mfcc)
            y_test.append(emotion)


X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

print("\nDataset loaded successfully!")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError(
        "Training or testing data is empty. "
        "Check the RAVDESS folder structure and actor names."
    )

# Normalize using training data only
print("\nNormalizing MFCC features...")

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

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

# Convert emotion names into numeric labels
label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\nEmotion classes:")
print(label_encoder.classes_)

num_classes = len(label_encoder.classes_)

y_train_cat = to_categorical(
    y_train_encoded,
    num_classes=num_classes
)

y_test_cat = to_categorical(
    y_test_encoded,
    num_classes=num_classes
)

# Build the CNN-LSTM model
model = Sequential([
    Input(shape=(40, MAX_MFCC_LENGTH)),

    Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.25),

    LSTM(64),
    Dropout(0.5),

    Dense(64, activation="relu"),
    Dropout(0.5),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

# Stop training when validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Reduce learning rate when progress slows
reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[
        early_stopping,
        reduce_learning_rate
    ],
    verbose=1
)

# Evaluate using actors excluded from training
print("\n====================================")
print("EVALUATING ON UNSEEN ACTORS")
print("====================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=1
)

print(f"\nTest loss: {test_loss:.4f}")
print(
    f"Speaker-independent test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

# Save model and matching preprocessing information
os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

np.savez(
    STATS_PATH,
    mean=mean.reshape(40, 1),
    std=std.reshape(40, 1)
)

np.save(
    CLASSES_PATH,
    label_encoder.classes_
)

print("\nModel and supporting files saved successfully!")
print("Model:", MODEL_PATH)
print("Normalization statistics:", STATS_PATH)
print("Emotion classes:", CLASSES_PATH)
