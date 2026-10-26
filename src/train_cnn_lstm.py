
import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# Dataset location
DATASET_PATH = "dataset/RAVDESS"

# Map emotion codes to emotion names
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

print("Loading dataset...")

for actor in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)
        parts = file.split("-")

        if len(parts) < 3 or parts[2] not in emotion_map:
            print("Skipping invalid filename:", file)
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

        X.append(mfcc)
        y.append(emotion)

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nDataset preparation completed!")
print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError(
        "No audio files found. Check DATASET_PATH and the RAVDESS folders."
    )

print("\nNormalizing MFCC features...")

# Split the data before calculating normalization statistics
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train_labels, y_test_labels = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Calculate normalization statistics using training data only
mean = np.mean(X_train, axis=(0, 2), keepdims=True)
std = np.std(X_train, axis=(0, 2), keepdims=True)

X_train = (X_train - mean) / (std + 1e-8)
X_test = (X_test - mean) / (std + 1e-8)

# Convert emotion labels to one-hot format
y_train = to_categorical(
    y_train_labels,
    num_classes=len(encoder.classes_)
)

y_test = to_categorical(
    y_test_labels,
    num_classes=len(encoder.classes_)
)

print("Normalization completed!")
print("Emotion classes:", encoder.classes_)
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

model = Sequential([
    Input(shape=(40, 174)),

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

    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Summary:")
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

print("\nStarting model training...")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        early_stopping,
        reduce_learning_rate
    ]
)

print("\nEvaluating model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n===================================")
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("===================================")

os.makedirs("models", exist_ok=True)

model.save(
    "models/cnn_lstm_emotion_model_improved.keras"
)

# Save the normalization statistics for future predictions
os.makedirs("models", exist_ok=True)

np.savez(
    "models/normalization_stats_improved.npz",
    mean=mean.reshape(40, 1),
    std=std.reshape(40, 1)
)

# Save the label order used by this model
np.save(
    "models/emotion_classes_improved.npy",
    encoder.classes_
)

print("\nImproved model saved successfully!")
print("Model: models/cnn_lstm_emotion_model_improved.keras")
print("Normalization: models/normalization_stats_improved.npz")
print("Emotion classes: models/emotion_classes_improved.npy")
