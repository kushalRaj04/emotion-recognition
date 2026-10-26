
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

DATASET_PATH = "dataset/RAVDESS"
MODEL_PATH = "models/cnn_lstm_actor_split.keras"
STATS_PATH = "models/normalization_stats.npz"
CLASSES_PATH = "models/emotion_classes.npy"
RESULTS_PATH = "results"

MAX_MFCC_LENGTH = 174

os.makedirs(RESULTS_PATH, exist_ok=True)

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


def extract_mfcc(file_path):
    """Extract 40 MFCC features and fix their length to 174 frames."""
    audio, sr = librosa.load(file_path, sr=22050)

    if audio.size == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
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

    return mfcc


# Step 1: Load test data from Actors 20–24
X_test = []
y_test = []

print("Loading test actors...")

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_PATH}"
    )

for actor in sorted(os.listdir(DATASET_PATH)):
    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    if not actor.startswith("Actor_"):
        continue

    try:
        actor_number = int(actor.split("_")[1])
    except (IndexError, ValueError):
        continue

    if actor_number < 20 or actor_number > 24:
        continue

    print(f"Processing {actor}...")

    for file in sorted(os.listdir(actor_path)):
        if not file.lower().endswith(".wav"):
            continue

        parts = file.split("-")

        if len(parts) < 3 or parts[2] not in emotion_map:
            print(f"Skipping invalid filename: {file}")
            continue

        file_path = os.path.join(actor_path, file)

        mfcc = extract_mfcc(file_path)
        emotion = emotion_map[parts[2]]

        X_test.append(mfcc)
        y_test.append(emotion)


if len(X_test) == 0:
    raise ValueError("No test audio files found for Actors 20–24.")

X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test)

print("\nTest data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)


# Step 2: Load normalization statistics from training
print("\nLoading training normalization statistics...")

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(
        f"Missing {STATS_PATH}. Run the actor-split training script first."
    )

stats = np.load(STATS_PATH)
mean = stats["mean"].reshape(40, 1)
std = stats["std"].reshape(40, 1)

X_test = (X_test - mean) / (std + 1e-8)

print("Normalization completed!")


# Step 3: Load the exact emotion class order used during training
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(
        f"Missing {CLASSES_PATH}. Run the updated actor-split "
        "training script that saves emotion_classes.npy."
    )

classes = np.load(CLASSES_PATH, allow_pickle=False).tolist()

label_encoder = LabelEncoder()
label_encoder.fit(list(emotion_map.values()))

if classes != label_encoder.classes_.tolist():
    raise ValueError(
        "Saved emotion classes do not match the current label mapping."
    )

y_encoded = label_encoder.transform(y_test)

print("\nEmotion classes:")
print(classes)


# Step 4: Load the trained actor-split model
print("\nLoading model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("Model loaded successfully!")


# Step 5: Make predictions
print("\nPredicting emotions...")

predictions = model.predict(X_test, verbose=1)

if predictions.shape[1] != len(classes):
    raise ValueError(
        "Model output count does not match the saved emotion classes."
    )

y_pred = np.argmax(predictions, axis=1)


# Step 6: Calculate accuracy
accuracy = accuracy_score(y_encoded, y_pred)

print("\n===================================")
print("SPEAKER-INDEPENDENT MODEL ACCURACY")
print("===================================")
print(f"Accuracy: {accuracy * 100:.2f}%")


# Step 7: Classification report
report = classification_report(
    y_encoded,
    y_pred,
    labels=np.arange(len(classes)),
    target_names=classes,
    zero_division=0
)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")
print(report)

report_path = os.path.join(
    RESULTS_PATH,
    "actor_split_classification_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write(report)

print("Classification report saved:", report_path)


# Step 8: Confusion matrix
cm = confusion_matrix(
    y_encoded,
    y_pred,
    labels=np.arange(len(classes))
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")
print(cm)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

disp.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False
)

ax.set_title("Speaker-Independent Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_PATH,
    "actor_split_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nConfusion matrix saved:", cm_path)
print("\nEvaluation completed successfully!")
