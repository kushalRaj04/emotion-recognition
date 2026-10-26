
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)


DATASET_PATH = "dataset/RAVDESS"
MODEL_PATH = "models/cnn_lstm_emotion_model_improved.keras"
STATS_PATH = "models/normalization_stats_improved.npz"
CLASSES_PATH = "models/emotion_classes_improved.npy"
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


# Step 1: Load the audio dataset
X = []
y = []

print("Loading dataset...")

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

        X.append(mfcc)
        y.append(emotion)


X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

if len(X) == 0:
    raise ValueError("No WAV files were loaded.")

print("\nDataset loaded!")
print("X shape:", X.shape)
print("y shape:", y.shape)


# Step 2: Encode emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("\nDataset emotion classes:")
print(label_encoder.classes_)


# Step 3: Reproduce the same random split used in training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Step 4: Load training normalization statistics
print("\nLoading normalization statistics...")

if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(
        f"Statistics file not found: {STATS_PATH}. "
        "Run the matching training script first."
    )

stats = np.load(STATS_PATH)

mean = stats["mean"].reshape(40, 1)
std = stats["std"].reshape(40, 1)

X_test = (X_test - mean) / (std + 1e-8)

print("Normalization completed!")


# Step 5: Load the matching trained model
print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = load_model(MODEL_PATH)

print("Model loaded successfully!")


# Step 6: Verify the model's emotion class order
if os.path.exists(CLASSES_PATH):
    saved_classes = np.load(
        CLASSES_PATH,
        allow_pickle=False
    ).tolist()

    if saved_classes != label_encoder.classes_.tolist():
        raise ValueError(
            "Saved emotion classes do not match the dataset labels."
        )

print("\nEmotion class order:")
print(label_encoder.classes_)


# Step 7: Predict emotions for the test set
print("\nMaking predictions...")

y_probability = model.predict(
    X_test,
    verbose=1
)

y_pred = np.argmax(y_probability, axis=1)


# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("MODEL ACCURACY")
print("===================================")
print(f"Accuracy: {accuracy * 100:.2f}%")


# Step 9: Generate the classification report
report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_)),
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n===================================")
print("CLASSIFICATION REPORT")
print("===================================")
print(report)

report_path = os.path.join(
    RESULTS_PATH,
    "classification_report.txt"
)

with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write(report)

print("Classification report saved to:", report_path)


# Step 10: Create the confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(label_encoder.classes_))
)

print("\n===================================")
print("CONFUSION MATRIX")
print("===================================")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=label_encoder.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))

disp.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False
)

plt.title("Speech Emotion Recognition - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    RESULTS_PATH,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nConfusion matrix saved to:", cm_path)

print("\nEvaluation completed successfully!")
