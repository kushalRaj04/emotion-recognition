import os
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


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


print("Features shape:", X.shape)
print("Labels shape:", y.shape)
  
encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

print("\nEmotion classes:")
print(encoder.classes_)
  
y_categorical = to_categorical(y_encoded)


print("\nEncoded labels shape:", y_categorical.shape)
  
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
  
model = Sequential([
    
    Dense(128, activation="relu", input_shape=(40,)),
    
    Dropout(0.3),
    
    Dense(64, activation="relu"),
    
    Dropout(0.3),
    
    Dense(8, activation="softmax")
])
  
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
  
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)
  
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)


print("\nTest Accuracy:", test_accuracy)
  
os.makedirs("models", exist_ok=True)

model.save("models/emotion_model.keras")

print("\nModel saved successfully!")