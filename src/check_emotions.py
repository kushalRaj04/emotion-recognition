import os

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

count = {}

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if file.endswith(".wav"):

            parts = file.split("-")

            emotion_code = parts[2]

            emotion = emotion_map[emotion_code]

            count[emotion] = count.get(emotion, 0) + 1


print("\nEmotion distribution:\n")

for emotion, total in count.items():
    print(emotion, ":", total)