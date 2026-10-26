import os

DATASET_PATH = "dataset/RAVDESS"

total_files = 0

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if os.path.isdir(actor_path):

        files = os.listdir(actor_path)

        wav_files = [
            file for file in files
            if file.endswith(".wav")
        ]

        print(actor, ":", len(wav_files), "audio files")

        total_files += len(wav_files)

print("\nTotal audio files:", total_files)