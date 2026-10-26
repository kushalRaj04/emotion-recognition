import librosa
import matplotlib.pyplot as plt


audio_path = "dataset/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav"


audio, sample_rate = librosa.load(audio_path, sr=None)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio length:", len(audio))


mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

print("MFCC shape:", mfcc.shape)


plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sample_rate
)

plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.show()