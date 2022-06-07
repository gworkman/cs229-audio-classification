import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Heiti TC']

CHUNK_SIZE = 1024


def get_spectrogram(path: Path):
    audio, sr = librosa.load(path, sr=None)
    audio = audio[10 * CHUNK_SIZE:-10 * CHUNK_SIZE]
    mfccs = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    return mfccs


data_path = Path('../data')
audio_path = data_path / 'audio/augmented'

df = pd.read_csv(data_path / 'single_characters.csv')
df['spectrogram'] = df.apply(lambda r: [get_spectrogram(p) for p in audio_path.glob(f'{r.pinyin}{r.tone}*.wav')], axis=1)

preview_index = 0

preview = df['spectrogram'].iloc[preview_index][0]
preview_db = librosa.power_to_db(preview, ref=np.max)

fig, ax = plt.subplots()
img = librosa.display.specshow(preview_db, x_axis='time', y_axis='mel', sr=44100, fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title=f'Mel frequency spectrogram for {df["hanzi"].iloc[preview_index]} (tone {df["tone"].iloc[preview_index]})')
plt.savefig('../images/spectrogram.png')

df.to_pickle(data_path / f'preprocessed/spectrogram.pkl')
