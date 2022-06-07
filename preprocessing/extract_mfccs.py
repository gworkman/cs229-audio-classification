import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['Heiti TC']

CHUNK_SIZE = 1024


def get_mfccs(path: Path, num_mfccs: int):
    audio, sr = librosa.load(path, sr=None)
    audio = audio[10 * CHUNK_SIZE:-10 * CHUNK_SIZE]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mels=128, fmax=8000, n_mfcc=num_mfccs)
    return mfccs


data_path = Path('../data')
audio_path = data_path / 'audio/augmented'

df = pd.read_csv(data_path / 'single_characters.csv')


for n_mfccs in [20, 30, 40, 50, 60]:
    df['mfccs'] = df.apply(lambda r: [get_mfccs(p, n_mfccs) for p in audio_path.glob(f'{r.pinyin}{r.tone}*.wav')], axis=1)

    preview_index = 0

    preview = df['mfccs'].iloc[preview_index][0]

    fig, ax = plt.subplots()
    img = librosa.display.specshow(preview, x_axis='time', y_axis='mel', sr=44100, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=f'{n_mfccs} MFCCs for {df["hanzi"].iloc[preview_index]} (tone {df["tone"].iloc[preview_index]})')
    plt.savefig(f'../images/mfccs_{n_mfccs}.png')

    df.to_pickle(data_path / f'preprocessed/mfccs_{n_mfccs}.pkl')
