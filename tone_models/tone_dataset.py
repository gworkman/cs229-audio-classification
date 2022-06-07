import torch
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def visualize(sample, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    img = librosa.display.specshow(sample['spectrogram'].squeeze(0).numpy(), x_axis='time', y_axis='mel', sr=44100, fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=f'Mel frequency spectrogram for sample (tone {sample["tone"]})')


class ToneDataset(Dataset):

    def __init__(self, pickle_path, transform=None):
        self.df = pd.read_pickle(pickle_path).explode('spectrogram')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.to_list()

        spectrogram = self.df['spectrogram'].iloc[index]
        spectrogram = torch.from_numpy(spectrogram)
        spectrogram = torch.unsqueeze(spectrogram, 0)
        tone = self.df['tone'].iloc[index]
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)

        sample = (spectrogram, torch.tensor(tone - 1, dtype=torch.long))
        return sample
