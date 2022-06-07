import random
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLE_RATE = 44100


def add_guassian_noise(audio: np.ndarray, volume: float):
    return audio + np.random.normal(0, volume, audio.shape)


def add_random_ambient(audio: np.ndarray, volume: float):
    ambient_paths = list((audio_path / 'ambient_sounds').glob('*.wav'))
    ambient, _sr = librosa.load(random.choice(ambient_paths), sr=SAMPLE_RATE)
    offset = random.randrange(0, len(ambient) - (len(audio) + 1))
    ambient_trimmed = ambient[offset:offset + len(audio)] * volume
    return audio + ambient_trimmed


data_path = Path('../data')
audio_path = data_path / 'audio'

for file_path in (audio_path / 'raw').glob('*.wav'):
    original_audio, _sr = librosa.load(file_path, sr=SAMPLE_RATE)
    sf.write(audio_path / 'augmented' / (file_path.stem + '_raw.wav'), original_audio, SAMPLE_RATE)

    for i, noise in enumerate([0.01, 0.05, 0.075]):
        augmented_noise = add_guassian_noise(original_audio, noise)
        sf.write(audio_path / 'augmented' / (file_path.stem + f'_noise_{i}.wav'), augmented_noise, SAMPLE_RATE)

    for i, volume in enumerate([0.5, 0.8, 1.0, 1.2, 1.5]):
        augmented_ambient = add_random_ambient(original_audio, volume)
        sf.write(audio_path / 'augmented' / (file_path.stem + f'_ambient_{i}.wav'), augmented_ambient, SAMPLE_RATE)
