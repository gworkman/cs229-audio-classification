import random
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLE_RATE = 44100
NOISE_SIGMA = 0.05


def add_guassian_noise(audio: np.ndarray):
    return audio + np.random.normal(0, NOISE_SIGMA, audio.shape)


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
    augmented_noise = add_guassian_noise(original_audio)
    augmented_ambient = add_random_ambient(original_audio, 0.8)
    sf.write(audio_path / 'augmented' / (file_path.stem + '_raw.wav'), original_audio, SAMPLE_RATE)
    sf.write(audio_path / 'augmented' / (file_path.stem + '_noise.wav'), augmented_noise, SAMPLE_RATE)
    sf.write(audio_path / 'augmented' / (file_path.stem + '_ambient.wav'), augmented_ambient, SAMPLE_RATE)
