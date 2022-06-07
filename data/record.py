import select
import sys
import pandas as pd
import pyaudio
import wave
import time
import shutil
from pathlib import Path


def keypress():
    ready = sys.stdin in select.select([sys.stdin], [], [], 0.001)[0]
    if ready:
        sys.stdin.readline()
    return ready


def init():
    data = pd.read_csv('single_characters.csv')
    audio_dir = Path('audio')
    if not audio_dir.exists():
        audio_dir.mkdir()
    else:
        clear_dir = input(f'The directory {audio_dir} already exists. Would you like to empty the directory to completely restart? [y/N] -> ')
        if clear_dir.lower() == 'y':
            shutil.rmtree(audio_dir)
            audio_dir.mkdir()
        else:
            filenames = [p.name.split('_')[0] for p in audio_dir.glob('*.wav')]
            tones = [int(f[-1]) for f in filenames]
            pinyins = [f[:-1] for f in filenames]
            for p, t in zip(pinyins, tones):
                data.drop(data.index[(data['pinyin'] == p) & (data['tone'] == t)], inplace=True)

    print('Instructions:')
    print('The program will prompt you with a word, with both the simplified characters and the pinyin')
    print('When you are ready, tap the Enter key to start recording. The recording will automatically stop after 1 second.')
    print('If you mess up, stop by pressing Ctrl-C, manually delete the audio recording that was incorrect, and start again. You will only be prompted for the words you did not complete')
    return data, audio_dir


recording = False

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100

p = pyaudio.PyAudio()
stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
data, audio_dir = init()

data_len = len(data)
notify_count = data_len / 10

for i, row in data.iterrows():
    hanzi = row['hanzi']
    pinyin = row['pinyin']
    tone = row['tone']

    print(f'Next word: {hanzi} ({pinyin}{tone})')

    frames = []

    while not keypress():
        try:
            stream.read(chunk)
        except OSError:
            stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)

    print('Recording...')
    for i in range(90):
        frames.append(stream.read(chunk))
    print('Done')

    record_time = time.monotonic_ns()

    wf = wave.open(str(audio_dir / f'{pinyin}{tone}_{record_time}.wav'), 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
