import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def pinyin_to_tone_number(pinyin):
    pinyin = unicodedata.normalize('NFD', pinyin)
    if '\u0304' in pinyin:
        return 1
    elif '\u0301' in pinyin:
        return 2
    elif '\u030c' in pinyin:
        return 3
    elif '\u0300' in pinyin:
        return 4
    else:
        return 5


def strip_tone_from_pinyin(pinyin):
    pinyin = unicodedata.normalize('NFD', pinyin)
    return pinyin.replace('\u0304', '').replace('\u0301', '').replace('\u030c', '').replace('\u0300', '')


if __name__ == '__main__':
    data_dir = Path(__file__).parent
    json_path = data_dir / 'hsk.json'

    df = pd.read_json(json_path)
    df = df.drop(['translations', 'level', 'id'], axis=1)

    # filter the df to only unique single-character entries
    df = df.loc[df['hanzi'].apply(lambda h: len(h)) == 1]
    df = df.drop_duplicates(subset=['pinyin'], keep='first')

    # get the tone number of the character
    df['tone'] = df['pinyin'].apply(lambda p: pinyin_to_tone_number(p))
    df['pinyin'] = df['pinyin'].apply(lambda p: strip_tone_from_pinyin(p))
    df = df.loc[df['tone'] != 5]

    pinyin_counts = df['pinyin'].value_counts()
    df = df.loc[df['pinyin'].apply(lambda p: pinyin_counts[p] > 2)]
    df = df.sort_values(by=['pinyin', 'tone'], axis=0)

    df.to_csv(data_dir / 'single_characters.csv', index=False)

    # plot the tone counts
    tone_counts = df['tone'].value_counts().sort_index()

    fig, ax = plt.subplots()
    plt.bar(tone_counts.index, tone_counts)
    plt.title('Tone counts for unique pronunciations extracted from HSK 1-6 data')
    plt.xlabel('Tone number')
    plt.ylabel('Count')
    plt.savefig('../images/tone_balance.png')
