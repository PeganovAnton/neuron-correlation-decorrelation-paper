from collections import Counter
from itertools import chain
from pathlib import Path

import numpy as np


class CharLanguageModellingVocabulary:
    def __init__(
            self,
            save_path,
            text_paths,
            max_num_characters=None,
            force_recollection=False
    ):
        self.save_path = Path(__file__).parent / Path(save_path).expanduser()
        self.text_paths = text_paths
        self.max_num_characters = max_num_characters
        self.vocab = None
        if force_recollection:
            self.create_vocab()
            self.save_vocab()
        elif self.save_path.is_file():
            self.load_vocab()
        else:
            self.create_vocab()
            self.save_vocab()

    def create_vocab(self):
        text_files = [
            Path(tp).expanduser().open(buffering=65536)
            for tp in self.text_paths]
        counter = Counter()
        for line in chain(*text_files):
            counter.update(line)
        for tf in text_files:
            tf.close()
        most_common, _ = zip(*counter.most_common(self.max_num_characters))
        most_common = list(most_common)
        most_common.append('<UNK>')
        self.vocab = most_common

    def save_vocab(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open('w') as f:
            for char in self.vocab:
                if char == '<UNK>':
                    f.write(char + '\n')
                else:
                    f.write(repr(char) + '\n')

    def load_vocab(self):
        self.vocab = []
        with self.save_path.open() as f:
            for line in f:
                line = line.strip()
                if line == '<UNK>':
                    self.vocab.append(line)
                else:
                    self.vocab.append(eval(line))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.vocab[key]
        elif isinstance(key, str):
            if key not in self.vocab:
                key = '<UNK>'
            return self.vocab.index(key)
        else:
            raise TypeError("Only `int` and `str` keys are supported")

    def __len__(self):
        return len(self.vocab)

    def text2indices(self, text):
        a = []
        for c in text:
            a.append(self[c])
        return np.array(a)

    def indices2text(self, indices):
        text = ''
        for i in indices:
            text += self[i]
        return text

