from collections import Counter
from itertools import chain
from pathlib import Path


class LanguageModellingVocabulary:
    def __init__(
            self,
            save_path,
            text_paths,
            max_num_characters,
            force_recollection=False
    ):
        self.save_path = Path(save_path).expanduser()
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
        most_common.append('<UNK>')
        self.vocab = most_common

    def save_vocab(self):
        with self.save_path.open() as f:
            for char in self.vocab:
                if char == '<UNK>':
                    f.write(char + '\n')
                else:
                    f.write(repr(char) + '\n')

    def load_vocab(self):
        self.vocab = []
        with self.save_path.open() as f:
            for line in self.vocab:
                if line.strip() == '<UNK>':
                    self.vocab.append(line)
                else:
                    self.vocab.append(eval(line))

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.vocab[key]
        else:
            return self.vocab.index(key)

    def __len__(self):
        return len(self.vocab)