from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset


class CharLanguageModellingDataset(IterableDataset):
    def __init__(
            self,
            vocab,
            path,
            start_character_idx,
            batch_size,
            bptt_len,
            dataset_len=None,
            loop=False
    ):
        self.n_examples = self.count_examples_in_dataset()
        self.vocab = vocab
        self.path = Path(path).expanduser()
        self.start_character_idx = start_character_idx
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.bptt_len = bptt_len
        if dataset_len is None:
            self.dataset_len = self.n_examples - self.start_character_idx
        else:
            self.dataset_len = dataset_len
        self.loop = loop
        if self.batch_size > self.n_examples:
            raise ValueError(
                f"The size of the dataset {self.n_examples} is greater than "
                f"batch size {self.batch_size}.")
        self.cursors = []
        self.step = self.n_examples // self.batch_size
        for i in range(self.batch_size):
            self.cursors.append(self.path.open(buffering=65536))
            self.cursors[i].seek(self.start_character_idx + i * self.step)

    def _read(self, file_object, length, end=None):
        if end is None:
            end = self.start_character_idx + self.dataset_len
        if self.loop:
            text = ''
            while len(text) < length:
                n_remained_characters_in_ds = end - file_object.tell()
                text += file_object.read(
                    min(length - len(text), n_remained_characters_in_ds))
                if len(text) < length:
                    file_object.seek(self.start_character_idx)
        else:
            n_remained_characters_in_ds = end - file_object.tell()
            text = file_object.read(min(length, n_remained_characters_in_ds))
        return text

    def _get_end_of_segment(self, i):
        if i < self.batch_size - 1:
            end = self.start_character_idx + (i + 1) * self.step
        else:
            end = self.start_character_idx + self.dataset_len
        return end

    def __getitem__(self, idx):
        batch = []
        with self.path.open() as f:
            for i in range(self.batch_size):
                f.seek(
                    self.start_character_idx
                    + i * self.step
                    + self.bptt_len * idx)
                end = None if self.loop else self._get_end_of_segment(i)
                text = self._read(f, self.bptt_len + 1, end)
                if text:
                    batch.append(self.vocab.text2indices(text))
        if not batch:
            raise IndexError(
                f"Index {idx} out of range for batch element {i}. The "
                f"length of dataset is {self.n_examples}.")
        return batch

    def count_examples_in_dataset(self):
        n_chars = 0
        with self.path.open(buffering=65536) as f:
            for line in f:
                n_chars += len(line)
        return np.ceil(n_chars / self.bptt_len)

    def __len__(self):
        return self.n_examples

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        for i, cursor in enumerate(self.cursors):
            end = None if self.loop else self._get_end_of_segment(i)
            text = self._read(cursor, self.bptt_len + 1, end)
            if text:
                batch.append(self.vocab.text2indices(text))
        if not batch:
            raise StopIteration
        return batch

