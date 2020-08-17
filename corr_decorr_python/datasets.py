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
            number_of_characters_in_dataset=None,
            loop=False
    ):
        self.read_count = 0
        self.path = Path(path).expanduser()
        self.bptt_len = bptt_len
        self.start_character_idx = start_character_idx
        self.number_of_characters_in_dataset = number_of_characters_in_dataset
        self.number_of_characters_in_file = self.count_characters_in_file()
        self.vocab = vocab
        self.batch_size = batch_size
        if number_of_characters_in_dataset is None:
            self.number_of_characters_in_dataset = \
                self.number_of_characters_in_file - self.start_character_idx
        else:
            self.number_of_characters_in_dataset = \
                number_of_characters_in_dataset
        self.n_examples = (self.number_of_characters_in_dataset - 1) \
            // (self.batch_size * self.bptt_len)
        self.loop = loop
        if self.n_examples < 1:
            raise ValueError("The size of text is too small")
        self.cursors = []
        self.segment_size = self.n_examples * self.bptt_len
        self._reset_cursors()

    def _reset_cursors(self):
        if self.cursors:
            for c in self.cursors:
                c.close()
        self.cursors = []
        for i in range(self.batch_size):
            self.cursors.append(self.path.open(buffering=65536))
            self.cursors[i].seek(
                self.start_character_idx + i * self.segment_size)

    def _read(self, file_object, length, end=None):
        if end is None:
            end = self.start_character_idx \
                  + self.number_of_characters_in_dataset
        if self.loop:
            text = ''
            while len(text) < length:
                n_remained_characters_in_ds = end - file_object.tell()
                if n_remained_characters_in_ds > 0:
                    text += file_object.read(
                        min(length - len(text), n_remained_characters_in_ds))
                if len(text) < length:
                    file_object.seek(self.start_character_idx)
        else:
            n_remained_characters_in_ds = end - file_object.tell()
            if n_remained_characters_in_ds > 0:
                text = file_object.read(
                    min(length, n_remained_characters_in_ds))
            else:
                text = ''
            self.read_count += 1
        return text

    def _get_end_of_segment(self, i):
        return self.start_character_idx + (i + 1) * self.segment_size

    def _get_inps_lbls(self, text_file, segment_idx):
        end = None if self.loop else self._get_end_of_segment(segment_idx)
        text = self._read(text_file, self.bptt_len + 1, end)
        if len(text) == self.bptt_len + 1:
            cur_pos = text_file.tell()
            if cur_pos != 0:
                text_file.seek(cur_pos)
            else:
                text_file.seek(0, 2)
                text_file.seek(text_file.tell()-1)
        if len(text) > 1:
            indices = self.vocab.text2indices(text)
            return indices[:-1], indices[1:]

    def __getitem__(self, idx):
        if idx > self.n_examples:
            raise IndexError(
                f"Index {idx} is out of range. Dataset has only "
                f"{self.n_examples}. Note that dataset batch size is "
                f"{self.batch_size} and dataset back propagation through time "
                f"length is {self.bptt_len}."
            )
        inputs = []
        labels = []
        with self.path.open() as f:
            for i in range(self.batch_size):
                f.seek(
                    self.start_character_idx
                    + i * self.segment_size
                    + self.bptt_len * idx)
                inps, lbls = self._get_inps_lbls(f, i)
                inputs.append(inps)
                labels.append(lbls)
        return inputs, labels

    def count_characters_in_file(self):
        n_chars = 0
        with self.path.open(buffering=65536) as f:
            for line in f:
                n_chars += len(line)
        return n_chars

    def __len__(self):
        return self.n_examples

    def __iter__(self):
        self._reset_cursors()
        return self

    def __next__(self):
        inputs, labels = [], []
        for i, cursor in enumerate(self.cursors):
            inps_lbls = self._get_inps_lbls(cursor, i)
            if inps_lbls is None:
                raise StopIteration
            inps, lbls = inps_lbls
            if len(inps) != self.bptt_len or len(lbls) != self.bptt_len:
                raise StopIteration
            inputs.append(inps)
            labels.append(lbls)
        assert len(inputs) == len(labels)
        inputs, labels = np.array(inputs), np.array(labels)
        return inputs, labels

    def __del__(self):
        for c in self.cursors:
            c.close()
