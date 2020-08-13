from pathlib import Path

from torch.utils.data import IterableDataset


class CharLanguageModellingDataset(IterableDataset):
    def __init__(self, vocab, path, start_character, bptt_len, loop):
        self.vocab = vocab
        self.path = Path(path).expanduser()
        self.start_character = start_character
        self.bptt_len = bptt_len
        self.loop = loop
        self.text_file = self.path.open()

    def __getitem__(self, idx):
        self.text_file.seek(self.start_character + self.bptt_len * idx)
        if self.loop:
            text = ''
            while len(text) < self.bptt_len + 1:
                text += self.text_file.read(self.bptt_len + 1 - len(text))
                if len(text) < self.bptt_len + 1:
                    self.text_file.seek(self.start_character)
        else:
            text = self.text_file.read(self.bptt_len + 1)
