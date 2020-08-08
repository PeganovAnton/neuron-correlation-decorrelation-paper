import torch
from pathlib import Path
from torchtext import data.datasets.WikiText2

class Text2Reader():

    def read(self, data_path: str):
        data_path = Path(data_path)

