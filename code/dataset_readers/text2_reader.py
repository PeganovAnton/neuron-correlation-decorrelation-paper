import torch
import torchtext
from pathlib import Path
from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader

class Text2Reader(torchtext.data.Dataset):

    def __init__(self, data_path: str, split, preproc):
        self.data_path = Path(data_path)
        
        wikitext2_dataset = WikiText2(data_path)
        
        return wikitext2_dataset
