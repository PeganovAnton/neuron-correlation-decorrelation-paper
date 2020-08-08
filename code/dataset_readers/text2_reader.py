import torch
from pathlib import Path
from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader

class Text2Reader():

    def read(self, data_path: str):
        data_path = Path(data_path)
        
        wikitext2_dataset = WikiText2(data_path)
        
        return wikitext2_dataset
