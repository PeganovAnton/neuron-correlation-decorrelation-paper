import numpy as np
import torch


def one_hot(x, depth, device):
    if isinstance(device, str):
        device = torch.device(device)
    t = torch.zeros(x.shape + (depth,), dtype=torch.float32, device=device)
    index = torch.as_tensor(np.reshape(x, x.shape + (1,)))
    t.scatter_(-1, index, value=1.)
    return t
