import torch


DTYPE = torch.float32


def tensor(x, device=None, dtype=DTYPE):
    return torch.tensor(x, dtype=dtype, device=device)