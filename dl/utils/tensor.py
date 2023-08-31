import torch


DTYPE = torch.float32


def tensor(x, device=None, dtype=DTYPE, requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)