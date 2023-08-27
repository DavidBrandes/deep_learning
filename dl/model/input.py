import torch
from torch import nn
from torch.nn import functional as F


class Content(nn.Module):
    def __init__(self, target, weight, slice):
        super().__init__()
        
        self._slice = slice
        if slice is not None:
            target = target[slice]

        self.register_buffer('_target', target)
        self.register_buffer('_weight', torch.tensor(weight))

        self.loss = None

    def forward(self, x):
        if self._slice:
            x_ = x[self._slice]
        else:
            x_ = x
            
        self.loss = self._weight * F.l1_loss(x_, self._target)

        return x


class Style(nn.Module):
    def __init__(self, target, weight, slice):
        super().__init__()
        
        self._slice = slice
        if slice is not None:
            target = target[slice]

        self.register_buffer('_target', self._gram_matrix(target))
        self.register_buffer('_weight', torch.tensor(weight))

        self.loss = None

    def _gram_matrix(self, x):
        n, c, w, h = x.size()
        features = x.view(n * c, w * h)

        G = torch.mm(features, features.t())

        return G.div(n * c * w * h)

    def forward(self, x):
        if self._slice:
            x_ = x[self._slice]
        else:
            x_ = x
           
        self.loss = self._weight * F.l1_loss(self._gram_matrix(x_), self._target)

        return x


class Dream(nn.Module):
    def __init__(self, target, weight, slice):
        super().__init__()

        self._slice = slice
        self.register_buffer('_weight', torch.tensor(weight))

        self.loss = None

    def forward(self, x):
        if self._slice:
            x_ = x[self._slice]
        else:
            x_ = x

        self.loss = self._weight * torch.mean(x_**2)

        return x


class Model:
    def __init__(self, model, modules):
        self._model = nn.Sequential()
        self._modules = []

        i, last_layer = 0, 0
        for layer_name, layer in model.named_children():
            self._model.add_module(layer_name, layer)

            i += 1

            for module, append_layer, target, weight, slice in modules:
                if append_layer == layer_name:
                    if target is not None:
                        target = self._model(target).detach()
                    if weight is None:
                        weight = 1

                    module = module(target, weight, slice)

                    self._model.add_module(f'{append_layer}_append_{i+1}', module)
                    print(f"Added {type(module).__name__} module to layer {append_layer}")
                    self._modules.append(module)

                    i += 1
                    last_layer = i

        self._model = self._model[:last_layer + 1]

    def to(self, device):
        self._model.to(device)
        
        return self

    def __call__(self, x):
        self._model(x)

        x = torch.tensor([0], dtype=x.dtype, device=x.device)

        for module in self._modules:
            x += module.loss

        return x
