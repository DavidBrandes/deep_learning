import torch
from torch import nn
from torch.nn import functional as F

from dl.utils import tensor as tensor_utils


class BaseModule(nn.Module):
    def __init__(self, target=None, slice=None, weight=1, append_layer="UNNAMED", debug=False):
        super().__init__()
        
        if target is not None:
            if slice is not None:
                target = target[slice]
            self.register_buffer('_target', target)
        self._slice = slice
        self.register_buffer('_weight', tensor_utils.tensor(weight))
        self._append_layer = append_layer
        self._debug = debug
        
        self.loss = None
        
    def forward(self, x):
        if self._slice:
            x_ = x[self._slice]
        else:
            x_ = x
            
        self.loss = self._weight * self._loss_function(x_)
        
        if self._debug:
            print(f"    Module {self.__class__.__name__} at layer {self._append_layer}:")
            print(f"      Loss {self.loss.item():.4e}")
            print(f"      Input {x.shape} -> Output {x_.shape}")
        
        return x
        

class ContentModule(BaseModule):
    def __init__(self, target=None, slice=None, weight=1, append_layer="UNNAMED", debug=False):
        super().__init__(target, slice, weight, append_layer, debug)
        
    def _loss_function(self, x):
        return F.l1_loss(x, self._target)


class StyleModule(BaseModule):
    def __init__(self, target=None, slice=None, weight=1, append_layer="UNNAMED", debug=False):
        super().__init__(target, slice, weight, append_layer, debug)

    def _gram_matrix(self, x):
        b, c, w, h = x.size()
        features = x.view(b * c, w * h)

        G = torch.mm(features, features.t())

        return G.div(b * c * w * h)
    
    def _loss_function(self, x):
        return F.l1_loss(self._gram_matrix(x), self._gram_matrix(self._target))


class DreamModule(BaseModule):
    def __init__(self, target=None, slice=None, weight=1, append_layer="UNNAMED", debug=False):
        super().__init__(target, slice, weight, append_layer, debug)
        
    def _loss_function(self, x):
        # we want to maximize the loss
        return -torch.mean(x**2)
    
    
class ActivationModule(BaseModule):
    def __init__(self, target=None, slice=None, weight=1, append_layer="UNNAMED", debug=False):
        super().__init__(target, slice, weight, append_layer, debug)
        
    def _loss_function(self, x):
        # Using minus here as we actually want to maximize the activation
        return -torch.mean(x)


class InputModel:
    def __init__(self, model, modules, verbose=True, debug=False):
        self._model = nn.Sequential()
        self._modules = []

        i, last_layer = 0, 0
        for layer_name, layer in model.named_children():
            self._model.add_module(layer_name, layer)

            i += 1

            for Module, append_layer, target, weight, slice in modules:
                if append_layer == layer_name:
                    if target is not None:
                        target = self._model(target).detach()
                        
                    if weight is None:
                        weight = 1

                    module = Module(target, slice, weight, append_layer, debug)
                    module_name = type(module).__name__

                    self._model.add_module(f'{append_layer}_append_{module_name}_{i + 1}', module)
                    if verbose or debug:
                        print(f"Added {module_name} module to layer {append_layer}")
                    self._modules.append(module)

                    i += 1
                    last_layer = i

        self._model = self._model[:last_layer + 1]

    def to(self, device):
        self._model.to(device)
        
        return self

    def __call__(self, x):
        self._model(x)

        x = tensor_utils.tensor([0], device=x.device, dtype=x.dtype, requires_grad=True)

        for module in self._modules:
            x = x + module.loss

        return x
