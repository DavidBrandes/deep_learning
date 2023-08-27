import torch


class FourierParameterization:
    def __init__(self, sigmoid=True, clip=True):
        self._sigmoid = sigmoid
        self._clip = clip

        self._eps = 1e-6

    def parameterize(self, x):
        if self._sigmoid:
            # clipping to avoid nans
            x = torch.clamp(x, self._eps, 1 - self._eps)
            x = torch.logit(x)

        return x

    def __call__(self, x):
        if self._sigmoid:
            x = torch.sigmoid(x)
        elif self._clip:
            # non inplace clipping produces bad results
            x.data.clamp_(0, 1)

        return x
    
    
class ImageParameterization:
    def __init__(self, clip=True):
        self._clip = clip

    def parameterize(self, x):
        return x

    def __call__(self, x):
        if self._clip:
            # non inplace clipping produces bad results
            x.data.clamp_(0, 1)

        return x