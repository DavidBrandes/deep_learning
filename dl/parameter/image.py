import torch
import numpy as np

from dl.utils import tensor as tensor_utils

# Taken from https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param/color.py
color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]])
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

# x_correlated = x_incorrelated @ M
M = color_correlation_normalized.T
M_inv = np.linalg.inv(color_correlation_normalized).T


class FourierParameterization:
    def __init__(self, fft=True, whiten=True):
        self._fft = fft
        self._whiten = whiten
        
        self._s = None

    def parameterize(self, x):
        if self._whiten:
            n, c, w, h = x.shape
            x = torch.transpose(x, 0, 1).view(3, -1).t()
            torch.matmul(x, tensor_utils.tensor(M_inv, device=x.device))
            x = torch.transpose(x.t().view((3, n, w, h)), 0, 1)
            
        if self._fft:
            self._s = x.shape[-2:]
            x = torch.fft.rfft2(x)
            
        return x

    def __call__(self, x):
        if self._fft:
            x = torch.fft.irfft2(x, s=self._s)
            
        if self._whiten:
            n, c, w, h = x.shape
            x = torch.transpose(x, 0, 1).view(3, -1).t()
            torch.matmul(x, tensor_utils.tensor(M, device=x.device))
            x = torch.transpose(x.t().view((3, n, w, h)), 0, 1)
            
        return x
    
    
class Normalization:
    def __init__(self, sigmoid=True, clip=True):
        self._sigmoid = sigmoid
        self._clip = clip

        self._eps = 1e-6

    def parameterize(self, x):
        if self._sigmoid:
            # clipping to avoid nans
            x = torch.clamp(x, self._eps, 1 - self._eps)
            x = torch.logit(x)
        elif self._clip:
            x = torch.clip(x, 0, 1)

        return x

    def __call__(self, x):
        if self._sigmoid:
            x = torch.sigmoid(x)
        elif self._clip:
            # non inplace clipping produces bad results
            x.data.clamp_(0, 1)

        return x
