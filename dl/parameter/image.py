import torch
import numpy as np

from dl.utils import tensor as tensor_utils

# Taken from https://github.com/tensorflow/lucid/blob/master/lucid/optvis/param

color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]])
max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
color_correlation_normalized = color_correlation_svd_sqrt / max_norm_svd_sqrt

# x_correlated = x_uncorrelated @ M
M = color_correlation_normalized.T
M_inv = np.linalg.inv(color_correlation_normalized).T


class FourierParameterization:
    def __init__(self):
        self._clamping_eps = 1e-6
        
        self._scale = None
        self._size = None
        
    def _set_params(self, x):
        (w, h) = x.shape[2:]
        
        fy = np.fft.fftfreq(w)[:, None]
        fx = np.fft.fftfreq(h)[: h // 2 + 1]
        f = np.sqrt(fx * fx + fy * fy)
        
        scale = 1.0 / np.maximum(f, 1.0 / max(w, h))
        scale *= np.sqrt(w * h)

        self._scale = scale
        self._size = (w, h)
    
    def parameterize(self, x):
        self._set_params(x)
        
        # normalize
        # clamping first to avoid infs
        x = torch.clamp(x, self._clamping_eps, 1 - self._clamping_eps)
        x = torch.logit(x)
                
        # decorelate colors
        b, c, w, h = x.shape
        x = torch.transpose(x, 0, 1).view(3, -1).t()
        x = torch.matmul(x, tensor_utils.tensor(M_inv, dtype=x.dtype, device=x.device))
        x = torch.transpose(x.t().view((c, b, w, h)), 0, 1)
        
        # into fourier space
        x = 4 * x
        x = torch.fft.rfft2(x)
        x = x / tensor_utils.tensor(self._scale, dtype=x.dtype, device=x.device)
        
        return x

    def __call__(self, x):
        # from fourier space
        x = x * tensor_utils.tensor(self._scale, dtype=x.dtype, device=x.device)
        x = torch.fft.irfft2(x, s=self._size)
        x = x / 4
        
        # into correlated color space
        b, c, w, h = x.shape
        x = torch.transpose(x, 0, 1).view(3, -1).t()
        x = torch.matmul(x, tensor_utils.tensor(M, dtype=x.dtype, device=x.device))
        x = torch.transpose(x.t().view((c, b, w, h)), 0, 1)
        
        # denormalize
        x = torch.sigmoid(x)
            
        return x
    
    
class UnitClipping:
    def __call__(self, x):
        # non inplace clipping produces bad results
        x.data.clamp_(0, 1)

        return x
