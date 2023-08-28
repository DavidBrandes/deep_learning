import kornia
import numpy as np
import torch

from dl.utils import tensor as tensor_utils


class RandomTransformation:
    def __init__(self, randomness=1, seed=1001, clip=True, sigmoid=True):
        self._clip = clip
        self._sigmoid = sigmoid
        
        self._randomness = randomness
        self._rng = np.random.default_rng(seed)
    
    def _get_random_values(self):
        translation = self._rng.uniform(-3 * self._randomness, 3 * self._randomness, (1, 2))
        scale = self._rng.uniform(1 - 0.01 * self._randomness, 1 + 0.01 * self._randomness, (1, 2))
        angle = self._rng.uniform(-1 * self._randomness, 1 * self._randomness, (1,))
        
        return tensor_utils.tensor(translation), tensor_utils.tensor(scale), tensor_utils.tensor(angle)
            
    def __call__(self, x):
        translation, scale, angle = self._get_random_values()
        
        x = kornia.geometry.transform.translate(x, translation, padding_mode="reflection")
        x = kornia.geometry.transform.scale(x, scale, padding_mode="reflection")
        x = kornia.geometry.transform.rotate(x, angle, padding_mode="reflection")
        
        if self._sigmoid:
            x = torch.sigmoid(x)
        
        elif self._clip:
            # non inplace clipping produces bad results
            x.data.clamp_(0, 1)

        return x
