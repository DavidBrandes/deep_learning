import kornia
import numpy as np

from dl.utils import tensor as tensor_utils


class RandomTransformation:
    def __init__(self, magnitude=1, seed=1001):
        self._magnitude = magnitude
        self._rng = np.random.default_rng(seed)
    
    def _get_random_values(self, x):
        translation_range = 4 * min(x.shape[-2:]) // 256
        translation = self._rng.uniform(-translation_range * self._magnitude, 
                                        translation_range * self._magnitude, (1, 2))
        scale = self._rng.uniform(1 - 0.02 * self._magnitude, 
                                  1 + 0.02 * self._magnitude, (1, 2))
        angle = self._rng.uniform(-2 * self._magnitude, 
                                  2 * self._magnitude, (1,))

        translation = tensor_utils.tensor(translation, device=x.device)
        scale = tensor_utils.tensor(scale, device=x.device)
        angle = tensor_utils.tensor(angle, device=x.device)
        
        return translation, scale, angle
            
    def __call__(self, x):
        translation, scale, angle = self._get_random_values(x)
        
        x = kornia.geometry.transform.translate(x, translation, padding_mode="reflection")
        x = kornia.geometry.transform.scale(x, scale, padding_mode="reflection")
        x = kornia.geometry.transform.rotate(x, angle, padding_mode="reflection")

        return x
