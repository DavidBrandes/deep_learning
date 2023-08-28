import kornia
import numpy as np

from dl.utils import tensor as tensor_utils


class RandomTransformation:
    def __init__(self, randomness=1, seed=1001):
        self._randomness = randomness
        self._rng = np.random.default_rng(seed)
    
    def _get_random_values(self, device):
        translation = self._rng.uniform(-3 * self._randomness, 3 * self._randomness, (1, 2))
        scale = self._rng.uniform(1 - 0.01 * self._randomness, 1 + 0.01 * self._randomness, (1, 2))
        angle = self._rng.uniform(-1 * self._randomness, 1 * self._randomness, (1,))

        translation = tensor_utils.tensor(translation, device=device)
        scale = tensor_utils.tensor(scale, device=device)
        angle = tensor_utils.tensor(angle, device=device)
        
        return translation, scale, angle
            
    def __call__(self, x):
        translation, scale, angle = self._get_random_values(x.device)
        
        x = kornia.geometry.transform.translate(x, translation, padding_mode="reflection")
        x = kornia.geometry.transform.scale(x, scale, padding_mode="reflection")
        x = kornia.geometry.transform.rotate(x, angle, padding_mode="reflection")

        return x
