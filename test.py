import torch
import numpy as np
np.random.seed(1001)

from dl.parameter.image import FourierParameterization
from dl.utils import tensor as tensor_utils



x = np.abs(np.random.randn(1, 3, 7, 8))
x = np.clip(x, 1e-4, 1 - 1e-4)

x_ = tensor_utils.tensor(x)
f = FourierParameterization()
y_ = f.parameterize(x_)
xx_ = f(y_)
xx = xx_.numpy()

print(x.shape, xx.shape)
print(np.max(np.abs(x - xx)))