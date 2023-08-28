from torch import nn

from dl.utils import tensor as tensor_utils



class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
    
        mean = tensor_utils.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor_utils.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    
        self.register_buffer('_mean', mean)
        self.register_buffer('_std', std)

    def forward(self, x):
        return (x - self._mean) / self._std