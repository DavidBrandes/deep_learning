from torch import nn
from torchvision.models import vgg19, VGG19_Weights

from dl.model.generic import ImageNetNormalize


def get_model():
    model = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

    block, number = 1, 1
    
    renamed = nn.Sequential()
    renamed.add_module("normalize", ImageNetNormalize())

    for layer in model.children():
        if isinstance(layer, nn.Conv2d):
            name = f'conv{block}_{number}'

        elif isinstance(layer, nn.ReLU):
            name = f'relu{block}_{number}'
            layer = nn.ReLU(inplace=False)
            number += 1

        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{block}'
            layer = nn.AvgPool2d(layer.kernel_size, layer.stride)
            block += 1
            number = 1

        else:
            raise RuntimeError(f'Unrecognized layer "{layer.__class__.__name__}""') 

        renamed.add_module(name, layer)

    return renamed
