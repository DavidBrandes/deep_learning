from torch import nn
from torchvision.models import vgg19, VGG19_Weights, googlenet, GoogLeNet_Weights

from dl.model.generic import ImageNetNormalize


def get_vgg19_model():
    # collect a alterd version of vgg19 in evaluation mode that has its layers renamed
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
    
    for parameter in renamed.parameters():
        parameter.requires_grad = False

    return renamed


def get_googlenet_model():
    # note that this model wont actaually work till the end, as in the recreation process
    # below we gte an issue with the final dropout layer for some reason. However as we are only
    # interested in its parts before that we leave it at this for now
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT).eval()
    
    adjusted = nn.Sequential()
    adjusted.add_module("normalize", ImageNetNormalize())

    for name, layer in model.named_children():
        adjusted.add_module(name, layer)  
        
    for parameter in adjusted.parameters():
        parameter.requires_grad = False

    return adjusted
