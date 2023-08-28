from torch.optim import Adam
from torchvision import transforms
import torch

from dl.utils import image as image_utils
from dl.model.vgg import get_model
from dl.model.input import Style, Dream, Content, Model, Activation
from dl.transform.image import RandomTransformation
from dl.parameter.image import FourierParameterization, Clipping
from dl.optimization.input import Optimizer


style_img_path = "/Users/david/Downloads/style.jpeg"
content_img_path = "/Users/david/Downloads/content.jpeg"
input_img_path = "/Users/david/Downloads/content.jpeg"
output_img_path = "/Users/david/Downloads/output.png"

EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMIZER_KWARGS = {"lr": 0.05}
SHAPE = (101, 151)

transform = transforms.Resize(SHAPE)

style_img = image_utils.load_image(style_img_path, transform=transform)
content_img = image_utils.load_image(content_img_path, transform=transform)
input_img = image_utils.load_image(input_img_path, transform=transform)
# input_img = image_utils.random_image(SHAPE)


def callback(epoch, loss, img):
    image_utils.save_image(output_img_path, img)


# module, append layer name, target, weight, slice
# modules = [(Dream, "relu5_1", None, -1, None)]
modules = [(Content, 'relu3_2', content_img, 1, None),
           (Style, 'relu1_1', style_img, 2000, None),
           (Style, 'relu2_1', style_img, 2000, None),
           (Style, 'relu3_1', style_img, 2000, None),
           (Style, 'relu4_1', style_img, 2000, None),
           (Style, 'relu5_1', style_img, 2000, None),]

vgg = get_model()
clipping = Clipping()
transformation = RandomTransformation()
parameterization = FourierParameterization()

model = Model(vgg, modules)
optimizer = Optimizer(model, Adam, optimizer_kwargs=OPTIMIZER_KWARGS, 
                      parameterization=parameterization, transformation=transformation,
                      clipping=clipping, epochs=EPOCHS, callback=callback, device=DEVICE, 
                      leave=False)

output_img = optimizer(input_img)

image_utils.save_image(output_img_path, output_img)
image_utils.show_image(output_img)
