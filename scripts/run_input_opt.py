from torch.optim import Adam
import torch

from dl.utils import image as image_utils, tensor as tensor_utils
from dl.model.convolutional.pretrained import get_vgg19_model, get_googlenet_model
from dl.model.input.input import StyleModule, DreamModule, ContentModule, InputModel, ActivationModule
from dl.transform.image import RandomTransformation
from dl.parameter.image import FourierParameterization, UnitClipping
from dl.optimization.input import InputOptimizer

tensor_utils.DTYPE = torch.float64


style_img_path = "/Users/david/Downloads/style.jpeg"
content_img_path = "/Users/david/Downloads/content.jpeg"
input_img_path = "/Users/david/Downloads/content.jpeg"
output_img_path = "/Users/david/Downloads/output.png"

EPOCHS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPTIMIZER_KWARGS = {"lr": 0.05}
MAX_SHAPE = 256

style_img_path = "/Users/david/Downloads/style.jpeg"
content_img_path = "/Users/david/Downloads/content.jpeg"
output_img_path = "/Users/david/Downloads/output.png"

style_img = image_utils.load_image(style_img_path)
content_img = image_utils.load_image(content_img_path)

(style_img, content_img), shape = image_utils.max_image_crops([style_img, content_img], max_size=MAX_SHAPE)
# input_img = image_utils.random_image(shape)
input_img = content_img

def callback(epoch, loss, img):
    image_utils.save_image(output_img_path, img)


# module, append layer name, target, weight, slice
modules = [(DreamModule, "inception5a", None, None, (0, 21))]
# modules = [(DreamModule, "inception4b", None, None, None)]
# modules = [(ContentModule, 'relu3_2', content_img, 1, None),
#             (StyleModule, 'relu1_1', style_img, 2000, None),
#             (StyleModule, 'relu2_1', style_img, 2000, None),
#             (StyleModule, 'relu3_1', style_img, 2000, None),
#             (StyleModule, 'relu4_1', style_img, 2000, None),
#             (StyleModule, 'relu5_1', style_img, 2000, None),]

# vgg19 = get_vgg19_model()
googlenet = get_googlenet_model()
clipping = UnitClipping()
transformation = RandomTransformation()
parameterization = FourierParameterization()

model = InputModel(googlenet, modules)
optimizer = InputOptimizer(model, Adam, optimizer_kwargs=OPTIMIZER_KWARGS, 
                      parameterization=parameterization, transformation=transformation,
                      clipping=None, epochs=EPOCHS, callback=callback, device=DEVICE, 
                      leave=False)

output_img = optimizer(input_img)

image_utils.save_image(output_img_path, output_img)
image_utils.show_image(output_img)
