from dl.utils import image as image_utils
from dl.transform.image import RandomTransformation

import torch
    
    
img_path = "/Users/david/Desktop/image0.jpeg"

img1 = image_utils.load_image(img_path)
t = RandomTransformation(sigmoid=False, randomness=2)
img2 = t(img1)
print(img1.shape, img2.shape)
print(torch.mean(torch.abs(img1 - img2)))
image_utils.show_image(img2)
