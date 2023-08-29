import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dl.utils import tensor as tensor_utils


def to_tensor(img):
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = tensor_utils.tensor(img)

    return img


def from_tensor(img):
    img = img.numpy()
    img = np.clip(img, 0, 1)
    img = img.squeeze(0)
    img = np.transpose(img, (1, 2, 0))
    
    return img


def load_image(path, transform=None):
    img = Image.open(path).convert("RGB")

    img = np.array(img) / 255
    img = to_tensor(img)
    if transform:
        img = transform(img)

    return img


def save_image(path, img):
    img = from_tensor(img)
    img = (img * 255).astype(np.uint8)

    img = Image.fromarray(img)
    img.save(path)


def random_image(shape, seed=1001):
    if type(shape) is int:
        shape = (shape, shape)
    elif type(shape) is tuple and len(shape) == 1:
        shape = shape + shape

    rng = np.random.default_rng(seed)

    img = rng.normal(0.5, 0.01, shape + (3,))
    img = np.clip(img, 0, 1)
    img = to_tensor(img)

    return img


def show_image(img):
    img = from_tensor(img)
    
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    plt.axis("off")
    plt.show()