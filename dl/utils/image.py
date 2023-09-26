import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from dl.utils import tensor as tensor_utils, generic as generic_utils


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
    shape = generic_utils.convert_to_shape(shape)

    rng = np.random.default_rng(seed)

    img = rng.normal(0.5, 0.01, shape + (3,))
    img = np.clip(img, 0, 1)
    img = to_tensor(img)

    return img


def show_image(img, title=None, figsize=(7, 7)):
    img = from_tensor(img)

    plt.figure(figsize=figsize)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.imshow(img)
    plt.show()


def max_image_crops(images, max_size=np.inf, best_ratio=True):
    """Crop images to have the same shape.
    
    Crop some images to have the same shape as good as possible. For best_ratio = False
    that is simply the maximum crop contained in all images while for best_ratio = True we
    try to maintain the image ratios as well as possible, possible shrinking the resulting
    image in that process. The crops are always taken from the center of the images.
    
    Parameters
    ----------
    images : List(tensors of shape (1, 3, n, m))
        The images to be cropped.
    max_size : int, optional
        The crops shape bwill e constrained by this value. The default is inf.
    best_ratio : TYPE, optional
        See the description above. The default is True.

    Returns
    -------
    List(tensors of shape (1, 3, n_, m_))
        The list of cropped images, all having the same shape.
    Tuple
        The shape (n_, m_) of the crops.

    """
    if not len(images):
        return [], (0, 0)
    
    ratio = (np.inf, np.inf)
    size = (max_size, max_size)
    
    for image in images:
        w, h = image.shape[-2:]
        
        ratio = (min(ratio[0], max(1, w / h)), min(ratio[1], max(1, h / w))) 
        size = (min(size[0], w), min(size[1], h))
        
    if best_ratio:
        min_side = min(size)
        
        size = (int(min_side * ratio[0]), int(min_side * ratio[1]))
        
        if max(size) > max_size:
            max_ratio = max(ratio)
            
            ratio = (ratio[0] / max_ratio, ratio[1] / max_ratio)  
            size = (int(max_size * ratio[0]), int(max_size * ratio[1]))
        
    image_crops = []
                
    for image in images:
        w, h = image.shape[-2:]
        
        crop_ratio = min(w / size[0], h / size[1])
        crop_size = (int(size[0] * crop_ratio), int(size[1] * crop_ratio))
        
        transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                        transforms.Resize(size)])
        
        image = transform(image)
        
        image_crops.append(image)
        
    return image_crops, size
