import kornia


class RandomTransformation:
    def __init__(self, randomness=1, clip=True):
        self._randomness = randomness
        self._clip = clip

    def __call__(self, x):
        transform = kornia.augmentation.container.AugmentationSequential(
            kornia.augmentation.RandomResizedCrop(
                size=x.shape[-2:], 
                scale=(1 - self._randomness * 0.025, 1), 
                ratio=(1 - self._randomness * 0.025, 1 + self._randomness * 0.025)),
            kornia.augmentation.RandomRotation(
                degrees=1 * self._randomness))

        x = transform(x)
        
        if self._clip:
            # non inplace clipping produces bad results
            x.data.clamp_(0, 1)

        return x
