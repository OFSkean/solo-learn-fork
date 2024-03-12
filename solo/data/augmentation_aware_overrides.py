from torchvision import transforms
from torchvision.transforms import functional as F
from typing import Callable, List, Optional, Sequence, Type, Union
import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from solo.data.pretrain_dataloader import NCropAugmentation, GaussianBlur, Equalization, Solarization
from itertools import chain

# MAKE get_params return the randomly sampled parameters
class verbose_get_params():
    def __init__(self, augmentation_name, original_get_params, normalization_function=None):
        self.augmentation_name = augmentation_name
        self.original_get_params = original_get_params
        self.normalization_function = normalization_function
        self.last_params = None

    def __call__(self, *args, **kwargs):
        real_params = self.original_get_params(*args, **kwargs)
        if self.normalization_function is not None:
            normalized_params = self.normalization_function(real_params, args)
        else:
            normalized_params = real_params

        self.last_params = normalized_params
        return real_params
    
    def clear_last_params(self):
        self.last_params = None
    
# OVERRIDE TRANSFORMS.__CALL__
def augaware_compose_transforms_call(self, img):
    params_dict = {
        'RandomResizedCrop': [0, 0, 0, 0],  # i, j, h, w
        'ColorJitter': [0, 0, 0, 0],        # brightness, contrast, saturation, hue
        'GaussianBlur': [0],                # sigma
        'Equalization': [0],                # true=1 / false=0
        'Solarization': [0],                # true=1 / false=0
        'RandomGrayScale':    [0],          # true=1 / false=0
        'RandomHorizontalFlip': [0],        # true=1 / false=0
    }

    def handle_get_params(base):
        params = base.last_params
        aug_name = base.augmentation_name
        base.clear_last_params()

        if params is not None:
            params_dict[aug_name] = params

    for t in self.transforms:
        img = t(img)

        # handle functions that have get_params method
        if hasattr(t, 'get_params') and hasattr(t.get_params, 'last_params'):
            handle_get_params(t.get_params)


        # check if transform is random apply and check out the actual transform
        elif isinstance(t, transforms.RandomApply) and \
            hasattr(t.transforms[0], 'get_params') and hasattr(t.transforms[0].get_params, 'last_params'):

            handle_get_params(t.transforms[0].get_params)

        # GrayScale special case
        elif isinstance(t, transforms.RandomGrayscale) and \
            hasattr(F, 'rgb_to_grayscale') and hasattr(F.rgb_to_grayscale, 'last_params'):

            handle_get_params(F.rgb_to_grayscale)
            
        # HorizontalFlip special case
        elif isinstance(t, transforms.RandomHorizontalFlip) and \
            hasattr(F, 'hflip') and hasattr(F.hflip, 'last_params'):

            handle_get_params(F.hflip)

            

    params_dict_values = list(chain(*params_dict.values()))
    return (img, params_dict_values)


transforms.Compose.__call__ = augaware_compose_transforms_call

transforms.RandomResizedCrop.get_params = verbose_get_params(
    'RandomResizedCrop',
    transforms.RandomResizedCrop.get_params,
    lambda p, args: (p[0]/args[0].size[1], 
                     p[1]/args[0].size[0], 
                     p[2]/args[0].size[1], 
                     p[3]/args[0].size[0])     # normalize params i,j,h,w wrt image size
)

transforms.ColorJitter.get_params = verbose_get_params(
    'ColorJitter',
    transforms.ColorJitter.get_params,
    lambda p, args: (p[1], p[2], p[3], p[4])  #brightness, contrast, saturation, hue
)

GaussianBlur.get_params = verbose_get_params(
    'GaussianBlur',
    GaussianBlur.get_params,
    lambda p, args: [p]               # sigma
)

Equalization.get_params = verbose_get_params(
    'Equalization',
    Equalization.get_params,
    lambda p, args: [1.]               # true=1
)

Solarization.get_params = verbose_get_params(
    'Solarization',
    Solarization.get_params,
    lambda p, args: [1.]               # true=1
)

# For RandomGrayScale
F.rgb_to_grayscale = verbose_get_params(
    'RandomGrayScale',
    transforms.functional.rgb_to_grayscale,
    lambda p, args: [1.]               # true=1
)

# For RandomHorizontalFlip
F.hflip = verbose_get_params(
    'RandomHorizontalFlip',
    transforms.functional.hflip,
    lambda p, args: [1.]               # true=1
)