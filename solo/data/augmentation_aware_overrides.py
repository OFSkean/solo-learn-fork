from torchvision import transforms
from torchvision.transforms import functional as F
from typing import Callable, List, Optional, Sequence, Type, Union
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFilter, ImageOps
from solo.data.pretrain_dataloader import NCropAugmentation, GaussianBlur, Equalization, Solarization
from itertools import chain
import pandas as pd

augmentation_error_names = [
    'rcc_i', 'rcc_j', 'rcc_h', 'rcc_w',
    'cj_brightness', 'cj_contrast', 'cj_saturation', 'cj_hue',
    'gb_sigma',
    'equalization',
    'solarization',
    'random_gray_scale',
    'random_horizontal_flip'
]

def adjustable_augmentation_dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.
    Allows augmentations to be adjusted per sample

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class AdjustableAugmentationDatasetWithIndex(DatasetClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_transforms = self.transform # change to initial_transforms to avoid conflicts in DatasetClass
            self.transform = None
            self.df = pd.read_csv("stl10_preprocessed_info.csv")

            self.__create_augmentation_error_df_columns()

        def __getitem__(self, index):
            data, target = super().__getitem__(index)
            data_row = self.df.iloc[index]
            data = self.initial_transforms(data, data_row=data_row)
            return (index, data, target)

        def __create_augmentation_error_df_columns(self):
            for name in augmentation_error_names:
                self.df[name] = 0.0

        def update_augmentation_parameters(self, batch_indices, augmentation_error_values, should_adjust=False):

            def update_errors(row):
                val_loc = batch_indices.index(row.name)
                row[augmentation_error_names] = augmentation_error_values[val_loc]
                return row
            
            def update_errors_and_vals(row, error_mean, error_std):
                val_loc = batch_indices.index(row.name)
                row[augmentation_error_names] = augmentation_error_values[val_loc]

                values = augmentation_error_values[val_loc]
                rcc_i_name = augmentation_error_names[0]
                rcc_j_name = augmentation_error_names[1]
                if any([abs(values[0] - error_mean[rcc_i_name]) > 2 * error_std[rcc_i_name],
                        abs(values[1] - error_mean[rcc_j_name]) > 2 * error_std[rcc_j_name]]):

                    if values[0] <= error_mean[rcc_i_name] and values[1] <= error_mean[rcc_j_name]:
                        # make harder
                        row['rrc_area_lower_bound'] = max(0.08, -0.01 + row['rrc_area_lower_bound'])
                    elif values[0] >= error_mean[rcc_i_name] and values[1] >= error_mean[rcc_j_name]:
                        # make easier
                        row['rrc_area_lower_bound'] = min(0.5, 0.01 + row['rrc_area_lower_bound'])

                return row


            if should_adjust:
                error_mean, error_std = self.get_augmentation_error_stats()

                self.df.loc[batch_indices] = self.df.loc[batch_indices].apply(update_errors_and_vals, axis=1, args=(error_mean, error_std))
            else:
                self.df.loc[batch_indices] = self.df.loc[batch_indices].apply(update_errors, axis=1)



        # get means and stds of the dataset errors
        # this can probably be sped up with an online batch-wise algorithm like Knuth or Welford
        def get_augmentation_error_stats(self):
            return self.df[augmentation_error_names].mean(), self.df[augmentation_error_names].std()

    return AdjustableAugmentationDatasetWithIndex

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
def augaware_compose_transforms_call(self, img, **kwargs):
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
        if kwargs != {}:
            if isinstance(t, torchvision.transforms.RandomResizedCrop):
                t.scale = (kwargs['data_row']["rrc_area_lower_bound"], 1.0)
            elif isinstance(t, torchvision.transforms.RandomGrayscale):
                t.p = 1. if kwargs['data_row']["is_grey_scale"] else t.p

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