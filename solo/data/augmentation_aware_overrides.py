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
import copy
import numpy as np

augmentation_error_names = [f'{x}_error' for x in [
    'rcc_i', 'rcc_j', 'rcc_h', 'rcc_w',
    'cj_brightness', 'cj_contrast', 'cj_saturation', 'cj_hue',
    'gb_sigma',
    'equalization',
    'solarization',
    'random_gray_scale',
    'random_horizontal_flip'
]]

def adjustable_augmentation_dataset_with_index(cfg, DatasetClass: Type[Dataset]) -> Type[Dataset]:
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
            
            self.df = pd.read_csv(cfg.data.adjustable_dataloader.initializer_csv_path)

            self.augs_to_adjust = cfg.data.adjustable_dataloader.adjustable_augmentations
            for aug in self.augs_to_adjust:
                assert aug in ['rrc', 'cj_brightness', 'cj_contrast', 'cj_saturation'], f"Adjusting for the Augmentation {aug} not supported"

            for aug in cfg.data.adjustable_dataloader.custom_defaults:
                assert aug in ['rrc', 'cj_brightness', 'cj_contrast', 'cj_saturation'], f"Custom defaults for the Augmentation {aug} not supported"

            if 'rrc' not in cfg.data.adjustable_dataloader.custom_defaults:
                self.df['rrc_area_lower_bound'] = 0.08
            if 'cj_brightness' not in cfg.data.adjustable_dataloader.custom_defaults:
                self.df['cj_brightness_lower_bound'] = 0.2
                self.df['cj_brightness_upper_bound'] = 1.8
            if 'cj_contrast' not in cfg.data.adjustable_dataloader.custom_defaults:
                self.df['cj_contrast_lower_bound'] = 0.2
                self.df['cj_contrast_upper_bound'] = 1.8
            if 'cj_saturation' not in cfg.data.adjustable_dataloader.custom_defaults:
                self.df['cj_saturation_lower_bound'] = 0.2
                self.df['cj_saturation_upper_bound'] = 1.8

            self.df['grey_scale_chance'] = np.where(self.df['is_grey_scale'] == True, 1.0, cfg.augmentations[0].grayscale.prob)

            # save originals in df
            for var_name in ['rrc_area_lower_bound', 'cj_brightness_lower_bound', 'cj_brightness_upper_bound',
                            'cj_contrast_lower_bound', 'cj_contrast_upper_bound', 'cj_saturation_lower_bound', 'cj_saturation_upper_bound']:
                self.df[f'{var_name}_original'] = self.df[var_name]

            self.__create_augmentation_error_df_columns()

        def __getitem__(self, index):
            data, target = super().__getitem__(index)
            data_row = self.df.iloc[index]
            data = self.initial_transforms(data, data_row=data_row)
            return (index, data, target)

        def __create_augmentation_error_df_columns(self):
            for name in augmentation_error_names:
                self.df[name] = 0.0
        
        def update_logging_path(self, path):
            self.logging_path = path

        def save_augmentation_parameters(self):
            self.df.to_csv(f"{self.logging_path}/{cfg.data.adjustable_dataloader.final_csv_path}", index=False, float_format='%g')

        def update_augmentation_parameters(self, batch_indices, augmentation_error_values, should_adjust=False):

            def update_errors(row):
                val_loc = batch_indices.index(row.name)
                row[augmentation_error_names] = augmentation_error_values[val_loc]
                return row
            
            def update_errors_and_vals(row, error_mean, error_std):
                val_loc = batch_indices.index(row.name)
                row[augmentation_error_names] = augmentation_error_values[val_loc]
                values = augmentation_error_values[val_loc]
                stepsize = cfg.data.adjustable_dataloader.stepsize

                if 'rrc' in self.augs_to_adjust:
                    # adjust rcc as needed
                    rcc_i_name = augmentation_error_names[0]
                    rcc_j_name = augmentation_error_names[1]
                    if any([abs(values[0] - error_mean[rcc_i_name]) > 2 * error_std[rcc_i_name],
                            abs(values[1] - error_mean[rcc_j_name]) > 2 * error_std[rcc_j_name]]):

                        if values[0] <= error_mean[rcc_i_name] or values[1] <= error_mean[rcc_j_name]:
                            # make harder
                            row['rrc_area_lower_bound'] = max(0.08, -stepsize + row['rrc_area_lower_bound'])
                        elif values[0] >= error_mean[rcc_i_name] or values[1] >= error_mean[rcc_j_name]:
                            # make easier
                            row['rrc_area_lower_bound'] = min(0.5, stepsize + row['rrc_area_lower_bound'])

                if 'cj_brightness' in self.augs_to_adjust:
                    # adjust brightness as needed
                    cj_b_name = augmentation_error_names[4]
                    if abs(values[4] - error_mean[cj_b_name]) > 2 * error_std[cj_b_name]:
                        if values[4] >= error_mean[cj_b_name]:
                            # make easier
                            row['cj_brightness_lower_bound'] = min(1.0, stepsize + row['cj_brightness_lower_bound'])
                            row['cj_brightness_upper_bound'] = max(1.0, -stepsize + row['cj_brightness_upper_bound'])
                        else:
                            # make harder
                            row['cj_brightness_lower_bound'] = max(0.2, -stepsize + row['cj_brightness_lower_bound'])
                            row['cj_brightness_upper_bound'] = min(1.8, stepsize + row['cj_brightness_upper_bound'])


                if 'cj_contrast' in self.augs_to_adjust:
                    # adjust contrast as needed
                    cj_c_name = augmentation_error_names[5]
                    if abs(values[5] - error_mean[cj_c_name]) > 2 * error_std[cj_c_name]:
                        if values[5] >= error_mean[cj_c_name]:
                            # make easier
                            row['cj_contrast_lower_bound'] = min(1.0, stepsize + row['cj_contrast_lower_bound'])
                            row['cj_contrast_upper_bound'] = max(1.0, -stepsize + row['cj_contrast_upper_bound'])
                        else:
                            # make harder
                            row['cj_contrast_lower_bound'] = max(0.2, -stepsize + row['cj_contrast_lower_bound'])
                            row['cj_contrast_upper_bound'] = min(1.8, stepsize + row['cj_contrast_upper_bound'])
                
                if 'cj_saturation' in self.augs_to_adjust:
                    # adjust saturation as needed
                    cj_s_name = augmentation_error_names[6]
                    if abs(values[6] - error_mean[cj_s_name]) > 2 * error_std[cj_s_name]:
                        if values[6] >= error_mean[cj_s_name]:
                            # make easier
                            row['cj_saturation_lower_bound'] = min(1.0, stepsize + row['cj_saturation_lower_bound'])
                            row['cj_saturation_upper_bound'] = max(1.0, -stepsize + row['cj_saturation_upper_bound'])
                        else:
                            # make harder
                            row['cj_saturation_lower_bound'] = max(0.2, -stepsize + row['cj_saturation_lower_bound'])
                            row['cj_saturation_upper_bound'] = min(1.8, stepsize + row['cj_saturation_upper_bound'])

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
                t.p = kwargs['data_row']["grey_scale_chance"]

            elif isinstance(t, torchvision.transforms.RandomApply):
                if isinstance(t.transforms[0], torchvision.transforms.ColorJitter):
                    t.transforms[0].brightness = (kwargs['data_row']["cj_brightness_lower_bound"], kwargs['data_row']["cj_brightness_upper_bound"])
                    t.transforms[0].contrast = (kwargs['data_row']["cj_contrast_lower_bound"], kwargs['data_row']["cj_contrast_upper_bound"])
                    t.transforms[0].saturation = (kwargs['data_row']["cj_saturation_lower_bound"], kwargs['data_row']["cj_saturation_upper_bound"])

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