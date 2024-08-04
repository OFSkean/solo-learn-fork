# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import os

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
    make_transforms_augmentation_aware
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous, omegaconf_select
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import torchvision
import tqdm
import numpy as np
import wandb
try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True
from torchmetrics.classification import MulticlassConfusionMatrix

import matplotlib.pyplot as plt
import numpy as np
import json

sweep_id = "cliplab/augrelius/p5edoice"

def download_sweep_runs():
    # get runs with sweep_id from wandb
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs = list(sweep.runs)
    sweep_run_ids = sorted([run.id for run in runs])

    with tqdm.tqdm(sweep_run_ids) as pbar:
        for run_id in pbar:
            if not os.path.exists(f"trained_models/augrelius/{run_id}"):
                pbar.set_description(f"Downloading {run_id}")
                os.system(f"scp -q -r ofsk222@lcc.uky.edu:/project/lsa273_uksr/frossl/augrelius/{run_id} trained_models/augrelius/{run_id}")

            pbar.update(1)

    return sweep_run_ids

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    cfg.return_train_outputs=True
    seed_everything(cfg.seed)

    # pretrain dataloader
    if cfg.data.augaware:
        make_transforms_augmentation_aware()
    if cfg.data.augadjustable:
        assert cfg.data.augaware, "augadjustable requires augaware to be enabled"
        make_dataset_augmentations_adjustable()

    # validation dataloader for when it is available
    if cfg.data.dataset == "custom" and (cfg.data.no_labels or cfg.data.val_path is None):
        val_loader = None
    elif cfg.data.dataset in ["imagenet100", "imagenet"] and cfg.data.val_path is None:
        val_loader = None
    else:
        if cfg.data.format == "dali":
            val_data_format = "image_folder"
        else:
            val_data_format = cfg.data.format

        _, val_loader = prepare_data_classification(
            cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            val_data_path=cfg.data.val_path,
            data_format=val_data_format,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

    # iterate through each run and get accuracies
    sweep_run_ids = download_sweep_runs()
    accuracies = np.zeros((len(sweep_run_ids), 12, 2))

    for idx, run_id in enumerate(sweep_run_ids):
        json_cfg = json.load(open(f"trained_models/augrelius/{run_id}/args.json"))
        min_crop_size = json_cfg["augmentations"][0]['rrc']['crop_min_scale']
        accuracies[idx, 0, 0] = min_crop_size

        # find ckpt file in folder
        ckpt_path = None
        for file in os.listdir(f"trained_models/augrelius/{run_id}"):
            if file.endswith(".ckpt"):
                ckpt_path = file
                break
        assert ckpt_path is not None, f"No ckpt file found in trained_models/augrelius/{run_id}"

        model = METHODS[cfg.method].load_from_checkpoint(f"trained_models/augrelius/{run_id}/{ckpt_path}", cfg=cfg)
        make_contiguous(model)
        model = model.cuda()

        all_targets = []
        preds = []
        for batch in tqdm.tqdm(val_loader):
            data, targets = batch
            all_targets.append(targets)

            outs = model.validation_step(batch, 0)
            
            predicted_label = torch.argmax(outs["shared_logits"], dim=1)
            preds.append(predicted_label)
        
        all_targets = torch.cat(all_targets).cuda()
        preds = torch.cat(preds).cuda()

        # compute overall accuracy
        accuracies[idx, 1, 0] = (preds == all_targets).float().mean().item() # mean
        accuracies[idx, 1, 1] = (preds == all_targets).float().std().item() # std

        # compute accuracy for each class
        for i in range(10):
            accuracies[idx, i+2] = (preds[all_targets == i] == i).float().mean().item() # mean
            accuracies[idx, i+2, 1] = (preds[all_targets == i] == i).float().std().item()


    np.save("rrc_sweep_accuracies.npy", accuracies)

def make_plots():
    accuracies = np.load("rrc_sweep_accuracies.npy")
    # sort by min crop size
    accuracies = accuracies[accuracies[:, 0, 0].argsort()]
    class_labels = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]

    min_crop_sizes = accuracies[:, 0, 0]
    overall_accuracies = accuracies[:, 1, 0]
    class_accuracies = accuracies[:, 2:, 0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    ax.plot(min_crop_sizes, overall_accuracies, label="Overall") # mean
    ax.fill_between(min_crop_sizes, overall_accuracies - accuracies[:, 1, 1]/10, overall_accuracies + accuracies[:, 1, 1]/10, alpha=0.2)

    for i in range(10):
        if class_labels[i] in ['cat', 'ship']:
            ax.plot(min_crop_sizes, class_accuracies[:, i], label=class_labels[i])
            ax.fill_between(min_crop_sizes, class_accuracies[:, i] - accuracies[:, i+2, 1]/10, class_accuracies[:, i] + accuracies[:, i+2, 1]/10, alpha=0.2)

    ax.invert_xaxis()
    ax.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    plt.tight_layout()    
    plt.savefig("rrc_sweep_accuracies.png")


if __name__ == "__main__":
    #if not os.path.exists("rrc_sweep_accuracies.npy"):
    main()
    make_plots()
