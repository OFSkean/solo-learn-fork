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

def visualize_multiview_dataset(batch, idx, pdiffhat, raw_images, sidesize=5):
    """
    Plot small batch of multiview dataset. image grid, Side-by-side of view 1 and view 2
    """
    x0 = [x[0][0] for x in batch]
    x1 = [x[0][1] for x in batch]

    w1 = [x[1][0].item() for x in batch]
    w2 = [x[1][1].item() for x in batch]

    img_size = x0[0].shape

    assert len(x0) >= sidesize**2, "sidesize**2 must be at most the batch size"

    view1_imgs = torch.zeros((sidesize**2, *img_size))
    view2_imgs = torch.zeros((sidesize**2, *img_size))
    raw_imgs =   torch.zeros((sidesize**2, *img_size))
    

    for i in range(sidesize**2):
        view1_imgs[i] = x0[i].reshape(img_size)
        view2_imgs[i] = x1[i].reshape(img_size)
        raw_imgs[i] = raw_images[i].reshape(img_size)


    view1_grid = torchvision.utils.make_grid(view1_imgs, nrow=sidesize, scale_each=True)
    view2_grid = torchvision.utils.make_grid(view2_imgs, nrow=sidesize, scale_each=True)
    raw_grid = torchvision.utils.make_grid(raw_imgs, nrow=sidesize, scale_each=True)

    data = np.random.random((sidesize, sidesize))

    fig, ax = plt.subplots(figsize=(13,13))
    # Using matshow here just because it sets the ticks up nicely. imshow is faster.
    ax.matshow(data, cmap='seismic')

    for (i, j), z in np.ndenumerate(data):
        pdiff = w1[i*sidesize+j] - w2[i*sidesize+j]
        ax.text(j, i, 'l={:0.2f} r={:0.2f}\np={:0.2f} phat={:0.2f}'.format(w1[i*sidesize+j], w2[i*sidesize+j], pdiff, pdiffhat[i*sidesize+j]), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.show()

    return [view1_grid, view2_grid, raw_grid, fig]

@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)
    cfg.return_train_outputs=True
    seed_everything(cfg.seed)


    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"

    if cfg.data.num_large_crops != 2:
        assert cfg.method in ["wmse", "mae", "augrelius"]

   
    # pretrain dataloader
    if cfg.data.augaware:
        make_transforms_augmentation_aware()
        
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


    if cfg.data.format == "dali":
        assert (
            _dali_avaliable
        ), "Dali is not currently avaiable, please install it first with pip3 install .[dali]."
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline_dali(
                        cfg.data.dataset, aug_cfg, dali_device=cfg.dali.device
                    ),
                    aug_cfg.num_crops,
                )
            )
        transform = FullTransformPipeline(pipelines)

        dali_datamodule = PretrainDALIDataModule(
            dataset=cfg.data.dataset,
            train_data_path=cfg.data.train_path,
            transforms=transform,
            num_large_crops=cfg.data.num_large_crops,
            num_small_crops=cfg.data.num_small_crops,
            num_workers=cfg.data.num_workers,
            batch_size=cfg.optimizer.batch_size,
            no_labels=cfg.data.no_labels,
            data_fraction=cfg.data.fraction,
            dali_device=cfg.dali.device,
            encode_indexes_into_labels=cfg.dali.encode_indexes_into_labels,
        )
        dali_datamodule.val_dataloader = lambda: val_loader
    else:
        pipelines = []
        for aug_cfg in cfg.augmentations:
            pipelines.append(
                NCropAugmentation(
                    build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops
                )
            )
        transform = FullTransformPipeline(pipelines)

        if cfg.debug_augmentations:
            print("Transforms:")
            print(transform)

        train_dataset = prepare_datasets(
            cfg.data.dataset,
            transform,
            train_data_path=cfg.data.train_path,
            data_format=cfg.data.format,
            no_labels=False,
            data_fraction=cfg.data.fraction,
            use_val=True
        )
        train_loader = prepare_dataloader(
            train_dataset, batch_size=cfg.optimizer.batch_size, num_workers=cfg.data.num_workers, shuffle=False
        )

    # 1.7 will deprecate resume_from_checkpoint, but for the moment
    # the argument is the same, but we need to pass it as ckpt_path to trainer.fit
    ckpt_path, wandb_run_id = None, None
    assert cfg.resume_from_checkpoint is not None
    ckpt_path = cfg.resume_from_checkpoint

    model = METHODS[cfg.method].load_from_checkpoint(ckpt_path, cfg=cfg)
    make_contiguous(model)
    # can provide up to ~20% speed up
    if not cfg.performance.disable_channel_last:
        model = model.to(memory_format=torch.channels_last)
        
    callbacks = []
    # wandb logging
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            name=cfg.name,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            offline=cfg.wandb.offline
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))


    trainer_kwargs = OmegaConf.to_container(cfg)
    # we only want to pass in valid Trainer args, the rest may be user specific
    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = {name: trainer_kwargs[name] for name in valid_kwargs if name in trainer_kwargs}
    trainer_kwargs.update(
        {
            "logger": wandb_logger if cfg.wandb.enabled else None,
            "callbacks": callbacks,
            "enable_checkpointing": False,
            "strategy": DDPStrategy(find_unused_parameters=False)
            if cfg.strategy == "ddp"
            else cfg.strategy,
            "max_epochs": 0
        }
    )


    # iterate over a few batches of the train loader
    augmentation_errors = []
    all_targets = []
    was_correct = []
    augmentation_params_view1 = []
    augmentation_params_view2 = []
    images_view1 = []
    images_view2 = []
    preds = []
    pdiffhats = []
    model = model.cuda()
    for batch in tqdm.tqdm(train_loader):
        batch_idx, data, targets = batch
        augmentation_params_view1.append(torch.stack([x.cpu().detach() for x in data[0][1]]).T)
        augmentation_params_view2.append(torch.stack([x.cpu().detach() for x in data[1][1]]).T)
        images_view1.append(data[0][0].cpu().detach())
        images_view2.append(data[1][0].cpu().detach())


        loss, outs = model.training_step(batch, 0)
        errors = outs["augmentation_errors"][0]
        augmentation_errors.append(outs["augmentation_errors"][0])
        all_targets.append(targets)
        
        predicted_phat = outs["phats"]
        pdiffhats.append(predicted_phat[0])

        predicted_label = torch.argmax(outs["shared_logits"][0], dim=1)
        preds.append(predicted_label)

        correct = predicted_label == targets.cuda()
        was_correct.append(correct)

    augmentation_names = [
        'rrc_i', 'rrc_j', 'rrc_h', 'rrc_w',
        'cj_b', 'cj_c', 'cj_s', 'cj_h',
        'gaussian_sigma', 
        'equalization', 'solarization', 'random_grayscale', 'random_horizontal_flip'
    ]

    augmentation_errors = torch.concat(augmentation_errors).cuda()
    targets = torch.concat(all_targets).cuda()
    was_correct = torch.concat(was_correct).cuda()
    preds = torch.concat(preds).cuda()
    pdiffhats = torch.concat(pdiffhats).cuda()
    augmentation_params_view1 = torch.concat(augmentation_params_view1).cuda()
    augmentation_params_view2 = torch.concat(augmentation_params_view2).cuda()
    images_view1 = torch.concat(images_view1).cuda()
    images_view2 = torch.concat(images_view2).cuda()

    mean_per_target = []


    # make histogram of augmentation errors
    fig, axs = plt.subplots(nrows=1, ncols=1)
    axs.hist(augmentation_errors.mean(dim=1).cpu().detach().numpy(), bins=20, alpha=0.5, color='b', label='All', density=True)
    axs.hist(augmentation_errors.mean(dim=1)[was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='g', label='Correct', density=True)
    axs.hist(augmentation_errors.mean(dim=1)[~was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='r', label='Incorrect', density=True)
    axs.set_title("Augmentation Errors Histogram")
    axs.set_xlabel("Augmentation Error")
    axs.set_ylabel("Frequency")
    axs.legend()
    wandb_logger.log_image("augmentation_errors_histogram", images=[wandb.Image(fig)])
    plt.close()

    # make histogram for each augmentation
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(10, 35))
    for i, ax in enumerate(axs.flat):
        if i == 0:
            data = augmentation_errors.mean(dim=1)
            ax.set_title('All')
        else:
            data = augmentation_errors[:, i-1]
            ax.set_title(augmentation_names[i-1])
        ax.hist(data.cpu().detach().numpy(), bins=20, alpha=0.5, color='b', label='All', density=True)
        ax.hist(data[was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='g', label='Correct', density=True)
        ax.hist(data[~was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='r', label='Incorrect', density=True)

        ax.set_xlabel("Augmentation Error")
        ax.set_ylabel("Frequency")
    wandb_logger.log_image("augmentation_errors_histogram_per_augmentation", images=[wandb.Image(fig)])
    plt.close()

    # make histogram for each augmentation prediction values
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(10, 35))
    for i, ax in enumerate(axs.flat):
        if i == 0:
            data = pdiffhats.mean(dim=1)
            ax.set_title('All')
        else:
            data = pdiffhats[:, i-1]
            ax.set_title(augmentation_names[i-1])
        ax.hist(data.cpu().detach().numpy(), bins=20, alpha=0.5, color='b', label='All', density=True)
        ax.hist(data[was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='g', label='Correct', density=True)
        ax.hist(data[~was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='r', label='Incorrect', density=True)

        ax.set_xlabel("Predicted values")
        ax.set_ylabel("Frequency")
    wandb_logger.log_image("predicted_values_histogram_per_augmentation", images=[wandb.Image(fig)])
    plt.close()

    # make histogram for each real augmentation parameters
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(10, 35))
    for i, ax in enumerate(axs.flat):
        if i == 0:
            data = augmentation_params_view1.mean(dim=1)
            ax.set_title('All')
        else:
            data = augmentation_params_view1[:, i-1]
            ax.set_title(augmentation_names[i-1])
        ax.hist(data.cpu().detach().numpy(), bins=20, alpha=0.5, color='b', label='All', density=True)
        ax.hist(data[was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='g', label='Correct', density=True)
        ax.hist(data[~was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='r', label='Incorrect', density=True)

        ax.set_xlabel("Real values")
        ax.set_ylabel("Frequency")
    wandb_logger.log_image("real_values_histogram_per_augmentation", images=[wandb.Image(fig)])
    plt.close()

    # make histogram for each real diffs augmentation parameters
    fig, axs = plt.subplots(nrows=7, ncols=2, figsize=(10, 35))
    for i, ax in enumerate(axs.flat):
        if i == 0:
            data = augmentation_params_view1.mean(dim=1) - augmentation_params_view2.mean(dim=1)
            ax.set_title('All')
        else:
            data = augmentation_params_view1[:, i-1] - augmentation_params_view2[:, i-1]
            ax.set_title(augmentation_names[i-1])
        ax.hist(data.cpu().detach().numpy(), bins=20, alpha=0.5, color='b', label='All', density=True)
        ax.hist(data[was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='g', label='Correct', density=True)
        ax.hist(data[~was_correct].cpu().detach().numpy(), bins=20, alpha=0.5, color='r', label='Incorrect', density=True)

        ax.set_xlabel("Real values")
        ax.set_ylabel("Frequency")
    wandb_logger.log_image("real_diffs_histogram_per_augmentation", images=[wandb.Image(fig)])
    plt.close()

    # PLOT CONFUSION MATRICES FOR HIGHEST AND LOWEST ERROR SAMPLES
    for choice in range(-1, len(augmentation_names)):
        if choice == -1:
            lowest_error_indices = augmentation_errors.mean(dim=1).abs().argsort()[:500]
            highest_error_indices = augmentation_errors.mean(dim=1).abs().argsort()[-500:]
        else:
            lowest_error_indices = augmentation_errors[:, choice].abs().argsort()[:500]
            highest_error_indices = augmentation_errors[:, choice].abs().argsort()[-500:]

        # lowest error plot
        y_true = targets[lowest_error_indices].cuda().float()
        y_shared_pred = preds[lowest_error_indices].cuda().float()
        conf_matrix = MulticlassConfusionMatrix(num_classes=10, normalize='true').to(y_true)
        conf_matrix.update(y_shared_pred, y_true)
        fig_lowest, ax_ = conf_matrix.plot(labels=val_loader.dataset.classes)

        conf_matrix = MulticlassConfusionMatrix(num_classes=10).to(y_true)
        conf_matrix.update(y_shared_pred, y_true)
        fig_lowest_unnormalized, ax_ = conf_matrix.plot(labels=val_loader.dataset.classes)

        # highest error plot
        y_true = targets[highest_error_indices].cuda().float()
        y_shared_pred = preds[highest_error_indices].cuda().float()
        conf_matrix = MulticlassConfusionMatrix(num_classes=10, normalize='true').to(y_true)
        conf_matrix.update(y_shared_pred, y_true)
        fig_highest, ax_ = conf_matrix.plot(labels=val_loader.dataset.classes)

        conf_matrix = MulticlassConfusionMatrix(num_classes=10).to(y_true)
        conf_matrix.update(y_shared_pred, y_true)
        fig_highest_unnormalized, ax_ = conf_matrix.plot(labels=val_loader.dataset.classes)

        # wandb_logger.log_image(f"aug. error confusions for aug {augmentation_names[choice] if choice != -1 else 'all'}", 
        #                         images=[wandb.Image(fig_lowest), wandb.Image(fig_lowest_unnormalized), wandb.Image(fig_highest), wandb.Image(fig_highest_unnormalized),], 
        #                         caption=["Norm. Lowest Error Confusion", "Lowest Error Confusion", "Norm. Highest Error Confusion", "Highest Error Confusion"])

        wandb_logger.log_image(f"aug. error confusions for aug {augmentation_names[choice] if choice != -1 else 'all'}", 
                                images=[wandb.Image(fig_lowest), wandb.Image(fig_highest)], 
                                caption=["Norm. Lowest Error Confusion", "Norm. Highest Error Confusion"])
        if choice >= 0:
            #low error samples
            lowest_samples = [[(images_view1[i], images_view2[i]), (augmentation_params_view1[i, choice], augmentation_params_view2[i, choice])] for i in lowest_error_indices.cpu()[0:25]]
            lowest_pdiffhats = pdiffhats[lowest_error_indices.cpu()[0:25], choice]
            raw_images = [val_loader.dataset[i][0][0] for i in lowest_error_indices.cpu()[0:25]]

            view1, view2, raw_grid, errors = visualize_multiview_dataset(lowest_samples, choice, lowest_pdiffhats, raw_images, sidesize=5)
            wandb_logger.log_image(f"aug. error lowest error samples for aug {augmentation_names[choice] if choice != -1 else 'all'}",
                                    images=[wandb.Image(view1), wandb.Image(view2), wandb.Image(raw_grid), wandb.Image(errors)],
                                    caption=["View 1", "View 2", "Raw", "Aug Params"])
            #high error samples
            highest_samples = [[(images_view1[i], images_view2[i]), (augmentation_params_view1[i, choice], augmentation_params_view2[i, choice])] for i in highest_error_indices.cpu()[-25:]]
            highest_pdiffhats = pdiffhats[highest_error_indices.cpu()[-25:], choice]
            raw_images = [val_loader.dataset[i][0][0] for i in highest_error_indices.cpu()[-25:]]
            view1, view2, raw_grid, errors = visualize_multiview_dataset(highest_samples, choice, highest_pdiffhats, raw_images, sidesize=5)
            wandb_logger.log_image(f"aug. error highest error samples for aug {augmentation_names[choice] if choice != -1 else 'all'}",
                                    images=[wandb.Image(view1), wandb.Image(view2), wandb.Image(raw_grid), wandb.Image(errors)],
                                    caption=["View 1", "View 2", "Raw", "Aug Params"])
    plt.close()

    # IMAGE FOR ALL SAMPLES
    # get augmentation errors per target
    for i in range(len(val_loader.dataset.classes)):
        target_errors = augmentation_errors[targets == i]
        mean_per_target.append(target_errors.mean(dim=0).cpu().detach().tolist())
    fig, axs = plt.subplots(nrows=1, ncols=1)
    im = axs.imshow(np.array(mean_per_target), vmin=0, vmax=1.0)
    axs.set_title("Augmentation Errors per Target")
    axs.set_ylabel("Target")
    axs.set_xlabel("Mean Augmentation Error")
    axs.set_xticks(range(len(augmentation_names)))
    axs.set_xticklabels(augmentation_names)
    axs.set_yticks(range(len(val_loader.dataset.classes)))
    axs.set_yticklabels(val_loader.dataset.classes)
    for tick in axs.get_xticklabels():
        tick.set_rotation(90)
    fig.colorbar(im)

    # IMAGE FOR ALL CORRECT SAMPLES
    # get augmentation errors per target
    mean_per_target= []
    for i in range(len(val_loader.dataset.classes)):
        # find rows where the target is i and the prediction is correct
        target_errors = augmentation_errors[(targets == i) & was_correct]
        mean_per_target.append(target_errors.mean(dim=0).cpu().detach().tolist())

    fig1, axs1 = plt.subplots(nrows=1, ncols=1)
    im = axs1.imshow(np.array(mean_per_target), vmin=0, vmax=1.0)
    axs1.set_title("Augmentation Errors per Target")
    axs1.set_ylabel("Target")
    axs1.set_xlabel("Mean Augmentation Error")
    axs1.set_xticks(range(len(augmentation_names)))
    axs1.set_xticklabels(augmentation_names)
    axs1.set_yticks(range(len(val_loader.dataset.classes)))
    axs1.set_yticklabels(val_loader.dataset.classes)
    for tick in axs1.get_xticklabels():
        tick.set_rotation(90)
    fig1.colorbar(im)

    # IMAGE FOR ALL INCORRECT SAMPLES
    # get augmentation errors per target
    mean_per_target= []
    for i in range(len(val_loader.dataset.classes)):
        # find rows where the target is i and the prediction is not correct
        target_errors = augmentation_errors[(targets == i) & ~was_correct]
        mean_per_target.append(target_errors.mean(dim=0).cpu().detach().tolist())

    fig2, axs2 = plt.subplots(nrows=1, ncols=1)
    im = axs2.imshow(np.array(mean_per_target), vmin=0, vmax=1.0)
    axs2.set_title("Augmentation Errors per Target")
    axs2.set_ylabel("Target")
    axs2.set_xlabel("Mean Augmentation Error")
    axs2.set_xticks(range(len(augmentation_names)))
    axs2.set_xticklabels(augmentation_names)
    axs2.set_yticks(range(len(val_loader.dataset.classes)))
    axs2.set_yticklabels(val_loader.dataset.classes)
    for tick in axs2.get_xticklabels():
        tick.set_rotation(90)
    fig2.colorbar(im)

    wandb_logger.log_image("augmentation_errors", 
                        images=[wandb.Image(fig), wandb.Image(fig1), wandb.Image(fig2)],
                        caption=["All", "Correct", "Incorrect"])
    plt.close()

    trainer = Trainer(**trainer_kwargs)
    trainer.validate(model, dataloaders=[val_loader], ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
