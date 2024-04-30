# %%
import argparse
import torch
import torchvision
from torchvision.datasets import STL10, ImageFolder, CIFAR10, CIFAR100
from solo.data.pretrain_dataloader import dataset_with_index
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from transformers import pipeline
import tqdm
from PIL import Image,ImageChops
import pandas as pd
import matplotlib.pyplot as plt
import os
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2ImageProcessor
from PIL import ImageDraw

MEANS_N_STD = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
    "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
}

def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='stl10', help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./datasets', help='directory to save dataset')
    parser.add_argument('--default_bbox_lower_bound', type=float, default=0.08, help='default lower bound for bounding box area')
    return parser.parse_args()

def get_dataloader(args):
    train_data_path = f"{args.data_dir}/{args.dataset}"
    dataset = args.dataset
    download = True

    mean, std = MEANS_N_STD[dataset]
    augmentations = []
    augmentations.append(transforms.ToTensor())
    augmentations.append(transforms.ToPILImage())
    augmentations = transforms.Compose(augmentations)
    augmentations = augmentations if augmentations != [] else None

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=augmentations,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=augmentations,
        )

    elif dataset in ["imagenet", "imagenet100"]:
        train_dataset = dataset_with_index(ImageFolder)(
            train_data_path,
            transform=augmentations)

    train_loader = DataLoader(
        train_dataset,
        batch_size=36,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        collate_fn=my_collate_fn
    )

    return train_loader

def bounding_box_for_image(args, model, processor, data, inputs, candidate_labels):
    outputs = model(**inputs)
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.1)

    labels = []
    areas = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    for idx, result in enumerate(results):
        if result['boxes'].tolist() == []:
            predicted_label = "none"
            best_area_lower_bound = args.default_bbox_lower_bound
            
            # put -1, which is impossible to get normally
            xmins.append(-1)
            ymins.append(-1)
            xmaxs.append(-1)
            ymaxs.append(-1)
        else:
            # find indxes with max score
            max_score_idx = result['scores'].argmax().item()
            score = result['scores'][max_score_idx].item()
            predicted_label = candidate_labels[result['labels'][max_score_idx]]
            xmin, ymin, xmax, ymax = result['boxes'][max_score_idx].tolist()

            xcenter, ycenter = (xmin + xmax) / 2, (ymin + ymax) / 2
            image_width, image_height = data[idx].size

            xleftdist = xcenter
            xrightdist = 1 - xcenter
            ytopdist = ycenter
            ybottomdist = 1 - ycenter

            best_distance = max(xleftdist, xrightdist, ytopdist, ybottomdist)
            best_area_lower_bound = best_distance**2

            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

        
        labels.append(predicted_label)
        areas.append(best_area_lower_bound)

    return areas, labels, xmins, ymins, xmaxs, ymaxs


def is_greyscale(im):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return False
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return False
    return True

def main():
    args = parse()
    csv_filename = f"{args.dataset}_preprocessed_info.csv"

    if os.path.exists(csv_filename):
        make_visualizations(csv_filename)
        return

    train_loader = get_dataloader(args)
    candidate_labels = train_loader.dataset.classes

    preprocessed_info = {}
    with torch.no_grad():

        checkpoint = "google/owlv2-base-patch16-ensemble"
        max_memory_mapping = {0: "24GB", 1: "24GB"}
        processor = Owlv2Processor.from_pretrained(checkpoint)
        model = Owlv2ForObjectDetection.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, max_memory=max_memory_mapping)

        for idx, batch in enumerate(tqdm.tqdm(train_loader)):
            batch_idx, data, _ = batch

            inputs = processor.__call__(text=[candidate_labels]*len(batch_idx),
                                         images=data, return_tensors='pt')
            if len(max_memory_mapping.keys()) == 1:
                inputs = inputs.to("cuda")


            rrc_area_lower_bound, predicted_label, xmins, ymins, xmaxs, ymaxs = bounding_box_for_image(args, model, processor, data, inputs, candidate_labels)
            is_grey_scale = [is_greyscale(im) for im in data]

            for a, i in enumerate(batch_idx):
                preprocessed_info[i] = {
                    "predicted_label": predicted_label[a],
                    "is_grey_scale": is_grey_scale[a],
                    "rrc_area_lower_bound": rrc_area_lower_bound[a],
                    "xmin": xmins[a],
                    "ymin": ymins[a],
                    "xmax": xmaxs[a],
                    "ymax": ymaxs[a]
                }

    # turn preprocessed_info into a pandas array
    preprocessed_info = pd.DataFrame.from_dict(preprocessed_info, orient="index")
    preprocessed_info.to_csv(csv_filename)

    make_visualizations(csv_filename)

def my_collate_fn(batch):
    indices = [idx for idx, _, _ in batch]
    data = [d for _, d, _ in batch]
    target = [t for _, _, t in batch]
    return indices, data, target

def make_visualizations(csv_filename):
    # make plot showing distributions of areas per class
    preprocessed_info = pd.read_csv(csv_filename)
    preprocessed_info = preprocessed_info.drop(columns="Unnamed: 0")

    # get list of classes from column
    classes = preprocessed_info["predicted_label"].unique()
    plt.figure(figsize=(10, 5))

    for c in classes:
        if c == "none":
            continue
        class_info = preprocessed_info[preprocessed_info["predicted_label"] == c]
        plt.hist(class_info["rrc_area_lower_bound"], bins=20, alpha=0.5, label=c)
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad=0)
    plt.tight_layout()
    plt.savefig(f"{csv_filename}_rrc_area_lower_bound_distributions.png")

if __name__ == "__main__":
    main()