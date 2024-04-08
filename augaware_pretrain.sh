DATASET="stl10"
CONFIG_NAME="augrelius.yaml"

# check if data is cifar
if [ "$DATASET" == "cifar10" ] || [ "$DATASET" == "cifar100" ]; then
    dataset_config_name="cifar"
else
    dataset_config_name="$DATASET"
fi
#CONFIG_PATH="scripts/pretrain/imagenet-100"
CONFIG_PATH="scripts/pretrain/$DATASET"

# echo "Preparing to start linear probe for experiment with name $EXPERIMENT_NAME"
# echo "on dataset $DATASET"
# echo "with config $LINEAR_CONFIG_PATH/$CONFIG_NAME"
# echo "seed is $SEED"
# echo $TRAINED_CHECKPOINT_PATH > last_ckpt.txt

# ####
# #### PRETRAIN LINEAR PROBE
# ####
CUDA_LAUNCH_BLOCKING=1 python3 -u main_pretrain.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++data.format="image_folder" \
    ++data.augaware=True \
    ++wandb.project="augrelius" \
    ++wandb.enabled=True \
    ++name="test_identity" \
    ++augmentations.0.num_crops=4 \
    # ++data.train_path="./datasets/imagenet100/train" \
    # ++data.val_path="./datasets/imagenet100/val" \
