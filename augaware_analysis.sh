DATASET="stl10"
CONFIG_NAME="augrelius.yaml"
CHECKPOINT="/home/AD/ofsk222/Research/opensource/solo-learn-fork/trained_models/augrelius/c2qdgx9a/augrelius-stl10-c2qdgx9a-ep=499.ckpt"
ESCAPED_CHECKPOINT=`echo "$CHECKPOINT" | sed -e 's@/@\\/@g'`

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
CUDA_LAUNCH_BLOCKING=1 python3 -u main_analysis.py \
    --config-path $CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++data.format="image_folder" \
    ++data.augaware=True \
    ++wandb.project="augrelius_analysis" \
    ++wandb.enabled=True \
    ++name="analysis" \
    ++method_kwargs.proj_output_dim=1024\