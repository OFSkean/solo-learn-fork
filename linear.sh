if [ $# -eq 0 ]
then
    echo "No arguments supplied. exiting"
    exit 1
fi

EXPERIMENT_NAME="$1"
SEED="$2"
DATASET="imagenet100"
CONFIG_NAME="barlow.yaml"
WANDBID=""
TRAINED_CHECKPOINT_PATH="/home/AD/ofsk222/Research/opensource/solo-learn-fork/trained_models/barlow_twins/sololearn-in100-checkpoint/barlow-400ep-imagenet100-fv1nl9um-ep=399.ckpt"

# check if data is cifar
if [ "$DATASET" == "cifar10" ] || [ "$DATASET" == "cifar100" ]; then
    dataset_config_name="cifar"
else
    dataset_config_name="$DATASET"
fi
LINEAR_CONFIG_PATH="scripts/linear/imagenet-100"

echo "Preparing to start linear probe for experiment with name $EXPERIMENT_NAME"
echo "on dataset $DATASET"
echo "with config $LINEAR_CONFIG_PATH/$CONFIG_NAME"
echo "seed is $SEED"
echo $TRAINED_CHECKPOINT_PATH > last_ckpt.txt

# ####
# #### PRETRAIN LINEAR PROBE
# ####
CUDA_LAUNCH_BLOCKING=1 python3 -u main_linear.py \
    --config-path $LINEAR_CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++name="$EXPERIMENT_NAME-linear-$WANDBID-$SEED" \
    ++seed="$SEED" \
    ++data.precompute_embeddings=True \
    ++data.format="image_folder" \
