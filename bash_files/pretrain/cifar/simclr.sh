python3 ../../../main_contrastive.py \
    --dataset $1 \
    --encoder resnet18 \
    --data_folder /data/datasets \
    --max_epochs 1000 \
    --gpus 1 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 5 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --name simclr-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --output_dim 256
