python3 ../../../main_contrastive.py \
    --dataset $1 \
    --encoder resnet18 \
    --data_folder ./datasets \
    --max_epochs 1000 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 5 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --asymmetric_augmentations \
    --name byol-$1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --method byol \
    --output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier
