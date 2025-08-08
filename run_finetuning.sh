#!/bin/bash

# Fine-tuning script for CoTracker on custom dataset
# Optimized for multiple videos

echo "Starting CoTracker fine-tuning on custom dataset..."

# Create checkpoint directory if it doesn't exist
mkdir -p ./checkpoints/my_finetuned_model

# Run training with batch size 1 for CoTracker3 compatibility
python train_custom_dataset.py \
    --model_name cotracker_three \
    --restore_ckpt ./checkpoints/michael_finetune_model/model_cotracker_three_000168.pth \
    --ckpt_path ./checkpoints/michael_finetune_model \
    --video_dir ./data/micheal_train \
    --batch_size 1 \
    --num_workers 2 \
    --lr 0.0005 \
    --wdecay 0.00001 \
    --num_steps 50000 \
    --sequence_len 16 \
    --crop_size 384 512 \
    --traj_per_sample 512 \
    --train_iters 4 \
    --save_every_n_epoch 2 \
    --evaluate_every_n_epoch 2 \
    --save_freq 200 \
    --eval_datasets tapvid_davis_first \
    --dataset_root ./data/eval \
    --limit_samples 5 \
    --mixed_precision

echo "Training completed!"
echo "Your fine-tuned model is saved in: ./checkpoints/my_finetuned_model" 