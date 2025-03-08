#!/bin/bash

# Exit on error
set -e

# Print commands before executing them
set -x

# Create directories if they don't exist
mkdir -p checkpoints

# Run the training script with the specified parameters
python train.py \
  --data_dir data/synthetic/synthetic_dataset \
  --input_channels 70 \
  --output_channels 300 \
  --seq_len 256 \
  --batch_size 32 \
  --epochs 100 \
  --save_dir checkpoints \
  --save_every 10 \
  --log_every 10 \
  --split_data \
  --val_ratio 0.2 \
  --input_file dataset_inputs.pt \
  --output_file dataset_outputs.pt \
  --wandb_watch=all

echo "Training completed successfully!" 