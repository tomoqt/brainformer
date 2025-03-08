#!/bin/bash

# Exit on error
set -e

# Print commands before executing them
set -x

# Create directories if they don't exist
mkdir -p checkpoints

# Run the training script with the specified parameters
python train.py \
  --data_dir eeg2meg_data \
  --input_channels 70 \
  --output_channels 306 \
  --seq_len 771 \
  --batch_size 16 \
  --epochs 100 \
  --save_dir checkpoints \
  --save_every 10 \
  --log_every 10 \
  --split_data \
  --val_ratio 0.2 \
  --input_file eeg2meg_inputs.pt \
  --output_file eeg2meg_outputs.pt \
  --wandb_watch=all \

echo "EEG2MEG training completed successfully!" 