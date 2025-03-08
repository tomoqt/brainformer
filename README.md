# BrainFormer

BrainFormer is a neural architecture that combines convolutional encoders/decoders with transformer-based models for time series modeling. It's designed for processing temporal signals with varying levels of complexity.

## Architecture

The BrainFormer model consists of three main components:

1. **Conv1DEncoder**: Processes input signals using 1D convolutional layers
2. **SimplifiedDecoder (Transformer)**: Processes encoded representations using a transformer architecture
3. **Conv1DDecoder**: Decodes transformer outputs to the desired output shape

## Requirements

Install the required dependencies:

```
pip install -r requirements.txt
```

## Data Preparation

Before training, you need to prepare your dataset. The model expects data in the following format:
- Four PyTorch tensor files in the specified data directory:
  - `train_inputs.pt`: Training input data [batch_size, seq_len, input_channels]
  - `train_outputs.pt`: Training output data [batch_size, seq_len, output_channels]
  - `val_inputs.pt`: Validation input data [batch_size, seq_len, input_channels]
  - `val_outputs.pt`: Validation output data [batch_size, seq_len, output_channels]

You can generate synthetic data using the script in `data/synthetic/generate_synthetic_timeseries.py`, which can be run separately.

## Training

The `train.py` script provides a framework for training the BrainFormer model on pre-generated data.

### Basic Usage

To train the BrainFormer model with default settings:

```bash
python train.py
```

By default, the script will look for data in the `data/synthetic/processed` directory.

### Custom Configuration

You can customize the training process using command-line arguments:

```bash
python train.py \
  --seq_len 256 \
  --input_channels 70 \
  --output_channels 300 \
  --batch_size 64 \
  --epochs 100 \
  --lr 3e-4 \
  --data_dir path/to/your/data
```

### Available Arguments

#### Data Parameters
- `--seq_len`: Sequence length (default: 256)
- `--input_channels`: Number of input channels (default: 70)
- `--output_channels`: Number of output channels (default: 300)
- `--data_dir`: Directory where data is located (default: 'data/synthetic/processed')

#### Model Parameters
- `--encoder_hidden_dims`: Comma-separated list of encoder hidden dimensions (default: '64,128,256,512')
- `--decoder_hidden_dims`: Comma-separated list of decoder hidden dimensions (default: '512,256,128,64')
- `--n_layer`: Number of transformer layers (default: 8)
- `--n_head`: Number of attention heads (default: 8)
- `--n_embd`: Embedding dimension (default: 512)
- `--dropout`: Dropout rate (default: 0.1)

#### Training Parameters
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 3e-4)
- `--weight_decay`: Weight decay (default: 0.01)
- `--beta1`: Beta1 for Adam optimizer (default: 0.9)
- `--beta2`: Beta2 for Adam optimizer (default: 0.99)
- `--epochs`: Number of training epochs (default: 100)
- `--save_dir`: Directory to save model checkpoints (default: 'checkpoints')
- `--save_every`: Save checkpoint every N epochs (default: 10)
- `--log_every`: Log training metrics every N batches (default: 10)

## Model Checkpoints

The training script saves checkpoints at regular intervals and whenever a new best model (based on validation loss) is found. Checkpoints are saved in the specified `--save_dir` directory.

## Synthetic Data Generation

For information on generating synthetic data, refer to the functions in `data/synthetic/generate_synthetic_timeseries.py`. This file contains tools to create synthetic time series data consisting of combinations of sine waves with varying frequencies, phases, and noise levels. 