# Synthetic Time Series Data Generator

This module provides a tool for generating synthetic multi-level time series data in PyTorch tensor format. The data consists of input and output pairs, where inputs are sine waves of various frequencies, and outputs are either noised versions of the inputs or linear combinations of the inputs.

## Features

- Generate time series data with configurable sequence length (default: 256)
- Control the number of input and output channels
- Values scaled between 0 and 255 for each channel
- Output channels can be:
  - Noised versions of input channels
  - Linear combinations of input channels (when output_channels > input_channels)
- Visualize sample data with included plotting functionality
- Save generated data as PyTorch tensors

## Usage

```bash
python generate_synthetic_timeseries.py [options]
```

### Command-line Options

- `--num_samples`: Number of samples to generate (default: 1000)
- `--seq_len`: Sequence length (default: 256)
- `--input_channels`: Number of input channels (default: 3)
- `--output_channels`: Number of output channels (default: 5)
- `--min_freq`: Minimum sine wave frequency (default: 0.5)
- `--max_freq`: Maximum sine wave frequency (default: 5.0)
- `--noise_level`: Noise level for output signals (default: 0.05)
- `--k_combinations`: Number of channels to combine for additional outputs (default: 2)
- `--save_dir`: Directory to save dataset (default: "./")
- `--prefix`: Prefix for saved files (default: "dataset")
- `--seed`: Random seed for reproducibility (default: None)

### Example

```bash
# Generate a dataset with 500 samples, 4 input channels, and 6 output channels
python generate_synthetic_timeseries.py --num_samples 500 --input_channels 4 --output_channels 6 --save_dir ./my_dataset
```

## Output Files

For a dataset with prefix "dataset", the following files will be generated:

- `dataset_inputs.pt`: PyTorch tensor containing input data
- `dataset_outputs.pt`: PyTorch tensor containing output data
- `dataset_sample_plot.png`: Visualization of a random sample
- `dataset_metadata.txt`: Information about the dataset

## Data Format

- Input tensor shape: `[num_samples, input_channels, seq_len]`
- Output tensor shape: `[num_samples, output_channels, seq_len]`
- All values are scaled to be between 0 and 255

## How It Works

1. For each input channel, a sine wave with random frequency and phase is generated
2. For the first `min(input_channels, output_channels)` output channels, a noised version of the corresponding input channel is created
3. If `output_channels > input_channels`, additional output channels are created as linear combinations of randomly selected input channels (with added noise)

## Requirements

- PyTorch
- NumPy
- Matplotlib 