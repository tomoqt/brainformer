import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Optional
import argparse
from tqdm import tqdm

def generate_sine_wave(seq_len: int, frequency: float, phase: float = 0.0) -> torch.Tensor:
    """
    Generate a sine wave with given frequency and phase.
    
    Args:
        seq_len: Length of the sequence
        frequency: Frequency of the sine wave (cycles per sequence)
        phase: Phase offset in radians
        
    Returns:
        torch.Tensor: A sine wave tensor of shape [seq_len]
    """
    t = torch.linspace(0, 2 * np.pi * frequency, seq_len)
    return torch.sin(t + phase)

def scale_to_range(tensor: torch.Tensor, min_val: float = 0.0, max_val: float = 255.0) -> torch.Tensor:
    """
    Scale the tensor to a range between min_val and max_val.
    
    Args:
        tensor: Input tensor
        min_val: Minimum value of the range
        max_val: Maximum value of the range
        
    Returns:
        torch.Tensor: Scaled tensor
    """
    min_tensor = tensor.min()
    max_tensor = tensor.max()
    
    # Avoid division by zero
    if max_tensor == min_tensor:
        return torch.ones_like(tensor) * ((max_val + min_val) / 2)
    
    scaled = (tensor - min_tensor) / (max_tensor - min_tensor)
    return scaled * (max_val - min_val) + min_val

def add_noise(tensor: torch.Tensor, noise_level: float = 0.05) -> torch.Tensor:
    """
    Add Gaussian noise to a tensor.
    
    Args:
        tensor: Input tensor
        noise_level: Standard deviation of the noise relative to the tensor range
        
    Returns:
        torch.Tensor: Noisy tensor
    """
    tensor_range = tensor.max() - tensor.min()
    noise = torch.randn_like(tensor) * (noise_level * tensor_range)
    noisy_tensor = tensor + noise
    return noisy_tensor

def generate_timeseries_dataset(
    num_samples: int,
    seq_len: int,
    input_channels: int,
    output_channels: int,
    min_freq: float = 0.5,
    max_freq: float = 5.0,
    noise_level: float = 0.05,
    k_combinations: int = 2,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic multi-level time series dataset.
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Length of each sequence
        input_channels: Number of input channels
        output_channels: Number of output channels
        min_freq: Minimum frequency of sine waves
        max_freq: Maximum frequency of sine waves
        noise_level: Noise level for output channels
        k_combinations: Number of input channels to combine for additional output channels
        seed: Random seed for reproducibility
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input and output tensors of shapes 
                                          [num_samples, input_channels, seq_len] and
                                          [num_samples, output_channels, seq_len]
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create empty tensors
    inputs = torch.zeros((num_samples, input_channels, seq_len))
    outputs = torch.zeros((num_samples, output_channels, seq_len))
    
    # Generate input data
    for i in tqdm(range(num_samples), desc="Generating input data", ncols=100):
        for c in range(input_channels):
            # Random frequency for this channel
            freq = torch.rand(1).item() * (max_freq - min_freq) + min_freq
            # Random phase for this channel
            phase = torch.rand(1).item() * 2 * np.pi
            
            # Generate sine wave
            sine_wave = generate_sine_wave(seq_len, freq, phase)
            # Scale to [0, 255]
            scaled_sine = scale_to_range(sine_wave, 0.0, 255.0)
            inputs[i, c] = scaled_sine
    
    # Generate output data
    min_channels = min(input_channels, output_channels)
    
    # First, copy and add noise to the common channels
    for i in tqdm(range(num_samples), desc="Generating base output channels", ncols=100):
        for c in range(min_channels):
            outputs[i, c] = add_noise(inputs[i, c], noise_level)
            # Ensure values stay in [0, 255] range
            outputs[i, c] = torch.clamp(outputs[i, c], 0.0, 255.0)
    
    # If output has more channels, create linear combinations
    if output_channels > input_channels:
        additional_channels = output_channels - input_channels
        # Create a tqdm progress bar for additional channels generation
        with tqdm(total=num_samples * additional_channels, desc="Generating additional output channels", ncols=100) as pbar:
            for i in range(num_samples):
                for c in range(input_channels, output_channels):
                    # Select k random input channels to combine
                    indices = torch.randperm(input_channels)[:k_combinations]
                    weights = torch.rand(k_combinations)  # Random weights
                    weights = weights / weights.sum()  # Normalize weights to sum to 1
                    
                    # Linear combination
                    for j, idx in enumerate(indices):
                        outputs[i, c] += weights[j] * inputs[i, idx]
                    
                    # Add some noise
                    outputs[i, c] = add_noise(outputs[i, c], noise_level)
                    # Ensure values stay in [0, 255] range
                    outputs[i, c] = torch.clamp(outputs[i, c], 0.0, 255.0)
                    
                    # Update progress bar
                    pbar.update(1)
    
    return inputs, outputs

def save_dataset(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    save_dir: str,
    prefix: str = "dataset",
    k_combinations: int = 2
) -> None:
    """
    Save the generated dataset to files.
    
    Args:
        inputs: Input tensor
        outputs: Output tensor
        save_dir: Directory to save the dataset
        prefix: Prefix for the filenames
        k_combinations: Number of channels combined for additional outputs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save tensors with progress tracking
    with tqdm(total=2, desc="Saving tensors", ncols=100) as save_progress:
        torch.save(inputs, os.path.join(save_dir, f"{prefix}_inputs.pt"))
        save_progress.update(1)
        torch.save(outputs, os.path.join(save_dir, f"{prefix}_outputs.pt"))
        save_progress.update(1)
    
    print("Creating visualization sample...")
    # Save a sample plot for visualization
    sample_idx = np.random.randint(0, inputs.shape[0])
    
    plt.figure(figsize=(15, 10))
    
    # Plot input channels
    for c in range(inputs.shape[1]):
        plt.subplot(2, max(inputs.shape[1], outputs.shape[1]), c + 1)
        plt.plot(inputs[sample_idx, c].numpy())
        plt.title(f"Input Channel {c+1}")
        plt.ylim(0, 255)
    
    # Plot output channels
    for c in range(outputs.shape[1]):
        plt.subplot(2, max(inputs.shape[1], outputs.shape[1]), 
                   outputs.shape[1] + c + 1)
        plt.plot(outputs[sample_idx, c].numpy())
        plt.title(f"Output Channel {c+1}")
        plt.ylim(0, 255)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_sample_plot.png"))
    
    # Save metadata
    with open(os.path.join(save_dir, f"{prefix}_metadata.txt"), "w") as f:
        f.write(f"Input shape: {inputs.shape}\n")
        f.write(f"Output shape: {outputs.shape}\n")
        if outputs.shape[1] > inputs.shape[1]:
            f.write(f"Additional output channels are linear combinations of {k_combinations} input channels with added noise.\n")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic time series data")
    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to generate")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--input_channels", type=int, default=70, help="Number of input channels")
    parser.add_argument("--output_channels", type=int, default=300, help="Number of output channels")
    parser.add_argument("--min_freq", type=float, default=0.5, help="Minimum sine wave frequency")
    parser.add_argument("--max_freq", type=float, default=5.0, help="Maximum sine wave frequency")
    parser.add_argument("--noise_level", type=float, default=0.05, help="Noise level for output signals")
    parser.add_argument("--k_combinations", type=int, default=2, help="Number of channels to combine for additional outputs")
    parser.add_argument("--save_dir", type=str, default="./", help="Directory to save dataset")
    parser.add_argument("--prefix", type=str, default="dataset", help="Prefix for saved files")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create progress tracking for overall process
    with tqdm(total=3, desc="Overall progress", position=0, ncols=100) as overall_progress:
        # Generate the dataset
        print(f"Generating dataset with {args.num_samples} samples...")
        inputs, outputs = generate_timeseries_dataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            input_channels=args.input_channels,
            output_channels=args.output_channels,
            min_freq=args.min_freq,
            max_freq=args.max_freq,
            noise_level=args.noise_level,
            k_combinations=args.k_combinations,
            seed=args.seed
        )
        overall_progress.update(1)
        
        # Save the dataset
        print(f"Saving dataset to {args.save_dir}...")
        save_dataset(
            inputs=inputs, 
            outputs=outputs, 
            save_dir=args.save_dir, 
            prefix=args.prefix,
            k_combinations=args.k_combinations
        )
        overall_progress.update(1)
        
        # Complete
        print("Dataset generation complete!")
        print(f"Input shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        overall_progress.update(1)

if __name__ == "__main__":
    main() 