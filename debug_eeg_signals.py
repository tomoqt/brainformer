#!/usr/bin/env python3
"""
Debug script for analyzing EEG signal shapes in the processed EEG2MEG dataset.
This script helps diagnose and fix the sigmoid-like pattern observed in the EEG data.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy import signal
from typing import Tuple, Optional, List


def load_dataset(data_dir: str, prefix: str = "eeg2meg_data") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the pre-processed EEG2MEG dataset.
    
    Args:
        data_dir: Directory where the dataset is saved
        prefix: Prefix for the filenames
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input (EEG) and output (MEG) tensors
    """
    inputs_path = os.path.join(data_dir, f"{prefix}_inputs.pt")
    outputs_path = os.path.join(data_dir, f"{prefix}_outputs.pt")
    
    if not os.path.exists(inputs_path) or not os.path.exists(outputs_path):
        raise FileNotFoundError(f"Dataset files not found in {data_dir}. Make sure the paths are correct.")
    
    print(f"Loading inputs from {inputs_path}")
    inputs = torch.load(inputs_path)
    
    print(f"Loading outputs from {outputs_path}")
    outputs = torch.load(outputs_path)
    
    print(f"Loaded dataset: Inputs {inputs.shape}, Outputs {outputs.shape}")
    return inputs, outputs


def transpose_signals(inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transpose the signals to match the expected format for BrainFormer model.
    
    The BrainFormer model expects inputs in shape [batch_size, seq_len, channels]
    but our data is in shape [batch_size, channels, seq_len].
    
    Args:
        inputs: Input tensor of shape [batch_size, channels, seq_len]
        outputs: Output tensor of shape [batch_size, channels, seq_len]
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Transposed input and output tensors
                                          with shape [batch_size, seq_len, channels]
    """
    print(f"Transposing signals from:")
    print(f"  - Inputs: {inputs.shape} (batch_size, channels, seq_len)")
    print(f"  - Outputs: {outputs.shape} (batch_size, channels, seq_len)")
    
    transposed_inputs = inputs.transpose(1, 2)
    transposed_outputs = outputs.transpose(1, 2)
    
    print(f"To:")
    print(f"  - Inputs: {transposed_inputs.shape} (batch_size, seq_len, channels)")
    print(f"  - Outputs: {transposed_outputs.shape} (batch_size, seq_len, channels)")
    
    return transposed_inputs, transposed_outputs


def save_transposed_dataset(inputs: torch.Tensor, outputs: torch.Tensor, output_dir: str, prefix: str = "eeg2meg_transposed") -> None:
    """
    Save the transposed dataset to disk.
    
    Args:
        inputs: Input tensor (transposed)
        outputs: Output tensor (transposed)
        output_dir: Directory to save the dataset
        prefix: Prefix for the filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    inputs_path = os.path.join(output_dir, f"{prefix}_inputs.pt")
    outputs_path = os.path.join(output_dir, f"{prefix}_outputs.pt")
    
    print(f"Saving transposed inputs to {inputs_path}")
    torch.save(inputs, inputs_path)
    
    print(f"Saving transposed outputs to {outputs_path}")
    torch.save(outputs, outputs_path)
    
    print(f"Transposed dataset saved successfully!")


def analyze_signal_properties(
    signal_data: torch.Tensor,
    sample_indices: Optional[List[int]] = None,
    channel_indices: Optional[List[int]] = None
) -> None:
    """
    Analyze and print key properties of the signal data.
    
    Args:
        signal_data: The signal data tensor [batch, channels, sequence]
        sample_indices: Indices of samples to analyze (default: first 3)
        channel_indices: Indices of channels to analyze (default: first 5)
    """
    if sample_indices is None:
        sample_indices = list(range(min(3, signal_data.shape[0])))
    
    if channel_indices is None:
        channel_indices = list(range(min(5, signal_data.shape[1])))
    
    print("\n=== Signal Properties Analysis ===")
    
    # Global statistics
    print(f"Global stats - Shape: {signal_data.shape}")
    print(f"Global stats - Min: {signal_data.min().item():.4f}, Max: {signal_data.max().item():.4f}")
    print(f"Global stats - Mean: {signal_data.mean().item():.4f}, Std: {signal_data.std().item():.4f}")
    
    # Sample & channel-specific statistics
    for sample_idx in sample_indices:
        for channel_idx in channel_indices:
            channel_data = signal_data[sample_idx, channel_idx]
            
            # Basic stats
            min_val = channel_data.min().item()
            max_val = channel_data.max().item()
            mean_val = channel_data.mean().item()
            std_val = channel_data.std().item()
            
            # Find trends (look for monotonic increases/decreases)
            diffs = np.diff(channel_data.numpy())
            increasing_trend = np.sum(diffs > 0) / len(diffs)
            
            # Check for sigmoid shape
            # A sigmoid would have mostly positive diffs in first half, negative in second half
            half_point = len(diffs) // 2
            first_half_increasing = np.sum(diffs[:half_point] > 0) / half_point
            second_half_decreasing = np.sum(diffs[half_point:] < 0) / (len(diffs) - half_point)
            
            # Detect DC offset or drift
            detrended = signal.detrend(channel_data.numpy())
            drift_magnitude = np.abs(channel_data.numpy() - detrended).mean()
            
            print(f"\nSample {sample_idx}, Channel {channel_idx}:")
            print(f"  Range: [{min_val:.4f}, {max_val:.4f}], Mean: {mean_val:.4f}, Std: {std_val:.4f}")
            print(f"  Trend analysis - Increasing trend: {increasing_trend:.2%}")
            print(f"  Sigmoid check - First half increasing: {first_half_increasing:.2%}, Second half decreasing: {second_half_decreasing:.2%}")
            print(f"  Drift magnitude: {drift_magnitude:.4f}")
            
            # Classify the shape
            if first_half_increasing > 0.7 and second_half_decreasing > 0.7:
                print("  Shape classification: Strong sigmoid-like pattern detected")
            elif drift_magnitude > 0.2 * std_val:
                print("  Shape classification: Significant drift detected")
            elif increasing_trend > 0.7 or increasing_trend < 0.3:
                print("  Shape classification: Strong directional trend detected")
            else:
                print("  Shape classification: No clear pathological pattern")


def visualize_signals(
    eeg_data: torch.Tensor,
    meg_data: torch.Tensor,
    sample_idx: int = 0,
    channels: List[int] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Create comprehensive visualizations of the signals.
    
    Args:
        eeg_data: The EEG data tensor [batch, channels, sequence]
        meg_data: The MEG data tensor [batch, channels, sequence]
        sample_idx: Index of the sample to visualize
        channels: List of channel indices to visualize (default: first 5)
        save_dir: Directory to save visualizations (if None, displays instead)
    """
    if channels is None:
        channels = list(range(min(5, eeg_data.shape[1])))
    
    # Convert tensors to numpy for plotting
    eeg_sample = eeg_data[sample_idx].numpy()
    meg_sample = meg_data[sample_idx].numpy()
    
    # Prepare x-axis for time series
    time_steps = np.arange(eeg_sample.shape[1])
    
    plt.figure(figsize=(15, 18))
    
    # 1. Raw EEG signals
    plt.subplot(4, 1, 1)
    plt.title(f"Raw EEG Signals (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        plt.plot(time_steps, eeg_sample[ch], label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Detrended EEG signals
    plt.subplot(4, 1, 2)
    plt.title(f"Detrended EEG Signals (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        plt.plot(time_steps, signal.detrend(eeg_sample[ch]), label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Highpass filtered EEG signals (1Hz cutoff assuming 100Hz sampling)
    plt.subplot(4, 1, 3)
    plt.title(f"Highpass Filtered EEG Signals (1Hz cutoff, Sample {sample_idx})", fontsize=14)
    for ch in channels:
        # Simple first-difference as high-pass filter
        filtered = np.zeros_like(eeg_sample[ch])
        filtered[1:] = eeg_sample[ch][1:] - eeg_sample[ch][:-1]
        # Rescale
        if np.std(filtered) > 0:
            scale = np.std(eeg_sample[ch]) / np.std(filtered)
            filtered *= scale
        plt.plot(time_steps, filtered, label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Spectrogram of first channel
    plt.subplot(4, 1, 4)
    plt.title(f"EEG Channel {channels[0]+1} Spectrogram", fontsize=14)
    fs = 100  # Assuming 100Hz sampling rate
    f, t, Sxx = signal.spectrogram(eeg_sample[channels[0]], fs=fs)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"eeg_signal_analysis_sample_{sample_idx}.png"))
        print(f"Saved signal visualization to {os.path.join(save_dir, f'eeg_signal_analysis_sample_{sample_idx}.png')}")
    else:
        plt.show()
    
    # Frequency domain analysis
    plt.figure(figsize=(15, 10))
    
    # 1. EEG Power Spectral Density
    plt.subplot(2, 1, 1)
    plt.title(f"EEG Power Spectral Density (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        f, Pxx = signal.welch(eeg_sample[ch], fs=fs, nperseg=256)
        plt.semilogy(f, Pxx, label=f"Ch {ch+1}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power/Frequency [V^2/Hz]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. MEG Power Spectral Density
    plt.subplot(2, 1, 2)
    plt.title(f"MEG Power Spectral Density (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        if ch < meg_sample.shape[0]:  # Ensure channel exists in MEG data
            f, Pxx = signal.welch(meg_sample[ch], fs=fs, nperseg=256)
            plt.semilogy(f, Pxx, label=f"Ch {ch+1}")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power/Frequency [V^2/Hz]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"eeg_frequency_analysis_sample_{sample_idx}.png"))
        print(f"Saved frequency analysis to {os.path.join(save_dir, f'eeg_frequency_analysis_sample_{sample_idx}.png')}")
    else:
        plt.show()


def visualize_transposed_signals(
    orig_inputs: torch.Tensor,
    orig_outputs: torch.Tensor,
    transposed_inputs: torch.Tensor,
    transposed_outputs: torch.Tensor,
    sample_idx: int = 0,
    channels: List[int] = None,
    time_steps: List[int] = None,
    save_dir: Optional[str] = None
) -> None:
    """
    Create visualizations comparing original and transposed signals.
    
    Args:
        orig_inputs: Original input tensor [batch, channels, sequence]
        orig_outputs: Original output tensor [batch, channels, sequence]
        transposed_inputs: Transposed input tensor [batch, sequence, channels]
        transposed_outputs: Transposed output tensor [batch, sequence, channels]
        sample_idx: Index of the sample to visualize
        channels: List of channel indices to visualize (default: first 5)
        time_steps: List of time steps to visualize (default: first 100)
        save_dir: Directory to save visualizations (if None, displays instead)
    """
    if channels is None:
        channels = list(range(min(5, orig_inputs.shape[1])))
    
    if time_steps is None:
        time_steps = list(range(min(100, orig_inputs.shape[2])))
    
    # Convert tensors to numpy for plotting
    orig_input_sample = orig_inputs[sample_idx].numpy()
    orig_output_sample = orig_outputs[sample_idx].numpy()
    
    # Get transposed data WITHOUT transposing it back
    trans_input_sample = transposed_inputs[sample_idx].numpy()
    trans_output_sample = transposed_outputs[sample_idx].numpy()
    
    # Create time axes
    orig_time_axis = np.arange(len(time_steps))
    
    # PLOT 1: Channel perspective (comparing same channels in original vs transposed)
    plt.figure(figsize=(20, 16))
    plt.suptitle("COMPARING SAME CHANNELS: Original vs Transposed", fontsize=16)
    
    # 1. Original Input (EEG) Signals - by channel
    plt.subplot(4, 1, 1)
    plt.title(f"Original EEG Signals [batch, channels, sequence] (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        plt.plot(orig_time_axis, orig_input_sample[ch, time_steps], label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Transposed Input (EEG) Signals - selecting same channels
    plt.subplot(4, 1, 2)
    plt.title(f"Transposed EEG Signals [batch, sequence, channels] (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        # Select data from transposed tensor for the same channel
        # In transposed data, sequence is dim 0, channels is dim 1
        plt.plot(orig_time_axis, trans_input_sample[time_steps, ch], label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Original Output (MEG) Signals - by channel
    plt.subplot(4, 1, 3)
    plt.title(f"Original MEG Signals [batch, channels, sequence] (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        if ch < orig_output_sample.shape[0]:  # MEG might have different channel count
            plt.plot(orig_time_axis, orig_output_sample[ch, time_steps], label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Transposed Output (MEG) Signals - selecting same channels
    plt.subplot(4, 1, 4)
    plt.title(f"Transposed MEG Signals [batch, sequence, channels] (Sample {sample_idx})", fontsize=14)
    for ch in channels:
        if ch < trans_output_sample.shape[1]:  # MEG might have different channel count
            plt.plot(orig_time_axis, trans_output_sample[time_steps, ch], label=f"Ch {ch+1}")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or display the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"transposed_comparison_by_channel_sample_{sample_idx}.png"))
        print(f"Saved transposed signal comparison by channel to {save_dir}/transposed_comparison_by_channel_sample_{sample_idx}.png")
    else:
        plt.show()
    
    # PLOT 2: Structure visualization (showing the actual data layout difference)
    plt.figure(figsize=(20, 16))
    plt.suptitle("DATA STRUCTURE VISUALIZATION: Original vs Transposed", fontsize=16)
    
    # 1. Original EEG structure - show first few channels and time steps as a heatmap
    plt.subplot(2, 2, 1)
    plt.title(f"Original EEG Structure [channels, sequence]", fontsize=14)
    display_channels = min(20, orig_input_sample.shape[0])
    display_steps = min(100, orig_input_sample.shape[1])
    plt.imshow(orig_input_sample[:display_channels, :display_steps], aspect='auto', cmap='viridis')
    plt.xlabel("Sequence (time steps)")
    plt.ylabel("Channels")
    plt.colorbar(label='Amplitude')
    
    # 2. Transposed EEG structure - as a heatmap
    plt.subplot(2, 2, 2)
    plt.title(f"Transposed EEG Structure [sequence, channels]", fontsize=14)
    display_steps_t = min(100, trans_input_sample.shape[0])
    display_channels_t = min(20, trans_input_sample.shape[1])
    plt.imshow(trans_input_sample[:display_steps_t, :display_channels_t], aspect='auto', cmap='viridis')
    plt.xlabel("Channels")
    plt.ylabel("Sequence (time steps)")
    plt.colorbar(label='Amplitude')
    
    # 3. Original MEG structure
    plt.subplot(2, 2, 3)
    plt.title(f"Original MEG Structure [channels, sequence]", fontsize=14)
    display_channels_meg = min(20, orig_output_sample.shape[0])
    display_steps_meg = min(100, orig_output_sample.shape[1])
    plt.imshow(orig_output_sample[:display_channels_meg, :display_steps_meg], aspect='auto', cmap='viridis')
    plt.xlabel("Sequence (time steps)")
    plt.ylabel("Channels")
    plt.colorbar(label='Amplitude')
    
    # 4. Transposed MEG structure
    plt.subplot(2, 2, 4)
    plt.title(f"Transposed MEG Structure [sequence, channels]", fontsize=14)
    display_steps_meg_t = min(100, trans_output_sample.shape[0])
    display_channels_meg_t = min(20, trans_output_sample.shape[1])
    plt.imshow(trans_output_sample[:display_steps_meg_t, :display_channels_meg_t], aspect='auto', cmap='viridis')
    plt.xlabel("Channels")
    plt.ylabel("Sequence (time steps)")
    plt.colorbar(label='Amplitude')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or display the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"transposed_structure_visualization_sample_{sample_idx}.png"))
        print(f"Saved data structure visualization to {save_dir}/transposed_structure_visualization_sample_{sample_idx}.png")
    else:
        plt.show()


def visualize_all_channels_heatmap(
    eeg_data: torch.Tensor,
    meg_data: torch.Tensor,
    sample_idx: int = 0,
    max_channels: int = 100,  # Limit number of channels to display
    time_steps: Optional[List[int]] = None,
    is_transposed: bool = False,
    save_dir: Optional[str] = None
) -> None:
    """
    Create heatmap visualizations of all EEG and MEG channels.
    
    Args:
        eeg_data: EEG data tensor [batch, channels, sequence] or [batch, sequence, channels] if transposed
        meg_data: MEG data tensor [batch, channels, sequence] or [batch, sequence, channels] if transposed
        sample_idx: Index of the sample to visualize
        max_channels: Maximum number of channels to display
        time_steps: List of time steps to visualize (default: all)
        is_transposed: Whether the data is already in transposed format [batch, sequence, channels]
        save_dir: Directory to save visualizations (if None, displays instead)
    """
    # Extract sample
    eeg_sample = eeg_data[sample_idx]
    meg_sample = meg_data[sample_idx]
    
    # Convert to numpy
    eeg_sample = eeg_sample.numpy()
    meg_sample = meg_sample.numpy()
    
    if is_transposed:
        # If transposed, channels are the second dimension
        eeg_channels = min(max_channels, eeg_sample.shape[1])
        meg_channels = min(max_channels, meg_sample.shape[1])
        
        # For heatmap, we want channels on Y-axis, time on X-axis, so we transpose
        eeg_heatmap_data = eeg_sample[:, :eeg_channels].T
        meg_heatmap_data = meg_sample[:, :meg_channels].T
        
        time_dimension = eeg_sample.shape[0]
    else:
        # If not transposed, channels are the first dimension
        eeg_channels = min(max_channels, eeg_sample.shape[0])
        meg_channels = min(max_channels, eeg_sample.shape[0])
        
        # Already in the right format for heatmap (channels on Y-axis, time on X-axis)
        eeg_heatmap_data = eeg_sample[:eeg_channels]
        meg_heatmap_data = meg_sample[:meg_channels]
        
        time_dimension = eeg_sample.shape[1]
    
    # Limit time steps if specified
    if time_steps is not None:
        if is_transposed:
            eeg_heatmap_data = eeg_heatmap_data[:, time_steps]
            meg_heatmap_data = meg_heatmap_data[:, time_steps]
            time_dimension = len(time_steps)
        else:
            eeg_heatmap_data = eeg_heatmap_data[:, time_steps]
            meg_heatmap_data = meg_heatmap_data[:, time_steps]
            time_dimension = len(time_steps)
    
    # Create figure for EEG and MEG heatmaps
    fig, axs = plt.subplots(2, 1, figsize=(20, 16))
    plt.suptitle(f"All Channels Heatmap Visualization (Sample {sample_idx})", fontsize=16)
    
    # Calculate aspect ratio - wider than tall for better visibility
    aspect_ratio = time_dimension / (eeg_channels * 4)
    
    # Plot EEG heatmap
    im1 = axs[0].imshow(eeg_heatmap_data, aspect=aspect_ratio, cmap='viridis', interpolation='none')
    axs[0].set_title(f"EEG Channels Heatmap ({'Transposed' if is_transposed else 'Original'} Format)", fontsize=14)
    axs[0].set_ylabel("Channel Index")
    axs[0].set_xlabel("Time Steps")
    fig.colorbar(im1, ax=axs[0], label="Amplitude")
    
    # Plot MEG heatmap
    im2 = axs[1].imshow(meg_heatmap_data, aspect=aspect_ratio, cmap='plasma', interpolation='none')
    axs[1].set_title(f"MEG Channels Heatmap ({'Transposed' if is_transposed else 'Original'} Format)", fontsize=14)
    axs[1].set_ylabel("Channel Index")
    axs[1].set_xlabel("Time Steps")
    fig.colorbar(im2, ax=axs[1], label="Amplitude")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save or display the plot
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data_format = "transposed" if is_transposed else "original"
        plt.savefig(os.path.join(save_dir, f"all_channels_heatmap_{data_format}_sample_{sample_idx}.png"))
        print(f"Saved all channels heatmap to {save_dir}/all_channels_heatmap_{data_format}_sample_{sample_idx}.png")
    else:
        plt.show()


def visualize_multiple_samples_heatmap(
    eeg_data: torch.Tensor,
    meg_data: torch.Tensor,
    sample_indices: List[int],
    max_channels: int = 100,
    time_steps: Optional[List[int]] = None,
    is_transposed: bool = False,
    save_dir: Optional[str] = None
) -> None:
    """
    Create heatmap visualizations for multiple samples.
    
    Args:
        eeg_data: EEG data tensor [batch, channels, sequence] or [batch, sequence, channels] if transposed
        meg_data: MEG data tensor [batch, channels, sequence] or [batch, sequence, channels] if transposed
        sample_indices: List of sample indices to visualize
        max_channels: Maximum number of channels to display
        time_steps: List of time steps to visualize (default: all)
        is_transposed: Whether the data is already in transposed format [batch, sequence, channels]
        save_dir: Directory to save visualizations (if None, displays instead)
    """
    num_samples = len(sample_indices)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    rows = grid_size
    cols = grid_size
    
    # Create figures for EEG and MEG
    fig_eeg, axs_eeg = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    fig_meg, axs_meg = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    
    fig_eeg.suptitle(f"EEG Channels Across Multiple Samples ({'Transposed' if is_transposed else 'Original'} Format)", fontsize=16)
    fig_meg.suptitle(f"MEG Channels Across Multiple Samples ({'Transposed' if is_transposed else 'Original'} Format)", fontsize=16)
    
    # Ensure axes are iterable even if there's only one subplot
    if num_samples == 1:
        axs_eeg = np.array([[axs_eeg]])
        axs_meg = np.array([[axs_meg]])
    elif rows == 1 or cols == 1:
        axs_eeg = np.array([axs_eeg]).reshape(rows, cols)
        axs_meg = np.array([axs_meg]).reshape(rows, cols)
    
    # Flatten axes for easier iteration
    axs_eeg_flat = axs_eeg.flatten()
    axs_meg_flat = axs_meg.flatten()
    
    # Process each sample
    for i, sample_idx in enumerate(sample_indices):
        if i >= len(axs_eeg_flat):  # Safety check
            break
            
        # Extract sample
        eeg_sample = eeg_data[sample_idx].numpy()
        meg_sample = meg_data[sample_idx].numpy()
        
        if is_transposed:
            # If transposed, channels are the second dimension
            eeg_channels = min(max_channels, eeg_sample.shape[1])
            meg_channels = min(max_channels, meg_sample.shape[1])
            
            # For heatmap, we want channels on Y-axis, time on X-axis, so we transpose
            eeg_heatmap_data = eeg_sample[:, :eeg_channels].T
            meg_heatmap_data = meg_sample[:, :meg_channels].T
            
            time_dimension = eeg_sample.shape[0]
        else:
            # If not transposed, channels are the first dimension
            eeg_channels = min(max_channels, eeg_sample.shape[0])
            meg_channels = min(max_channels, meg_sample.shape[0])
            
            # Already in the right format for heatmap (channels on Y-axis, time on X-axis)
            eeg_heatmap_data = eeg_sample[:eeg_channels]
            meg_heatmap_data = meg_sample[:meg_channels]
            
            time_dimension = eeg_sample.shape[1]
        
        # Limit time steps if specified
        if time_steps is not None:
            if is_transposed:
                eeg_heatmap_data = eeg_heatmap_data[:, time_steps]
                meg_heatmap_data = meg_heatmap_data[:, time_steps]
                time_dimension = len(time_steps)
            else:
                eeg_heatmap_data = eeg_heatmap_data[:, time_steps]
                meg_heatmap_data = meg_heatmap_data[:, time_steps]
                time_dimension = len(time_steps)
        
        # Calculate aspect ratio - wider than tall for better visibility
        aspect_ratio = time_dimension / (eeg_channels * 4)
        
        # Plot EEG heatmap for this sample
        im_eeg = axs_eeg_flat[i].imshow(eeg_heatmap_data, aspect=aspect_ratio, cmap='viridis', interpolation='none')
        axs_eeg_flat[i].set_title(f"Sample {sample_idx}", fontsize=12)
        axs_eeg_flat[i].set_ylabel("Channel Index")
        axs_eeg_flat[i].set_xlabel("Time Steps")
        
        # Plot MEG heatmap for this sample
        im_meg = axs_meg_flat[i].imshow(meg_heatmap_data, aspect=aspect_ratio, cmap='plasma', interpolation='none')
        axs_meg_flat[i].set_title(f"Sample {sample_idx}", fontsize=12)
        axs_meg_flat[i].set_ylabel("Channel Index")
        axs_meg_flat[i].set_xlabel("Time Steps")
    
    # Hide unused subplots
    for i in range(num_samples, len(axs_eeg_flat)):
        axs_eeg_flat[i].axis('off')
        axs_meg_flat[i].axis('off')
    
    # Add colorbars for each figure
    fig_eeg.colorbar(im_eeg, ax=axs_eeg.ravel().tolist(), label="Amplitude")
    fig_meg.colorbar(im_meg, ax=axs_meg.ravel().tolist(), label="Amplitude")
    
    # Adjust layout
    fig_eeg.tight_layout()
    fig_meg.tight_layout()
    fig_eeg.subplots_adjust(top=0.92)
    fig_meg.subplots_adjust(top=0.92)
    
    # Save or display the plots
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        data_format = "transposed" if is_transposed else "original"
        sample_str = "_".join(map(str, sample_indices))
        
        fig_eeg.savefig(os.path.join(save_dir, f"multi_sample_eeg_heatmap_{data_format}_samples_{sample_str}.png"))
        fig_meg.savefig(os.path.join(save_dir, f"multi_sample_meg_heatmap_{data_format}_samples_{sample_str}.png"))
        
        print(f"Saved multi-sample EEG heatmap to {save_dir}/multi_sample_eeg_heatmap_{data_format}_samples_{sample_str}.png")
        print(f"Saved multi-sample MEG heatmap to {save_dir}/multi_sample_meg_heatmap_{data_format}_samples_{sample_str}.png")
    else:
        plt.show()


def apply_fixes_and_save(
    eeg_data: torch.Tensor,
    meg_data: torch.Tensor,
    output_dir: str,
    prefix: str = "eeg2meg_fixed",
    fix_method: str = "highpass"
) -> None:
    """
    Apply various fixes to the EEG data and save the results.
    
    Args:
        eeg_data: The EEG data tensor [batch, channels, sequence]
        meg_data: The MEG data tensor [batch, channels, sequence]
        output_dir: Directory to save the fixed dataset
        prefix: Prefix for the output files
        fix_method: Method to fix the EEG data ('normalize', 'detrend', 'highpass', 'all')
    """
    print(f"\nApplying fix method: {fix_method}")
    
    # Create a copy to avoid modifying the original data
    fixed_eeg = eeg_data.clone()
    
    # Apply the selected fix method
    if fix_method in ["normalize", "all"]:
        print("Applying z-score normalization...")
        for b in range(fixed_eeg.shape[0]):
            for c in range(fixed_eeg.shape[1]):
                channel = fixed_eeg[b, c]
                mean = channel.mean()
                std = channel.std()
                if std > 0:
                    fixed_eeg[b, c] = (channel - mean) / std
    
    if fix_method in ["detrend", "all"]:
        print("Applying detrending...")
        for b in range(fixed_eeg.shape[0]):
            for c in range(fixed_eeg.shape[1]):
                fixed_eeg[b, c] = torch.tensor(signal.detrend(fixed_eeg[b, c].numpy()), dtype=fixed_eeg.dtype)
    
    if fix_method in ["highpass", "all"]:
        print("Applying high-pass filtering...")
        for b in range(fixed_eeg.shape[0]):
            for c in range(fixed_eeg.shape[1]):
                # Simple first-difference as high-pass filter
                channel = fixed_eeg[b, c].numpy()
                filtered = np.zeros_like(channel)
                filtered[1:] = channel[1:] - channel[:-1]
                
                # Rescale to maintain similar amplitude characteristics
                if np.std(filtered) > 0:
                    scale = np.std(channel) / np.std(filtered)
                    filtered *= scale
                
                fixed_eeg[b, c] = torch.tensor(filtered, dtype=fixed_eeg.dtype)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the fixed EEG data and the original MEG data
    print(f"Saving fixed dataset to {output_dir}...")
    torch.save(fixed_eeg, os.path.join(output_dir, f"{prefix}_inputs.pt"))
    torch.save(meg_data, os.path.join(output_dir, f"{prefix}_outputs.pt"))
    print("Dataset saved successfully!")
    
    # Visualize a sample with the fix applied
    visualize_signals(
        fixed_eeg, meg_data, 
        sample_idx=0, 
        save_dir=output_dir
    )


def main():
    parser = argparse.ArgumentParser(description="Debug and fix EEG signal shape issues")
    parser.add_argument("--data_dir", type=str, default="./eeg2meg_data", help="Directory where the dataset is saved")
    parser.add_argument("--output_dir", type=str, default="./eeg2meg_fixed", help="Directory to save fixed dataset")
    parser.add_argument("--prefix", type=str, default="eeg2meg", help="Prefix for the dataset files")
    parser.add_argument("--fix_method", type=str, default="highpass", 
                        choices=["normalize", "detrend", "highpass", "all"],
                        help="Method to fix the EEG signals")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--multi_samples", type=int, nargs="+", default=None, help="Multiple sample indices to visualize together")
    parser.add_argument("--visualize_only", action="store_true", help="Only visualize without fixing")
    parser.add_argument("--transpose", action="store_true", help="Transpose signals for BrainFormer model")
    parser.add_argument("--transpose_only", action="store_true", help="Only transpose signals without fixing")
    parser.add_argument("--plot_transposed", action="store_true", help="Plot both original and transposed signals")
    parser.add_argument("--plot_heatmap", action="store_true", help="Plot all channels as 2D heatmaps")
    parser.add_argument("--max_channels", type=int, default=100, help="Maximum number of channels to display in heatmaps")
    parser.add_argument("--channels", type=int, nargs="+", default=None, help="Specific channels to visualize")
    parser.add_argument("--time_steps", type=int, nargs="+", default=None, help="Specific time steps to visualize")
    
    args = parser.parse_args()
    
    # Load the dataset
    inputs, outputs = load_dataset(args.data_dir, args.prefix)
    
    # Transpose signals if requested
    if args.transpose or args.transpose_only or args.plot_transposed:
        transposed_inputs, transposed_outputs = transpose_signals(inputs, outputs)
        
        # Visualize original vs transposed signals if requested
        if args.plot_transposed:
            visualize_transposed_signals(
                inputs, outputs,
                transposed_inputs, transposed_outputs,
                sample_idx=args.sample_idx,
                channels=args.channels,
                time_steps=args.time_steps,
                save_dir=args.output_dir if not args.visualize_only else None
            )
        
        # Visualize as heatmaps if requested
        if args.plot_heatmap:
            if args.multi_samples:
                # Plot multiple samples as heatmaps
                visualize_multiple_samples_heatmap(
                    inputs, outputs,
                    sample_indices=args.multi_samples,
                    max_channels=args.max_channels,
                    time_steps=args.time_steps,
                    is_transposed=False,
                    save_dir=args.output_dir if not args.visualize_only else None
                )
                
                visualize_multiple_samples_heatmap(
                    transposed_inputs, transposed_outputs,
                    sample_indices=args.multi_samples,
                    max_channels=args.max_channels,
                    time_steps=args.time_steps,
                    is_transposed=True,
                    save_dir=args.output_dir if not args.visualize_only else None
                )
            else:
                # Plot original data heatmap
                visualize_all_channels_heatmap(
                    inputs, outputs,
                    sample_idx=args.sample_idx,
                    max_channels=args.max_channels,
                    time_steps=args.time_steps,
                    is_transposed=False,
                    save_dir=args.output_dir if not args.visualize_only else None
                )
                
                # Plot transposed data heatmap
                visualize_all_channels_heatmap(
                    transposed_inputs, transposed_outputs,
                    sample_idx=args.sample_idx,
                    max_channels=args.max_channels,
                    time_steps=args.time_steps,
                    is_transposed=True,
                    save_dir=args.output_dir if not args.visualize_only else None
                )
        
        if args.transpose_only:
            # Save transposed dataset and exit
            save_transposed_dataset(
                transposed_inputs, transposed_outputs,
                output_dir=args.output_dir,
                prefix=f"{args.prefix}_transposed"
            )
            return
            
        # Update the inputs/outputs for further processing if --transpose is used
        if args.transpose:
            inputs, outputs = transposed_inputs, transposed_outputs
    
    # Plot heatmap of original data (if not already done with transpose option)
    elif args.plot_heatmap:
        if args.multi_samples:
            # Plot multiple samples as heatmaps
            visualize_multiple_samples_heatmap(
                inputs, outputs,
                sample_indices=args.multi_samples,
                max_channels=args.max_channels,
                time_steps=args.time_steps,
                is_transposed=False,
                save_dir=args.output_dir if not args.visualize_only else None
            )
        else:
            visualize_all_channels_heatmap(
                inputs, outputs,
                sample_idx=args.sample_idx,
                max_channels=args.max_channels,
                time_steps=args.time_steps,
                is_transposed=False,
                save_dir=args.output_dir if not args.visualize_only else None
            )
    
    # Analyze signal properties
    analyze_signal_properties(inputs)
    
    # Visualize signals (skip if we already visualized other plots)
    if not (args.plot_transposed or args.plot_heatmap):
        visualize_signals(
            inputs, outputs, 
            sample_idx=args.sample_idx,
            channels=args.channels,
            save_dir=args.output_dir if not args.visualize_only else None
        )
    
    # Apply fixes and save (unless visualize_only is specified)
    if not args.visualize_only:
        apply_fixes_and_save(
            inputs, outputs, 
            output_dir=args.output_dir,
            prefix=f"{args.prefix}_{'transposed_' if args.transpose else ''}fixed",
            fix_method=args.fix_method
        )


if __name__ == "__main__":
    main() 