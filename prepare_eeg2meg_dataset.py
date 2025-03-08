import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datasets import load_dataset
from tqdm import tqdm
from typing import Tuple, Dict, Optional

def load_eeg2megset(
    cache_dir: Optional[str] = None,
    split: str = "train"
) -> Dict:
    """
    Load the EEG2MEG dataset from Hugging Face.
    
    Args:
        cache_dir: Directory to cache the dataset
        split: Dataset split to load ('train', 'validation', or 'test')
        
    Returns:
        Dict: The loaded dataset
    """
    print(f"Loading EEG2MEG dataset (split: {split})...")
    dataset = load_dataset(
        "fracapuano/eeg2meg-medium-tokenized", 
        split=split,
        cache_dir=cache_dir
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    return dataset

def process_dataset(
    dataset: Dict,
    expected_seq_len: int = 771
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process the EEG2MEG dataset by converting integers to floats and ensuring correct shapes.
    
    Args:
        dataset: The loaded dataset
        expected_seq_len: Expected sequence length
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Input (EEG) and output (MEG) tensors
    """
    num_samples = len(dataset)
    
    # Determine input and output channels from the first sample
    eeg_channels = len(dataset[0]['eeg']) if isinstance(dataset[0]['eeg'], list) else 1
    meg_channels = len(dataset[0]['meg']) if isinstance(dataset[0]['meg'], list) else 1
    
    # Create empty tensors to store the processed data
    inputs = []
    outputs = []
    
    # Process each sample
    for i in tqdm(range(num_samples), desc="Processing dataset", ncols=100):
        # Get EEG data (input)
        eeg = dataset[i]['eeg']
        # Get MEG data (output)
        meg = dataset[i]['meg']
        
        # Convert to float tensors
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        meg_tensor = torch.tensor(meg, dtype=torch.float32)
        
        # Check shapes and reshape if necessary
        if eeg_tensor.ndim == 1 and eeg_tensor.shape[0] != expected_seq_len:
            print(f"Warning: Sample {i} EEG has unexpected shape {eeg_tensor.shape}, expected length {expected_seq_len}")
            # Pad or truncate to expected length
            if eeg_tensor.shape[0] < expected_seq_len:
                eeg_tensor = torch.nn.functional.pad(eeg_tensor, (0, expected_seq_len - eeg_tensor.shape[0]))
            else:
                eeg_tensor = eeg_tensor[:expected_seq_len]
        
        if meg_tensor.ndim == 1 and meg_tensor.shape[0] != expected_seq_len:
            print(f"Warning: Sample {i} MEG has unexpected shape {meg_tensor.shape}, expected length {expected_seq_len}")
            # Pad or truncate to expected length
            if meg_tensor.shape[0] < expected_seq_len:
                meg_tensor = torch.nn.functional.pad(meg_tensor, (0, expected_seq_len - meg_tensor.shape[0]))
            else:
                meg_tensor = meg_tensor[:expected_seq_len]
        
        # Handle multi-channel data if needed
        if eeg_tensor.ndim > 1:
            # If data is already multi-channel, just ensure the sequence dim is correct
            if eeg_tensor.shape[-1] != expected_seq_len:
                eeg_tensor = eeg_tensor[..., :expected_seq_len]
        else:
            # If single channel, reshape to [1, seq_len]
            eeg_tensor = eeg_tensor.unsqueeze(0)
        
        if meg_tensor.ndim > 1:
            if meg_tensor.shape[-1] != expected_seq_len:
                meg_tensor = meg_tensor[..., :expected_seq_len]
        else:
            meg_tensor = meg_tensor.unsqueeze(0)
        
        inputs.append(eeg_tensor)
        outputs.append(meg_tensor)
    
    # Stack all samples
    try:
        inputs_tensor = torch.stack(inputs)
        outputs_tensor = torch.stack(outputs)
    except RuntimeError as e:
        print(f"Error stacking tensors: {e}")
        print(f"Input shapes: {[x.shape for x in inputs[:5]]} (showing first 5)")
        print(f"Output shapes: {[x.shape for x in outputs[:5]]} (showing first 5)")
        # Try to standardize shapes
        fixed_inputs = []
        fixed_outputs = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            if inp.shape[0] != inputs[0].shape[0]:
                if inp.shape[0] < inputs[0].shape[0]:
                    # Pad with zeros
                    padded = torch.zeros(inputs[0].shape[0], inp.shape[1], dtype=inp.dtype)
                    padded[:inp.shape[0]] = inp
                    fixed_inputs.append(padded)
                else:
                    # Truncate
                    fixed_inputs.append(inp[:inputs[0].shape[0]])
            else:
                fixed_inputs.append(inp)
                
            if out.shape[0] != outputs[0].shape[0]:
                if out.shape[0] < outputs[0].shape[0]:
                    # Pad with zeros
                    padded = torch.zeros(outputs[0].shape[0], out.shape[1], dtype=out.dtype)
                    padded[:out.shape[0]] = out
                    fixed_outputs.append(padded)
                else:
                    # Truncate
                    fixed_outputs.append(out[:outputs[0].shape[0]])
            else:
                fixed_outputs.append(out)
                
        inputs_tensor = torch.stack(fixed_inputs)
        outputs_tensor = torch.stack(fixed_outputs)
    
    print(f"Processed data shapes: Inputs {inputs_tensor.shape}, Outputs {outputs_tensor.shape}")
    return inputs_tensor, outputs_tensor

def save_dataset(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    save_dir: str,
    prefix: str = "eeg2meg"
) -> None:
    """
    Save the processed dataset to files.
    
    Args:
        inputs: Input tensor (EEG data)
        outputs: Output tensor (MEG data)
        save_dir: Directory to save the dataset
        prefix: Prefix for the filenames
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
    
    # Plot input channels (up to 10 for clarity)
    num_input_channels_to_plot = min(10, inputs.shape[1])
    for c in range(num_input_channels_to_plot):
        plt.subplot(2, num_input_channels_to_plot, c + 1)
        plt.plot(inputs[sample_idx, c].numpy())
        plt.title(f"EEG Channel {c+1}")
    
    # Plot output channels (up to 10 for clarity)
    num_output_channels_to_plot = min(10, outputs.shape[1])
    for c in range(num_output_channels_to_plot):
        plt.subplot(2, num_output_channels_to_plot, 
                   num_input_channels_to_plot + c + 1)
        plt.plot(outputs[sample_idx, c].numpy())
        plt.title(f"MEG Channel {c+1}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix}_sample_plot.png"))
    
    # Save metadata
    with open(os.path.join(save_dir, f"{prefix}_metadata.txt"), "w") as f:
        f.write(f"Input (EEG) shape: {inputs.shape}\n")
        f.write(f"Output (MEG) shape: {outputs.shape}\n")
        f.write(f"Input type: {inputs.dtype}\n")
        f.write(f"Output type: {outputs.dtype}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare EEG2MEG dataset from Hugging Face")
    parser.add_argument("--save_dir", type=str, default="./eeg2meg", help="Directory to save dataset")
    parser.add_argument("--prefix", type=str, default="eeg2meg", help="Prefix for saved files")
    parser.add_argument("--seq_len", type=int, default=771, help="Expected sequence length")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache the dataset")
    
    args = parser.parse_args()
    
    # Create progress tracking for overall process
    with tqdm(total=3, desc="Overall progress", position=0, ncols=100) as overall_progress:
        # Load the dataset
        dataset = load_eeg2megset(cache_dir=args.cache_dir, split=args.split)
        overall_progress.update(1)
        
        # Process the dataset
        print("Processing dataset...")
        inputs, outputs = process_dataset(dataset, expected_seq_len=args.seq_len)
        overall_progress.update(1)
        
        # Save the dataset
        print(f"Saving processed dataset to {args.save_dir}...")
        save_dataset(
            inputs=inputs, 
            outputs=outputs, 
            save_dir=args.save_dir, 
            prefix=args.prefix
        )
        overall_progress.update(1)
        
        # Complete
        print("Dataset preparation complete!")
        print(f"Input (EEG) shape: {inputs.shape}")
        print(f"Output (MEG) shape: {outputs.shape}")

if __name__ == "__main__":
    main() 