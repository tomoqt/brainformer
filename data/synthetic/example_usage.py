import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from generate_synthetic_timeseries import generate_timeseries_dataset

def main():
    # Generate a small dataset
    print("Generating a small example dataset...")
    inputs, outputs = generate_timeseries_dataset(
        num_samples=5,
        seq_len=256,
        input_channels=3,
        output_channels=5,
        noise_level=0.1,
        k_combinations=2,
        seed=42  # For reproducibility
    )
    
    print(f"Input tensor shape: {inputs.shape}")
    print(f"Output tensor shape: {outputs.shape}")
    
    # Save the example dataset
    if not os.path.exists("example_data"):
        os.makedirs("example_data")
    
    torch.save(inputs, "example_data/example_inputs.pt")
    torch.save(outputs, "example_data/example_outputs.pt")
    
    # Load the saved data
    print("\nLoading the saved data...")
    loaded_inputs = torch.load("example_data/example_inputs.pt")
    loaded_outputs = torch.load("example_data/example_outputs.pt")
    
    print(f"Loaded input tensor shape: {loaded_inputs.shape}")
    print(f"Loaded output tensor shape: {loaded_outputs.shape}")
    
    # Visualize one sample
    sample_idx = 0
    
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
    plt.savefig("example_data/example_visualization.png")
    plt.close()
    
    print("\nExample visualization saved to 'example_data/example_visualization.png'")
    
    # Demonstrate how the first two channels of the outputs are noised versions of inputs
    print("\nDemonstrating noise relationship between inputs and outputs:")
    for c in range(min(inputs.shape[1], 2)):
        mean_diff = torch.mean(torch.abs(inputs[sample_idx, c] - outputs[sample_idx, c])).item()
        print(f"Mean absolute difference between input channel {c+1} and output channel {c+1}: {mean_diff:.2f}")
    
    # Demonstrate how additional output channels are linear combinations of inputs
    if outputs.shape[1] > inputs.shape[1]:
        print("\nAdditional output channels are linear combinations of input channels.")
        print("Let's try to find the approximate weights:")
        
        for c in range(inputs.shape[1], min(outputs.shape[1], inputs.shape[1] + 2)):
            print(f"\nFor output channel {c+1}:")
            
            # Simple linear regression to find weights
            X = inputs[sample_idx].numpy().T  # Shape: [seq_len, input_channels]
            y = outputs[sample_idx, c].numpy()  # Shape: [seq_len]
            
            # Add a column of ones for the intercept
            X_with_intercept = np.column_stack([X, np.ones(X.shape[0])])
            
            # Solve for weights: X_with_intercept * weights = y
            try:
                weights = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                
                for i in range(inputs.shape[1]):
                    print(f"  Weight for input channel {i+1}: {weights[i]:.4f}")
                print(f"  Intercept: {weights[-1]:.4f}")
            except np.linalg.LinAlgError:
                print("  Could not determine weights (singular matrix)")

if __name__ == "__main__":
    main() 