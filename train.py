import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import wandb

from models.brainformer import BrainFormer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train BrainFormer on synthetic data')
    
    # Data parameters
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length for time series data')
    parser.add_argument('--input_channels', type=int, default=70, help='Number of input channels')
    parser.add_argument('--output_channels', type=int, default=300, help='Number of output channels')
    parser.add_argument('--data_dir', type=str, default='data/synthetic/processed', help='Directory to load data from')
    parser.add_argument('--split_data', action='store_true', default=True, help='Split raw data into train/val (if only inputs.pt/outputs.pt exist)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation data ratio when splitting raw data')
    parser.add_argument('--input_file', type=str, default='inputs.pt', help='Filename of input data (default: inputs.pt)')
    parser.add_argument('--output_file', type=str, default='outputs.pt', help='Filename of output data (default: outputs.pt)')
    
    # Model parameters
    parser.add_argument('--encoder_hidden_dims', type=str, default='256,256,256,256', help='Comma-separated list of encoder hidden dimensions')
    parser.add_argument('--decoder_hidden_dims', type=str, default='256,256,256,256', help='Comma-separated list of decoder hidden dimensions')
    parser.add_argument('--n_layer', type=int, default=8, help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.99, help='Beta2 for Adam optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--save_checkpoints', action='store_true', default=False, help='Whether to save checkpoints during training')
    parser.add_argument('--log_every', type=int, default=10, help='Log training metrics every N batches')
    parser.add_argument('--viz_every', type=int, default=1, help='Visualize sample predictions every N epochs')
    
    # Wandb parameters
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Whether to use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='brainformer', help='Wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb_watch', type=str, default='gradients', help='Wandb watch mode: gradients, parameters, or all')
    parser.add_argument('--wandb_watch_log_freq', type=int, default=100, help='Frequency of logging gradients and parameters')
    
    return parser.parse_args()

def prepare_data(args):
    """Load datasets for training and validation"""
    # Check if we need to split raw data
    inputs_path = os.path.join(args.data_dir, args.input_file)
    outputs_path = os.path.join(args.data_dir, args.output_file)
    
    train_data_path = os.path.join(args.data_dir, 'train_inputs.pt')
    train_label_path = os.path.join(args.data_dir, 'train_outputs.pt')
    val_data_path = os.path.join(args.data_dir, 'val_inputs.pt')
    val_label_path = os.path.join(args.data_dir, 'val_outputs.pt')
    
    # If split_data is True and raw files exist but split files don't, perform the split
    if args.split_data and os.path.exists(inputs_path) and os.path.exists(outputs_path) and \
       (not os.path.exists(train_data_path) or not os.path.exists(val_data_path)):
        logger.info(f"Raw data found: {inputs_path} and {outputs_path}. Splitting into train and validation sets...")
        
        # Load raw data
        inputs = torch.load(inputs_path)
        outputs = torch.load(outputs_path)
        
        # Determine split indices
        dataset_size = inputs.shape[0]
        indices = list(range(dataset_size))
        np.random.shuffle(indices)
        
        val_size = int(args.val_ratio * dataset_size)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        # Split the data
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        val_inputs = inputs[val_indices]
        val_outputs = outputs[val_indices]
        
        # Save the split data
        logger.info(f"Saving split data to {args.data_dir}")
        torch.save(train_inputs, train_data_path)
        torch.save(train_outputs, train_label_path)
        torch.save(val_inputs, val_data_path)
        torch.save(val_outputs, val_label_path)
        
        logger.info(f"Train data: inputs {train_inputs.shape}, outputs {train_outputs.shape}")
        logger.info(f"Validation data: inputs {val_inputs.shape}, outputs {val_outputs.shape}")
    else:
        missing_files = []
        for path in [train_data_path, train_label_path, val_data_path, val_label_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            # Check if raw data exists
            if os.path.exists(inputs_path) and os.path.exists(outputs_path):
                logger.info("Raw data found but split_data is False. Set --split_data to automatically split the data.")
            raise FileNotFoundError(
                f"Data files not found: {', '.join(missing_files)}. "
                f"Please ensure data exists in {args.data_dir} before training or use --split_data."
            )
        
        logger.info("Loading pre-split data...")
        train_inputs = torch.load(train_data_path)
        train_outputs = torch.load(train_label_path)
        val_inputs = torch.load(val_data_path)
        val_outputs = torch.load(val_label_path)
        
        logger.info(f"Train data: inputs {train_inputs.shape}, outputs {train_outputs.shape}")
        logger.info(f"Validation data: inputs {val_inputs.shape}, outputs {val_outputs.shape}")
    
    # Create datasets and dataloaders
    # Check data dimensions and ensure correct shape [batch, seq_len, channels]
    # Conv1DEncoder expects [batch, seq_len, channels] which it will transpose to [batch, channels, seq_len]
    logger.info("Checking data dimensions...")
    
    # Ensure input data has shape [N, seq_len, input_channels]
    if train_inputs.dim() == 3:
        if train_inputs.shape[2] != args.input_channels and train_inputs.shape[1] == args.input_channels:
            logger.info(f"Transposing train inputs from {train_inputs.shape} to ensure seq_len is dim 1 and channels is dim 2")
            train_inputs = train_inputs.transpose(1, 2)
            val_inputs = val_inputs.transpose(1, 2)
    
    # Ensure output data has shape [N, seq_len, output_channels]
    if train_outputs.dim() == 3:
        if train_outputs.shape[2] != args.output_channels and train_outputs.shape[1] == args.output_channels:
            logger.info(f"Transposing train outputs from {train_outputs.shape} to ensure seq_len is dim 1 and channels is dim 2")
            train_outputs = train_outputs.transpose(1, 2)
            val_outputs = val_outputs.transpose(1, 2)
    
    logger.info(f"Final train data shapes: inputs {train_inputs.shape}, outputs {train_outputs.shape}")
    logger.info(f"Final validation data shapes: inputs {val_inputs.shape}, outputs {val_outputs.shape}")
    
    train_dataset = TensorDataset(train_inputs, train_outputs)
    val_dataset = TensorDataset(val_inputs, val_outputs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader

def create_model(args):
    """Create a BrainFormer model with the specified configuration"""
    # Parse hidden dimension lists
    encoder_hidden_dims = [int(d) for d in args.encoder_hidden_dims.split(',')]
    decoder_hidden_dims = [int(d) for d in args.decoder_hidden_dims.split(',')]
    
    # Create model
    model = BrainFormer(
        input_channels=args.input_channels,
        encoder_hidden_dims=encoder_hidden_dims,
        encoder_kernel_sizes=[3, 3, 3, 3],
        encoder_dilations=[1, 1, 1, 1],
        
        block_size=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        transformer_dropout=args.dropout,
        bias=True,
        
        decoder_hidden_dims=decoder_hidden_dims,
        decoder_kernel_sizes=[3, 3, 3, 3],
        decoder_dilations=[1, 1, 1, 1],
        output_channels=args.output_channels
    )
    
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, args, epoch):
    """Train the model for one epoch"""
    model.train()
    epoch_loss = 0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Log batch metrics
        if batch_idx % args.log_every == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
            
        # Log batch metrics to wandb
        if args.use_wandb and batch_idx % args.log_every == 0:
            wandb.log({
                'batch_loss': loss.item(),
                'batch': batch_idx + epoch * num_batches,
                'epoch': epoch
            })
    
    return epoch_loss / num_batches

def validate(model, dataloader, criterion, device):
    """Validate the model on the validation set"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            val_loss += loss.item()
    
    # Calculate average validation loss
    val_loss /= len(dataloader)
    return val_loss

def save_checkpoint(model, optimizer, epoch, loss, args, is_best=False):
    """Save model checkpoint"""
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Checkpoint filename
    filename = os.path.join(args.save_dir, f"brainformer_epoch_{epoch+1}.pt")
    
    # Save checkpoint
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }, filename)
    
    logger.info(f"Checkpoint saved to {filename}")
    
    # Save best model if this is the best so far
    if is_best:
        best_filename = os.path.join(args.save_dir, "brainformer_best.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'args': vars(args)
        }, best_filename)
        logger.info(f"Best model saved to {best_filename}")

def plot_training_history(train_losses, val_losses, args):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    os.makedirs(args.save_dir, exist_ok=True)
    plot_path = os.path.join(args.save_dir, 'loss_history.png')
    plt.savefig(plot_path)
    logger.info(f"Loss history plot saved to {plot_path}")

def visualize_predictions(model, dataloader, device, epoch, args):
    """
    Visualize sample input, ground truth, and predicted output
    
    Args:
        model: The trained model
        dataloader: Data loader containing validation data
        device: The device to use for inference
        epoch: Current epoch number
        args: Command line arguments
    """
    model.eval()
    
    # Get a batch of data
    inputs, targets = next(iter(dataloader))
    
    # Select just one sample to visualize
    input_sample = inputs[0:1].to(device)
    target_sample = targets[0:1].to(device)
    
    # Generate prediction
    with torch.no_grad():
        prediction = model(input_sample)
    
    # Move tensors to CPU for plotting
    input_sample = input_sample.cpu().numpy()[0]  # Shape: [seq_len, input_channels]
    target_sample = target_sample.cpu().numpy()[0]  # Shape: [seq_len, output_channels]
    prediction = prediction.cpu().numpy()[0]  # Shape: [seq_len, output_channels]
    
    # Create directory for visualization
    viz_dir = os.path.join(args.save_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create a multi-panel figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    
    # Plot input channels (reducing dimensionality if needed)
    axs[0].set_title('Input Signal')
    if input_sample.shape[1] > 10:
        # If too many channels, just plot first 10
        channels_to_plot = min(10, input_sample.shape[1])
        for i in range(channels_to_plot):
            axs[0].plot(input_sample[:, i], label=f'Ch {i+1}')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()
    else:
        im = axs[0].imshow(input_sample.T, aspect='auto', interpolation='none')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Channel')
        plt.colorbar(im, ax=axs[0])
    
    # Plot target output (heatmap for many channels)
    axs[1].set_title('Target Output')
    im = axs[1].imshow(target_sample.T, aspect='auto', interpolation='none')
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Channel')
    plt.colorbar(im, ax=axs[1])
    
    # Plot predicted output
    axs[2].set_title('Predicted Output')
    im = axs[2].imshow(prediction.T, aspect='auto', interpolation='none')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Channel')
    plt.colorbar(im, ax=axs[2])
    
    # Add overall title
    plt.suptitle(f'Sample Visualization - Epoch {epoch+1}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    viz_path = os.path.join(viz_dir, f'visualization_epoch_{epoch+1}.png')
    plt.savefig(viz_path)
    plt.close()
    
    logger.info(f"Sample visualization saved to {viz_path}")
    
    # Log to wandb if enabled
    if args.use_wandb:
        wandb.log({f"visualization_epoch_{epoch+1}": wandb.Image(viz_path)})

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args)
        )
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Determine the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Prepare data
    train_loader, val_loader = prepare_data(args)
    
    # Create model
    model = create_model(args)
    model = model.to(device)
    
    # Print model summary
    logger.info(f"BrainFormer model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Set up wandb to watch model gradients if wandb is enabled
    if args.use_wandb:
        wandb.watch(
            model,
            log=args.wandb_watch,
            log_freq=args.wandb_watch_log_freq,
            log_graph=True
        )
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    
    # Configure optimizer using the model's method
    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        betas=(args.beta1, args.beta2),
        device_type='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1}/{args.epochs} completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb_metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch + 1
            }
            wandb.log(wandb_metrics)
        
        # Check if this is the best model so far
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
        
        # Save checkpoint
        if args.save_checkpoints and ((epoch + 1) % args.save_every == 0 or is_best or epoch == args.epochs - 1):
            save_checkpoint(model, optimizer, epoch, val_loss, args, is_best)
        
        # Visualize predictions
        if (epoch + 1) % args.viz_every == 0:
            visualize_predictions(model, val_loader, device, epoch, args)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, args)
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    logger.info("Training completed.")

if __name__ == "__main__":
    main() 