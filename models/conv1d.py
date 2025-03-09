import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    A single convolutional block with layer normalization and skip connection
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.same_channels = in_channels == out_channels
        
        # Padding to maintain sequence length (valid only for odd kernel sizes)
        padding = (kernel_size - 1) // 2 * dilation
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            dilation=dilation
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        
        # If channels don't match, use a 1x1 conv for the skip connection
        if not self.same_channels:
            self.channel_map = nn.Conv1d(in_channels, out_channels, kernel_size=1)
            
    def forward(self, x):
        # Store the input for skip connection
        identity = x
        
        # Apply convolution
        out = self.conv(x)
        
        # Apply channel mapping if needed
        if not self.same_channels:
            identity = self.channel_map(identity)
            
        # Add skip connection
        out = out + identity
        
        # Transpose for layer norm (LayerNorm expects [batch, seq_len, features])
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)  # Back to [batch, channels, seq_len]
        
        # Apply activation
        out = self.activation(out)
        
        return out


class Conv1DEncoder(nn.Module):
    """
    1D Convolutional Encoder with layer normalization and skip connections
    Can optionally use an MLP instead of convolutional blocks
    """
    def __init__(
        self, 
        input_channels=1,
        hidden_dims=[32, 64, 128, 256],
        kernel_sizes=[3, 3, 3, 3],
        dilations=[1, 1, 1, 1],
        output_channels=None,
        use_mlp=False  # Default to using MLP
    ):
        super().__init__()
        
        self.use_mlp = use_mlp
        self.output_channels = output_channels
        
        if use_mlp:
            # Create MLP layers
            self.mlp = nn.Sequential()
            
            # Input projection
            self.mlp.add_module('input_proj', nn.Linear(input_channels, hidden_dims[0]))
            self.mlp.add_module('input_relu', nn.ReLU())
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                self.mlp.add_module(f'hidden_{i}', nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.mlp.add_module(f'relu_{i}', nn.ReLU())
            
            # Output projection layer if output_channels is specified
            if output_channels is not None:
                self.output_proj = nn.Linear(hidden_dims[-1], output_channels)
        else:
            # Original convolutional implementation
            # Input projection
            layers = [nn.Conv1d(input_channels, hidden_dims[0], kernel_size=1)]
            
            # Add convolutional blocks
            in_channels = hidden_dims[0]
            for i, (hidden_dim, kernel_size, dilation) in enumerate(zip(hidden_dims, kernel_sizes, dilations)):
                layers.append(
                    ConvBlock(in_channels, hidden_dim, kernel_size, dilation)
                )
                in_channels = hidden_dim
            
            self.encoder = nn.Sequential(*layers)
            
            # Output projection layer if output_channels is specified
            if output_channels is not None:
                self.output_proj = nn.Conv1d(hidden_dims[-1], output_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, sequence_length, input_channels]
        """
        if self.use_mlp:
            # Process each position in the sequence independently using the MLP
            # x is [batch_size, sequence_length, input_channels]
            
            # Process through MLP
            encoded = self.mlp(x)
            
            # Apply output projection if specified
            if self.output_channels is not None:
                encoded = self.output_proj(encoded)
            
            # Output is already in [batch_size, sequence_length, channels] format
        else:
            # Original convolutional implementation
            # Convert to [batch_size, input_channels, sequence_length]
            x = x.transpose(1, 2)
            
            # Apply encoder
            encoded = self.encoder(x)
            
            # Apply output projection if specified
            if self.output_channels is not None:
                encoded = self.output_proj(encoded)
            
            # Return in format [batch_size, sequence_length, channels]
            encoded = encoded.transpose(1, 2)
        
        return encoded


class Conv1DDecoder(nn.Module):
    """
    1D Convolutional Decoder that mirrors the encoder structure
    Takes transformer outputs and processes them to a specified output dimension
    Can optionally use an MLP instead of convolutional blocks
    """
    def __init__(
        self, 
        input_channels,
        hidden_dims=[256, 128, 64, 32],
        kernel_sizes=[3, 3, 3, 3],
        dilations=[1, 1, 1, 1],
        output_channels=1,
        use_mlp=False  # Default to using MLP
    ):
        super().__init__()
        
        self.use_mlp = use_mlp
        
        if use_mlp:
            # Create MLP layers
            self.mlp = nn.Sequential()
            
            # Input projection
            self.mlp.add_module('input_proj', nn.Linear(input_channels, hidden_dims[0]))
            self.mlp.add_module('input_relu', nn.ReLU())
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                self.mlp.add_module(f'hidden_{i}', nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                self.mlp.add_module(f'relu_{i}', nn.ReLU())
            
            # Output projection
            self.output_proj = nn.Linear(hidden_dims[-1], output_channels)
        else:
            # Original convolutional implementation
            # Input projection
            layers = [nn.Conv1d(input_channels, hidden_dims[0], kernel_size=1)]
            
            # Add convolutional blocks
            in_channels = hidden_dims[0]
            for i, (hidden_dim, kernel_size, dilation) in enumerate(zip(hidden_dims, kernel_sizes, dilations)):
                layers.append(
                    ConvBlock(in_channels, hidden_dim, kernel_size, dilation)
                )
                in_channels = hidden_dim
            
            self.decoder = nn.Sequential(*layers)
            
            # Output projection layer
            self.output_proj = nn.Conv1d(hidden_dims[-1], output_channels, kernel_size=1)
    
    def forward(self, x):
        """
        x: Input tensor of shape [batch_size, sequence_length, input_channels]
        """
        if self.use_mlp:
            # Process each position in the sequence independently using the MLP
            # x is [batch_size, sequence_length, input_channels]
            
            # Process through MLP
            decoded = self.mlp(x)
            
            # Apply output projection
            decoded = self.output_proj(decoded)
            
            # Output is already in [batch_size, sequence_length, output_channels] format
        else:
            # Original convolutional implementation
            # Convert to [batch_size, input_channels, sequence_length]
            x = x.transpose(1, 2)
            
            # Apply decoder
            decoded = self.decoder(x)
            
            # Apply ReLU activation
            decoded = F.relu(decoded)
            
            # Apply output projection
            decoded = self.output_proj(decoded)
            
            # Return in format [batch_size, sequence_length, channels]
            decoded = decoded.transpose(1, 2)
        
        return decoded


# Example usage
if __name__ == "__main__":
    # Input shape: [batch_size, sequence_length, channels]
    batch_size, seq_len, channels = 8, 128, 1
    x = torch.randn(batch_size, seq_len, channels)
    
    # Create model (customize parameters as needed)
    model = Conv1DEncoder(
        input_channels=channels,
        hidden_dims=[32, 64, 128, 256],
        kernel_sizes=[3, 5, 7, 9],
        dilations=[1, 2, 4, 8],
        output_channels=512  # Explicitly specify output channels
    )
    
    # Forward pass
    encoded = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {encoded.shape}")
    
    # Example without output projection (uses last hidden dim as output)
    model_no_proj = Conv1DEncoder(
        input_channels=channels,
        hidden_dims=[32, 64, 128, 256],
        kernel_sizes=[3, 5, 7, 9],
        dilations=[1, 2, 4, 8]
    )
    
    encoded_no_proj = model_no_proj(x)
    print(f"Encoded shape (no output projection): {encoded_no_proj.shape}")