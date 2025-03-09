import torch
import torch.nn as nn
from models.conv1d import Conv1DEncoder, ConvBlock, Conv1DDecoder
from models.simplified_decoder import SimplifiedDecoder, DecoderConfig

class BrainFormer(nn.Module):
    """
    Combined model that uses:
    1. Conv1DEncoder to process input signals
    2. SimplifiedDecoder (transformer) to process the encoded representation
    3. Conv1DDecoder to process the transformer output to the desired shape
    
    The model ensures dimensional compatibility between components
    """
    def __init__(
        self,
        # Encoder params
        input_channels=1,
        encoder_hidden_dims=[32, 64, 128, 256],
        encoder_kernel_sizes=[3, 3, 3, 3],
        encoder_dilations=[1, 1, 1, 1],
        
        # Transformer params
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        transformer_dropout=0.0,
        bias=True,
        bidirectional=False,
        
        # Decoder params
        decoder_hidden_dims=[256, 128, 64, 32],
        decoder_kernel_sizes=[3, 3, 3, 3],
        decoder_dilations=[1, 1, 1, 1],
        output_channels=1
    ):
        super().__init__()
        
        # Set the transformer embedding dimension as output for the encoder
        self.n_embd = n_embd
        
        # Create the encoder
        self.encoder = Conv1DEncoder(
            input_channels=input_channels,
            hidden_dims=encoder_hidden_dims,
            kernel_sizes=encoder_kernel_sizes,
            dilations=encoder_dilations,
            output_channels=n_embd  # Set to match transformer embedding dimension
        )
        
        # Create the transformer decoder
        decoder_config = DecoderConfig(
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=transformer_dropout,
            bias=bias,
            bidirectional=bidirectional
        )
        self.transformer = SimplifiedDecoder(decoder_config)
        
        # Create the convolutional decoder
        self.decoder = Conv1DDecoder(
            input_channels=n_embd,
            hidden_dims=decoder_hidden_dims,
            kernel_sizes=decoder_kernel_sizes,
            dilations=decoder_dilations,
            output_channels=output_channels
        )
        
    def forward(self, x):
        """
        Forward pass through the entire model chain
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_channels]
            
        Returns:
            Output tensor of shape [batch_size, sequence_length, output_channels]
        """
        # Ensure the input is in the expected format
        batch_size, seq_len, channels = x.size()
        
        # Step 1: Pass through the encoder
        encoded = self.encoder(x)  # Shape: [batch_size, seq_len, n_embd]
        
        # Step 2: Pass through the transformer
        # The encoder already outputs in the format [batch_size, seq_len, n_embd]
        transformed = self.transformer(encoded)  # Shape: [batch_size, seq_len, n_embd]
        
        # Step 3: Pass through the decoder
        # The transformer outputs in the format [batch_size, seq_len, n_embd]
        decoded = self.decoder(transformed)  # Shape: [batch_size, seq_len, output_channels]
        
        return decoded

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_muon=False, muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5, rank=0, world_size=1):
        """
        Configure optimizers for the entire model
        """
        # Collect all parameters
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Create optimizer groups
        # Weight decay for 2D parameters (weights), no decay for 1D parameters (biases, LayerNorms)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # Print parameter stats
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # If Muon is requested, use it for 2D parameters (typically weights) and AdamW for the rest
        if use_muon:
            from muon import Muon
            
            # Create Muon optimizer for 2D parameters
            muon_optimizer = Muon(
                decay_params,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_steps=muon_ns_steps,
                rank=rank,  # Use the provided rank
                world_size=world_size  # Use the provided world_size
            )
            print(f"Using Muon optimizer for {len(decay_params)} 2D parameter tensors (rank={rank}, world_size={world_size})")
            
            # Create AdamW optimizer for 1D parameters
            # Use fused AdamW if available
            fused_available = (
                'fused' in torch.__dict__ and
                hasattr(torch.optim, 'AdamW') and
                hasattr(torch.optim.AdamW.__init__.__code__, 'co_varnames') and
                'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
            )
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            
            adamw_optimizer = torch.optim.AdamW(
                [{'params': nodecay_params, 'weight_decay': 0.0}],
                lr=learning_rate, 
                betas=betas, 
                **extra_args
            )
            print(f"Using AdamW optimizer for {len(nodecay_params)} non-2D parameter tensors")
            print(f"using fused AdamW: {use_fused}")
            
            # Return a list of optimizers (for use with PyTorch's ZeroRedundancyOptimizer)
            return [muon_optimizer, adamw_optimizer]
        else:
            # Standard optimization with AdamW
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            
            # Use fused AdamW if available
            fused_available = (
                'fused' in torch.__dict__ and
                hasattr(torch.optim, 'AdamW') and
                hasattr(torch.optim.AdamW.__init__.__code__, 'co_varnames') and
                'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
            )
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
            
            return optimizer 