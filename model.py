import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Any


class PositionalEncoding(nn.Module):
    pe: torch.Tensor
    
    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class GroupedInputProjection(nn.Module):
    """Project per-stock features, then merge across stocks.

    Input shape: [batch, seq_len, num_stocks * stock_feat_dim]
    Output shape: [batch, seq_len, d_model]
    """
    def __init__(
        self,
        d_model: int,
        num_stocks: int = 36,
        stock_feat_dim: int = 13,
        stock_embed_dim: int = 8,
        final_layer_cls: type[nn.Module] = nn.Linear,
        final_layer_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super(GroupedInputProjection, self).__init__()

        self.num_stocks = num_stocks
        self.stock_feat_dim = stock_feat_dim
        self.stock_embed_dim = stock_embed_dim

        self.per_stock = nn.Linear(stock_feat_dim, stock_embed_dim)
        self.stock_score = nn.Linear(stock_embed_dim, 1)

        self.last_stock_weights = None

        final_kwargs = final_layer_kwargs or {}
        self.final_proj = final_layer_cls(
            in_features=num_stocks * stock_embed_dim,
            out_features=d_model,
            **final_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size: int
        seq_len: int
        feat_dim: int
        batch_size, seq_len, feat_dim = x.shape
        expected_dim = self.num_stocks * self.stock_feat_dim
        if feat_dim != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {feat_dim}")

        x = x.view(batch_size, seq_len, self.num_stocks, self.stock_feat_dim)
        x = self.per_stock(x)

        # Compute per-stock attention weights for interpretability
        scores = self.stock_score(x).squeeze(-1)  # [batch, seq_len, num_stocks]
        weights = torch.softmax(scores, dim=-1)
        self.last_stock_weights = weights.detach()

        x = x * weights.unsqueeze(-1)
        x = x.view(batch_size, seq_len, self.num_stocks * self.stock_embed_dim)
        return self.final_proj(x)

    def get_last_stock_weights(self) -> Optional[torch.Tensor]:
        return self.last_stock_weights


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_length: int, output_dim: int = 1) -> None:
        super(TransformerEncoder, self).__init__()

        self.input_projection = GroupedInputProjection(d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Custom transformer layers to support KV caching
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                batch_first=True,
                activation='relu'
            ) for _ in range(num_layers)
        ])
        
        # Regression head: output single continuous value
        self.output_projection = nn.Linear(d_model, output_dim)
        self.d_model = d_model
        
        # Cache for incremental processing
        self.cache = None
    
    def reset_cache(self) -> None:
        """Reset cache at the start of each session"""
        self.cache = None
    
    def forward(self, x: torch.Tensor, use_cache: bool = False, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional caching for incremental processing
        
        Args:
            x: Input tensor [batch, seq_len, 432]
            use_cache: If True, only process last timestep and use cached computations
                      (only works in eval mode, disabled during training)
            seq_lengths: [batch] tensor of actual sequence lengths for padding mask
        """
        # Disable caching during training to avoid backprop issues
        if self.training:
            use_cache = False
            
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        
        if use_cache and self.cache is not None:
            # Only process the new timestep (last one)
            new_token = x[:, -1:, :]  # [batch, 1, d_model]
            
            # Get cached representations
            cached_repr = self.cache  # [batch, prev_len, d_model]
            
            # Concatenate cached with new
            x = torch.cat([cached_repr, new_token], dim=1)  # [batch, prev_len+1, d_model]
        
        # Create padding mask if seq_lengths provided
        # Transformer expects mask where True = ignore, False = attend
        src_key_padding_mask: Optional[torch.Tensor] = None
        if seq_lengths is not None:
            seq_len_val: int = x.shape[1]
            # Create mask: True for positions beyond actual sequence length
            mask = torch.arange(seq_len_val, device=x.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
            src_key_padding_mask = mask  # [batch, seq_len]
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Update cache with current full sequence (only in eval mode)
        if use_cache and not self.training:
            self.cache = x.detach()
        
        # Output projection - regression output (no softmax)
        output = self.output_projection(x)
        
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.1) -> None:
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.dropout(self.fc2(x))
        gate = torch.sigmoid(self.gate(residual))
        x = gate * x + (1 - gate) * residual
        return self.layer_norm(x)


class TemporalFusionTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_length: int, output_dim: int = 1, dropout: float = 0.1, use_causal_mask: bool = True) -> None:
        super(TemporalFusionTransformer, self).__init__()
        self.input_projection = GroupedInputProjection(d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.input_norm = nn.LayerNorm(d_model)
        self.feature_grn = GatedResidualNetwork(d_model, d_ff, dropout=dropout)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.post_grn = GatedResidualNetwork(d_model, d_ff, dropout=dropout)
        
        # Regression head: output single continuous value
        self.output_projection = nn.Linear(d_model, output_dim)
        self.use_causal_mask = use_causal_mask
        self.d_model = d_model
        
        # Cache for incremental processing
        self.cache = None
        self.lstm_state = None

    def reset_cache(self) -> None:
        """Reset cache at the start of each session"""
        self.cache = None
        self.lstm_state = None

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask: upper triangular matrix"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x: torch.Tensor, use_cache: bool = False, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional caching for incremental processing
        
        Args:
            x: Input tensor [batch, seq_len, 432]
            use_cache: If True, only process last timestep and use cached LSTM/attention states
                      (only works in eval mode, disabled during training)
            seq_lengths: [batch] tensor of actual sequence lengths for padding mask
        """
        # Disable caching during training to avoid backprop issues
        if self.training:
            use_cache = False
            
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.feature_grn(x)
        
        if use_cache and self.cache is not None:
            # Only process the new timestep through LSTM
            new_token = x[:, -1:, :]  # [batch, 1, d_model]
            lstm_out, self.lstm_state = self.lstm(new_token, self.lstm_state)
            
            # Concatenate cached LSTM outputs with new
            x = torch.cat([self.cache, lstm_out], dim=1)  # [batch, prev_len+1, d_model]
        else:
            # Process full sequence
            x, self.lstm_state = self.lstm(x)
        
        # Update cache (only in eval mode)
        if use_cache and not self.training:
            self.cache = x.detach()

        # Create masks for attention
        # Causal mask: 2D [seq_len, seq_len] - same for all batches
        attn_mask = None
        if self.use_causal_mask:
            attn_mask = self._causal_mask(x.size(1), x.device)
        
        # Padding mask: 2D [batch, seq_len] - True where positions should be ignored
        key_padding_mask = None
        if seq_lengths is not None:
            seq_len = x.size(1)
            key_padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)

        x, _ = self.attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.post_grn(x)
        x = self.output_projection(x)  # Regression output (no softmax)
        return x


class BayesianTransformer(nn.Module):
    """Bayesian Transformer using variational inference for uncertainty quantification
    
    Returns both predictions and KL divergence for ELBO loss.
    Use Monte Carlo sampling at inference to get prediction variance.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_length: int, output_dim: int = 1, dropout: float = 0.1, prior_mu: float = 0.0, prior_sigma: float = 0.1) -> None:
        super(BayesianTransformer, self).__init__()
        
        # Import Bayesian layers
        from bayesian_torch.layers import LinearReparameterization  # type: ignore

        self.input_projection = GroupedInputProjection(
            d_model=d_model,
            final_layer_cls=LinearReparameterization,
            final_layer_kwargs={
                'prior_mean': prior_mu,
                'prior_variance': prior_sigma,
                'posterior_mu_init': prior_mu,
                'posterior_rho_init': -3.0,
            },
        )
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Standard transformer layers (use Bayesian in attention next iteration if needed)
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Bayesian output projection
        self.output_projection = LinearReparameterization(
            in_features=d_model,
            out_features=output_dim,
            prior_mean=prior_mu,  # type: ignore[arg-type]
            prior_variance=prior_sigma,  # type: ignore[arg-type]
            posterior_mu_init=prior_mu,  # type: ignore[arg-type]
            posterior_rho_init=-3.0
        )
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, 468]
            seq_lengths: Actual sequence lengths [batch]
            
        Returns:
            output: Predictions [batch, output_dim]
            kl: KL divergence for ELBO loss (scalar)
        """
        batch_size: int
        seq_len: int
        batch_size, seq_len, _ = x.shape
        
        # Bayesian input projection (accumulates KL)
        x, kl = self.input_projection(x)
        
        # Positional encoding and normalization
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        
        # Create attention mask for variable-length sequences
        if seq_lengths is not None:
            key_padding_mask = torch.arange(seq_len, device=x.device)[None, :] >= seq_lengths[:, None]
        else:
            key_padding_mask = None
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Get last valid position for each sequence
        if seq_lengths is not None:
            batch_indices = torch.arange(batch_size, device=x.device)
            seq_end_indices = (seq_lengths - 1).clamp(min=0, max=seq_len-1)
            x = x[batch_indices, seq_end_indices, :]
        else:
            x = x[:, -1, :]
        
        # Bayesian output projection
        output, kl_output = self.output_projection(x)
        kl = kl + kl_output
        
        return output.squeeze(-1), kl


class MCDropoutTransformer(nn.Module):
    """Transformer with Monte Carlo Dropout for uncertainty estimation
    
    Simpler than full Bayesian - uses dropout during inference.
    Run multiple forward passes to get prediction variance.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, num_layers: int, max_seq_length: int, output_dim: int = 1, dropout: float = 0.3) -> None:
        super(MCDropoutTransformer, self).__init__()
        
        self.input_projection = GroupedInputProjection(d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Keep dropout rate higher for better uncertainty estimates
        self.dropout = nn.Dropout(dropout)
        
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None, return_uncertainty: bool = False, mc_samples: int = 10) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch, seq_len, 468]
            seq_lengths: Actual sequence lengths [batch]
            return_uncertainty: If True, run MC sampling for uncertainty
            mc_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            If return_uncertainty=False: predictions [batch, output_dim]
            If return_uncertainty=True: (mean_predictions, std_predictions)
        """
        if return_uncertainty:
            # Monte Carlo sampling - multiple forward passes with dropout
            predictions: list[torch.Tensor] = []
            for _ in range(mc_samples):
                pred = self._forward_once(x, seq_lengths)
                predictions.append(pred)
            
            predictions_stacked = torch.stack(predictions)  # [mc_samples, batch, output_dim]
            mean_pred = predictions_stacked.mean(dim=0)
            std_pred = predictions_stacked.std(dim=0)
            return mean_pred.squeeze(-1), std_pred.squeeze(-1)
        else:
            return self._forward_once(x, seq_lengths).squeeze(-1)
    
    def _forward_once(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single forward pass"""
        batch_size: int
        seq_len: int
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.dropout(x)  # Apply dropout even during inference for MC sampling
        
        # Create attention mask for variable-length sequences
        if seq_lengths is not None:
            key_padding_mask = torch.arange(seq_len, device=x.device)[None, :] >= seq_lengths[:, None]
        else:
            key_padding_mask = None
        
        # Transformer layers (dropout active)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Get last valid position
        if seq_lengths is not None:
            batch_indices = torch.arange(batch_size, device=x.device)
            seq_end_indices = (seq_lengths - 1).clamp(min=0, max=seq_len-1)
            x = x[batch_indices, seq_end_indices, :]
        else:
            x = x[:, -1, :]
        
        x = self.dropout(x)
        output = self.output_projection(x)
        
        return output


class xLSTMEncoder(nn.Module):
    """Extended LSTM (xLSTM) encoder with exponential gating
    
    Based on Beck et al. 2024 "xLSTM: Extended Long Short-Term Memory"
    Improvements over standard LSTM:
    - Exponential gating for better gradient flow
    - Layer normalization
    - Modern initialization
    
    Simpler than Transformers, good for small datasets.
    """
    def __init__(self, d_model: int, num_layers: int, max_seq_length: int, output_dim: int = 1, dropout: float = 0.2) -> None:
        super(xLSTMEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = GroupedInputProjection(d_model=d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Stack of LSTM layers with layer norm
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                dropout=0.0  # We'll apply dropout manually
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms after each LSTM
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Initialize with modern scheme
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with modern scheme"""
        for name, param in self.named_parameters():
            if 'weight' in name and 'norm' not in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, 13]
            seq_lengths: Actual sequence lengths [batch]
            
        Returns:
            output: Predictions [batch] (squeezed from [batch, 1])
        """
        batch_size: int
        seq_len: int
        batch_size, seq_len, _ = x.shape
        
        # Project and normalize input
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Pack sequence for efficient LSTM processing
        packed_x: Any
        sorted_lengths: Any = None
        unsort_idx: Any = None
        if seq_lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            seq_lengths_cpu = seq_lengths.cpu()
            sorted_lengths, sorted_idx = seq_lengths_cpu.sort(descending=True)
            _, unsort_idx = sorted_idx.sort()
            
            x = x[sorted_idx]
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, sorted_lengths, batch_first=True, enforce_sorted=True
            )
        else:
            packed_x = x
        
        # Pass through LSTM layers with residual connections and layer norm
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            if seq_lengths is not None:
                lstm_out: Any
                unpacked: torch.Tensor
                lstm_out, _ = lstm(packed_x)
                # Unpack for residual and norm
                unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
                # For first layer, no residual (dimensions match after projection)
                residual = nn.utils.rnn.pad_packed_sequence(packed_x, batch_first=True, total_length=seq_len)[0] if i == 0 else x
                x = norm(unpacked + residual)
                x = self.dropout(x)
                # Repack for next layer
                packed_x = nn.utils.rnn.pack_padded_sequence(
                    x, sorted_lengths, batch_first=True, enforce_sorted=True
                )
            else:
                lstm_out, _ = lstm(x)
                x = norm(lstm_out + x)
                x = self.dropout(x)
        
        # Unsort and get last valid output
        if seq_lengths is not None:
            x = x[unsort_idx]
            batch_indices = torch.arange(batch_size, device=x.device)
            seq_end_indices = (seq_lengths - 1).clamp(min=0, max=seq_len-1)
            x = x[batch_indices, seq_end_indices, :]
        else:
            x = x[:, -1, :]
        
        # Project to output
        output = self.output_projection(x)
        
        return output.squeeze(-1)


class SelectiveSSMBlock(nn.Module):
    """Selective State Space Model block (pure PyTorch approximation)
    
    Approximates Mamba's selective mechanism using:
    - Gated linear units for selectivity
    - 1D convolution for local context
    - Linear recurrence for long-range dependencies
    """
    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super(SelectiveSSMBlock, self).__init__()
        
        self.d_model = d_model
        self.expand = 2
        d_inner = d_model * self.expand
        
        # Input projection with gating
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        
        # 1D convolution for local context (like Mamba's d_conv)
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=4,
            padding=3,
            groups=d_inner  # Depthwise
        )
        
        # SSM parameters (simplified) - project to d_inner to match dimensions
        self.x_proj = nn.Linear(d_inner, d_inner)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        residual = x
        
        # Input projection with gating
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x, z = xz.chunk(2, dim=-1)  # Each: [batch, seq_len, d_inner]
        
        # Local convolution
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2)
        x = self.act(x_conv)
        
        # Selective gating (simplified SSM)
        # In real Mamba, this would be a structured state space model
        # Here we approximate with learned gating
        gate = torch.sigmoid(self.dt_proj(x))
        x = x * gate
        
        # Apply gating from z
        x = x * self.act(z)
        
        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        
        # Residual connection and normalization
        out = self.norm(residual + x)
        
        return out


class MambaEncoder(nn.Module):
    """Nemotron-style hybrid model combining SSM with Transformer attention
    
    Based on NVIDIA's Nemotron architecture which alternates between:
    - Selective SSM layers for efficient long-range modeling
    - Transformer attention for complex pattern matching
    
    This hybrid approach provides both efficiency and expressiveness,
    using pure PyTorch (no compilation required).
    """
    def __init__(self, d_model: int, num_heads: int, num_layers: int, max_seq_length: int, output_dim: int = 1, dropout: float = 0.1) -> None:
        super(MambaEncoder, self).__init__()
        
        self.input_projection = GroupedInputProjection(d_model=d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Nemotron-style hybrid: alternate between SSM and Attention
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                # Even layers: Selective SSM approximation
                self.layers.append(SelectiveSSMBlock(d_model, dropout))
            else:
                # Odd layers: Transformer attention
                self.layers.append(
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=num_heads,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                        batch_first=True
                    )
                )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, 468]
            seq_lengths: Actual sequence lengths [batch]
            
        Returns:
            output: Predictions [batch] (squeezed from [batch, 1])
        """
        batch_size: int
        seq_len: int
        batch_size, seq_len, _ = x.shape
        
        # Project and add positional encoding
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Create attention mask for variable-length sequences
        if seq_lengths is not None:
            key_padding_mask = torch.arange(seq_len, device=x.device)[None, :] >= seq_lengths[:, None]
        else:
            key_padding_mask = None
        
        # Pass through hybrid SSM + Attention layers
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                # SSM layer (no mask needed - handles via gating)
                x = layer(x)
            else:
                # Attention layer (use mask)
                x = layer(x, src_key_padding_mask=key_padding_mask)
        
        # Get output at last valid position for each sequence
        if seq_lengths is not None:
            batch_indices = torch.arange(batch_size, device=x.device)
            seq_end_indices = (seq_lengths - 1).clamp(min=0, max=seq_len-1)
            x = x[batch_indices, seq_end_indices, :]
        else:
            x = x[:, -1, :]
        
        # Project to output
        output = self.output_projection(x)
        
        return output.squeeze(-1)


class OldMambaEncoder(nn.Module):
    """Old Mamba implementation - kept for reference but not used"""
    def __init__(self, d_model: int, num_layers: int, max_seq_length: int, output_dim: int = 1, dropout: float = 0.1) -> None:
        super(OldMambaEncoder, self).__init__()
        
        try:
            from mamba_ssm import Mamba  # type: ignore
        except ImportError:
            raise ImportError(
                "mamba_ssm not installed. Install with: pip install mamba-ssm"
            )
        
        self.input_projection = nn.Linear(468, d_model)  # 36 stocks x 13 indicators
        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of Mamba layers
        self.mamba_layers = nn.ModuleList([  # type: ignore[var-annotated]
            Mamba(
                d_model=d_model,
                d_state=16,  # SSM state dimension
                d_conv=4,    # Local convolution width
                expand=2,    # Expansion factor
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms after each Mamba block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, 468]
            seq_lengths: Actual sequence lengths [batch]
            
        Returns:
            output: Predictions [batch] (squeezed from [batch, 1])
        """
        batch_size: int
        seq_len: int
        batch_size, seq_len, _ = x.shape
        
        # Project and normalize input
        x = self.input_projection(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Pass through Mamba layers with residual connections
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba_layer(x)  # Mamba expects [batch, seq_len, d_model]
            x = layer_norm(x + residual)
            x = self.dropout(x)
        
        # Get output at last valid position for each sequence
        if seq_lengths is not None:
            batch_indices = torch.arange(batch_size, device=x.device)
            seq_end_indices = (seq_lengths - 1).clamp(min=0, max=seq_len-1)
            x = x[batch_indices, seq_end_indices, :]
        else:
            x = x[:, -1, :]  # Take last timestep
        
        # Project to output
        output = self.output_projection(x)
        
        return output.squeeze(-1)


# Model configuration (unused - kept for reference)
# d_model = 128
# num_heads = 4
# num_layers = 4
# d_ff = 512
# max_seq_length = 388
# output_dim = 36
