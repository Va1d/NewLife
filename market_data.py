"""
Raw market data extraction and normalization for backtesting.

Converts TheSetGPU sequences into normalized OHLCV features suitable for bot trading.
"""

import torch
import numpy as np
from typing import Tuple, List
from loader import TheSetGPU


class RawMarketDataProvider:
    """
    Provides normalized raw market data for backtesting NEAT trading bots.
    
    Extracts key market features from multi-step sequences:
    - Price changes (normalized)
    - Volume changes (normalized)
    - Momentum indicators
    - Volatility
    
    All features normalized to [-1, 1] range.
    """
    
    def __init__(self, device: str = 'cuda:1', target_stock_idx: int = 10, normalize: bool = True):
        """
        Initialize market data provider.
        
        Args:
            device: GPU device for data loading
            target_stock_idx: Which stock index to use
            normalize: Whether to normalize features to [-1, 1]
        """
        self.device = device
        self.target_stock_idx = target_stock_idx
        self.normalize = normalize
        
        print(f"Loading TheSetGPU for Stock #{target_stock_idx}...")
        self.dataset = TheSetGPU(device=device, target_stock_idx=target_stock_idx)
        
        # Pre-compute all raw features and returns
        self._prepare_data()
    
    def _prepare_data(self):
        """Pre-compute normalized features and returns for all sequences."""
        print(f"Pre-processing {len(self.dataset)} sequences...")
        
        all_features = []
        all_returns = []
        
        # Limit to first 10 sequences for speed (still ~2500 samples)
        num_seqs = min(10, len(self.dataset))
        
        for i in range(num_seqs):
            x_batch, y_batch, seq_lengths = self.dataset[i]
            
            # Extract features from sequence
            features = self._extract_features(x_batch)  # [256, n_features]
            returns = self._extract_returns(x_batch)    # [256]
            
            all_features.append(features)
            all_returns.append(returns)
            
            print(f"  Processed {i+1}/{num_seqs} sequences...")
        
        # Stack all
        self.features = torch.cat(all_features, dim=0)  # [256*N, n_features]
        self.returns = torch.cat(all_returns, dim=0)     # [256*N]
        
        # Normalize features
        if self.normalize:
            self._normalize_features()
        
        print(f"\nMarket data ready:")
        print(f"  Total steps: {len(self.features):,}")
        print(f"  Features: {self.features.shape[1]} dimensions")
        print(f"  Return stats: mean={self.returns.mean():.4f}, std={self.returns.std():.4f}")
        print(f"  Asset growth: {(1 + self.returns).prod().item():.2f}x (buy & hold)")
    
    def _extract_features(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract raw market features from sequence.
        
        Args:
            x_batch: [256 steps, 388 time_steps, 468 features]
        
        Returns:
            features: [256, n_features] normalized to approximately [-1, 1]
        """
        n_steps = x_batch.shape[0]
        
        # For each prediction step, compute market statistics
        features_list = []
        
        for step_idx in range(n_steps):
            step_seq = x_batch[step_idx]  # [388 time, 468 features]
            
            # Extract key market indicators from the enriched features
            # Assuming first features are price-related, then volume, then derived metrics
            
            # Price momentum: last - first prices (normalized by mean)
            price_change = (step_seq[-1, 0] - step_seq[0, 0]) / (step_seq[:, 0].abs().mean() + 1e-8)
            
            # Volume trend: avg volume in last half vs first half
            mid = step_seq.shape[0] // 2
            vol_early = step_seq[:mid, 1].mean() if step_seq.shape[1] > 1 else torch.tensor(0.0, device=self.device)
            vol_late = step_seq[mid:, 1].mean() if step_seq.shape[1] > 1 else torch.tensor(0.0, device=self.device)
            vol_trend = (vol_late - vol_early) / (vol_early.abs() + 1e-8)
            
            # Volatility: standard deviation of price in this window
            volatility = step_seq[:, 0].std() / (step_seq[:, 0].abs().mean() + 1e-8)
            
            # Momentum: average price velocity
            price_diffs = torch.diff(step_seq[:, 0])
            momentum = price_diffs.mean() / (price_diffs.abs().mean() + 1e-8)
            
            # Average features (if more than 2 dimensions available)
            avg_features = step_seq.mean(dim=0)[:min(5, step_seq.shape[1])]  # Top 5 features averaged
            
            # Combine all indicators
            step_features = torch.cat([
                torch.tensor([price_change, vol_trend, volatility, momentum], device=self.device),
                avg_features
            ])
            
            features_list.append(step_features)
        
        # Pad to same size
        max_len = max(f.shape[0] for f in features_list)
        padded = []
        for f in features_list:
            if f.shape[0] < max_len:
                padding = torch.zeros(max_len - f.shape[0], device=self.device)
                f = torch.cat([f, padding])
            padded.append(f)
        
        return torch.stack(padded)  # [256, max_len]
    
    def _extract_returns(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract forward returns (what happens to price next step).
        
        Since enriched features may not be raw prices, we use the first feature
        (price changes) which should be normalized.
        
        Args:
            x_batch: [256 steps, 388 time_steps, 468 features]
        
        Returns:
            returns: [256] - normalized price change from this step to next
        """
        n_steps = x_batch.shape[0]
        returns = []
        
        for step_idx in range(n_steps - 1):
            # Use momentum/price change from enriched features
            # Average the price changes across the time window
            curr_changes = x_batch[step_idx, :, 0]  # First feature: price change
            next_changes = x_batch[step_idx + 1, :, 0]
            
            # Calculate simple average return
            ret = (next_changes.mean() - curr_changes.mean()) / (curr_changes.abs().mean() + 1e-8)
            returns.append(ret)
        
        # Last step has no forward return, use 0
        returns.append(torch.tensor(0.0, device=self.device))
        
        return torch.stack(returns)
    
    def _normalize_features(self):
        """Normalize all features to approximately [-1, 1] range."""
        print("Normalizing features...")
        
        mean = self.features.mean(dim=0)
        std = self.features.std(dim=0) + 1e-8
        
        self.features = (self.features - mean) / std
        
        # Clip to [-3, 3] to handle outliers
        self.features = torch.clamp(self.features, -3, 3)
        
        print("  ✓ Features normalized")
    
    def get_batch(self, start_idx: int, end_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of features and returns.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index (exclusive)
        
        Returns:
            (features, returns) - normalized market data and forward returns
        """
        return self.features[start_idx:end_idx], self.returns[start_idx:end_idx]
    
    def __len__(self) -> int:
        """Total number of trading steps available."""
        return len(self.features)
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all features and returns for backtesting."""
        return self.features, self.returns


if __name__ == "__main__":
    # Test data provider
    print("Testing RawMarketDataProvider...\n")
    
    provider = RawMarketDataProvider(device='cuda:1', target_stock_idx=10, normalize=True)
    
    # Sample some data
    features, returns = provider.get_batch(0, 100)
    
    print(f"\nSample data (first 100 steps):")
    print(f"  Features shape: {features.shape}")
    print(f"  Returns shape: {returns.shape}")
    print(f"  Feature stats: mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"  Return stats: mean={returns.mean():.4f}, std={returns.std():.4f}")
    
    # Test profitability
    cumulative_return = (1 + returns).prod().item()
    print(f"\n  Buy & hold return: {(cumulative_return - 1) * 100:.2f}%")
