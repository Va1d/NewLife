"""Analyze which stocks show the most bot-like trading patterns"""

import torch
from loader import Extender
import numpy as np
from pathlib import Path

def analyze_bot_activity():
    """Compute bot activity metrics for each of the 36 stocks"""
    
    ex = Extender()
    
    # Number of stocks
    num_stocks = ex.data.shape[2]
    print(f"Analyzing {num_stocks} stocks for bot activity patterns...\n")
    
    metrics = []
    
    for stock_idx in range(num_stocks):
        # Extract data for this stock (all days, all minutes)
        volume = ex.volume[:, :, stock_idx, 0]
        trade_count = ex.trade_count[:, :, stock_idx, 0]
        vwap = ex.vwap[:, :, stock_idx, 0]
        close = ex.close[:, :, stock_idx, 0]
        mask = ex.mask[:, :, stock_idx, 0]
        
        # Only analyze valid data
        valid_mask = mask == 1
        
        if valid_mask.sum() < 100:  # Skip if too little data
            continue
        
        # 1. Small trade size indicator (bots = many small trades)
        valid_volume = volume[valid_mask]
        valid_trades = trade_count[valid_mask]
        avg_trade_size = (valid_volume / (valid_trades + 1e-8)).mean().item()
        
        # 2. VWAP reversion rate (mean reversion bots)
        valid_vwap = vwap[valid_mask]
        valid_close = close[valid_mask]
        
        # Calculate how often price moves toward VWAP
        vwap_dev = (valid_close - valid_vwap) / (valid_vwap + 1e-8)
        # Check next-step reversion (if we have enough data)
        if len(vwap_dev) > 1:
            curr_dev_abs = torch.abs(vwap_dev[:-1])
            next_dev_abs = torch.abs(vwap_dev[1:])
            reversion_rate = (next_dev_abs < curr_dev_abs).float().mean().item()
        else:
            reversion_rate = 0.0
        
        # 3. Trade frequency consistency (bots trade regularly)
        # Standard deviation of trade count (lower = more consistent = bots)
        trade_consistency = 1.0 / (valid_trades.std().item() + 1e-8)
        
        # 4. Volume concentration (what % of time has high volume)
        volume_95th = torch.quantile(valid_volume, 0.95)
        volume_concentration = (valid_volume > volume_95th).float().mean().item()
        
        # 5. VWAP tracking tightness (how closely price follows VWAP)
        vwap_tracking = 1.0 / (torch.abs(vwap_dev).mean().item() + 1e-8)
        
        # 6. High-frequency trading indicator (high trade_count/volume)
        hft_ratio = (valid_trades / (valid_volume + 1e-8)).mean().item()
        
        # Composite bot score (weighted combination)
        bot_score = (
            0.2 * (1.0 / (avg_trade_size + 1e-8)) +  # Smaller trades = more bot-like
            0.25 * reversion_rate +                   # VWAP reversion
            0.15 * trade_consistency +                # Consistent trading
            0.15 * vwap_tracking +                    # Tight VWAP tracking
            0.25 * hft_ratio                          # High frequency
        )
        
        metrics.append({
            'stock_idx': stock_idx,
            'bot_score': bot_score,
            'avg_trade_size': avg_trade_size,
            'reversion_rate': reversion_rate,
            'trade_consistency': trade_consistency,
            'volume_concentration': volume_concentration,
            'vwap_tracking': vwap_tracking,
            'hft_ratio': hft_ratio,
            'valid_samples': valid_mask.sum().item()
        })
    
    # Sort by bot score
    metrics.sort(key=lambda x: x['bot_score'], reverse=True)
    
    print("=" * 100)
    print("BOT ACTIVITY RANKING (Higher score = more bot-like behavior)")
    print("=" * 100)
    print(f"{'Rank':<6} {'Stock':<8} {'Bot Score':<12} {'Reversion':<12} {'HFT Ratio':<12} {'Trade Size':<12} {'Valid Samples':<15}")
    print("-" * 100)
    
    for rank, m in enumerate(metrics[:36], 1):
        print(f"{rank:<6} {m['stock_idx']:<8} {m['bot_score']:<12.4f} {m['reversion_rate']:<12.3f} "
              f"{m['hft_ratio']:<12.6f} {m['avg_trade_size']:<12.1f} {m['valid_samples']:<15}")
    
    print("\n" + "=" * 100)
    print("KEY METRICS EXPLAINED:")
    print("=" * 100)
    print("Bot Score      : Composite metric (higher = more algorithmic trading)")
    print("Reversion      : How often price reverts to VWAP (mean reversion bots)")
    print("HFT Ratio      : Trades per volume unit (higher = smaller trades = scalping)")
    print("Trade Size     : Average $ per trade (lower = more bot-like)")
    print("Valid Samples  : Number of valid data points")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATION:")
    print("=" * 100)
    top_stock = metrics[0]
    print(f"Stock #{top_stock['stock_idx']} shows the strongest bot behavior:")
    print(f"  - VWAP reversion rate: {top_stock['reversion_rate']:.1%} (mean reversion bots)")
    print(f"  - HFT ratio: {top_stock['hft_ratio']:.6f} (high-frequency scalping)")
    print(f"  - Avg trade size: ${top_stock['avg_trade_size']:.0f} (small trades)")
    print(f"  - Bot score: {top_stock['bot_score']:.4f}")
    print(f"\nStart training with this stock for best chance of detecting bot patterns.")
    
    # Save results
    results_file = Path("/home/bo/Py/NewLife/bot_activity_analysis.txt")
    with open(results_file, 'w') as f:
        f.write("BOT ACTIVITY ANALYSIS\n")
        f.write("=" * 100 + "\n\n")
        for rank, m in enumerate(metrics[:36], 1):
            f.write(f"Rank {rank}: Stock {m['stock_idx']}\n")
            f.write(f"  Bot Score: {m['bot_score']:.4f}\n")
            f.write(f"  Reversion Rate: {m['reversion_rate']:.3f}\n")
            f.write(f"  HFT Ratio: {m['hft_ratio']:.6f}\n")
            f.write(f"  Avg Trade Size: {m['avg_trade_size']:.1f}\n")
            f.write(f"  Valid Samples: {m['valid_samples']}\n\n")
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return metrics

if __name__ == "__main__":
    analyze_bot_activity()
