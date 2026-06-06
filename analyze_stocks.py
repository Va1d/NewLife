"""
Analyze all 36 stocks to find the one with cleanest bot activity signals
"""
import torch
import numpy as np
from loader import Extender
import time

print("=" * 100)
print("STOCK SIGNAL QUALITY ANALYSIS")
print("=" * 100)
print("\nAnalyzing bot activity signals across all 36 stocks...")
print("Loading data (this takes ~30 seconds)...\n")

# Store results for all stocks
results = []

for stock_idx in range(36):
    print(f"Stock {stock_idx:2d}: ", end="", flush=True)
    start = time.time()
    
    try:
        # Load only this stock's data
        ex = Extender(input_stock_idx=stock_idx, target_stock_idx=stock_idx)
        
        # Get all features (squeeze out stock/feature dimensions to get 1D time series)
        volume_velocity = ex.volume_velocity.squeeze()  # Remove singleton dimensions
        trade_velocity = ex.trade_velocity.squeeze()
        log_return = ex.log_return.squeeze()
        spread_ratio = ex.spread_ratio.squeeze()
        volume = ex.volume.squeeze()
        trade_count = ex.trade_count.squeeze()
        minute = ex.minute
        
        # Define potential bot signals (as bool first, then convert)
        volume_spike = (volume_velocity > 2.0)
        trade_spike = (trade_velocity > 1.5)
        price_stable = (torch.abs(log_return) < 0.005)
        v_spike_stable = (volume_spike & price_stable).float()
        
        # Ensemble: volume+stable OR trade spike
        ensemble = ((volume_spike & price_stable) | trade_spike).float()
        
        # Calculate statistics
        v_spike_rate = volume_spike.float().mean().item()
        t_spike_rate = trade_spike.float().mean().item()
        p_stable_rate = price_stable.float().mean().item()
        v_spike_stable_rate = v_spike_stable.mean().item()
        ensemble_rate = ensemble.mean().item()
        
        # Check for clustering (good signal) vs randomness (bad)
        ensemble_binary = (ensemble > 0).long()
        
        # Find positive streaks (if bots act consistently, we see streaks)
        if len(ensemble_binary) > 0:
            diffs = torch.diff(ensemble_binary)
            transitions = (diffs != 0).sum().item()
        else:
            transitions = 0
        
        # Positive run length statistics
        positive_indices = torch.where(ensemble > 0)[0]
        if len(positive_indices) > 0:
            # Calculate gap between positive samples
            gaps = torch.diff(positive_indices).float()
            avg_gap = gaps.mean().item() if len(gaps) > 0 else 0
            max_gap = gaps.max().item() if len(gaps) > 0 else 0
        else:
            avg_gap = float('inf')
            max_gap = float('inf')
        
        # Correlation between different signals
        v_spike_arr = volume_spike.float().numpy().flatten()
        t_spike_arr = trade_spike.float().numpy().flatten()
        p_stable_arr = price_stable.float().numpy().flatten()
        
        if v_spike_rate > 0.01 and t_spike_rate > 0.01 and p_stable_rate > 0.01:
            try:
                corr_vol_trade = np.corrcoef(v_spike_arr, t_spike_arr)[0, 1]
                corr_vol_stable = np.corrcoef(v_spike_arr, p_stable_arr)[0, 1]
            except:
                corr_vol_trade = 0
                corr_vol_stable = 0
        else:
            corr_vol_trade = 0
            corr_vol_stable = 0
        
        # Data quality metric: lower variance in rates = cleaner signal
        signal_variance = np.var([v_spike_rate, t_spike_rate, p_stable_rate, ensemble_rate])
        
        elapsed = time.time() - start
        
        results.append({
            'stock_idx': stock_idx,
            'vol_spike_pct': v_spike_rate,
            'trade_spike_pct': t_spike_rate,
            'price_stable_pct': p_stable_rate,
            'vol_spike_stable_pct': v_spike_stable_rate,
            'ensemble_pct': ensemble_rate,
            'transitions': transitions,
            'avg_gap': avg_gap,
            'corr_vol_trade': corr_vol_trade,
            'corr_vol_stable': corr_vol_stable,
            'signal_variance': signal_variance,
            'total_samples': len(ensemble),
        })
        
        # Print quick stats
        print(f"Vol:{v_spike_rate*100:5.1f}% Trade:{t_spike_rate*100:5.1f}% Stable:{p_stable_rate*100:5.1f}% Ensem:{ensemble_rate*100:5.1f}%")
        
    except Exception as e:
        print(f"ERROR: {str(e)[:50]}")
        continue

print("\n" + "=" * 100)
print("ANALYSIS RESULTS - Ranked by Signal Quality")
print("=" * 100)

# Rank by ensemble label rate closest to 50% (clearest non-trivial signal)
results_sorted = sorted(results, key=lambda x: abs(x['ensemble_pct'] - 0.5))

print("\nTop 10 Stocks by Ensemble Signal Quality (50% is ideal for binary classification):\n")
print("Rank | Stock | Ensemble% | Vol% | Trade% | P-Stable% | Transitions | Avg Gap | Signal Var | Quality\n")
print("-" * 120)

quality_scores = []
for rank, r in enumerate(results_sorted[:10]):
    ensemble_quality = 1.0 - abs(r['ensemble_pct'] - 0.5) * 2  # 1.0 at 50%, 0 at 0% or 100%
    
    # Bonus for consistency (good transitions = signal clusters)
    clustering_bonus = min(1.0, r['transitions'] / 1000.0) * 0.2  # Fewer transitions = more clustering
    
    # Bonus for signal correlation (signals reinforce each other)
    correlation_bonus = min(0.1, abs(r['corr_vol_stable']))  # Some correlation is good
    
    quality = (ensemble_quality * 0.7 + clustering_bonus + correlation_bonus)
    quality_scores.append(quality)
    
    print(f"{rank+1:3d}  | {r['stock_idx']:3d}   | {r['ensemble_pct']*100:7.1f}  | {r['vol_spike_pct']*100:5.1f} | {r['trade_spike_pct']*100:6.1f} | {r['price_stable_pct']*100:8.1f} | {r['transitions']:11d} | {r['avg_gap']:7.1f} | {r['signal_variance']:9.4f} | {quality:.3f}")

# Re-rank by quality score
results_by_quality = list(zip(results_sorted[:10], quality_scores))
results_by_quality.sort(key=lambda x: x[1], reverse=True)

print("\n" + "=" * 100)
print("FINAL RANKING - Best Stocks for Bot Activity Detection")
print("=" * 100)
print("\nRank | Stock | Quality | Ensemble% | Rationale\n")

best_stock = None
for rank, (r, quality) in enumerate(results_by_quality):
    reasoning = f"Ensemble: {r['ensemble_pct']*100:.1f}%"
    
    if r['ensemble_pct'] > 0.55:
        reasoning += " ✓ Strong signal"
    elif r['ensemble_pct'] < 0.35:
        reasoning += " ✗ Weak signal"
    else:
        reasoning += " ≈ Balanced"
    
    if r['transitions'] < 800:
        reasoning += ", Clustered"
    
    print(f"{rank+1:3d}  | {r['stock_idx']:3d}   | {quality:.3f}  | {r['ensemble_pct']*100:7.1f}   | {reasoning}")
    
    if rank == 0:
        best_stock = r['stock_idx']

print("\n" + "=" * 100)
print(f"RECOMMENDATION: Use Stock #{best_stock}")
print("=" * 100)

# Print detailed stats for best stock
if len(results_by_quality) > 0:
    best_r = results_by_quality[0][0]
    print(f"\nDetailed Stats for Stock #{best_stock}:")
    print(f"  Ensemble Signal Rate: {best_r['ensemble_pct']*100:.1f}%")
    print(f"  → Volume spike + price stable: {best_r['vol_spike_stable_pct']*100:.1f}%")
    print(f"  → Trade spike: {best_r['trade_spike_pct']*100:.1f}%")
    print(f"  Volume Spike Rate: {best_r['vol_spike_pct']*100:.1f}%")
    print(f"  Trade Spike Rate: {best_r['trade_spike_pct']*100:.1f}%")
    print(f"  Price Stable Rate: {best_r['price_stable_pct']*100:.1f}%")
    print(f"  Signal Clustering: {best_r['transitions']} transitions in {best_r['total_samples']} samples")
    print(f"  Avg time between signals: {best_r['avg_gap']:.1f} steps")

    print("\n✓ This stock has the cleanest bot activity signals!")
    print(f"\nTo use this stock, update loader.py to use stock_idx={best_stock} by default")
    print("and implement the ensemble bot activity label.")
else:
    print("\n✗ No valid stocks found in analysis. Check error messages above.")
print("=" * 100)
