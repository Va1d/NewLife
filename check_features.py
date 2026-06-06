"""Find which feature has extreme values."""

from loader import Extender
import torch

ex = Extender()

features = {
    'volume_zs': ex.volume_zs,
    'trade_count_zs': ex.trade_count_zs,
   'vwap_zs': ex.vwap_zs,
    'vwap_deviation': ex.vwap_deviation,
    'volume_velocity': ex.volume_velocity,
    'trade_velocity': ex.trade_velocity,
    'trade_size': ex.trade_size,
    'rsi': ex.rsi,
    'true_range_zs': ex.true_range_zs,
    'spread_ratio': ex.spread_ratio,
    'log_return': ex.log_return,
    'flow': ex.flow,
    'minute': ex.minute
}

print("Feature statistics:")
print("=" * 100)
print(f"{'Feature':<20} {'Min':<15} {'Max':<15} {'Mean':<15} {'Std':<15}")
print("-" * 100)

for name, feat in features.items():
    print(f"{name:<20} {feat.min().item():<15.4f} {feat.max().item():<15.4f} {feat.mean().item():<15.4f} {feat.std().item():<15.4f}")
