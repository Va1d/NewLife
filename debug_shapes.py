"""
Debug script to check tensor shapes
"""
import torch
from loader import Extender

print("Loading Extender...")
ex = Extender(target_stock_idx=10)

print(f"\nvola_velocity shape: {ex.volume_velocity.shape}")
print(f"trade_velocity shape: {ex.trade_velocity.shape}")
print(f"log_return shape: {ex.log_return.shape}")
print(f"close shape: {ex.close.shape}")

# Check what happens when we create signals
v_spike = ex.volume_velocity > 2.0
p_stable = torch.abs(ex.log_return) < 0.005
t_spike = ex.trade_velocity > 1.5

print(f"\nv_spike shape: {v_spike.shape}")
print(f"p_stable shape: {p_stable.shape}")
print(f"t_spike shape: {t_spike.shape}")

combined = (v_spike & p_stable) | t_spike
print(f"combined shape: {combined.shape}")

# Try the max approach
result = combined.squeeze(-1).max(dim=2)[0].float()
print(f"result (after max) shape: {result.shape}")

# Check the last 256
target_256 = result[:, -256:]
print(f"target_256 shape: {target_256.shape}")
print(f"target_256 positive rate: {target_256.float().mean().item():.1%}")
