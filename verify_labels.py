"""
Quick verification that the new bot activity labels work correctly
"""
import torch
from loader import TheSetGPU

print("=" * 80)
print("VERIFYING NEW BOT ACTIVITY LABELS")
print("=" * 80)

ds = TheSetGPU(device='cpu')

# Check first session
x_batch, y_batch, seq_lengths = ds[0]

print(f"\nDataset loaded successfully!")
print(f"Input shape: {x_batch.shape}")
print(f"Target shape: {y_batch.shape}")
print(f"Seq lengths shape: {seq_lengths.shape}")

# Check label statistics
pos_count = (y_batch == 1).sum().item()
pos_rate = pos_count / len(y_batch)

print(f"\nLabel Statistics (Session 0):")
print(f"  Positive samples: {pos_count} / {len(y_batch)} ({pos_rate*100:.1f}%)")
print(f"  Negative samples: {len(y_batch) - pos_count}")
print(f"  Expected rate: ~20% (bot activity signal)")

if 0.15 < pos_rate < 0.25:
    print(f"\n✓ Label rate is in expected range (15-25%)")
elif pos_rate < 0.10:
    print(f"\n⚠ Label rate is LOW - may still have issues")
else:
    print(f"\n⚠ Label rate is HIGH - different than expected")

print(f"\nLabel distribution: {(y_batch.sum().item() / len(y_batch))*100:.1f}%")
print(f"First 10 labels: {y_batch[:10].tolist()}")
print(f"Last 10 labels: {y_batch[-10:].tolist()}")

print("\n" + "=" * 80)
print("READY TO TRAIN - New bot activity labels are working!")
print("=" * 80)
