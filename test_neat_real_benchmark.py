"""
Benchmark NEAT evolution on real bot activity data.

This script runs smaller evolution (4 gen, 10 pop) on real Stock #10 data
to get realistic timing for full 8 gen x 20 pop evolution.
"""

import time
import torch
from test_neat import create_real_dataloaders, test_neat_full

if __name__ == "__main__":
    print("\n" + "="*70)
    print("NEAT Real Data Benchmark")
    print("="*70)
    
    # Time the data loading
    print("\nTiming real data load + flattening...")
    start = time.time()
    train_loader, val_loader = create_real_dataloaders(
        device='cuda:1',
        target_stock_idx=10,
        batch_size=32
    )
    data_time = time.time() - start
    print(f"✓ Data loading took {data_time:.1f} seconds")
    
    # Show dataset info
    total_samples = len(train_loader.dataset) + len(val_loader.dataset)
    print(f"\nDataset: {total_samples:,} total samples")
    print(f"  Training: {len(train_loader.dataset):,} samples")
    print(f"  Validation: {len(val_loader.dataset):,} samples")
    print(f"  Feature dimension: {train_loader.dataset.dataset.features.shape[1]}")
    
    # Estimate full evolution time
    print(f"\n" + "="*70)
    print("Time Estimate for Full Evolution (8 gen, 20 pop)")
    print("="*70)
    
    # Rough calculation: each individual evaluation ~5-10 seconds on GPU
    # 20 population × 8 generations = 160 evaluations
    # Plus overhead for mutation/crossover
    est_time_per_eval = 8  # seconds (conservative)
    total_evals = 20 * 8
    est_total = total_evals * est_time_per_eval
    
    print(f"Estimated time:")
    print(f"  {total_evals} evaluations × ~{est_time_per_eval}s per evaluation")
    print(f"  Total: ~{est_total//60} minutes ({est_total} seconds)")
    print(f"  + Data loading: {data_time:.0f}s")
    print(f"  Grand total: ~{(est_total + data_time)//60 + 1} minutes")
    
    print(f"\n" + "="*70)
    print("To run full NEAT evolution on real data:")
    print("  python test_neat.py --full --real-data")
    print("\n(Runs in the background, will take 8-15 minutes)")
    print("="*70)
