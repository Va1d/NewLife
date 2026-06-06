"""
Optimal Model Sizing Analysis for All Models
Uses correct methodology: find minimum model that CAN overfit, then reduce by 50-70%
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import (
    TransformerEncoder, TemporalFusionTransformer, BayesianTransformer,
    MCDropoutTransformer, MambaEncoder, xLSTMEncoder
)
from loader import TheSetGPU
import time

device = torch.device('cuda:1')
dataset = TheSetGPU(device=str(device))

print("=" * 100)
print("OPTIMAL MODEL SIZING ANALYSIS")
print("=" * 100)
print(f"\nDataset Characteristics:")
print(f"  Total sessions: {len(dataset)}")
print(f"  Samples per session: 256")
print(f"  Total training samples: ~{len(dataset) * 256 / 1000:.0f}K")
print(f"  Input dimension: 468")
print(f"  Sequence length: max 388")
print(f"  Output dimension: 1 (binary)")
print(f"  Label positive rate: ~18% (bot activity)")

# Dataset
tiny_train_idx = [0, 1]  # Just 2 sessions = ~512 samples
tiny_test_idx = 0
num_epochs = 50

print(f"\nOverfitting Test Configuration:")
print(f"  Train sessions: {len(tiny_train_idx)} (~{len(tiny_train_idx)*256} samples)")
print(f"  Test session: 1 (~256 samples)")
print(f"  Goal: Find minimum model that CAN overfit (train loss <0.2, test loss >0.5)")

# Get a sample batch to verify shapes
x_sample, y_sample, seq_lengths_sample = dataset[0]
print(f"\nSample batch shapes:")
print(f"  X: {x_sample.shape}")
print(f"  Y: {y_sample.shape}")
print(f"  Seq lengths: {seq_lengths_sample.shape}")

results = {}

# ============================================================================
# Model 1: TransformerEncoder
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 1: TransformerEncoder")
print("=" * 100)

model_configs = [
    {'name': 'Tiny', 'd_model': 32, 'num_heads': 2, 'd_ff': 64, 'num_layers': 1},
    {'name': 'Small', 'd_model': 64, 'num_heads': 4, 'd_ff': 256, 'num_layers': 2},
    {'name': 'Medium', 'd_model': 128, 'num_heads': 8, 'd_ff': 512, 'num_layers': 3},
    {'name': 'Large', 'd_model': 256, 'num_heads': 8, 'd_ff': 1024, 'num_layers': 4},
]

print("\nSize | d_model | heads | ff   | layers | Params  | Train Loss | Test Loss | Can Overfit?")
print("-" * 90)

for config in model_configs:
    model = TransformerEncoder(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_length=388,
        output_dim=1
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    
    # Train on tiny set
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idx in tiny_train_idx:
            x_batch, y_batch, seq_lengths = dataset[idx]
            logits = model(x_batch, seq_lengths=seq_lengths)
            batch_indices = torch.arange(256, device=device)
            seq_end_indices = seq_lengths - 1
            logits_end = logits[batch_indices, seq_end_indices, :].squeeze(-1)
            loss = criterion(logits_end, y_batch.float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tiny_train_idx)
    
    # Test
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits = model(x_batch, seq_lengths=seq_lengths)
        batch_indices = torch.arange(256, device=device)
        seq_end_indices = seq_lengths - 1
        logits_end = logits[batch_indices, seq_end_indices, :].squeeze(-1)
        test_loss = criterion(logits_end, y_batch.float()).mean().item()
    
    can_overfit = train_loss < 0.2 and test_loss > 0.5
    
    print(f"{config['name']:4s} | {config['d_model']:7d} | {config['num_heads']:5d} | {config['d_ff']:4d} | {config['num_layers']:6d} | {num_params:7d} | {train_loss:10.4f} | {test_loss:9.4f} | {'✓ YES' if can_overfit else '✗ NO'}")
    
    results[f"TransformerEncoder-{config['name']}"] = {
        'params': num_params,
        'config': config,
        'can_overfit': can_overfit,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

# ============================================================================
# Model 2: TemporalFusionTransformer
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 2: TemporalFusionTransformer (TFT)")
print("=" * 100)

model_configs = [
    {'name': 'Tiny', 'd_model': 32, 'num_heads': 2, 'd_ff': 64, 'num_layers': 1},
    {'name': 'Small', 'd_model': 64, 'num_heads': 4, 'd_ff': 256, 'num_layers': 2},
    {'name': 'Medium', 'd_model': 128, 'num_heads': 8, 'd_ff': 512, 'num_layers': 2},
    {'name': 'Large', 'd_model': 256, 'num_heads': 8, 'd_ff': 1024, 'num_layers': 2},
]

print("\nSize | d_model | heads | ff   | layers | Params  | Train Loss | Test Loss | Can Overfit?")
print("-" * 90)

for config in model_configs:
    model = TemporalFusionTransformer(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_length=388,
        output_dim=1,
        dropout=0.1,
        use_causal_mask=True
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idx in tiny_train_idx:
            x_batch, y_batch, seq_lengths = dataset[idx]
            logits = model(x_batch, seq_lengths=seq_lengths)
            batch_indices = torch.arange(256, device=device)
            seq_end_indices = seq_lengths - 1
            logits_end = logits[batch_indices, seq_end_indices, :].squeeze(-1)
            loss = criterion(logits_end, y_batch.float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tiny_train_idx)
    
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits = model(x_batch, seq_lengths=seq_lengths)
        batch_indices = torch.arange(256, device=device)
        seq_end_indices = seq_lengths - 1
        logits_end = logits[batch_indices, seq_end_indices, :].squeeze(-1)
        test_loss = criterion(logits_end, y_batch.float()).mean().item()
    
    can_overfit = train_loss < 0.2 and test_loss > 0.5
    
    print(f"{config['name']:4s} | {config['d_model']:7d} | {config['num_heads']:5d} | {config['d_ff']:4d} | {config['num_layers']:6d} | {num_params:7d} | {train_loss:10.4f} | {test_loss:9.4f} | {'✓ YES' if can_overfit else '✗ NO'}")
    
    results[f"TFT-{config['name']}"] = {
        'params': num_params,
        'config': config,
        'can_overfit': can_overfit,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

# ============================================================================
# Model 3: BayesianTransformer
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 3: BayesianTransformer")
print("=" * 100)

model_configs = [
    {'name': 'Tiny', 'd_model': 32, 'num_heads': 2, 'd_ff': 64, 'num_layers': 1},
    {'name': 'Small', 'd_model': 64, 'num_heads': 4, 'd_ff': 256, 'num_layers': 2},
    {'name': 'Medium', 'd_model': 128, 'num_heads': 8, 'd_ff': 512, 'num_layers': 2},
    {'name': 'Large', 'd_model': 256, 'num_heads': 8, 'd_ff': 1024, 'num_layers': 3},
]

print("\nSize | d_model | heads | ff   | layers | Params  | Train Loss | Test Loss | Can Overfit?")
print("-" * 90)

for config in model_configs:
    model = BayesianTransformer(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_length=388,
        output_dim=1,
        dropout=0.05,
        prior_mu=0.0,
        prior_sigma=1.0
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idx in tiny_train_idx:
            x_batch, y_batch, seq_lengths = dataset[idx]
            logits, kl = model(x_batch, seq_lengths=seq_lengths)
            bce = criterion(logits, y_batch.float()).mean()
            # Very weak KL for overfitting test
            loss = bce + 1e-4 * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tiny_train_idx)
    
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        bce = criterion(logits, y_batch.float()).mean()
        test_loss = (bce + 1e-4 * kl).item()
    
    can_overfit = train_loss < 0.2 and test_loss > 0.5
    
    print(f"{config['name']:4s} | {config['d_model']:7d} | {config['num_heads']:5d} | {config['d_ff']:4d} | {config['num_layers']:6d} | {num_params:7d} | {train_loss:10.4f} | {test_loss:9.4f} | {'✓ YES' if can_overfit else '✗ NO'}")
    
    results[f"BayesianTransformer-{config['name']}"] = {
        'params': num_params,
        'config': config,
        'can_overfit': can_overfit,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

# ============================================================================
# Model 4: MCDropoutTransformer
# ============================================================================
print("\n" + "=" * 100)
print("MODEL 4: MCDropoutTransformer")
print("=" * 100)

model_configs = [
    {'name': 'Tiny', 'd_model': 32, 'num_heads': 2, 'd_ff': 64, 'num_layers': 1},
    {'name': 'Small', 'd_model': 64, 'num_heads': 4, 'd_ff': 256, 'num_layers': 2},
    {'name': 'Medium', 'd_model': 128, 'num_heads': 8, 'd_ff': 512, 'num_layers': 2},
    {'name': 'Large', 'd_model': 256, 'num_heads': 8, 'd_ff': 1024, 'num_layers': 3},
]

print("\nSize | d_model | heads | ff   | layers | Params  | Train Loss | Test Loss | Can Overfit?")
print("-" * 90)

for config in model_configs:
    model = MCDropoutTransformer(
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        max_seq_length=388,
        output_dim=1,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idx in tiny_train_idx:
            x_batch, y_batch, seq_lengths = dataset[idx]
            logits = model(x_batch, seq_lengths=seq_lengths)
            loss = criterion(logits, y_batch.float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(tiny_train_idx)
    
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits = model(x_batch, seq_lengths=seq_lengths)
        test_loss = criterion(logits, y_batch.float()).mean().item()
    
    can_overfit = train_loss < 0.2 and test_loss > 0.5
    
    print(f"{config['name']:4s} | {config['d_model']:7d} | {config['num_heads']:5d} | {config['d_ff']:4d} | {config['num_layers']:6d} | {num_params:7d} | {train_loss:10.4f} | {test_loss:9.4f} | {'✓ YES' if can_overfit else '✗ NO'}")
    
    results[f"MCDropout-{config['name']}"] = {
        'params': num_params,
        'config': config,
        'can_overfit': can_overfit,
        'train_loss': train_loss,
        'test_loss': test_loss
    }

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)

# Summary and recommendations
print("\n" + "=" * 100)
print("RECOMMENDATIONS FOR PRODUCTION TRAINING")
print("=" * 100)

print("\nRationale:")
print("  • Minimum overfitting capability: Model should pass tiny data test (train<0.2, test>0.5)")
print("  • Production size: Use 70-80% of parameters that CAN overfit")
print("  • This prevents overfitting on full dataset while maintaining capacity to learn")

model_types = {
    'TransformerEncoder': [(k, v) for k, v in results.items() if 'TransformerEncoder' in k],
    'TemporalFusionTransformer': [(k, v) for k, v in results.items() if 'TFT-' in k],
    'BayesianTransformer': [(k, v) for k, v in results.items() if 'BayesianTransformer' in k],
    'MCDropoutTransformer': [(k, v) for k, v in results.items() if 'MCDropout' in k],
}

for model_name, configs in model_types.items():
    print(f"\n{model_name}:")
    
    # Find smallest that can overfit
    can_overfit_configs = [c for c in configs if c[1]['can_overfit']]
    
    if can_overfit_configs:
        smallest_overfit = min(can_overfit_configs, key=lambda x: x[1]['params'])
        config_name, data = smallest_overfit
        
        params = data['params']
        recommended_params = int(params * 0.75)  # 75% of max
        
        print(f"  ✓ Can overfit: Smallest is {config_name.split('-')[1]} ({params:,} params)")
        print(f"  → Production config (75% of capacity): {data['config']}")
        print(f"  → Estimated production params: ~{recommended_params:,}")
    else:
        largest = max(configs, key=lambda x: x[1]['params'])
        print(f"  ✗ Cannot overfit even at largest. Consider:")
        print(f"    - Increase num_layers")
        print(f"    - Reduce regularization more")
        print(f"    - Use current largest: {largest[0].split('-')[1]} ({largest[1]['params']:,} params)")

print("\n" + "=" * 100)
