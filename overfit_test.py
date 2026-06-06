"""
Overfitting diagnostic test - determines if model can learn at all
Run this to identify whether regularization is blocking learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import BayesianTransformer
from loader import TheSetGPU
import sys

# Setup
device = torch.device('cuda:1')
dataset = TheSetGPU(device=str(device))

print("=" * 80)
print("OVERFITTING DIAGNOSTIC TEST")
print("=" * 80)

# First: check class distribution
print("\n1. Checking class distribution in training data...")
for idx in range(min(5, len(dataset))):
    _, y_batch, _ = dataset[idx]
    positive_rate = (y_batch == 1).sum().item() / len(y_batch)
    print(f"   Session {idx}: {positive_rate:.1%} positive samples ({(y_batch == 1).sum().item()}/256)")

# Second: tiny overfit test with ZERO regularization
print("\n2. Tiny subset overfitting test (2 training sessions vs 1 test session)...")
print("   Configuration: NO weight decay, NO dropout, NO label smoothing")
print("   Strong KL weight to test Bayesian impact\n")

tiny_train_idx = [0, 1]
tiny_test_idx = 0

model = BayesianTransformer(
    d_model=256, 
    num_heads=8, 
    d_ff=1024,
    num_layers=3, 
    max_seq_length=388, 
    output_dim=1,
    dropout=0.0,           # NO DROPOUT - force overfitting
    prior_mu=0.0,
    prior_sigma=1.0        # Weak prior to allow learning
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)  # High LR, no decay
criterion = nn.BCEWithLogitsLoss()

print("Epoch | Train Loss | Test Loss | Gap (Test-Train) | Status")
print("-" * 65)

best_test_loss = float('inf')
train_losses = []
test_losses = []

for epoch in range(100):
    train_loss = 0.0
    
    # Train on tiny set
    for idx in tiny_train_idx:
        x_batch, y_batch, seq_lengths = dataset[idx]
        
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        
        # BCE loss
        bce_loss = criterion(logits, y_batch.float()).mean()
        
        # Add KL divergence - TEST WITH STRONG WEIGHT
        kl_weight = 1.0  # Strong KL
        loss = bce_loss + kl_weight * kl
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(tiny_train_idx)
    train_losses.append(train_loss)
    
    # Test
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        bce_loss = criterion(logits, y_batch.float()).mean()
        kl_weight = 1.0
        test_loss = (bce_loss + kl_weight * kl).item()
    
    test_losses.append(test_loss)
    gap = test_loss - train_loss
    
    best_test_loss = min(best_test_loss, test_loss)
    
    # Status indicators
    if train_loss < 0.1 and test_loss > 0.7:
        status = "✓ OVERFITTING DETECTED"
    elif train_loss > test_loss:
        status = "✗ Test better than train (random)"
    elif train_loss < 0.1:
        status = "✓ Training converging"
    else:
        status = ""
    
    if epoch % 10 == 0 or epoch < 5:
        print(f"{epoch:3d}  | {train_loss:10.4f} | {test_loss:9.4f} | {gap:16.4f} | {status}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

# Interpret results
min_train = min(train_losses)
min_test = min(test_losses)
final_train = train_losses[-1]
final_test = test_losses[-1]

print(f"\nFinal metrics (epoch 99):")
print(f"  Train Loss: {final_train:.4f}")
print(f"  Test Loss:  {final_test:.4f}")
print(f"  Gap:        {final_test - final_train:.4f}")

print(f"\nBest values across 100 epochs:")
print(f"  Best Train Loss: {min_train:.4f}")
print(f"  Best Test Loss:  {min_test:.4f}")

if min_train < 0.1 and min_test > 0.6:
    print("\n✓ DIAGNOSIS: Model CAN overfit when regularization removed")
    print("  → Problem: REGULARIZATION IS TOO STRONG in main training")
    print("  → Solution: Reduce weight_decay, dropout, label_smoothing, KL weight")
elif final_test < final_train:
    print("\n✗ DIAGNOSIS: Test loss BETTER than train (not learning)")
    print("  → Problem: KL divergence or optimizer config")
    print("  → Next test: Try without KL divergence")
else:
    print("\n? DIAGNOSIS: Model not converging even on tiny data")
    print("  → Problem: Data signal quality or label definition")
    print("  → Action: Check if you can manually classify these samples")

print("\n" + "=" * 80)
print("\nRECOMMENDATION:")
print("If overfitting detected above:")
print("  1. Run main training with: weight_decay=1e-6, dropout=0.05, label_smooth=0")
print("  2. Reduce KL weight scaling (use 1.0/len(train_indices) instead of scaling)")
print("  3. Increase learning_rate to 0.001")
print("\nIf overfitting NOT detected:")
print("  1. KL divergence may be too strong - test with prior_sigma=10.0")
print("  2. Or label definition is too noisy - check y_batch manually")
print("=" * 80)
