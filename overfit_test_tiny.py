"""
Diagnostic 3: Tiny model without KL to test label definition quality
If even a small model can't overfit, the problem is the LABELS, not model size
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import BayesianTransformer
from loader import TheSetGPU

device = torch.device('cuda:1')
dataset = TheSetGPU(device=str(device))

print("\n" + "=" * 80)
print("DIAGNOSTIC 3: TINY MODEL (d_model=32) WITHOUT KL")
print("Testing if label definition is too noisy for ANY model to learn")
print("=" * 80)

tiny_train_idx = [0, 1]
tiny_test_idx = 0

# MUCH smaller model
model = BayesianTransformer(
    d_model=32,           # 256 → 32
    num_heads=4,          # 8 → 4
    d_ff=128,             # 1024 → 128
    num_layers=1,         # 3 → 1
    max_seq_length=388, 
    output_dim=1,
    dropout=0.0,
    prior_mu=0.0,
    prior_sigma=1.0
).to(device)

# Higher learning rate for smaller model
optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=0.0)
criterion = nn.BCEWithLogitsLoss()

print("\nEpoch | Train Loss | Test Loss | Train Acc | Test Acc | Status")
print("-" * 70)

for epoch in range(100):
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for idx in tiny_train_idx:
        x_batch, y_batch, seq_lengths = dataset[idx]
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        
        loss = criterion(logits, y_batch.float()).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
        
        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        train_correct += (preds == y_batch).sum().item()
        train_total += len(y_batch)
    
    train_loss /= len(tiny_train_idx)
    train_acc = train_correct / train_total
    
    # Test
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        test_loss = criterion(logits, y_batch.float()).mean().item()
        
        preds = (torch.sigmoid(logits) > 0.5).float()
        test_acc = (preds == y_batch).sum().item() / len(y_batch)
    
    if train_loss < 0.1 and test_loss > 0.3 and train_acc > 0.9:
        status = "✓ OVERFITTING"
    elif train_acc > 0.95:
        status = "✓ MEMORIZING"
    else:
        status = ""
    
    if epoch % 10 == 0 or epoch < 5:
        print(f"{epoch:3d}  | {train_loss:10.4f} | {test_loss:9.4f} | {train_acc:9.1%} | {test_acc:8.1%} | {status}")

print("\n" + "=" * 80)
print("RESULT:")
print("=" * 80)
print(f"Final: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
print(f"       Train Acc={train_acc:.1%}, Test Acc={test_acc:.1%}")

if train_acc > 0.9:
    print("\n✓ Even TINY model can memorize training data")
    print("  → Problem: Your labels contain too much noise/information")
    print("  → Solution: Check label definition - is VWAP mean reversion predictable?")
else:
    print("\n✗ Even TINY model cannot memorize training data")
    print("  → Problem: Severe label noise or data pipeline issue")
    print("  → Solution: Manually inspect y_batch and x_batch to verify correlation")

print("=" * 80)
