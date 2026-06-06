"""
Second diagnostic: Test WITHOUT KL divergence to isolate the problem
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import BayesianTransformer
from loader import TheSetGPU

device = torch.device('cuda:1')
dataset = TheSetGPU(device=str(device))

print("\n" + "=" * 80)
print("DIAGNOSTIC 2: SAME TEST BUT WITHOUT KL DIVERGENCE")
print("=" * 80)

tiny_train_idx = [0, 1]
tiny_test_idx = 0

model = BayesianTransformer(
    d_model=256, 
    num_heads=8, 
    d_ff=1024,
    num_layers=3, 
    max_seq_length=388, 
    output_dim=1,
    dropout=0.0,
    prior_mu=0.0,
    prior_sigma=1.0
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
criterion = nn.BCEWithLogitsLoss()

print("\nEpoch | Train Loss | Test Loss | Gap (Test-Train) | Status")
print("-" * 65)

for epoch in range(100):
    train_loss = 0.0
    
    for idx in tiny_train_idx:
        x_batch, y_batch, seq_lengths = dataset[idx]
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        
        # ONLY BCE LOSS - NO KL
        loss = criterion(logits, y_batch.float()).mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(tiny_train_idx)
    
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        logits, kl = model(x_batch, seq_lengths=seq_lengths)
        test_loss = criterion(logits, y_batch.float()).mean().item()
    
    gap = test_loss - train_loss
    
    if train_loss < 0.1 and test_loss > 0.6:
        status = "✓ CLEAR OVERFITTING"
    elif train_loss > test_loss:
        status = "?"
    elif train_loss < 0.2:
        status = "✓ Converging"
    else:
        status = ""
    
    if epoch % 10 == 0 or epoch < 5:
        print(f"{epoch:3d}  | {train_loss:10.4f} | {test_loss:9.4f} | {gap:16.4f} | {status}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print(f"\nFinal Train Loss (without KL): {train_loss:.4f}")
print(f"Final Test Loss (without KL):  {test_loss:.4f}")
print(f"\nComparison with KL version:")
print(f"  With KL (~5.6):    Model stuck, random predictions")
print(f"  Without KL:        Model can {'LEARN' if train_loss < 0.3 else 'NOT LEARN'}")
print("\nACTION: Reduce KL divergence weight in main training")
print("=" * 80)
