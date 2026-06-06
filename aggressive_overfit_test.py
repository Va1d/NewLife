"""
Aggressive Overfitting Test - No dropout, extreme learning rate, 100 epochs
This will show us the true capacity ceiling
"""
import torch
import torch.nn as nn
import torch.optim as optim
from model import (
    TransformerEncoder, TemporalFusionTransformer, BayesianTransformer,
    MCDropoutTransformer
)
from loader import TheSetGPU

device = torch.device('cuda:1')
dataset = TheSetGPU(device=str(device))

print("=" * 100)
print("AGGRESSIVE OVERFITTING TEST - Extreme Training")
print("=" * 100)
print("\nConfiguration:")
print("  • Learning rate: 0.1 (extreme)")
print("  • Dropout: DISABLED")
print("  • Regularization: None")
print("  • Epochs: 100")
print("  • Batch: 2 training sessions (~512 samples)")
print("  • Goal: Force memorization\n")

tiny_train_idx = [0, 1]
tiny_test_idx = 0
num_epochs = 100

models_to_test = [
    ('TransformerEncoder-Tiny', lambda: TransformerEncoder(
        d_model=32, num_heads=2, d_ff=64, num_layers=1,
        max_seq_length=388, output_dim=1
    )),
    ('TransformerEncoder-Large', lambda: TransformerEncoder(
        d_model=256, num_heads=8, d_ff=1024, num_layers=4,
        max_seq_length=388, output_dim=1
    )),
    ('BayesianTransformer-Small', lambda: BayesianTransformer(
        d_model=64, num_heads=4, d_ff=256, num_layers=2,
        max_seq_length=388, output_dim=1, dropout=0.0,
        prior_mu=0.0, prior_sigma=100.0  # VERY weak prior
    )),
    ('BayesianTransformer-Large', lambda: BayesianTransformer(
        d_model=256, num_heads=8, d_ff=1024, num_layers=3,
        max_seq_length=388, output_dim=1, dropout=0.0,
        prior_mu=0.0, prior_sigma=100.0  # VERY weak prior
    )),
    ('MCDropout-Large', lambda: MCDropoutTransformer(
        d_model=256, num_heads=8, d_ff=1024, num_layers=3,
        max_seq_length=388, output_dim=1, dropout=0.0  # NO dropout
    )),
]

print("Model | Params | Epoch 10 | Epoch 50 | Epoch 100 | Final Test | Status")
print("-" * 80)

for model_name, model_fn in models_to_test:
    model = model_fn().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # Extreme LR
    criterion = nn.BCEWithLogitsLoss()
    
    train_losses_10 = []
    train_losses = []
    test_loss_final = None
    
    for epoch in range(num_epochs):
        train_loss = 0.0
        for idx in tiny_train_idx:
            x_batch, y_batch, seq_lengths = dataset[idx]
            
            if 'Bayesian' in model_name:
                logits, kl = model(x_batch, seq_lengths=seq_lengths)
                loss = criterion(logits, y_batch.float()).mean() + 0 * kl  # NO KL weight
            else:
                logits = model(x_batch, seq_lengths=seq_lengths)
                batch_indices = torch.arange(256, device=device)
                seq_end_indices = seq_lengths - 1
                logits = logits[batch_indices, seq_end_indices, :].squeeze(-1)
            
            loss = criterion(logits, y_batch.float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(tiny_train_idx)
        train_losses.append(train_loss)
        
        if epoch == 9:
            train_losses_10 = train_loss
        if epoch == 49:
            train_losses_50 = train_loss
    
    # Final test
    with torch.no_grad():
        x_batch, y_batch, seq_lengths = dataset[tiny_test_idx]
        
        if 'Bayesian' in model_name:
            logits, kl = model(x_batch, seq_lengths=seq_lengths)
            test_loss_final = criterion(logits, y_batch.float()).mean().item()
        else:
            logits = model(x_batch, seq_lengths=seq_lengths)
            batch_indices = torch.arange(256, device=device)
            seq_end_indices = seq_lengths - 1
            logits = logits[batch_indices, seq_end_indices, :].squeeze(-1)
            test_loss_final = criterion(logits, y_batch.float()).mean().item()
    
    # Status
    if train_losses[-1] < 0.1 and test_loss_final > 0.5:
        status = "✓ OVERFITS"
    elif train_losses[-1] < 0.2:
        status = "~ Trains well"
    elif train_losses[-1] < 0.3:
        status = "⚠ Trains slowly"
    else:
        status = "✗ Stuck"
    
    print(f"{model_name:25s} | {num_params:7d} | {train_losses_10:8.4f} | {train_losses_50:8.4f} | {train_losses[-1]:9.4f} | {test_loss_final:10.4f} | {status}")

print("\n" + "=" * 100)
print("INTERPRETATION:")
print("=" * 100)
print("""
If all models show:
  Train loss: 0.45-0.50
  Test loss: 0.45-0.50
  
→ The model IS learning the data, but the bot activity signal has LIMITED capacity
  (It captures ~18% patterns but not enough to fully separate train/test)
  
→ This is GOOD NEWS: means the label isn't pure noise, but has structured signal

Recommendations:
  1. Use models that train well (not stuck at 0.5)
  2. Size them to balance accuracy vs overfitting
  3. Focus on regularization to prevent false patterns
""")
