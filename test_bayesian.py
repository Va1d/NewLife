"""Test for NaN/Inf in data and Bayesian model."""

from loader import TheSetGPU
from model import BayesianTransformer
import torch

print('=' * 80)
print('TESTING DATA LOADER')
print('=' * 80)
ds = TheSetGPU(device='cuda:0')
x, y, seq_lengths = ds[0]

print(f'Input shape: {x.shape}')
print(f'Target shape: {y.shape}')
print(f'Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}, mean={x.mean().item():.4f}, std={x.std().item():.4f}')
print(f'Has NaN in input: {torch.isnan(x).any().item()}')
print(f'Has Inf in input: {torch.isinf(x).any().item()}')
print(f'Target stats: min={y.min().item():.4f}, max={y.max().item():.4f}, mean={y.mean().item():.4f}')
print(f'Has NaN in target: {torch.isnan(y).any().item()}')
print(f'Target class balance: {y.sum().item()}/{len(y)} = {y.mean().item():.3f}')

print('\n' + '=' * 80)
print('TESTING BAYESIAN MODEL FORWARD PASS')
print('=' * 80)

model = BayesianTransformer(
    d_model=64,
    num_heads=4,
    d_ff=256,
    num_layers=2,
    max_seq_length=388,
    output_dim=1,
    dropout=0.2,
    prior_mu=0.0,
    prior_sigma=0.1
).to('cuda:0')

print(f'Model created successfully')

# Test forward pass
with torch.no_grad():
    logits, kl = model(x, seq_lengths=seq_lengths)
    
print(f'\nLogits shape: {logits.shape}')
print(f'Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, mean={logits.mean().item():.4f}')
print(f'Has NaN in logits: {torch.isnan(logits).any().item()}')
print(f'Has Inf in logits: {torch.isinf(logits).any().item()}')
print(f'KL divergence: {kl.item():.4f}')
print(f'Has NaN in KL: {torch.isnan(kl).any().item()}')

# Test loss computation
bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
loss_per_sample = bce_loss(logits, y)
print(f'\nLoss per sample stats: min={loss_per_sample.min().item():.4f}, max={loss_per_sample.max().item():.4f}, mean={loss_per_sample.mean().item():.4f}')
print(f'Has NaN in loss: {torch.isnan(loss_per_sample).any().item()}')

# Test backward pass
print('\n' + '=' * 80)
print('TESTING BACKWARD PASS')
print('=' * 80)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()

logits, kl = model(x[:10], seq_lengths=seq_lengths[:10])  # Small batch
bce_loss_val = bce_loss(logits, y[:10]).mean()
total_loss = bce_loss_val + 0.001 * kl

print(f'BCE loss: {bce_loss_val.item():.4f}')
print(f'KL: {kl.item():.4f}')
print(f'Total loss: {total_loss.item():.4f}')

total_loss.backward()

# Check gradients
max_grad = 0.0
has_nan_grad = False
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if torch.isnan(param.grad).any():
            print(f'NaN gradient in {name}')
            has_nan_grad = True
        if grad_norm > max_grad:
            max_grad = grad_norm

print(f'\nMax gradient norm: {max_grad:.4f}')
print(f'Has NaN gradients: {has_nan_grad}')

if max_grad > 100:
    print('WARNING: Gradient explosion detected!')
elif has_nan_grad:
    print('WARNING: NaN gradients detected!')
else:
    print('✓ Gradients look healthy')
