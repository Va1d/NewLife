import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple
from model import TransformerEncoder, TemporalFusionTransformer, BayesianTransformer, MCDropoutTransformer, MambaEncoder, xLSTMEncoder
from loader import TheSetGPU
from progressbar import progressbar  # type: ignore[import]
import time
import shutil
import random
import subprocess
import sys
import csv
from pathlib import Path
import plotly.graph_objects as go  # type: ignore[import-not-found]
from PIL import Image  # type: ignore[import-not-found]
import io

has_pynvml = False
nvmlInit = None  # type: ignore[assignment]
nvmlDeviceGetHandleByIndex = None  # type: ignore[assignment]
nvmlDeviceGetMemoryInfo = None  # type: ignore[assignment]
nvmlDeviceGetUtilizationRates = None  # type: ignore[assignment]

try:
    from pynvml import nvmlInit as _nvmlInit, nvmlDeviceGetHandleByIndex as _nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo as _nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates as _nvmlDeviceGetUtilizationRates  # type: ignore[import]
    nvmlInit = _nvmlInit  # type: ignore[assignment]
    nvmlDeviceGetHandleByIndex = _nvmlDeviceGetHandleByIndex  # type: ignore[assignment]
    nvmlDeviceGetMemoryInfo = _nvmlDeviceGetMemoryInfo  # type: ignore[assignment]
    nvmlDeviceGetUtilizationRates = _nvmlDeviceGetUtilizationRates  # type: ignore[assignment]
    has_pynvml = True
    nvmlInit()
except ImportError:
    has_pynvml = False
    print("Warning: pynvml not installed. GPU metrics will not be logged. Install with: pip install nvidia-ml-py")

parser = argparse.ArgumentParser(description='Train transformer model with optional Bayesian inference')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--log-dir', type=str, default='/home/bo/Py/TB_Log')
parser.add_argument('--model', type=str, default='bayesian', choices=['transformer', 'tft', 'bayesian', 'mcdropout', 'mamba', 'xlstm'])
parser.add_argument('--run-all', action='store_true', help='Run all models sequentially')
parser.add_argument('--skip-log-wipe', action='store_true', help='Do not wipe log dir when using default path')
args = parser.parse_args()

def run_all_models() -> None:
    models = ['transformer', 'tft', 'bayesian', 'mcdropout', 'mamba', 'xlstm']
    print("Running all models sequentially:", models)
    for model_name in models:
        print(f"\n=== Starting model: {model_name} ===")
        cmd: list[str] = [
            sys.executable,
            __file__,
            '--device', args.device,
            '--model', model_name,
            '--log-dir', args.log_dir,
            '--skip-log-wipe',
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Model {model_name} exited with code {result.returncode}; continuing.")

if args.run_all:
    run_all_models()
    sys.exit(0)

# Device setup
device = torch.device(args.device)
print(f"Using device ({device}) for training and testing")

# TensorBoard logging
default_log_dir = '/home/bo/Py/TB_Log'
log_root = Path(args.log_dir)

# If using default log directory, wipe entire folder; otherwise preserve other runs
if args.log_dir == default_log_dir and log_root.exists() and not args.skip_log_wipe:
    shutil.rmtree(log_root)
    print(f"Cleared entire default log directory: {log_root}")

log_root.mkdir(parents=True, exist_ok=True)
device_tag = str(device).replace(':', '')
run_name = f"{args.model}_{device_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
log_dir = log_root / run_name
log_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(log_dir=str(log_dir))
print(f"TensorBoard logs will be saved to: {log_dir}")

# Hyperparameter Configuration
config: dict[str, int | float | str] = {
    'num_epochs': 70,
    'accumulation_steps': 4,
    'learning_rate': 0.001,  # Increased from 0.0001
    'warmup_epochs': 1,
    'weight_decay': 1e-6,  # Reduced from 1e-5
    'max_grad_norm': 1.0,
    'label_smoothing': 0.0,  # Disabled - turning off from 0.1
    'patience': 16,  # Early stopping patience - higher for small datasets to avoid premature stopping
    'resume_lr_factor': 0.5,
    'advanced_log_every': 5,
    'log_dir': str(log_dir),
    'checkpoint_dir': '/home/bo/Py/NewLife/checkpoint',
}

# Model setup
def build_model_and_optim() -> Tuple[nn.Module, optim.Optimizer, optim.lr_scheduler.LambdaLR, GradScaler]:
    if args.model == 'tft':
        model = TemporalFusionTransformer(
            d_model=96,  # Optimized from 64
            num_heads=8,  # More heads for different bot strategies
            d_ff=384,  # Optimized from 256
            num_layers=2,
            max_seq_length=388,
            output_dim=1,  # Binary classification: single logit
            dropout=0.1,  # Reduced from 0.2
            use_causal_mask=True
        ).to(device)
    elif args.model == 'bayesian':
        model = BayesianTransformer(
            d_model=96,  # Reduced from 256 - Bayesian already complex
            num_heads=8,  # Each head can specialize on different bot patterns
            d_ff=384,  # Reduced from 1024
            num_layers=4,  # Reduced from 3
            max_seq_length=388,
            output_dim=1,  # Binary classification: single logit
            dropout=0.05,  # Keep reduced
            prior_mu=0.0,
            prior_sigma=1.0  # Weakened prior from 0.1 to allow learning
        ).to(device)
    elif args.model == 'mcdropout':
        model = MCDropoutTransformer(
            d_model=128,  # Increased from 64 - MCDropout is stable
            num_heads=8,  # More heads for bot diversity
            d_ff=512,  # Increased from 256
            num_layers=2,
            max_seq_length=388,
            output_dim=1,  # Binary classification: single logit
            dropout=0.2  # Reduced from 0.3 for stability
        ).to(device)
    elif args.model == 'mamba':
        model = MambaEncoder(
            d_model=128,  # Increased for better capacity
            num_heads=8,  # For attention layers in hybrid
            num_layers=3,  # Reduced from 4 for stability
            max_seq_length=388,
            output_dim=1,  # Binary classification: single logit
            dropout=0.1  # Standard level
        ).to(device)
    elif args.model == 'xlstm':
        model = xLSTMEncoder(
            d_model=128,  # Increased from 64
            num_layers=3,  # Keep for depth
            max_seq_length=388,
            output_dim=1,  # Binary classification: single logit
            dropout=0.1  # Reduced from 0.2
        ).to(device)
    else:
        model = TransformerEncoder(
            d_model=128,  # Increased from 64 - optimized size
            num_heads=8,  # More heads to detect different bot algorithms
            d_ff=512,  # Increased from 256
            num_layers=3,  # Increased from 2
            max_seq_length=388,
            output_dim=1  # Binary classification: single logit
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']), weight_decay=float(config['weight_decay']))

    # Learning rate scheduler with warmup
    total_steps = 4000  # Approximate steps (will be recalculated)
    warmup_steps = int(total_steps * int(config['warmup_epochs']) // int(config['num_epochs']))
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, (step + 1) / max(1, warmup_steps)) * (0.5 ** ((step + 1 - warmup_steps) / max(1, total_steps - warmup_steps)))
    )

    scaler = GradScaler('cuda')

    return model, optimizer, scheduler, scaler

model, optimizer, scheduler, scaler = build_model_and_optim()

# Log model architecture
writer.add_text(tag='Model/Architecture', text_string=str(model)) # type: ignore
writer.add_text(tag='Run/Config', text_string=f"model={args.model}, device={args.device}, log_dir={log_dir}") # type: ignore

# Dataset - load all data onto GPU once
dataset = TheSetGPU(device=str(device))
num_test_sessions = 16

# Randomly select test days to avoid black-swan bias
all_indices = list(range(len(dataset)))
random.shuffle(all_indices)  # Shuffle in-place
test_indices = all_indices[:num_test_sessions]
train_indices = all_indices[num_test_sessions:]

print(f"Randomly selected test sessions: {sorted(test_indices)}")
print(f"Total: {len(train_indices)} training sessions, {len(test_indices)} test sessions")

# Compute class weights from training data for balanced BCE loss
print("Computing class weights from training data...")
total_positives = 0
total_samples = 0

for idx in train_indices:
    _, y_batch, _ = dataset[idx]  # No target_valid_mask anymore
    total_positives += y_batch.sum().item()
    total_samples += y_batch.numel()
    
    # Old masked version - commented out for potential future use:
    # _, y_batch, target_valid_mask, _ = dataset[idx]
    # valid_targets = y_batch[target_valid_mask > 0]
    # total_positives += valid_targets.sum().item()
    # total_samples += valid_targets.numel()

total_negatives = total_samples - total_positives

if total_positives > 0 and total_negatives > 0:
    pos_weight = total_negatives / total_positives
else:
    pos_weight = 1.0

print(f"Training data: {total_positives:.0f} positives, {total_negatives:.0f} negatives (ratio: {pos_weight:.4f})")

# Loss function for binary classification with class weighting
# BCEWithLogitsLoss combines sigmoid + BCE for numerical stability
criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([pos_weight], device=device))

# Checkpoint paths and helpers
checkpoint_dir = Path(str(config['checkpoint_dir']))
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_ckpt_path = checkpoint_dir / 'best_model.pth'

def archive_best_checkpoint() -> None:
    if best_ckpt_path.exists():
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        archived = checkpoint_dir / f"best_model_{timestamp}.pth"
        shutil.move(best_ckpt_path, archived)
        print(f"Archived best checkpoint to {archived}")

def resume_from_best() -> Optional[float]:
    if not best_ckpt_path.exists():
        return None

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.to(device)

    for group in optimizer.param_groups:
        group['lr'] *= float(config['resume_lr_factor'])

    return checkpoint.get('test_loss', None)

# Best model tracking
best_test_loss = float('inf')
patience_counter = 0
resume_attempted = False

# Training loop
model.train()
global_step = 0
prev_epoch_loss = None

for epoch in range(int(config['num_epochs'])):
    print(f"\nEpoch {epoch + 1}/{int(config['num_epochs'])}")
    epoch_start_time = time.time()
    epoch_loss = 0
    num_samples = 0
    num_valid_samples = 0  # Count of samples with mask=1
    grad_norm_sum = 0.0
    grad_norm_steps = 0
    
    # Binary classification metrics
    train_correct = 0
    train_true_positives = 0
    train_false_positives = 0
    train_true_negatives = 0
    train_false_negatives = 0
    
    # Shuffle training indices for this epoch
    random.shuffle(train_indices)
    
    # Training phase
    for session_idx in progressbar(train_indices):  # type: ignore[assignment,misc]
        # Get entire session data: [256, max_seq_len, 432], [256], [256]
        # Gaps are treated as 0 (no event) rather than being masked out
        x_batch, y_batch, seq_lengths = dataset[session_idx]
        # Old masked version: x_batch, y_batch, target_valid_mask, seq_lengths = dataset[session_idx]
        
        # Forward pass with mixed precision - process all 256 steps at once
        with torch.amp.autocast('cuda'): # type: ignore
            model_output = model(x_batch, seq_lengths=seq_lengths)
            
            # Handle different model types
            kl = None
            if args.model == 'bayesian':
                logits, kl = model_output  # Bayesian returns (predictions, kl_divergence)
            elif args.model in ['mcdropout', 'mamba', 'xlstm']:
                logits = model_output  # These models return predictions directly (already squeezed)
            else:
                outputs = model_output  # Standard models return outputs directly
                # Get predictions at actual sequence end for each step
                batch_indices = torch.arange(256, device=device)
                seq_end_indices = seq_lengths - 1
                logits = outputs[batch_indices, seq_end_indices, :].squeeze(-1)  # Shape: [256]
            
            # Compute BCE loss - all samples are valid (gaps treated as 0)
            bce_loss = criterion(logits, y_batch).mean()  # Average over all 256 samples
            
            # Add KL divergence for Bayesian models (ELBO loss)
            if args.model == 'bayesian' and kl is not None:
                kl_weight = 1e-4  # Very weak KL - let model focus on BCE loss
                loss = bce_loss + (kl_weight * kl)
            else:
                loss = bce_loss
            
            # Old masked version - commented out for potential future use:
            # loss_per_sample = criterion(logits, y_batch)  # Shape: [256]
            # masked_loss = loss_per_sample * target_valid_mask
            # num_valid = target_valid_mask.sum().item()
            # if num_valid > 0:
            #     loss = masked_loss.sum() / num_valid
            # else:
            #     loss = None
        
        # Backward pass - always have valid samples now
        scaler.scale(loss).backward() # type: ignore
        
        epoch_loss += bce_loss.item() * 256  # Track BCE loss for metrics
        num_samples += 256
        num_valid_samples += 256  # All samples are valid now
        
        # Binary classification metrics (all samples)
        with torch.no_grad():
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            train_correct += (predictions == y_batch).sum().item()
            train_true_positives += ((predictions == 1) & (y_batch == 1)).sum().item()
            train_false_positives += ((predictions == 1) & (y_batch == 0)).sum().item()
            train_true_negatives += ((predictions == 0) & (y_batch == 0)).sum().item()
            train_false_negatives += ((predictions == 0) & (y_batch == 1)).sum().item()
            
            # Old masked version - commented out:
            # valid_logits = logits[target_valid_mask > 0]
            # valid_target = y_batch[target_valid_mask > 0]
            # valid_pred = (torch.sigmoid(valid_logits) > 0.5).float()
            # train_correct += (valid_pred == valid_target).sum().item()
            # train_true_positives += ((valid_pred == 1) & (valid_target == 1)).sum().item()
            # train_false_positives += ((valid_pred == 1) & (valid_target == 0)).sum().item()
            # train_true_negatives += ((valid_pred == 0) & (valid_target == 0)).sum().item()
            # train_false_negatives += ((valid_pred == 0) & (valid_target == 1)).sum().item()
        
        global_step += 1
        
        # Update weights periodically with gradient clipping
        if global_step % int(config['accumulation_steps']) == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['max_grad_norm']))
                grad_norm_sum += grad_norm.item()
                grad_norm_steps += 1
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
    
    # Step scheduler once per epoch
    scheduler.step()
    
    avg_epoch_loss = epoch_loss / num_valid_samples if num_valid_samples > 0 else 0.0
    train_accuracy = train_correct / num_valid_samples if num_valid_samples > 0 else 0.0
    train_precision = train_true_positives / (train_true_positives + train_false_positives) if (train_true_positives + train_false_positives) > 0 else 0.0
    train_sensitivity = train_true_positives / (train_true_positives + train_false_negatives) if (train_true_positives + train_false_negatives) > 0 else 0.0
    train_specificity = train_true_negatives / (train_true_negatives + train_false_positives) if (train_true_negatives + train_false_positives) > 0 else 0.0
    train_f1 = 2 * train_precision * train_sensitivity / (train_precision + train_sensitivity) if (train_precision + train_sensitivity) > 0 else 0.0
    valid_sample_ratio = num_valid_samples / num_samples if num_samples > 0 else 0.0
    epoch_time = time.time() - epoch_start_time
    avg_grad_norm = (grad_norm_sum / grad_norm_steps) if grad_norm_steps > 0 else 0.0
    loss_slope = (avg_epoch_loss - prev_epoch_loss) if prev_epoch_loss is not None else 0.0
    
    print(f"Epoch {epoch + 1} Training Loss: {avg_epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Spec: {train_specificity:.4f}, Valid: {num_valid_samples}/{num_samples}, Time: {epoch_time:.1f}s")
    
    # Log epoch metrics and learning rate (scalars only)
    writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)  # type: ignore[misc]
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/train_precision', train_precision, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/train_sensitivity', train_sensitivity, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/train_specificity', train_specificity, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/train_f1', train_f1, epoch)  # type: ignore[misc]
    writer.add_scalar('Train/valid_sample_ratio', valid_sample_ratio, epoch)  # type: ignore[misc]
    writer.add_scalar('Learning_rate', float(optimizer.param_groups[0]['lr']), epoch)  # type: ignore[misc]
    writer.add_scalar('Train/avg_grad_norm', avg_grad_norm, epoch)  # type: ignore[misc]
    writer.add_scalar('Train/loss_slope', loss_slope, epoch)  # type: ignore[misc]
    
    # Log GPU metrics for training GPU
    if has_pynvml and nvmlDeviceGetHandleByIndex is not None and nvmlDeviceGetMemoryInfo is not None and nvmlDeviceGetUtilizationRates is not None:
        try:
            handle_gpu1 = nvmlDeviceGetHandleByIndex(1)  # type: ignore[misc]
            mem_info = nvmlDeviceGetMemoryInfo(handle_gpu1)  # type: ignore[misc]
            util = nvmlDeviceGetUtilizationRates(handle_gpu1)  # type: ignore[misc]
            writer.add_scalar('GPU/GPU1_memory_used_GB', mem_info.used / 1e9, epoch)  # type: ignore[misc]
            writer.add_scalar('GPU/GPU1_memory_percent', (mem_info.used / mem_info.total) * 100, epoch)  # type: ignore[misc]
            writer.add_scalar('GPU/GPU1_utilization_percent', util.gpu, epoch)  # type: ignore[misc]
        except Exception as e:
            print(f"Warning: Could not read GPU 1 metrics: {e}")
    
    # Testing phase
    model.eval()
    test_loss = 0
    test_samples = 0
    test_valid_samples = 0

    stock_weight_sum = None
    stock_weight_count = 0
    
    # Binary classification metrics
    test_correct = 0
    test_true_positives = 0
    test_false_positives = 0
    test_true_negatives = 0
    test_false_negatives = 0
    
    # For confusion matrix
    all_predictions_list: list[torch.Tensor] = []
    all_targets_list: list[torch.Tensor] = []  # type: ignore[var-annotated]
    
    with torch.no_grad():
        for session_idx in progressbar(test_indices):  # type: ignore[assignment,misc]
            # Get entire session data: [256, max_seq_len, 468], [256], [256]
            # Gaps are treated as 0 (no event) rather than being masked out
            session_idx_int: int = int(session_idx)  # type: ignore[arg-type]
            x_batch, y_batch, seq_lengths = dataset[session_idx_int]
            # Old masked version: x_batch, y_batch, target_valid_mask, seq_lengths = dataset[session_idx]
            
            # Forward pass - process all 256 steps at once
            with torch.amp.autocast('cuda'): # type: ignore
                model_output = model(x_batch, seq_lengths=seq_lengths)
                
                # Handle different model types
                if args.model == 'bayesian':
                    logits, kl = model_output  # Bayesian returns (predictions, kl_divergence)
                elif args.model in ['mcdropout', 'mamba', 'xlstm']:
                    logits = model_output  # These models return predictions directly (already squeezed)
                else:
                    outputs = model_output
                    # Get predictions at actual sequence end for each step
                    batch_indices = torch.arange(256, device=device)
                    seq_end_indices = seq_lengths - 1
                    logits = outputs[batch_indices, seq_end_indices, :].squeeze(-1)  # Shape: [256]
                
                # Compute BCE loss - all samples are valid
                bce_loss = criterion(logits, y_batch).mean()
                test_loss += bce_loss.item() * 256
                test_valid_samples += 256

                # Collect per-stock weights for interpretability
                if hasattr(model, 'input_projection'):
                    input_proj = getattr(model, 'input_projection', None)
                    if input_proj is not None and hasattr(input_proj, 'get_last_stock_weights'):
                        get_weights_method = getattr(input_proj, 'get_last_stock_weights', None)
                        if callable(get_weights_method):
                            weights = get_weights_method()
                            if weights is not None and isinstance(weights, torch.Tensor):
                                # weights: [batch, seq_len, num_stocks]
                                summed: torch.Tensor = weights.sum(dim=(0, 1)).cpu()
                                count = weights.shape[0] * weights.shape[1]
                                if stock_weight_sum is None:
                                    stock_weight_sum = summed
                                else:
                                    stock_weight_sum += summed
                                stock_weight_count += count
                
                # Old masked version - commented out:
                # loss_per_sample = criterion(logits, y_batch)
                # masked_loss = loss_per_sample * target_valid_mask
                # num_valid = target_valid_mask.sum().item()
                # if num_valid > 0:
                #     loss = masked_loss.sum() / num_valid
                #     test_loss += loss.item() * num_valid
                #     test_valid_samples += num_valid
            
            test_samples += 256
            
            # Collect binary classification metrics (all samples)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            test_correct += (predictions == y_batch).sum().item()
            test_true_positives += ((predictions == 1) & (y_batch == 1)).sum().item()
            test_false_positives += ((predictions == 1) & (y_batch == 0)).sum().item()
            test_true_negatives += ((predictions == 0) & (y_batch == 0)).sum().item()
            test_false_negatives += ((predictions == 0) & (y_batch == 1)).sum().item()
            
            all_predictions_list.append(predictions.cpu())  # type: ignore[misc]
            all_targets_list.append(y_batch.cpu())  # type: ignore[misc]
            
            # Old masked version - commented out:
            # has_valid_targets = target_valid_mask > 0
            # if has_valid_targets.sum() > 0:
            #     valid_logits = logits[has_valid_targets]
            #     valid_target = y_batch[has_valid_targets]
            #     valid_pred = (torch.sigmoid(valid_logits) > 0.5).float()
            #     test_correct += (valid_pred == valid_target).sum().item()
            #     test_true_positives += ((valid_pred == 1) & (valid_target == 1)).sum().item()
            #     test_false_positives += ((valid_pred == 1) & (valid_target == 0)).sum().item()
            #     test_true_negatives += ((valid_pred == 0) & (valid_target == 0)).sum().item()
            #     test_false_negatives += ((valid_pred == 0) & (valid_target == 1)).sum().item()
            #     all_predictions.append(valid_pred.cpu())
            #     all_targets.append(valid_target.cpu())
    
    avg_test_loss = test_loss / test_valid_samples if test_valid_samples > 0 else 0.0
    test_accuracy = test_correct / test_valid_samples if test_valid_samples > 0 else 0.0
    test_precision = test_true_positives / (test_true_positives + test_false_positives) if (test_true_positives + test_false_positives) > 0 else 0.0
    test_sensitivity = test_true_positives / (test_true_positives + test_false_negatives) if (test_true_positives + test_false_negatives) > 0 else 0.0
    test_specificity = test_true_negatives / (test_true_negatives + test_false_positives) if (test_true_negatives + test_false_positives) > 0 else 0.0
    test_f1 = 2 * test_precision * test_sensitivity / (test_precision + test_sensitivity) if (test_precision + test_sensitivity) > 0 else 0.0
    test_valid_ratio = test_valid_samples / test_samples if test_samples > 0 else 0.0
    
    print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
          f"Train Loss: {avg_epoch_loss:.4f}, Acc: {train_accuracy:.4f}, Prec: {train_precision:.4f}, Sens: {train_sensitivity:.4f}, Spec: {train_specificity:.4f}, F1: {train_f1:.4f} | "
          f"Test Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.4f}, Prec: {test_precision:.4f}, Sens: {test_sensitivity:.4f}, Spec: {test_specificity:.4f}, F1: {test_f1:.4f}")
    
    # Compute confusion matrix
    if len(all_predictions_list) > 0: # type: ignore
        all_predictions = torch.cat(all_predictions_list)  # type: ignore[arg-type]
        all_targets = torch.cat(all_targets_list)  # type: ignore[arg-type]
        
        # Build 2x2 confusion matrix
        # Row 0: actual negative (0), Row 1: actual positive (1)
        # Col 0: predicted negative (0), Col 1: predicted positive (1)
        confusion_matrix = torch.zeros(2, 2, dtype=torch.long)
        for t, p in zip(all_targets, all_predictions):
            confusion_matrix[int(t.item()), int(p.item())] += 1
        
        # Normalize confusion matrix by row (true class)
        cm_normalized = confusion_matrix.float() / confusion_matrix.sum(dim=1, keepdim=True).clamp(min=1)
        
        # Create confusion matrix heatmap
        fig = go.Figure(data=go.Heatmap(  # type: ignore[misc]
            z=cm_normalized.numpy(),
            x=['Predicted Down (0)', 'Predicted Up (1)'],
            y=['Actual Down (0)', 'Actual Up (1)'],
            colorscale='Blues',
            colorbar=dict(title='Proportion'),
            text=confusion_matrix.numpy(),
            texttemplate='%{text}',
            textfont=dict(size=16)
        ))
        
        fig.update_layout(  # type: ignore[misc]
            title=f'Confusion Matrix - Epoch {epoch + 1}<br>TN={test_true_negatives}, FP={test_false_positives}, FN={test_false_negatives}, TP={test_true_positives}',
            xaxis_title='Predicted Class',
            yaxis_title='Actual Class',
            width=800,
            height=800
        )
        
        # Convert plotly figure to image for TensorBoard
        img_bytes = fig.to_image(format="png", width=800, height=800)  # type: ignore[misc]
        img = Image.open(io.BytesIO(img_bytes))  # type: ignore[misc]
        
        # Convert PIL image to tensor for TensorBoard
        import numpy as np
        img_array = np.array(img)  # type: ignore[misc]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # type: ignore[arg-type] # HWC -> CHW
        
        # Log to TensorBoard
        writer.add_image('Test/confusion_matrix', img_tensor, epoch)  # type: ignore[misc]
    
    # Log test metrics
    writer.add_scalar('Loss/test_epoch', avg_test_loss, epoch)  # type: ignore[misc]
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/test_precision', test_precision, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/test_sensitivity', test_sensitivity, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/test_specificity', test_specificity, epoch)  # type: ignore[misc]
    writer.add_scalar('Metrics/test_f1', test_f1, epoch)  # type: ignore[misc]
    writer.add_scalar('Test/valid_sample_ratio', test_valid_ratio, epoch)  # type: ignore[misc]

    # Log per-stock attention weights (average over test set)
    if stock_weight_sum is not None and stock_weight_count > 0:
        avg_weights = (stock_weight_sum / stock_weight_count).numpy()
        for i, weight in enumerate(avg_weights):
            writer.add_scalar(f"StockWeights/stock_{i}", float(weight), epoch)  # type: ignore[misc]

        # Append to CSV for offline analysis
        csv_path = log_dir / "stock_weights.csv"
        file_exists = csv_path.exists()
        with open(csv_path, mode='a', newline='') as f:
            writer_csv = csv.writer(f)
            if not file_exists:
                header = ['epoch'] + [f"stock_{i}" for i in range(len(avg_weights))]
                writer_csv.writerow(header)
            writer_csv.writerow([epoch + 1] + [float(w) for w in avg_weights])

    # Advanced logging every N epochs
    advanced_log_every: int = int(config['advanced_log_every'])  # type: ignore[arg-type]
    if (epoch + 1) % advanced_log_every == 0:
        weight_norm_sq: float = 0.0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            writer.add_histogram(f"Weights/{name}", param.detach().float().cpu(), epoch)  # type: ignore[misc]
            weight_norm_sq += param.detach().float().norm().item() ** 2  # type: ignore[misc,operator]

        weight_norm: float = weight_norm_sq ** 0.5  # type: ignore[assignment]
        update_ratio: float = 0.0  # type: ignore[assignment]
        if weight_norm > 0.0:
            update_ratio = (optimizer.param_groups[0]['lr'] * avg_grad_norm) / weight_norm  # type: ignore[assignment]

        writer.add_scalar('Train/weight_norm', weight_norm, epoch)  # type: ignore[misc]
        writer.add_scalar('Train/update_to_weight_ratio', update_ratio, epoch)  # type: ignore[misc]
    
    # Log GPU metrics for testing GPU
    if has_pynvml:
        try:
            handle_gpu1 = nvmlDeviceGetHandleByIndex(1)  # type: ignore[misc]
            mem_info = nvmlDeviceGetMemoryInfo(handle_gpu1)  # type: ignore[misc]
            util = nvmlDeviceGetUtilizationRates(handle_gpu1)  # type: ignore[misc]
            writer.add_scalar('GPU/GPU1_test_memory_used_GB', mem_info.used / 1e9, epoch)  # type: ignore[misc]
            writer.add_scalar('GPU/GPU1_test_memory_percent', (mem_info.used / mem_info.total) * 100, epoch)  # type: ignore[misc]
            writer.add_scalar('GPU/GPU1_test_utilization_percent', util.gpu, epoch)  # type: ignore[misc]
        except Exception as e:
            print(f"Warning: Could not read GPU 1 metrics: {e}")
    
    # (No need to move model, it's already on GPU 1)
    
    # Switch back to training mode
    model.train()
    
    # Best model checkpointing with resume/restart logic
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        patience_counter = 0
        resume_attempted = False

        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_epoch_loss,
            'test_loss': avg_test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'test_sensitivity': test_sensitivity,
            'test_specificity': test_specificity,
        }, str(best_ckpt_path))
        print(f"Best model saved with test loss: {avg_test_loss:.4f}, accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, sensitivity: {test_sensitivity:.4f}, specificity: {test_specificity:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter}/{config['patience']} epochs")

    # Resume from best after 5 bad epochs; restart if it happens again
    if patience_counter >= int(config['patience']):
        if not resume_attempted:
            resumed_loss = resume_from_best()
            if resumed_loss is not None:
                best_test_loss = resumed_loss
                patience_counter = 0
                resume_attempted = True
                model.train()
                print("Resumed from best checkpoint and reduced LR.")
            else:
                print("Best checkpoint not found. Continuing without resume.")
        else:
            archive_best_checkpoint()
            model, optimizer, scheduler, scaler = build_model_and_optim()
            model.train()
            best_test_loss = float('inf')
            patience_counter = 0
            resume_attempted = False
            print("Restarted training from scratch.")
    
    # Save checkpoint every epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_epoch_loss,
        'test_loss': avg_test_loss,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_f1': train_f1,
        'test_f1': test_f1,
    }, f"{config['checkpoint_dir']}/epoch_{epoch}.pth")

    prev_epoch_loss = avg_epoch_loss

print("\nTraining complete!")
print(f"Best test loss achieved: {best_test_loss:.4f}")
writer.close()