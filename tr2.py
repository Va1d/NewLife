# Initialize Cache as None
past_key_values = None 

# 1. First Pass: Encode the first 256 chunks (the 'Context')
# This populates the cache with everything the model already "knows"
e_logits, s_logits, past_key_values = model(
    input_events, 
    input_sources, 
    past_key_values=None
)

# 2. Sequential Loop: Predict & Override
for i in range(split_idx, session_events.size(1) - 1):
    # Get the Ground Truth for the CURRENT step (the one we are 'at')
    # This is the "Truth Override"
    current_e = session_events[:, i].unsqueeze(1)
    current_s = session_sources[:, i].unsqueeze(1)
    
    # 3. Forward Pass: ONLY send the single current token + the CACHE
    # The model only calculates Attention for this 1 token against the past
    e_logits, s_logits, past_key_values = model(
        current_e, 
        current_s, 
        past_key_values=past_key_values
    )
    
    # Prediction for the NEXT token (i + 1)
    next_e_pred = e_logits[:, -1, :] 
    target_e = session_events[:, i + 1]
    
    # Loss & Backward
    loss = criterion(next_e_pred, target_e)
    loss.backward() 
    
    # IMPORTANT: Detach the cache if you are not doing BPTT across steps
    # This prevents the computational graph from growing infinitely
    past_key_values = [(k.detach(), v.detach()) for k, v in past_key_values]

out, k_cache, v_cache = model.prime(context_events, context_sources)

for i in range(split_idx, total_len - 1):
    # Truth Override: Take the REAL token at i
    current_e = session_events[:, i].unsqueeze(1)
    current_s = session_sources[:, i].unsqueeze(1)
    
    # Forward: Only process 1 token!
    # Returns prediction for i+1 AND updated cache
    logits, k_cache, v_cache = model.step(current_e, current_s, k_cache, v_cache)
    
    # Calculate loss for the prediction of i+1
    target_e = session_events[:, i+1]
    loss = criterion(logits, target_e)
    
    # Backward: Gradient flows through the model weights
    # Note: We detach cache to prevent backprop through the entire history 
    # (which would consume all 48GB VRAM quickly)
    loss.backward()
    k_cache = k_cache.detach()
    v_cache = v_cache.detach()

class TransformerBlockWithCache(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, past_k=None, past_v=None):
        # x is the single new "Truth" token: (batch, 1, d_model)
        
        # 1. Self-Attention
        x_norm = self.ln1(x)
        
        # Project current token to K, V (using internal weights)
        # For simplicity in raw PyTorch, we often just concatenate the raw X 
        # and let MHA handle projections, but this is less efficient.
        # Efficient way:
        if past_k is not None:
            combined_k = torch.cat([past_k, x_norm], dim=1)
            combined_v = torch.cat([past_v, x_norm], dim=1)
        else:
            combined_k, combined_v = x_norm, x_norm
            
        attn_out, _ = self.mha(x_norm, combined_k, combined_v)
        x = x + attn_out
        
        # 2. Feed Forward
        x = x + self.ff(self.ln2(x))
        
        return x, combined_k, combined_v
    

import torch
import torch.nn as nn

class StatefulEventTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings for your specific vocab
        self.event_emb = nn.Embedding(10, d_model)   # 8 events + shift + pad
        self.source_emb = nn.Embedding(258, d_model) # 256 sources + pad
        
        # We use a custom list of layers to manage the cache manually
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True) 
            for _ in range(num_layers)
        ])
        
        self.event_head = nn.Linear(d_model, 10)
        self.source_head = nn.Linear(d_model, 258)

    def forward(self, event_ids, source_ids, past_kv=None):
        """
        past_kv: List of (Key, Value) tensors for each layer
        Returns: event_logits, source_logits, new_past_kv
        """
        # 1. Combine Feature Embeddings
        x = self.event_emb(event_ids) + self.source_emb(source_ids)
        
        new_past_kv = []
        
        # 2. Process through layers with KV-Caching
        for i, layer in enumerate(self.layers):
            # PyTorch DecoderLayer usually expects (tgt, memory)
            # Here we treat it as self-attention (tgt=x, memory=history)
            
            if past_kv is not None:
                # Retrieve the history for this specific layer
                prev_kv = past_kv[i] 
                # Concatenate current token features with history
                # Note: This is a conceptual simplification of KV-caching
                # In a high-perf setup, you'd concatenate inside the MHA projections.
                full_context = torch.cat([prev_kv, x], dim=1)
                
                # Self-attention: query is current 'x', key/value is 'full_context'
                x = layer(x, full_context)
                new_past_kv.append(full_context.detach()) 
            else:
                # If no cache, this is the 'Prime' phase (processing first 256 chunks)
                # Use a causal mask here to maintain autoregressive logic
                mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
                x = layer(x, x, tgt_mask=mask)
                new_past_kv.append(x.detach())

        e_logits = self.event_head(x)
        s_logits = self.source_head(x)
        
        return e_logits, s_logits, new_past_kv
def train_high_precision(model, session_events, session_sources, split_idx, optimizer, criterion):
    model.train()
    
    # --- PHASE 1: PRIME ---
    # Feed the first 256 chunks all at once to build the initial cache
    ctx_e = session_events[:, :split_idx]
    ctx_s = session_sources[:, :split_idx]
    
    # We don't backprop through the 256 chunks, just get the starting cache
    with torch.no_grad():
        _, _, cache = model(ctx_e, ctx_s, past_kv=None)

    # --- PHASE 2: TRUTH OVERRIDE LOOP ---
    total_loss = 0
    # Predict from split_idx to the end of the session
    for i in range(split_idx, session_events.size(1) - 1):
        
        # The token we ARE at (the truth)
        current_e = session_events[:, i].unsqueeze(1)
        current_s = session_sources[:, i].unsqueeze(1)
        
        # The token we WANT to predict (the target)
        target_e = session_events[:, i+1]
        target_s = session_sources[:, i+1]
        
        # Forward pass: Only 1 token processed!
        e_logits, s_logits, cache = model(current_e, current_s, past_kv=cache)
        
        # Compute loss for the NEXT token prediction
        loss = criterion(e_logits[:, -1, :], target_e) + \
               criterion(s_logits[:, -1, :], target_s)
        
        # Backprop for this specific prediction
        loss.backward()
        
        # Clip and Step (or accumulate)
        # On a 4090, you can step every N tokens or every session
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # IMPORTANT: Detach cache to stop gradient history from eating VRAM
        cache = [kv.detach() for kv in cache]

    return total_loss

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size, dataset):
    setup(rank, world_size)
    
    # 1. Initialize Model on current GPU
    model = StatefulEventTransformer().to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 2. Loop through Sessions
    # Note: Use a DistributedSampler to ensure GPUs see different sessions
    for session_events, session_sources, split_idx in dataset:
        session_events = session_events.to(rank)
        session_sources = session_sources.to(rank)
        
        # Phase A: Prime Context (First 256 chunks)
        # We process this as a block to populate the cache
        ctx_e = session_events[:, :split_idx]
        ctx_s = session_sources[:, :split_idx]
        
        # Initial forward to get the cache
        # No grad here to save memory on the 4090s
        with torch.no_grad():
            _, _, cache = model.module.prime(ctx_e, ctx_s)

        # Phase B: The 128-Chunk "Truth Override" Loop
        # Iterate from the end of context to the end of the session
        for i in range(split_idx, session_events.size(1) - 1):
            
            # Current "Truth" token
            curr_e = session_events[:, i].unsqueeze(1)
            curr_s = session_sources[:, i].unsqueeze(1)
            
            # Target (Next Token)
            target_e = session_events[:, i+1]
            target_s = session_sources[:, i+1]

            # Forward pass: Single-step with cache
            e_logits, s_logits, cache = model.module.step(curr_e, curr_s, cache)
            
            # Compute step-wise loss
            loss = criterion(e_logits[:, -1, :], target_e) + \
                   criterion(s_logits[:, -1, :], target_s)
            
            # Backward: Gradients accumulate in model.parameters()
            # Divide by total steps to normalize the gradient magnitude
            (loss / (session_events.size(1) - split_idx)).backward()

            # DETACH: Crucial for VRAM stability
            cache = [kv.detach() for kv in cache]

        # 3. Synchronize & Update
        # DDP handles the gradient averaging across the two 4090s here
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    dist.destroy_process_group()


from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

def calculate_metrics(all_preds, all_targets, num_classes):
    # Flatten everything for sklearn
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    
    # F1 Macro gives an average across all event types
    f1 = f1_score(targets, preds, average='macro')
    
    # Specific FP/FN for the time_shift (ID 0)
    # This tells you if the model is failing to "close" chunks correctly
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    tp = cm[0, 0]
    fp = cm[:, 0].sum() - tp
    fn = cm[0, :].sum() - tp
    
    return f1, fp, fn

def validate(model, val_loader, criterion, device, writer, epoch):
    model.eval()
    total_val_loss = 0
    all_e_preds, all_e_targets = [], []

    with torch.no_grad():
        for events, sources, split_idx in val_loader:
            events, sources = events.to(device), sources.to(device)
            
            # 1. Prime the model
            _, _, cache = model.module.prime(events[:, :split_idx], sources[:, :split_idx])
            
            # 2. Autoregressive Generation (No Truth Override)
            curr_e = events[:, split_idx].unsqueeze(1)
            curr_s = sources[:, split_idx].unsqueeze(1)
            
            for i in range(split_idx, events.size(1) - 1):
                e_logits, s_logits, cache = model.module.step(curr_e, curr_s, cache)
                
                # Get prediction
                pred_e = torch.argmax(e_logits[:, -1, :], dim=-1)
                pred_s = torch.argmax(s_logits[:, -1, :], dim=-1)
                
                # Store for metrics
                all_e_preds.append(pred_e.item())
                all_e_targets.append(events[:, i+1].item())
                
                # FEED PREDICTION BACK (Inference Mode)
                curr_e = pred_e.unsqueeze(1)
                curr_s = pred_s.unsqueeze(1)
                
                # Detach cache to save VRAM
                cache = [kv.detach() for kv in cache]

    # Calculate metrics
    f1, fp, fn = calculate_metrics(all_e_preds, all_e_targets, 10)
    
    # Log to TensorBoard
    writer.add_scalar('Val/F1_Event', f1, epoch)
    writer.add_scalar('Val/TimeShift_FP', fp, epoch) # Model predicted shift too early
    writer.add_scalar('Val/TimeShift_FN', fn, epoch) # Model missed the shift


from torch.utils.tensorboard import SummaryWriter

if rank == 0:
    writer = SummaryWriter(log_dir="runs/transformer_experiment_1")

# Inside training loop:
if rank == 0 and step % 100 == 0:
    writer.add_scalar('Train/Loss', loss.item(), global_step)
    # Log the learning rate (important for warmup debugging)
    writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

import math
from torch.optim.lr_scheduler import LambdaLR

def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear Warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine Decay
        progress = float(current_step - num_warmup_steps) / \
                   float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)

# Usage:
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_scheduler(optimizer, num_warmup_steps=2000, num_training_steps=50000)

# Step the scheduler AFTER every batch/session update
optimizer.step()
scheduler.step()

# 1. The Peak Learning Rate (
# )
# For Transformers, the "optimal" learning rate is almost always lower than for CNNs. 
# Safe Starting Point: 
#  (
# ) is the industry standard for small-to-medium Transformers.
# Search Range: If the model converges too slowly, try up to 
# . If it's unstable (loss spikes), drop to 
# .
# Hardware Factor: Since you have dual 4090s, if you increase your total batch size, you can usually tolerate a slightly higher 
# . 
# 2. Warmup Steps (
# )
# Warmup prevents the Adam optimizer from making destructive updates when its "moments" are still uninitialized. 
# The 10% Rule: A common heuristic is to set warmup steps to 10% of your total training steps.
# Small Datasets: If your dataset is small (e.g., fewer than 50,000 steps), you can be more aggressive with a 2–5% warmup.
# Stability indicator: If your loss is extremely high or "chaotic" in the first epoch, increase the warmup duration to give the model more time to find a stable area of the loss landscape. 
# 3. The Minimum Learning Rate (
# ) 
# The cosine curve shouldn't necessarily hit zero, as the model still needs to make minor refinements at the end of training. 
# Recommended Value: Set 
#  to 1% to 10% of your 
#  (e.g., if 
#  is 
# , 
#  should be 
#  or 
# ). 
# 4. Weight Decay (Regularization)
# Weight decay is critical for preventing Transformers from overfitting on the specific sequences in your training set. 
# Standard Value: Use 
#  or 
#  with the AdamW optimizer.
# Precision Tip: Do not apply weight decay to LayerNorm weights or Embedding bias terms—this is a common trick to keep training stable. 


class StatefulEventTransformer(nn.Module):
    # ... previous __init__ code ...

    def prime(self, events, sources, padding_mask=None):
        """
        padding_mask: (Batch, Seq_Len) - True for [PAD] tokens
        """
        x = self.event_emb(events) + self.source_emb(sources)
        cache = []
        
        # Create Causal Mask (Triangular)
        sz = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(sz).to(x.device)

        for layer in self.layers:
            # We pass both the Causal Mask (what it can't see) 
            # and the Padding Mask (what doesn't exist)
            x = layer(x, x, tgt_mask=causal_mask, tgt_key_padding_mask=padding_mask)
            cache.append(x) 
            
        return x, cache

    def step(self, event, source, cache, padding_mask=None):
        """
        Note: During step-wise truth override, the 'padding_mask' 
        usually isn't needed unless the context itself was padded.
        """
        x = self.event_emb(event) + self.source_emb(source)
        new_cache = []
        
        for i, layer in enumerate(self.layers):
            combined = torch.cat([cache[i], x], dim=1)
            
            # Since 'combined' grows, the mask must also grow or be ignored.
            # Usually, during truth override on valid sessions, padding isn't an issue.
            x = layer(x, combined, tgt_key_padding_mask=padding_mask)
            new_cache.append(combined)
            
        return x, new_cache

events, sources, split_indices, pad_mask = next(iter(train_loader))
events, sources, pad_mask = events.to(device), sources.to(device), pad_mask.to(device)

# 1. Prime the whole batch
# Note: Use the mask so the context 256 chunks don't attend to padding
_, cache = model.prime(events[:, :max(split_indices)], 
                       sources[:, :max(split_indices)], 
                       padding_mask=pad_mask[:, :max(split_indices)])

# 2. Step Loop
# For simplicity with batching, loop until the absolute max length
for i in range(max(split_indices), events.size(1) - 1):
    e_logits, s_logits, cache = model.step(events[:, i:i+1], sources[:, i:i+1], cache)
    
    # Target
    target_e = events[:, i+1]
    
    # CRITICAL: Use 'reduction=none' so we can zero out padding loss
    loss_val = criterion(e_logits[:, -1, :], target_e)
    
    # Zero out loss for sessions that have already hit padding
    active_mask = ~pad_mask[:, i+1] 
    loss_val = (loss_val * active_mask).sum() / active_mask.sum()
    
    loss_val.backward()
    # ... optimizer step logic ...

from torch.utils.tensorboard import SummaryWriter

def log_experiment(hparams, metrics, run_name):
    # run_name should be unique (e.g., 'gpu0_d128_do01')
    with SummaryWriter(f'runs/{run_name}') as writer:
        # hparam_dict: Your configuration
        # metric_dict: Final results for this run
        writer.add_hparams(
            hparam_dict={
                'd_model': hparams['d_model'],
                'dropout': hparams['dropout'],
                'lr': hparams['lr'],
                'batch_size': 32
            },
            metric_dict={
                'hparam/f1_score': metrics['f1'],
                'hparam/timeshift_fp': metrics['fp'],
                'hparam/timeshift_fn': metrics['fn']
            }
        )
import torch.multiprocessing as mp
import itertools

def run_experiment(gpu_id, hparams):
    """Function executed on a specific GPU"""
    torch.cuda.set_device(gpu_id)
    
    # Generate a unique name for TensorBoard
    run_name = f"model_d{hparams['d_model']}_dr{hparams['dropout']}_lr{hparams['lr']}"
    print(f"Starting {run_name} on GPU {gpu_id}")
    
    # Your training function (the one we built earlier)
    # Ensure it returns final metrics (f1, fp, fn)
    metrics = train_model_standard_pytorch(gpu_id, hparams, run_name)
    
    # Log to HParams at the very end
    log_hparams(hparams, metrics, run_name)

if __name__ == "__main__":
    # Define your search space
    search_space = {
        'd_model': [128, 256],
        'dropout': [0.1, 0.2, 0.3],
        'lr': [1e-4, 5e-5]
    }
    
    # Create all combinations (Grid)
    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    # Use a pool or a simple loop to manage GPUs
    processes = []
    for i, hparam_set in enumerate(experiments):
        gpu_to_use = i % 2  # Alternate between 4090 (0) and 4090 (1)
        
        # Wait if both GPUs are busy (simple join)
        if len(processes) >= 2:
            for p in processes:
                p.join()
            processes = []
            
        p = mp.Process(target=run_experiment, args=(gpu_to_use, hparam_set))
        p.start()
        processes.append(p)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_metric):
        # We assume higher is better (like F1-Score)
        # For FN/FP (lower is better), flip the logic: if current < best
        if self.best_score is None:
            self.best_score = current_metric
        elif current_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_metric
            self.counter = 0

def get_structural_metrics(preds, targets):
    # preds/targets are flat lists from the autoregressive val loop
    # shift_id is (0,0)
    
    # 1. Total Shifts expected
    total_shifts = targets.count(0)
    
    # 2. False Positives: predicted 0, but truth was an event ID
    fps = sum(1 for p, t in zip(preds, targets) if p == 0 and t != 0)
    
    # 3. False Negatives: predicted event, but truth was a shift
    fns = sum(1 for p, t in zip(preds, targets) if p != 0 and t == 0)
    
    return {
        'early_term_rate': fps / total_shifts,
        'run_on_rate': fns / total_shifts
    }
early_stopper = EarlyStopping(patience=5)

for epoch in range(max_epochs):
    train_loss = train_one_epoch(...)
    val_metrics = validate_autoregressive(...) # Structural metrics here
    
    # Log to TensorBoard as scalars
    writer.add_scalar('Val/RunOnRate', val_metrics['run_on_rate'], epoch)
    
    # Check if we should stop
    early_stopper(val_metrics['f1_score'])
    if early_stopper.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

# Final metrics for HParams are the 'best_score' found
log_hparams(hparams, {'f1': early_stopper.best_score, ...}, run_name)

import torch

@torch.no_grad()
def predict_future_chunks(model_path, initial_events, initial_sources, max_chunks=128):
    """
    initial_events/sources: The first 256 chunks of the session (tensors)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    checkpoint = torch.load(model_path)
    # Re-initialize model using saved hparams
    model = StatefulEventTransformer(**checkpoint['hparams']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Prime the Context (Chunks 1-256)
    # This generates the 'past_kv' cache so we don't re-process the history
    _, cache = model.prime(initial_events.to(device), initial_sources.to(device))
    
    generated_events = []
    generated_sources = []
    chunks_completed = 0
    
    # Starting token for generation is the last event of the context
    curr_e = initial_events[:, -1:]
    curr_s = initial_sources[:, -1:]

    # 3. Autoregressive Loop
    # We stop once we've seen 128 time_shift tokens (0,0)
    while chunks_completed < max_chunks:
        # Step forward with 1 token
        e_logits, s_logits, cache = model.step(curr_e, curr_s, cache)
        
        # Greedy search for top precision (pick the most likely next event)
        pred_e = torch.argmax(e_logits[:, -1, :], dim=-1)
        pred_s = torch.argmax(s_logits[:, -1, :], dim=-1)
        
        # Track the generation
        generated_events.append(pred_e.item())
        generated_sources.append(pred_s.item())
        
        # If we predicted a time_shift (0,0), increment chunk counter
        if pred_e.item() == 0 and pred_s.item() == 0:
            chunks_completed += 1
            
        # FEED BACK: Use our own prediction as the next input
        curr_e = pred_e.unsqueeze(0)
        curr_s = pred_s.unsqueeze(0)
        
        # Break if the model goes into an infinite loop (safety)
        if len(generated_events) > 2000: 
            break

    return generated_events, generated_sources
# In your training loop
if val_f1 > best_val_f1:
    best_val_f1 = val_f1
    # Save the "Gold" model for inference
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hparams': hparams,
    }, f"checkpoints/{run_name}_best.pt")
    print(f"Saved new best model with F1: {val_f1:.4f}")



if __name__ == "__main__":
    mp.set_start_method('spawn')
    world_size = 2 # Your two 4090s
    # mp.spawn(train, args=(world_size, my_dataset), nprocs=world_size)
