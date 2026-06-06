import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Injects information about the relative position of measurements in the series."""
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

class TransformerVQVAE(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_embeddings):
        super().__init__()
        self.d_model = d_model
        
        # 1. Project input floats to Transformer dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. TRANSFORMER ENCODER: Global context awareness
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. VQ LAYER: The Codebook
        self.embedding = nn.Embedding(num_embeddings, d_model)
        
        # 4. TRANSFORMER DECODER: Contextual reconstruction
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, 
                                                   dim_feedforward=d_model*4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, input_dim)

    def quantize(self, z):
        # z: (Batch, 32, d_model)
        z_flattened = z.view(-1, self.d_model)
        
        # Distance calculation
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices).view(z.shape)
        
        # Straight-through estimator (gradient bypass)
        z_q = z + (z_q - z).detach()
        return z_q, indices

    def forward(self, x):
        # Encode
        x_proj = self.pos_encoder(self.input_projection(x))
        z = self.transformer_encoder(x_proj)
        
        # Quantize
        z_q, indices = self.quantize(z)
        
        # Decode
        # We use z_q as both 'tgt' and 'memory' for a strong reconstruction
        recon_out = self.transformer_decoder(z_q, z_q)
        reconstruction = self.output_projection(recon_out)
        
        return reconstruction, indices.view(x.size(0), -1), z, z_q


def vq_loss_function(reconstruction, original, encoded, quantized, beta=0.25):
    # 1. Reconstruction Loss
    recon_loss = F.mse_loss(reconstruction, original)
    
    # 2. VQ Losses (Dictionary + Commitment)
    # Dictionary loss: optimize the codebook
    dict_loss = F.mse_loss(quantized, encoded.detach())
    # Commitment loss: force encoder to stick to a prototype
    commitment_loss = F.mse_loss(encoded, quantized.detach())
    
    return recon_loss + dict_loss + beta * commitment_loss

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        # 1. Forward Pass
        # Note: Modify your model forward to return 'encoded' and 'quantized' 
        recon, int_features, encoded, quantized = model(batch)
        
        # 2. Calculate Loss
        loss = vq_loss_function(recon, batch, encoded, quantized)
        
        # 3. Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        # Check how many unique integers are used in a validation batch
        unique_ids = torch.unique(int_features).numel()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}, Unique IDs used: {unique_ids}/256")

# --- Usage Example ---
input_dim = 16   # size of your float vector
hidden_dim = 32  # internal processing size
vocab_size = 256 # 0-255 integer range

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

def train_transformer_vqvae(model, dataloader, epochs=100, lr=1e-4, device="cuda"):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    writer = SummaryWriter("runs/semantic_compressor")
    
    # 1. LR Warmup Schedule (Linear warmup for first 10% of steps)
    warmup_steps = len(dataloader) * (epochs // 10)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, step / warmup_steps))
    
    global_step = 0
    beta = 0.25 # Commitment loss weight

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_vq = 0, 0, 0
        used_indices = set()

        for batch in dataloader:
            batch = batch.to(device) # Shape: (B, 32, input_dim)
            
            # Forward Pass
            recon, indices, z_e, z_q = model(batch)
            
            # 2. Multi-Part Loss Function
            recon_loss = F.mse_loss(recon, batch)
            # VQ Loss: Dictionary (move embedding to encoder) + Commitment (move encoder to embedding)
            vq_loss = F.mse_loss(z_q.detach(), z_e) + beta * F.mse_loss(z_q, z_e.detach())
            
            loss = recon_loss + vq_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping is essential for Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Tracking Metrics
            used_indices.update(indices.view(-1).cpu().numpy())
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            global_step += 1

        # 3. Log Progress
        avg_loss = total_loss / len(dataloader)
        perplexity = len(used_indices) # How many "integers" were actually used
        
        writer.add_scalar("Loss/Total", avg_loss, epoch)
        writer.add_scalar("Loss/Reconstruction", total_recon / len(dataloader), epoch)
        writer.add_scalar("Metrics/Codebook_Usage", perplexity, epoch)
        writer.add_scalar("Metrics/Learning_Rate", scheduler.get_last_lr()[0], epoch)

        print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Used IDs: {perplexity}/{model.embedding.num_embeddings}")

        # 4. Automatic Codebook Reset (If usage is low)
        if perplexity < (model.embedding.num_embeddings * 0.1):
            reset_unused_embeddings(model, used_indices)

    writer.close()

def train_step(model, batch, criterion, optimizer, scheduler):
    model.train()
    optimizer.zero_grad()
    
    # Forward
    recon, indices, z_e, z_q = model(batch)
    
    # Calculate modular loss
    loss, recon_l, dict_l, commit_l = criterion(recon, batch, z_e, z_q)
    
    # Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    return {
        "loss": loss.item(),
        "recon": recon_l.item(),
        "indices": indices
    }
 class RobustTransformerVQVAE(TransformerVQVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Learnable vector that represents a 'missing measurement'
        self.mask_token = nn.Parameter(torch.randn(1, 1, kwargs['d_model']))

    def forward(self, x, mask=None):
        # x: (Batch, 32, input_dim)
        x_proj = self.input_projection(x)
        
        # If a mask is provided, replace masked positions with our learnable token
        if mask is not None:
            # mask is (Batch, 32), 0 at masked positions
            mask = mask.unsqueeze(-1) # (Batch, 32, 1)
            x_proj = x_proj * mask + self.mask_token * (1 - mask)
            
        x_pos = self.pos_encoder(x_proj)
        z_e = self.transformer_encoder(x_pos)
        
        z_q, indices = self.quantize(z_e)
        
        # Decoder attempts to reconstruct the FULL sequence, 
        # including the parts that were masked out.
        recon_out = self.transformer_decoder(z_q, z_q)
        reconstruction = self.output_projection(recon_out)
        
        return reconstruction, indices.view(x.size(0), -1), z_e, z_q   
    
class RobustVQLoss(nn.Module):
    def __init__(self, beta=0.25, imputation_weight=2.0):
        super().__init__()
        self.beta = beta
        self.imputation_weight = imputation_weight

    def forward(self, reconstruction, original, z_e, z_q, mask):
        # 1. Weighted Reconstruction Loss
        # We penalize errors on missing data more heavily to force 'understanding'
        mse_elementwise = F.mse_loss(reconstruction, original, reduction='none')
        
        # Increase weight for masked indices (mask == 0)
        weights = torch.ones_like(mask)
        weights[mask == 0] = self.imputation_weight
        
        recon_loss = (mse_elementwise * weights.unsqueeze(-1)).mean()
        
        # 2. Standard VQ Losses
        dict_loss = F.mse_loss(z_q, z_e.detach())
        commit_loss = F.mse_loss(z_e, z_q.detach())
        
        total_loss = recon_loss + dict_loss + self.beta * commit_loss
        return total_loss, recon_loss, dict_loss, commit_loss    
class VQLoss(nn.Module):
    def __init__(self, beta=0.25):
        super().__init__()
        self.beta = beta

    def forward(self, reconstruction, original, z_encoder, z_quantized):
        """
        reconstruction: output of the decoder
        original: the raw input float vectors
        z_encoder: the continuous output from the transformer encoder
        z_quantized: the vectors from the codebook selected by the indices
        """
        # 1. Reconstruction Loss: How well do we recreate the floats?
        recon_loss = F.mse_loss(reconstruction, original)
        
        # 2. Dictionary Loss: Move the Codebook weights closer to the Encoder outputs
        # We detach z_encoder because we only want to update the Embedding layer here
        dict_loss = F.mse_loss(z_quantized, z_encoder.detach())
        
        # 3. Commitment Loss: Force the Encoder to output vectors near the Codebook
        # We detach z_quantized because we only want to update the Encoder here
        commitment_loss = F.mse_loss(z_encoder, z_quantized.detach())
        
        total_loss = recon_loss + dict_loss + self.beta * commitment_loss
        
        return total_loss, recon_loss, dict_loss, commitment_loss
def reset_unused_embeddings(model, used_indices):
    """Re-initializes unused embeddings to prevent dead codebook entries."""
    n_embeddings = model.embedding.num_embeddings
    all_indices = set(range(n_embeddings))
    unused_indices = list(all_indices - used_indices)
    
    if len(unused_indices) > 0:
        with torch.no_grad():
            # Replace unused with random samples from the 'used' ones plus noise
            used_list = list(used_indices)
            replacement_indices = torch.tensor([used_list[i % len(used_list)] for i in range(len(unused_indices))])
            new_weights = model.embedding.weight[replacement_indices] + torch.randn_like(model.embedding.weight[replacement_indices]) * 0.01
            model.embedding.weight[unused_indices] = new_weights