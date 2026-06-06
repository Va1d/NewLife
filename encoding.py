import torch
import torch.nn as nn

#embeddings = model.device_id_embedding.weight.detach().cpu()

# 2. Add them to TensorBoard
#writer.add_embedding(embeddings, metadata=list(range(256)), tag="Device_Personalities")
def masked_mse_loss(y_pred, y_true, mask):
    """
    Args:
        y_pred: Model predictions
        y_true: Ground truth targets
        mask: Binary tensor (1 for valid, 0 for gap)
    """
    # 1. Compute element-wise squared error
    loss = F.mse_loss(y_pred, y_true, reduction='none')
    
    # 2. Zero out the loss at gap locations
    masked_loss = loss * mask.float()
    
    # 3. Average only over valid (non-gap) elements
    # Add a small epsilon to denominator to avoid division by zero
    return masked_loss.sum() / (mask.float().sum() + 1e-8)

def save_checkpoint(model, optimizer, scaler, epoch, path="checkpoint.pt"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Save your scaler constants so you can denormalize on any machine
        'scaler_mean': scaler.mean,
        'scaler_std': scaler.std,
        'target_indices': [0, 1, 2] # Useful to remember what you were predicting
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer, scaler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Reload scaler constants
    scaler.mean = checkpoint['scaler_mean']
    scaler.std = checkpoint['scaler_std']
    
    return checkpoint['epoch']

class PerDeviceScaler:
    def __init__(self, num_devices=256, feature_dim=24):
        # Shape: [256, 24]
        self.means = torch.zeros(num_devices, feature_dim)
        self.stds = torch.ones(num_devices, feature_dim)

    def fit(self, train_data):
        """
        train_data: [Total_Samples, 256, 24]
        """
        # Calculate mean/std across the 'Samples' dimension (dim 0)
        self.means = train_data.mean(dim=0)
        self.stds = train_data.std(dim=0)
        self.stds[self.stds < 1e-7] = 1.0 # Stability

    def transform(self, x):
        """
        x: [Batch, 256, 24]
        Broadcasting magic: PyTorch automatically aligns the 256x24 
        stats with the last two dims of the batch.
        """
        device = x.device
        return (x - self.means.to(device)) / self.stds.to(device)

    def inverse_transform_subset(self, x_scaled_subset, indices):
        """
        If you only predict a delta for indices [0, 1, 2], 
        use this to get back to physical units.
        """
        device = x_scaled_subset.device
        m = self.means[:, indices].to(device)
        s = self.stds[:, indices].to(device)
        return (x_scaled_subset * s) + m

class ConditionedSystemAutoencoder(nn.Module):
    def __init__(self, num_devices=256, source_dim=24, id_emb_dim=8, latent_dim=128):
        super().__init__()
        
        # 1. SHARED IDENTITY: One table for both Encoding and Decoding
        self.device_emb = nn.Embedding(num_devices, id_emb_dim)
        
        # 2. ENCODER: (Source + ID) -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(source_dim + id_emb_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # 3. DECODER: (Latent + ID) -> Reconstruction
        # We re-inject the ID here so the decoder knows the device's "personality"
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + id_emb_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, source_dim) 
        )

    def forward(self, x, device_ids):
        """
        x: [Batch, 256, 24] (Scaled input)
        device_ids: [256] (0 to 255)
        """
        b, s, f = x.shape
        
        # Get IDs for the whole batch: [B, 256, id_emb_dim]
        ids = device_ids.unsqueeze(0).expand(b, -1)
        id_feat = self.device_emb(ids)
        
        # --- ENCODE ---
        enc_input = torch.cat([x, id_feat], dim=-1)
        # Flatten B and S for the shared MLP
        latent = self.encoder(enc_input.view(b * s, -1))
        
        # --- DECODE ---
        # We must re-attach the ID feature to the latent vector
        # This tells the decoder: "Here is a thought, now express it as Device #42"
        id_feat_flat = id_feat.view(b * s, -1)
        dec_input = torch.cat([latent, id_feat_flat], dim=-1)
        
        reconstruction = self.decoder(dec_input)
        
        # Return reconstruction reshaped and the latents for the Transformer
        return reconstruction.view(b, s, f), latent.view(b, s, -1)


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        # Ensure weights is a tensor of shape [State_Dim]
        self.register_buffer('weights', weights)

    def forward(self, pred, target):
        # (pred - target)^2 calculates the square error for every element
        elementwise_sq_error = (pred - target) ** 2
        
        # Multiply each dimension's error by its specific weight
        weighted_error = elementwise_sq_error * self.weights
        
        return weighted_error.mean()
