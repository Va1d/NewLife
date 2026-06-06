import torch
import torch.nn as nn

class AcousticPolisher(nn.Module):
    def __init__(self, vocab_size=1024, n_embd=512, n_layers=6):
        super().__init__()
        # Input is Codebook 1
        self.embedding = nn.Embedding(vocab_size, n_embd)
        
        # Transformer Encoder (looks at surrounding semantic context)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 8 Heads: each head predicts one of the 8 acoustic codebooks
        self.heads = nn.ModuleList([
            nn.Linear(n_embd, vocab_size) for _ in range(8)
        ])

    def forward(self, cb1):
        # cb1: [Batch, Seq_Len]
        x = self.embedding(cb1) 
        x = self.transformer(x) # [Batch, Seq_Len, n_embd]
        
        # Predict 8 codebooks simultaneously
        logits = [head(x) for head in self.heads] 
        # Returns list of 8 tensors: [Batch, Seq_Len, 1024]
        return torch.stack(logits, dim=1) 