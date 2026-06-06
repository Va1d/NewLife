import torch
from torch.utils.data import Dataset, DataLoader

class SessionDataset(Dataset):
    def __init__(self, raw_data):
        # raw_data: your list of 1,500 sessions
        self.events = []
        self.sources = []
        self.split_indices = []

        for session in raw_data:
            # 1. Flatten the session tuples into tensors
            # 2. Find the split_idx for the 256th chunk
            # 3. Append to lists
            self.events.append(torch.tensor(session['event_ids'], dtype=torch.long))
            self.sources.append(torch.tensor(session['source_ids'], dtype=torch.long))
            self.split_indices.append(session['split_idx'])
            
        # These are now sitting in your 128GB RAM as a list of tensors
        # No more disk access!

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        return self.events[idx], self.sources[idx], self.split_indices[idx]
    
train_loader = DataLoader(
    train_dataset,
    batch_size=32, 
    shuffle=True,
    num_workers=0,      # Keep at 0 since data is already in RAM; avoids multi-process overhead
    pin_memory=True,    # Fast-track data from 128GB RAM to 4090 VRAM
    collate_fn=my_collate_fn # To handle variable lengths
)

from torch.nn.utils.rnn import pad_sequence

def my_collate_fn(batch):
    events, sources, splits = zip(*batch)
    
    # Pad sequences with your PAD_ID (e.g., 9 for events, 257 for sources)
    events_padded = pad_sequence(events, batch_first=True, padding_value=9)
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=257)
    
    # Create a Padding Mask (True where padding exists)
    # The Transformer needs this to ignore the padded tokens
    padding_mask = (events_padded == 9)
    
    return events_padded, sources_padded, torch.tensor(splits), padding_mask