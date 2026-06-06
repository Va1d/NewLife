import numpy as np
import os

def create_memmap_dataset(token_folder, output_path="dataset.bin"):
    # We only use the FIRST codebook (index 0) for 'meaning'
    # Each token is an integer up to 1024, so uint16 is perfect (saves 75% space)
    all_files = [f for f in os.listdir(token_folder) if f.endswith('.pt')]
    
    # Pre-calculate total length to allocate file
    total_len = 0
    for f in all_files:
        t = torch.load(os.path.join(token_folder, f))
        total_len += t.shape[1] 

    # Create the empty file on disk
    fp = np.memmap(output_path, dtype='uint16', mode='w+', shape=(total_len,))
    
    # Fill it
    offset = 0
    for f in all_files:
        t = torch.load(os.path.join(token_folder, f))[0].numpy().astype('uint16')
        length = t.shape[0]
        fp[offset:offset+length] = t
        offset += length
    
    fp.flush() # Write to disk
    print(f"Dataset created: {total_len} tokens (~{total_len*2 / 1e9:.2f} GB)")
