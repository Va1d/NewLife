import torch
import subprocess
import numpy as np
import dac
import math

# 1. Initialize DAC on a 4090
device = "cuda:0"
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)
model.to(device).eval()

def load_wav_with_ffmpeg(path, target_sr=44100):
    """Bypasses torchaudio/torchcodec using FFmpeg pipe."""
    command = [
        'ffmpeg', '-i', path,
        '-ac', '1', '-ar', str(target_sr),
        '-f', 'f32le', '-acodec', 'pcm_f32le', '-'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate()
    waveform = torch.from_numpy(np.frombuffer(out, dtype=np.float32)).clone()
    return waveform # Returns [T]

def tokenize_wav(wav_path, chunk_minutes=5):
    # 1. Load full audio into System RAM (CPU)
    print(f"Loading {wav_path} via FFmpeg...")
    full_waveform = load_wav_with_ffmpeg(wav_path)
    sr = 44100
    
    # 2. Calculate chunk sizes
    samples_per_chunk = int(chunk_minutes * 60 * sr)
    total_samples = full_waveform.shape[0]
    num_chunks = math.ceil(total_samples / samples_per_chunk)
    
    all_tokens = []
    
    print(f"Processing {num_chunks} chunks of ~{chunk_minutes} min each on {device}...")
    
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * samples_per_chunk
            end = min(start + samples_per_chunk, total_samples)
            
            # Slice and move ONLY the chunk to GPU
            chunk = full_waveform[start:end].to(device).unsqueeze(0).unsqueeze(0) # [1, 1, T]
            
            # DAC Preprocess & Encode
            # Note: DAC adds padding to handle model alignment
            x = model.preprocess(chunk, sr)
            _, codes, _, _, _ = model.encode(x)
            
            # Move tokens back to CPU to save VRAM
            all_tokens.append(codes.squeeze(0).cpu())
            
            print(f" - Chunk {i+1}/{num_chunks} complete.")
            
            # Clear GPU cache to be safe
            torch.cuda.empty_cache()

    # 3. Concatenate tokens along the time dimension (dim=1)
    # codes shape is [Codebooks, Time]
    final_tokens = torch.cat(all_tokens, dim=1)
    return final_tokens
# Run it!
try:
    tokens = tokenize_wav("/home/bo/Py/WAV/Uo4X4C_-580.wav", 5)
    print(f"Success! Tokens extracted: {tokens.shape}")
    torch.save(tokens, "voice_tokens.pt")
except Exception as e:
    print(f"Failed: {e}")
# FORCE torchaudio to use a stable backend (ffmpeg or sox)
# This bypasses the automatic attempt to use torchcodec
# if "ffmpeg" in torchaudio.list_audio_backends():
#     torchaudio.set_audio_backend("ffmpeg")
# else:
#     # If ffmpeg isn't listed, we'll use the soundfile backend
#     # pip install soundfile
#     torchaudio.set_audio_backend("soundfile")

# device = "cuda:0"
# model = dac.DAC.load(dac.utils.download(model_type="44khz"))
# model.to(device).eval()

# def tokenize_wav(wav_path):
#     # Use the 'backend' parameter to be absolutely sure
#     # This prevents the 'load_with_torchcodec' trigger
#     backend = torchaudio.get_audio_backend()
#     waveform, sr = torchaudio.load(wav_path, backend=backend)
#     waveform = waveform.to(device)
    
#     # 1. Stereo to Mono
#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)
            
#     # 2. Resample to 44.1kHz
#     if sr != 44100:
#         waveform = torchaudio.functional.resample(waveform, sr, 44100)

#     # 3. DAC Encoding
#     if waveform.dim() == 2:
#         waveform = waveform.unsqueeze(0) 
    
#     with torch.no_grad():
#         # model.preprocess handles normalization and padding
#         x = model.preprocess(waveform, 44100)
#         _, codes, _, _, _ = model.encode(x)
        
#     return codes.squeeze(0).cpu()

# torchaudio.list_audio_backends()
# Example usage
# tokens = tokenize_wav("/home/bo/Py/WAV/UjoTq-MJCRk.wav")
# print(f"Token shape: {tokens.shape}") # e.g., [1, 9, 450] for ~10s of audio