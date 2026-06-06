import torch
import dac
import torchaudio

@torch.no_grad()
def generate_and_reconstruct(prompt_wav, mind_model, polisher_model, dac_model, duration_secs=10):
    device = "cuda:0"
    
    # 1. Get Prompt Tokens (The "Seed")
    # (Use your previous tokenize_wav_clean function here)
    prompt_tokens = tokenize_wav_clean(prompt_wav) # Shape: [9, Seq_Len]
    cb1_prompt = prompt_tokens[0:1, :].to(device) # Just the 1st codebook
    
    # 2. Mind Model: Generate Continuation
    # We want to predict the next ~1000 tokens (10 seconds)
    generated_cb1 = cb1_prompt
    for _ in range(duration_secs * 100):
        # Predict next token (standard autoregressive sampling)
        logits, _ = mind_model(generated_cb1[:, -1024:]) # Context window
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_cb1 = torch.cat([generated_cb1, next_token], dim=1)
    
    # 3. Polisher: Add Acoustic Detail
    # Input generated CB1 -> Output all 9 layers
    # Generated_cb1 shape: [1, Seq_Len]
    all_logits = polisher_model(generated_cb1) # [1, 8, Seq_Len, 1024]
    cb2_to_9 = torch.argmax(all_logits, dim=-1) # [1, 8, Seq_Len]
    
    # 4. Final Reconstruction
    # Stitch CB1 and the Polished layers back together
    final_9_layers = torch.cat([generated_cb1.unsqueeze(1), cb2_to_9], dim=1)
    
    # Use DAC to turn tokens back into sound
    z_quantized = dac_model.quantizer.from_codes(final_9_layers)[0]
    generated_wav = dac_model.decode(z_quantized)
    
    # 5. Save to disk
    torchaudio.save("ai_monologue.wav", generated_wav.squeeze(0).cpu(), 44100)
    print("Creation complete: ai_monologue.wav")
