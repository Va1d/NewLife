def sample_top_p(logits, p=0.9, temperature=1.0):
    # Apply temperature to control 'creativity'
    logits = logits / temperature
    
    # Sort logits and calculate cumulative probabilities
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold (Nucleus)
    sorted_indices_to_remove = cumulative_probs > p
    # Shift to include the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Model Scale ---")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Params:    {trainable_params:,}")
    print(f"Model Size (MB):     {total_params * 4 / (1024**2):.2f} MB (at float32)")
    return total_params


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"--- Model Scale ---")
    print(f"Total Parameters:    {total_params:,}")
    print(f"Trainable Params:    {trainable_params:,}")
    print(f"Model Size (MB):     {total_params * 4 / (1024**2):.2f} MB (at float32)")
    return total_params