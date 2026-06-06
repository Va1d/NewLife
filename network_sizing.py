"""
Network sizing calculation for byte-input spiking neurons with Hamming receptive fields.

Each neuron has 4 prototypes and responds to patterns within Hamming distance 1 of each.
Calculate minimum network size to cover 50% of the dataset's byte distribution.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Set

import torch  # type: ignore[import-not-found]

from seq_mnist import get_sequential_mnist_loaders  # type: ignore[import-not-found]


def hamming_distance(x: int, y: int, num_bits: int = 4) -> int:
    """Calculate Hamming distance between two 4-bit shade values (0-15)."""
    xor: int = x ^ y
    distance: int = 0
    while xor:
        distance += xor & 1
        xor >>= 1
    return distance


def get_receptive_field(prototypes: List[int], max_distance: int = 1) -> Set[int]:
    """
    Get all 4-bit shade values (0-15) within Hamming distance max_distance from any prototype.
    
    Args:
        prototypes: List of 4 prototype shade values (0-15)
        max_distance: Max Hamming distance to include (default 1)
    
    Returns:
        Set of all shade values in receptive field
    """
    field: Set[int] = set()
    for proto in prototypes:
        for candidate in range(16):  # 0-15 for 4-bit
            if hamming_distance(proto, candidate, 4) <= max_distance:
                field.add(candidate)
    return field


def sample_dataset_bytes(train_loader: Any, num_samples: int = 1000) -> Counter:  # type: ignore[misc]
    """
    Sample 4-bit shade values from Sequential MNIST training set.
    
    Args:
        train_loader: DataLoader for Sequential MNIST
        num_samples: Number of samples to process
    
    Returns:
        Counter of shade value frequencies
    """
    shade_counter: Counter = Counter()
    
    samples_processed: int = 0
    for seq, _label in train_loader:
        if samples_processed >= num_samples:
            break
        
        seq_tensor: torch.Tensor = seq.squeeze(0)  # (784, 1) or (784,)
        seq_len: int = seq_tensor.shape[0]
        
        # Process 4 pixels at a time as 4-bit shades
        for t in range(0, seq_len, 4):
            end_idx: int = min(t + 4, seq_len)
            chunk: torch.Tensor = seq_tensor[t:end_idx].squeeze(-1)
            
            # Pad if needed
            if chunk.shape[0] < 4:
                padding: torch.Tensor = torch.zeros(4 - chunk.shape[0])
                chunk = torch.cat([chunk, padding])
            
            # Convert to 4-bit shade
            bits: torch.Tensor = (chunk > 0.5).float()
            powers: torch.Tensor = torch.pow(2.0, torch.arange(3, -1, -1, dtype=torch.float32))
            shade_val: torch.Tensor = torch.sum(bits * powers)
            shade_counter[int(shade_val.item())] += 1  # type: ignore[misc]
        
        samples_processed += 1
    
    return shade_counter


def calculate_network_size(shade_counter: Counter, coverage_target: float = 0.5) -> dict:  # type: ignore[misc]
    """
    Calculate network size needed to cover coverage_target of the dataset.
    
    Args:
        shade_counter: Counter of shade value frequencies (0-15)
        coverage_target: Target coverage (0-1), default 0.5 for 50%
    
    Returns:
        Dict with sizing information
    """
    total_shades: int = sum(shade_counter.values())  # type: ignore[misc]
    target_count: int = int(total_shades * coverage_target)
    
    # Sort by frequency
    sorted_shades: List[tuple] = shade_counter.most_common()
    
    # Greedy: pick neurons with 4 prototypes each to maximize coverage
    neurons_needed: int = 0
    coverage_count: int = 0
    used_shades: set = set()
    prototypes_list: List[List[int]] = []
    
    for shade_val, freq in sorted_shades:
        if coverage_count >= target_count:
            break
        
        if shade_val not in used_shades:
            # Start a new neuron with 4 prototypes
            prototypes: List[int] = []
            available_shades: List[tuple] = [
                (s, f) for s, f in sorted_shades 
                if s not in used_shades and s not in prototypes
            ]
            
            # Pick top 4 unused shades as prototypes
            for i in range(min(4, len(available_shades))):
                proto_shade, proto_freq = available_shades[i]
                prototypes.append(proto_shade)
                used_shades.add(proto_shade)
            
            # Get receptive field for this neuron
            receptive_field: Set[int] = get_receptive_field(prototypes, max_distance=1)
            
            # Count coverage from this neuron
            field_coverage: int = sum(shade_counter[s] for s in receptive_field if s in shade_counter)
            coverage_count += field_coverage
            
            prototypes_list.append(prototypes)
            neurons_needed += 1
    
    # Calculate actual coverage percentage
    actual_coverage: float = min(coverage_count / total_shades, 1.0)
    
    # Calculate average receptive field size
    avg_field_size: float = sum(
        len(get_receptive_field(protos, max_distance=1)) 
        for protos in prototypes_list
    ) / len(prototypes_list) if prototypes_list else 0
    
    return {
        "neurons_needed": neurons_needed,
        "target_coverage": coverage_target,
        "actual_coverage": actual_coverage,
        "actual_coverage_pct": actual_coverage * 100,
        "shades_covered": coverage_count,
        "total_shades_in_dataset": total_shades,
        "unique_shade_values": len(shade_counter),
        "avg_receptive_field_size": avg_field_size,
        "prototypes": prototypes_list,
    }


if __name__ == "__main__":
    print("="*70)
    print("Network Sizing Calculation (4-bit shades: 0-15)")
    print("="*70)
    
    # Load data
    print("\nLoading Sequential MNIST dataset...")
    train_loader, _ = get_sequential_mnist_loaders(batch_size=1, train_split=1.0)  # type: ignore[misc]
    
    # Sample shade distribution
    print("Sampling shade distribution from 1000 training samples...")
    shade_counter: Counter = sample_dataset_bytes(train_loader, num_samples=1000)
    
    print(f"Found {len(shade_counter)} unique shade values in dataset")
    print(f"Total shade observations: {sum(shade_counter.values())}")
    print(f"Top 10 most common shades (0-15):")
    for shade_val, count in shade_counter.most_common(10):
        print(f"  Shade {shade_val:2d}: {count:6d} times")
    
    # Calculate for 50% coverage
    print("\n" + "="*70)
    print("NETWORK SIZING FOR 50% COVERAGE:")
    print("="*70)
    result: dict = calculate_network_size(shade_counter, coverage_target=0.5)  # type: ignore[misc]
    
    print(f"\nNeurons needed: {result['neurons_needed']}")
    print(f"Target coverage: {result['target_coverage']*100:.1f}%")
    print(f"Actual coverage: {result['actual_coverage_pct']:.1f}%")
    print(f"Shades covered: {result['shades_covered']} / {result['total_shades_in_dataset']}")
    print(f"Unique values in dataset: {result['unique_shade_values']}")
    print(f"Avg receptive field per neuron: {result['avg_receptive_field_size']:.1f} shade values")
    
    print("\n" + "="*70)
    print("PROTOTYPES PER NEURON:")
    print("="*70)
    for i, protos in enumerate(result["prototypes"]):
        field_size: int = len(get_receptive_field(protos, max_distance=1))
        print(f"Neuron {i+1}: prototypes={protos}, receptive_field_size={field_size}")
    
    print("\n" + "="*70)
    print(f"RECOMMENDATION: Use {result['neurons_needed']} neurons")
    print("="*70)
