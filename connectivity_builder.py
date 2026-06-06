"""Build sparse connectivity matrices for 96-neuron Iris network

- 96 neurons total
- Each neuron connects to 24 of the other 95 neurons (sparse)
- 1/3 of neurons (32) receive external inputs (4 features)
- Memory capacity per neuron: 2 patterns (half of single neuron's 4)
"""

from __future__ import annotations

import numpy as np
import torch


def build_sparse_adjacency(num_neurons: int = 96, connections_per_neuron: int = 24, seed: int = 42) -> torch.Tensor:
    """
    Build sparse adjacency matrix where each neuron connects to specified number of others.
    
    Args:
        num_neurons: Total neurons
        connections_per_neuron: Out-degree for each neuron (connections to other neurons)
        seed: Random seed for reproducibility
    
    Returns:
        (num_neurons, num_neurons) adjacency matrix with 0s and 1s
    """
    np.random.seed(seed)
    adjacency = np.zeros((num_neurons, num_neurons), dtype=np.float32)
    
    for i in range(num_neurons):
        # Each neuron i connects to connections_per_neuron random other neurons
        # Exclude self-connections
        other_neurons = [j for j in range(num_neurons) if j != i]
        targets = np.random.choice(other_neurons, size=connections_per_neuron, replace=False)
        adjacency[i, targets] = 1.0
    
    return torch.tensor(adjacency, dtype=torch.float32)


def build_input_map(num_neurons: int = 96, num_features: int = 4, input_fraction: float = 1/3) -> torch.Tensor:
    """
    Build input routing matrix where only a fraction of neurons receive external inputs.
    
    Distributes input neurons evenly across the population (every 3rd neuron).
    Each input-receiving neuron gets all 4 features.
    
    Args:
        num_neurons: Total neurons (96)
        num_features: Number of input features (4 for Iris)
        input_fraction: Fraction of neurons that receive inputs (1/3 = 32 neurons)
    
    Returns:
        (num_neurons, num_features) input routing matrix
    """
    input_map = np.zeros((num_neurons, num_features), dtype=np.float32)
    
    num_input_neurons = int(num_neurons * input_fraction)
    step = num_neurons // num_input_neurons  # Should be 3 for 96 neurons
    
    # Place input neurons evenly spaced
    input_neuron_indices = list(range(0, num_neurons, step))[:num_input_neurons]
    
    # Each input-receiving neuron gets all features equally
    for idx in input_neuron_indices:
        input_map[idx, :] = 1.0 / num_features
    
    return torch.tensor(input_map, dtype=torch.float32)


def print_connectivity_stats(adjacency: torch.Tensor, input_map: torch.Tensor) -> None:
    """Print statistics about the connectivity"""
    num_neurons = adjacency.shape[0]
    num_features = input_map.shape[1]
    
    # Adjacency stats
    total_connections = adjacency.sum().item()
    avg_out_degree = total_connections / num_neurons
    
    # Input map stats
    input_receiving_neurons = (input_map.sum(dim=1) > 0).sum().item()
    
    print("=" * 70)
    print("CONNECTIVITY STATISTICS")
    print("=" * 70)
    print(f"Total neurons: {num_neurons}")
    print(f"Total features: {num_features}")
    print()
    print(f"Adjacency matrix (neuron-to-neuron):")
    print(f"  Shape: {adjacency.shape}")
    print(f"  Total connections: {int(total_connections)}")
    print(f"  Avg out-degree per neuron: {avg_out_degree:.1f}")
    print(f"  Sparsity: {(1 - total_connections / (num_neurons * num_neurons)):.1%}")
    print()
    print(f"Input map (external features):")
    print(f"  Shape: {input_map.shape}")
    print(f"  Neurons receiving inputs: {input_receiving_neurons} ({input_receiving_neurons/num_neurons:.1%})")
    print(f"  Input neurons indices: {[i for i in range(num_neurons) if input_map[i].sum() > 0]}")
    print()
    print(f"Capacity estimates:")
    print(f"  Patterns per neuron: 2 (half of original 4)")
    print(f"  Total pattern capacity: {num_neurons * 2} = {96 * 2}")
    print(f"  Iris samples: 150")
    print(f"  Coverage: {(96*2)/150:.1%} with capacity to spare for generalization")
    print("=" * 70)


if __name__ == "__main__":
    # Build matrices
    adjacency = build_sparse_adjacency(num_neurons=96, connections_per_neuron=24, seed=42)
    input_map = build_input_map(num_neurons=96, num_features=4, input_fraction=1/3)
    
    print_connectivity_stats(adjacency, input_map)
    
    # Save for use in iris_sandbox.py
    torch.save({"adjacency": adjacency, "input_map": input_map}, "/home/bo/Py/NewLife/.venv/src/connectivity_96.pt")
    print("\nSaved connectivity matrices to connectivity_96.pt")
