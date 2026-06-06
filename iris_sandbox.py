"""Iris dataset with spiking neural network

Load Iris, quantize to 6 bits per feature (4 features = 24 total bits),
feed into spiking network as sequence of byte inputs, measure per-class spike patterns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
from sklearn import datasets  # type: ignore[import-not-found]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

from spiking_sandbox import SimpleThresholdNeuron, SpikeConfig, SpikingNetwork


def load_and_prepare_iris(num_bits: int = 6) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Load Iris dataset and quantize to num_bits per feature.
    
    Args:
        num_bits: Bits per feature (6 bits = 0-63 range)
    
    Returns:
        (X_quantized, y, class_names)
        - X_quantized: (150, 4) array of byte values 0-63 (4 features per sample)
        - y: (150,) class labels 0-2
        - class_names: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    """
    # Load Iris
    iris = datasets.load_iris()
    X = iris.data  # (150, 4)
    y = iris.target  # (150,)
    
    # Normalize features to 0-1
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)  # (150, 4)
    X_01 = (X_norm - X_norm.min(axis=0)) / (X_norm.max(axis=0) - X_norm.min(axis=0))  # 0-1 range
    
    # Quantize to num_bits
    max_val = (2 ** num_bits) - 1  # e.g., 63 for 6 bits
    X_quantized = (X_01 * max_val).astype(np.int32)  # (150, 4) with values 0-63
    
    class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
    
    return X_quantized, y, class_names


def run_iris_experiment(
    num_neurons: int = 96,
    decay_factor: int = 95,
    num_bits: int = 6,
    connectivity_file: str = "/home/bo/Py/NewLife/.venv/src/connectivity_96.pt",
) -> Dict[str, Any]:
    """
    Run spiking network on Iris dataset.
    
    Feed each of 4 features as separate timesteps.
    Neuron learns its own threshold (no hardcoded value).
    
    Args:
        num_neurons: Number of spiking neurons
        decay_factor: Decay percentage (0-100), neuron handles threshold internally
        num_bits: Quantization bits per feature
        connectivity_file: Path to saved connectivity matrices
    
    Returns:
        Dictionary with per-class statistics
    """
    # Load Iris
    X_quantized, y, class_names = load_and_prepare_iris(num_bits=num_bits)
    num_samples, num_features = X_quantized.shape
    
    # Create config (threshold removed - neuron handles internally)
    config = SpikeConfig(
        num_neurons=num_neurons,
        num_input_bits=8,  # Independent of Iris features
        threshold=32,  # Neuron will adapt; this is initial guess
        decay_factor=decay_factor,
    )
    
    # Load sparse connectivity matrices
    connectivity = torch.load(connectivity_file)
    adjacency = connectivity["adjacency"]  # (96, 96)
    input_map = connectivity["input_map"]  # (96, 4)
    
    print("=" * 70)
    print(f"Iris Spiking Network Experiment (Dynamic Neuron)")
    print(f"Features per sample: {num_features}")
    print(f"Quantization: {num_bits} bits per feature (0-{(2**num_bits)-1})")
    print(f"Network: {num_neurons} neurons, decay={decay_factor}%")
    print(f"  Capacity per neuron: 2 patterns")
    print(f"  Total capacity: {num_neurons * 2} patterns")
    print(f"  Input neurons: {int((input_map.sum(dim=1) > 0).sum())} (1/3 of population)")
    print(f"  Connectivity: Sparse, ~24 connections per neuron (75% sparse)")
    print(f"Samples: {num_samples} (50 per class)")
    print(f"Process: Feed 4 features sequentially as timesteps")
    print("=" * 70)
    
    # Run network on each sample
    spike_counts_per_class: Dict[int, List[List[int]]] = {0: [], 1: [], 2: []}
    spike_patterns_per_class: Dict[int, List[List[int]]] = {0: [], 1: [], 2: []}
    
    for sample_idx in range(num_samples):
        sample = X_quantized[sample_idx]  # (4,) array of ints 0-63
        label = int(y[sample_idx])
        
        # Create fresh network for this sample
        network = SpikingNetwork(
            neuron_class=SimpleThresholdNeuron,
            num_neurons=num_neurons,
            adjacency=adjacency,
            input_map=input_map,
            config=config,
        )
        
        # Feed each of 4 features as separate timestep
        for feature_idx in range(num_features):
            feature_byte = torch.tensor([sample[feature_idx]], dtype=torch.uint8)
            _ = network.process_step(feature_byte)
        
        # Record spike counts per neuron
        spike_counts = network.spike_counts.copy()
        spike_counts_per_class[label].append(spike_counts)
        
        # Record spike pattern (binary: 1 if spiked, 0 otherwise)
        spike_pattern = [1 if cnt > 0 else 0 for cnt in spike_counts]
        spike_patterns_per_class[label].append(spike_pattern)
    
    # Compute per-class statistics
    stats: Dict[int, Dict[str, Any]] = {}
    for class_id in [0, 1, 2]:
        spike_counts = spike_counts_per_class[class_id]  # List[List[int]]
        spike_patterns = spike_patterns_per_class[class_id]  # List[List[int]]
        
        # Convert to arrays
        spike_counts_arr = np.array(spike_counts)  # (50, num_neurons)
        spike_patterns_arr = np.array(spike_patterns)  # (50, num_neurons)
        
        # Compute avg spike counts per neuron across samples of this class
        avg_spike_counts = spike_counts_arr.mean(axis=0)  # (num_neurons,)
        
        # Compute spike activation rate per neuron (how many samples triggered this neuron)
        spike_activation_rate = spike_patterns_arr.mean(axis=0)  # (num_neurons,) in [0, 1]
        
        # Find neurons most responsive to this class
        top_neurons = np.argsort(-avg_spike_counts)[:5]  # Top 5 by avg spike count
        
        stats[class_id] = {
            "class_name": class_names[class_id],
            "num_samples": len(spike_counts),
            "avg_spike_counts": avg_spike_counts.tolist(),
            "spike_activation_rate": spike_activation_rate.tolist(),
            "top_responsive_neurons": top_neurons.tolist(),
            "avg_total_spikes": spike_counts_arr.sum() / len(spike_counts),
        }
    
    return stats


def print_stats(stats: Dict[int, Dict[str, Any]]) -> None:
    """Pretty-print per-class statistics"""
    print("\n" + "=" * 70)
    print("PER-CLASS SPIKE STATISTICS")
    print("=" * 70)
    
    for class_id in [0, 1, 2]:
        s = stats[class_id]
        print(f"\n{s['class_name'].upper()} ({s['num_samples']} samples)")
        print(f"  Avg total spikes per sample: {s['avg_total_spikes']:.1f}")
        print(f"  Top 5 responsive neurons: {s['top_responsive_neurons']}")
        
        # Show per-neuron activation rates for top neurons
        for neuron_id in s["top_responsive_neurons"]:
            act_rate = s["spike_activation_rate"][neuron_id]
            avg_count = s["avg_spike_counts"][neuron_id]
            print(f"    Neuron {neuron_id}: active in {act_rate*100:.1f}% of samples, avg {avg_count:.1f} spikes")
    
    print("\n" + "=" * 70)
    print("CLASS DISCRIMINABILITY (by neuron)")
    print("=" * 70)
    
    # Find neurons that differ most across classes
    all_neurons = list(range(len(stats[0]["avg_spike_counts"])))
    neuron_discriminability = []
    
    for neuron_id in all_neurons:
        counts_by_class = [stats[cid]["avg_spike_counts"][neuron_id] for cid in [0, 1, 2]]
        variance = np.var(counts_by_class)
        neuron_discriminability.append((neuron_id, variance, counts_by_class))
    
    neuron_discriminability.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 discriminative neurons (highest spike variance across classes):")
    for rank, (neuron_id, variance, counts) in enumerate(neuron_discriminability[:5]):
        print(f"  {rank+1}. Neuron {neuron_id}: variance={variance:.2f}")
        print(f"     Setosa={counts[0]:.1f}, Versicolor={counts[1]:.1f}, Virginica={counts[2]:.1f}")


if __name__ == "__main__":
    # Run experiment with sparse 96-neuron network
    stats = run_iris_experiment(
        num_neurons=96,
        decay_factor=95,
        num_bits=6,
    )
    
    print_stats(stats)
