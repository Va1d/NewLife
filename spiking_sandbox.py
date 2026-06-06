"""Discrete Spiking Neural Network Sandbox

Neurons process bit-stream inputs sequentially, spike when threshold crossed.
No backprop—just threshold mechanics and optional local learning.
Measures emergent spike patterns for classification without explicit loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Tuple

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]


@dataclass
class SpikeConfig:
    """Configuration for discrete spiking network"""
    num_neurons: int = 64
    num_input_bits: int = 8  # Number of input bits per timestep (unrelated to pixel quantization)
    threshold: int = 32  # Byte threshold (0-255)
    decay_factor: int = 95  # Decay as percentage (0-100), applied as (potential * decay_factor) // 100
    refractory_period: int = 1  # Steps before neuron can spike again
    device: str = "cpu"


class SpikingNeuron(Protocol):
    """Interface for spiking neuron"""

    def reset(self) -> None:
        """Reset internal state"""
        ...

    def receive_input(self, inputs: List[int]) -> None:
        """
        Receive vector of byte inputs from all sources.
        
        Args:
            inputs: List of byte inputs (0-255), sources agnostic (external, recurrent, etc.)
        """
        ...

    def process_step(self) -> int:
        """
        Process current timestep: check threshold, decide spike
        
        Returns:
            spike: 0 or 1 (whether neuron spiked this step)
        """
        ...


class SimpleThresholdNeuron:
    """Simple integrate-and-fire spiking neuron (all-byte communication)"""

    def __init__(self, config: SpikeConfig, num_inputs: int) -> None:
        self.config: SpikeConfig = config
        self.num_inputs: int = num_inputs  # Expected number of input bytes
        self.membrane_potential: int = 0  # 0-255 byte range
        self.refractory_counter: int = 0

    def reset(self) -> None:
        """Reset internal state for new sequence"""
        self.membrane_potential = 0
        self.refractory_counter = 0

    def receive_input(self, inputs: List[int]) -> None:
        """Store inputs (bytes) for integration in process_step"""
        # Sum all inputs, clamp to byte range
        total: int = sum(max(0, min(255, inp)) for inp in inputs)  # type: ignore[misc]
        self.membrane_potential += total
        if self.membrane_potential > 255:
            self.membrane_potential = 255
        elif self.membrane_potential < 0:
            self.membrane_potential = 0

    def process_step(self) -> int:
        """
        Check threshold and decide spike using only byte arithmetic.
        
        Returns:
            spike: 1 if neuron fired, 0 otherwise
        """
        # Refractory period: can't spike yet
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            # Apply decay to potential during refractory period
            self.membrane_potential = (self.membrane_potential * self.config.decay_factor) // 100
            return 0

        # Check threshold
        spike: int = 1 if self.membrane_potential > self.config.threshold else 0

        # If spiked, reset potential and enter refractory period
        if spike:
            self.membrane_potential = 0
            self.refractory_counter = self.config.refractory_period

        # Decay potential
        self.membrane_potential = (self.membrane_potential * self.config.decay_factor) // 100

        return spike


class SpikingNetwork:
    """Encapsulates a complete spiking neural network with local connectivity"""

    def __init__(
        self,
        neuron_class: type,
        num_neurons: int,
        adjacency: torch.Tensor,
        input_map: torch.Tensor,
        config: SpikeConfig,
    ) -> None:
        """
        Args:
            neuron_class: Spiking neuron implementation (e.g., SimpleThresholdNeuron)
            num_neurons: Number of neurons in network
            adjacency: (num_neurons, num_neurons) connectivity matrix
            input_map: (num_neurons, input_size) input routing matrix
            config: SpikeConfig with threshold, decay, refractory settings
        """
        self.num_neurons: int = num_neurons
        self.adjacency: torch.Tensor = adjacency
        self.input_map: torch.Tensor = input_map
        self.config: SpikeConfig = config
        self.device: str = config.device

        # Initialize neuron pool with config and input connectivity info
        self.neurons: List[SimpleThresholdNeuron] = []
        for i in range(num_neurons):
            # Count incoming connections from adjacency matrix for neuron i
            incoming_connections: int = int((adjacency[i, :] != 0).sum().item())  # type: ignore[misc]
            # Total inputs: 1 external + incoming from other neurons
            num_inputs: int = 1 + incoming_connections
            self.neurons.append(neuron_class(config, num_inputs))

        # Metrics tracking
        self.spike_raster: torch.Tensor = torch.zeros((num_neurons, 0), device=self.device, dtype=torch.uint8)
        self.spike_counts: List[int] = [0] * num_neurons
        self.step_count: int = 0
        self.last_spikes: List[int] = [0] * num_neurons  # Track spikes for recurrent routing

    def reset(self) -> None:
        """Reset all neurons and metrics"""
        for neuron in self.neurons:
            neuron.reset()  # type: ignore[misc]
        self.spike_raster = torch.zeros((self.num_neurons, 0), device=self.device, dtype=torch.uint8)
        self.spike_counts = [0] * self.num_neurons
        self.step_count = 0
        self.last_spikes = [0] * self.num_neurons  # type: ignore[misc]

    def process_step(self, external_input: torch.Tensor) -> torch.Tensor:
        """
        Process one timestep through the network.

        Args:
            external_input: (1,) byte-valued input tensor

        Returns:
            spikes: (num_neurons,) spike outputs
        """
        self.step_count += 1
        external_input_byte: int = int(external_input[0].item())  # type: ignore[misc]
        external_input_byte = max(0, min(255, external_input_byte))  # Ensure byte range

        # Collect spikes from all neurons
        spikes_curr: List[int] = []
        for i in range(self.num_neurons):
            # Build input list: external + simple recurrent scaling
            inputs: List[int] = [external_input_byte]
            
            # Simple recurrence: multiply last spike by some strength (avoid full matrix for speed)
            recurrent_strength: int = 50  # Fixed coupling strength as byte value
            for j in range(self.num_neurons):
                if self.last_spikes[j] == 1:  # type: ignore[misc]
                    inputs.append(recurrent_strength)
            
            # Provide inputs and get spike
            self.neurons[i].receive_input(inputs)  # type: ignore[misc]
            spike: int = self.neurons[i].process_step()  # type: ignore[misc]
            spikes_curr.append(spike)
            self.last_spikes[i] = spike  # type: ignore[misc]

        spikes_tensor: torch.Tensor = torch.tensor(
            spikes_curr,  # type: ignore[misc]
            device=self.device,
            dtype=torch.uint8,
        )

        # Track metrics
        self.spike_raster = torch.cat(
            [self.spike_raster, spikes_tensor.unsqueeze(-1)], dim=1
        )
        for i in range(self.num_neurons):
            self.spike_counts[i] += spikes_tensor[i].item()  # type: ignore[misc]

        return spikes_tensor

    def get_spike_rate(self) -> int:
        """Return average spike rate as integer percentage (0-255)"""
        if self.step_count == 0:
            return 0
        total_spikes: int = sum(self.spike_counts)  # type: ignore[misc]
        # Return as percentage scaled to byte range
        rate_percent: int = (total_spikes * 255) // (self.step_count * self.num_neurons)
        return max(0, min(255, rate_percent))

    def get_spike_raster(self) -> torch.Tensor:
        """Return (num_neurons, num_steps) spike raster"""
        return self.spike_raster

    def get_metrics(self) -> Dict[str, Any]:
        """Return dict of current network metrics"""
        return {
            "step_count": self.step_count,
            "spike_counts": self.spike_counts.copy(),
            "avg_spike_rate": self.get_spike_rate(),
            "sparsity": float((torch.tensor(self.spike_counts) < 0.01 * self.step_count).sum().item()),  # type: ignore[misc]
        }


def run_spiking_experiment(
    dataset_loader: Any,
    network: SpikingNetwork,
) -> Tuple[List[List[int]], List[Tuple[torch.Tensor, int]], Dict[int, Dict[str, Any]]]:
    """
    Run spiking network on dataset, measure spike patterns per sample and per digit.

    Args:
        dataset_loader: Yields (seq, label) from Sequential MNIST
        network: SpikingNetwork instance to run experiment with

    Returns:
        spike_rates: List[List[int]] - per-neuron spike counts per sample
        spike_rasters: List[Tuple] - (spike_data, digit_label) for visualization
        summary_by_digit: Dict - average spike rate per digit class
    """
    spike_rates: List[List[int]] = []
    spike_rasters: List[Tuple[torch.Tensor, int]] = []
    digit_summaries: Dict[int, Dict[str, List[Any]]] = {d: {"spike_counts": [], "sequence_length": []} for d in range(10)}

    for sample_idx, (seq, label) in enumerate(dataset_loader):
        # seq shape: (1, 784, 1) if batch_size=1
        seq_device: torch.Tensor = seq.squeeze(0).to(network.device)  # (784, 1)
        seq_len: int = seq_device.shape[0]

        # Reset network for new sequence
        network.reset()

        # Process sequence
        for t in range(0, seq_len, 4):  # Group 4 pixels per timestep for 4-bit quantization
            # Extract chunk of 4 pixels from sequence
            end_idx: int = min(t + 4, seq_len)
            chunk: torch.Tensor = seq_device[t:end_idx].squeeze(-1)  # (4,)
            
            # Pad if needed (last chunk may be shorter)
            if chunk.shape[0] < 4:
                padding: torch.Tensor = torch.zeros(4 - chunk.shape[0], device=seq_device.device)
                chunk = torch.cat([chunk, padding])
            
            # Quantize to discrete 0/1 bits
            bits: torch.Tensor = (chunk > 0.5).float()  # shape (4,)
            
            # Convert bits to 4-bit integer (0-15 shade)
            powers: torch.Tensor = torch.pow(2.0, torch.arange(3, -1, -1, device=bits.device, dtype=torch.float32))  # [8, 4, 2, 1]
            shade_value: torch.Tensor = torch.sum(bits * powers)  # scalar in [0, 15]
            
            # Scale to byte range [0, 255] for internal representation
            byte_value: torch.Tensor = shade_value * 16  # 0-15 becomes 0-240
            
            # Feed through network
            _ = network.process_step(byte_value.unsqueeze(0))

        # Get results for this sample
        spike_raster: torch.Tensor = network.get_spike_raster()
        spike_counts: List[int] = network.spike_counts.copy()

        # Compute per-neuron spike rates (as integer counts normalized by network steps)
        network_steps: int = network.step_count
        spike_rate_counts: List[int] = spike_counts  # Keep raw counts for analysis
        spike_rates.append(spike_rate_counts)
        spike_rasters.append((spike_raster, label.item()))  # type: ignore[misc]

        # Summary per digit
        digit_label: int = label.item()  # type: ignore[misc]
        digit_summaries[digit_label]["spike_counts"].append(sum(spike_counts))
        digit_summaries[digit_label]["sequence_length"].append(network_steps)

        if (sample_idx + 1) % 100 == 0:
            print(f"  Processed {sample_idx + 1} samples")

    # Aggregate per-digit statistics
    summary_by_digit: Dict[int, Dict[str, Any]] = {}
    for digit, data in digit_summaries.items():
        if data["spike_counts"]:
            avg_total_spikes: int = sum(data["spike_counts"]) // len(data["spike_counts"])  # type: ignore[misc]
            seq_len_first: int = data["sequence_length"][0]
            avg_spike_rate: int = (avg_total_spikes * 255) // (seq_len_first * network.num_neurons)  # As byte percentage
            summary_by_digit[digit] = {
                "avg_total_spikes": avg_total_spikes,
                "avg_spike_rate_per_neuron": avg_spike_rate,
                "num_samples": len(data["spike_counts"]),
            }

    return spike_rates, spike_rasters, summary_by_digit


if __name__ == "__main__":
    # Test with Sequential MNIST
    print("Discrete Spiking Neural Network Sandbox")
    print("=" * 70)

    from seq_mnist import get_sequential_mnist_loaders  # type: ignore[import-not-found]

    # Setup
    config_: SpikeConfig = SpikeConfig(num_neurons=32, num_input_bits=8, threshold=32, decay_factor=95)
    num_neurons_: int = config_.num_neurons
    num_input_bits_: int = config_.num_input_bits
    
    # Simple connectivity: ring topology (each neuron connects to 2 neighbors)
    adjacency_: torch.Tensor = torch.eye(num_neurons_) * 0.1  # Self-recurrence
    for i in range(num_neurons_):
        adjacency_[i, (i + 1) % num_neurons_] = 0.2  # Forward connection
        adjacency_[i, (i - 1) % num_neurons_] = 0.2  # Backward connection

    # Input mapping: distribute single integer (byte) input across all neurons equally
    input_map_: torch.Tensor = torch.ones((num_neurons_, 1)) / num_neurons_

    # Load small subset of MNIST for testing
    train_loader, _ = get_sequential_mnist_loaders(batch_size=1, train_split=0.01)  # type: ignore[misc]  # 1% = ~600 samples

    print(f"Config: {num_neurons_} neurons, {num_input_bits_} input bits, threshold={config_.threshold}, decay={config_.decay_factor}%")
    print(f"Adjacency shape: {adjacency_.shape}")
    print(f"Input map shape: {input_map_.shape}")
    print(f"Dataset: Sequential MNIST, ~600 training samples (1% subset)")
    print("=" * 70)

    # Create network
    network_: SpikingNetwork = SpikingNetwork(
        neuron_class=SimpleThresholdNeuron,
        num_neurons=num_neurons_,
        adjacency=adjacency_,
        input_map=input_map_,
        config=config_,
    )

    # Run experiment
    spike_rates, spike_rasters, summary_by_digit = run_spiking_experiment(
        train_loader,
        network_,
    )

    print("\n" + "=" * 70)
    print("RESULTS BY DIGIT:")
    print("=" * 70)
    for digit in sorted(summary_by_digit.keys()):
        info: Dict[str, Any] = summary_by_digit[digit]
        print(
            f"Digit {digit}: "
            f"avg_spike_rate={info['avg_spike_rate_per_neuron']}/255, "
            f"total_spikes_per_seq={info['avg_total_spikes']}, "
            f"samples={info['num_samples']}"
        )

    print("\n" + "=" * 70)
    print("Overall statistics:")
    overall_spike_counts: List[int] = [sum([summary_by_digit[d].get("avg_total_spikes", 0) for d in range(10)])]  # type: ignore[misc]
    print(f"  Total samples processed: {len(spike_rates)}")
    print("=" * 70)
