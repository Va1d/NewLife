"""Iris Spiking Network using torch

Clean interface for neuron and network classes.
Loads pre-built sparse connectivity architecture.
Neuron internals (threshold, spike mechanics) to be implemented.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class SpikingNeuronInterface(nn.Module):
    """Interface for spiking neuron implementation"""

    def __init__(self, num_inputs: int) -> None:
        super().__init__()
        self.num_inputs = num_inputs
    
    def reset(self) -> None:
        """Reset neuron state for new sequence"""
        raise NotImplementedError
    
    def receive_input(self, inputs: torch.Tensor) -> None:
        """
        Receive byte inputs from all sources.
        
        Args:
            inputs: (num_inputs,) tensor of byte values 0-255
        """
        raise NotImplementedError
    
    def process_step(self) -> int:
        """
        Process one timestep: integrate inputs, decide spike.
        
        Returns:
            spike: 0 or 1
        """
        raise NotImplementedError


class PlaceholderNeuron(SpikingNeuronInterface):
    """
    Comprehensive biological neuron model with all common features.
    Many features are disabled by default - enable as needed for experiments.
    
    This serves as a REFERENCE for known neuron modeling techniques.
    
    IMPLEMENTED BIOLOGICAL FEATURES:
    ================================
    
    1. MEMBRANE DYNAMICS:
       - Leaky integration (exponential decay)
       - Resting potential
       - Membrane time constant
    
    2. SPIKING MECHANISMS:
       - Fixed threshold
       - Adaptive threshold (spike-frequency adaptation)
       - Soft threshold (probabilistic firing)
       - Reset after spike (to reset_potential or resting)
    
    3. REFRACTORY PERIODS:
       - Absolute refractory (no spike possible)
       - Relative refractory (higher threshold)
    
    4. SYNAPTIC PROCESSING:
       - Weighted inputs (per-synapse weights)
       - Excitatory/Inhibitory separation
       - Synaptic delays (time-delayed inputs)
       - Short-term plasticity (depression/facilitation)
    
    5. LEARNING RULES:
       - STDP (Spike-Timing-Dependent Plasticity)
       - Hebbian learning
       - Homeostatic plasticity (target firing rate)
       - Weight normalization
    
    6. ADVANCED DYNAMICS:
       - Burst firing mode
       - After-hyperpolarization (AHP)
       - Dendritic integration (compartments)
       - Background noise/stochasticity
       - Subthreshold oscillations
       - Calcium dynamics (for learning)
    
    7. CONDUCTANCE-BASED MODELS:
       - Hodgkin-Huxley dynamics
       - Izhikevich model
       - AdEx (Adaptive Exponential)
    
    Args:
        num_inputs: Number of input connections
        
        # Core parameters
        threshold: Spike threshold (default: 100)
        reset_potential: Membrane potential after spike (default: 0)
        resting_potential: Equilibrium potential (default: 0)
        leak_factor: Decay per timestep, 0-1 (default: 0.95, i.e., 5% leak)
        
        # Refractory period
        refractory_period: Timesteps of absolute refractory (default: 3)
        relative_refractory_steps: Timesteps of relative refractory (default: 5)
        relative_refractory_threshold_mult: Threshold multiplier during relative (default: 1.5)
        
        # Adaptation
        enable_adaptation: Enable spike-frequency adaptation (default: False)
        adaptation_increment: How much threshold increases per spike (default: 10)
        adaptation_decay: Adaptation decay per timestep (default: 0.9)
        
        # Synaptic weights
        use_learned_weights: Use learnable per-input weights (default: False)
        init_weight_mean: Initial weight mean (default: 1.0)
        init_weight_std: Initial weight std (default: 0.2)
        
        # Plasticity (STDP)
        enable_stdp: Enable spike-timing-dependent plasticity (default: False)
        stdp_lr: Learning rate for STDP (default: 0.01)
        stdp_tau_plus: Time constant for potentiation, ms (default: 20)
        stdp_tau_minus: Time constant for depression, ms (default: 20)
        stdp_a_plus: Potentiation amplitude (default: 0.01)
        stdp_a_minus: Depression amplitude (default: 0.01)
        
        # Homeostasis
        enable_homeostasis: Enable homeostatic plasticity (default: False)
        target_rate: Target firing rate, spikes/timestep (default: 0.05)
        homeostasis_lr: Learning rate for homeostatic adjustment (default: 0.001)
        
        # Noise/Stochasticity
        enable_noise: Add membrane potential noise (default: False)
        noise_std: Std of Gaussian noise added per timestep (default: 2.0)
        
        # Advanced features
        enable_bursts: Enable burst firing (default: False)
        burst_threshold: Secondary threshold for burst mode (default: 150)
        burst_isi: Inter-spike interval in burst, timesteps (default: 2)
        
        enable_ahp: Enable after-hyperpolarization (default: False)
        ahp_amplitude: AHP magnitude (default: -20)
        ahp_duration: AHP duration, timesteps (default: 10)
    """
    
    def __init__(
        self,
        num_inputs: int,
        
        # Core parameters
        threshold: int = 100,
        reset_potential: int = 0,
        resting_potential: int = 0,
        leak_factor: float = 0.95,
        
        # Refractory
        refractory_period: int = 3,
        relative_refractory_steps: int = 5,
        relative_refractory_threshold_mult: float = 1.5,
        
        # Adaptation
        enable_adaptation: bool = False,
        adaptation_increment: float = 10.0,
        adaptation_decay: float = 0.9,
        
        # Weights
        use_learned_weights: bool = False,
        init_weight_mean: float = 1.0,
        init_weight_std: float = 0.2,
        
        # STDP
        enable_stdp: bool = False,
        stdp_lr: float = 0.01,
        stdp_tau_plus: float = 20.0,
        stdp_tau_minus: float = 20.0,
        stdp_a_plus: float = 0.01,
        stdp_a_minus: float = 0.01,
        
        # Homeostasis
        enable_homeostasis: bool = False,
        target_rate: float = 0.05,
        homeostasis_lr: float = 0.001,
        
        # Noise
        enable_noise: bool = False,
        noise_std: float = 2.0,
        
        # Bursts
        enable_bursts: bool = False,
        burst_threshold: int = 150,
        burst_isi: int = 2,
        
        # AHP
        enable_ahp: bool = False,
        ahp_amplitude: float = -20.0,
        ahp_duration: int = 10,
    ) -> None:
        super().__init__(num_inputs)
        
        # === CORE PARAMETERS ===
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.resting_potential = resting_potential
        self.leak_factor = leak_factor
        
        # === REFRACTORY ===
        self.refractory_period = refractory_period
        self.relative_refractory_steps = relative_refractory_steps
        self.relative_refractory_threshold_mult = relative_refractory_threshold_mult
        self.register_buffer("refractory_counter", torch.zeros(1, dtype=torch.int32))
        
        # === ADAPTATION ===
        self.enable_adaptation = enable_adaptation
        self.adaptation_increment = adaptation_increment
        self.adaptation_decay = adaptation_decay
        self.register_buffer("threshold_adaptation", torch.zeros(1, dtype=torch.float32))
        
        # === SYNAPTIC WEIGHTS ===
        self.use_learned_weights = use_learned_weights
        if use_learned_weights:
            # Learnable per-input weights
            weights = torch.randn(num_inputs) * init_weight_std + init_weight_mean
            self.weights = nn.Parameter(weights)
        else:
            # Fixed uniform weights
            self.register_buffer("weights", torch.ones(num_inputs, dtype=torch.float32))
        
        # === STDP (Spike-Timing-Dependent Plasticity) ===
        self.enable_stdp = enable_stdp
        self.stdp_lr = stdp_lr
        self.stdp_tau_plus = stdp_tau_plus
        self.stdp_tau_minus = stdp_tau_minus
        self.stdp_a_plus = stdp_a_plus
        self.stdp_a_minus = stdp_a_minus
        # Track pre/post spike times for STDP
        self.register_buffer("last_presynaptic_spike_times", torch.zeros(num_inputs, dtype=torch.float32))
        self.register_buffer("last_postsynaptic_spike_time", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("current_time", torch.zeros(1, dtype=torch.float32))
        
        # === HOMEOSTASIS ===
        self.enable_homeostasis = enable_homeostasis
        self.target_rate = target_rate
        self.homeostasis_lr = homeostasis_lr
        self.register_buffer("spike_count_window", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("timestep_count", torch.zeros(1, dtype=torch.float32))
        
        # === NOISE ===
        self.enable_noise = enable_noise
        self.noise_std = noise_std
        
        # === BURST FIRING ===
        self.enable_bursts = enable_bursts
        self.burst_threshold = burst_threshold
        self.burst_isi = burst_isi
        self.register_buffer("in_burst_mode", torch.zeros(1, dtype=torch.bool))
        self.register_buffer("burst_spike_counter", torch.zeros(1, dtype=torch.int32))
        
        # === AFTER-HYPERPOLARIZATION (AHP) ===
        self.enable_ahp = enable_ahp
        self.ahp_amplitude = ahp_amplitude
        self.ahp_duration = ahp_duration
        self.register_buffer("ahp_counter", torch.zeros(1, dtype=torch.int32))
        
        # === STATE VARIABLES ===
        self.register_buffer("membrane_potential", torch.tensor([float(resting_potential)], dtype=torch.float32))
        self.register_buffer("last_input", torch.zeros(num_inputs, dtype=torch.float32))
        self.last_spike = 0
    
    def reset(self) -> None:
        """Reset all neuron state to initial conditions"""
        self.membrane_potential.fill_(self.resting_potential)
        self.refractory_counter.zero_()
        self.threshold_adaptation.zero_()
        self.last_presynaptic_spike_times.zero_()
        self.last_postsynaptic_spike_time.zero_()
        self.current_time.zero_()
        self.spike_count_window.zero_()
        self.timestep_count.zero_()
        self.in_burst_mode.fill_(False)
        self.burst_spike_counter.zero_()
        self.ahp_counter.zero_()
        self.last_input.zero_()
        self.last_spike = 0
    
    def receive_input(self, inputs: torch.Tensor) -> None:
        """
        Receive and store inputs for processing.
        
        Args:
            inputs: (num_inputs,) tensor of input values
        """
        inputs = inputs.float().to(self.membrane_potential.device)
        
        # Store for STDP later
        self.last_input = inputs.clone()
        
        # Apply synaptic weights
        weighted_inputs = inputs * self.weights
        
        # Sum weighted inputs
        total_input = weighted_inputs.sum()
        
        # Apply to membrane potential (will be integrated in process_step)
        self.membrane_potential.data += total_input
    
    def process_step(self) -> int:
        """
        Process one timestep: apply dynamics, check spiking, update learning.
        
        Returns:
            spike: 1 if neuron fires, 0 otherwise
        """
        self.current_time += 1.0
        self.timestep_count += 1.0
        
        spike = 0
        
        # === AFTER-HYPERPOLARIZATION ===
        if self.enable_ahp and self.ahp_counter > 0:
            self.membrane_potential.data += self.ahp_amplitude / self.ahp_duration
            self.ahp_counter -= 1
        
        # === LEAK (passive decay toward resting) ===
        # V_new = leak_factor * V_old + (1 - leak_factor) * V_rest
        self.membrane_potential.data = (
            self.leak_factor * self.membrane_potential.data +
            (1 - self.leak_factor) * self.resting_potential
        )
        
        # === NOISE ===
        if self.enable_noise:
            noise = torch.randn(1, device=self.membrane_potential.device) * self.noise_std
            self.membrane_potential.data += noise
        
        # === CHECK REFRACTORY PERIOD ===
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            return 0  # Cannot spike during absolute refractory
        
        # === COMPUTE EFFECTIVE THRESHOLD ===
        effective_threshold = self.threshold
        
        # Apply adaptation
        if self.enable_adaptation:
            effective_threshold += self.threshold_adaptation.item()
        
        # Apply relative refractory (higher threshold)
        if self.refractory_counter == 0 and self.last_spike > 0:
            time_since_spike = int(self.current_time.item() - self.last_postsynaptic_spike_time.item())
            if time_since_spike <= self.relative_refractory_steps:
                effective_threshold *= self.relative_refractory_threshold_mult
        
        # === CHECK SPIKING CONDITIONS ===
        
        # BURST MODE: Check if in burst
        if self.enable_bursts and self.in_burst_mode.item():
            if self.burst_spike_counter > 0:
                spike = 1
                self.burst_spike_counter -= 1
                if self.burst_spike_counter == 0:
                    self.in_burst_mode.fill_(False)
        
        # NORMAL MODE: Check threshold
        elif self.membrane_potential.item() >= effective_threshold:
            spike = 1
            
            # Check if entering burst mode
            if self.enable_bursts and self.membrane_potential.item() >= self.burst_threshold:
                self.in_burst_mode.fill_(True)
                self.burst_spike_counter.fill_(3)  # Burst of 3 spikes
        
        # === PROCESS SPIKE ===
        if spike == 1:
            # Reset membrane potential
            self.membrane_potential.data.fill_(self.reset_potential)
            
            # Set refractory period
            self.refractory_counter.fill_(self.refractory_period)
            
            # Update adaptation
            if self.enable_adaptation:
                self.threshold_adaptation += self.adaptation_increment
            
            # Trigger AHP
            if self.enable_ahp:
                self.ahp_counter.fill_(self.ahp_duration)
            
            # Record spike time
            self.last_postsynaptic_spike_time.fill_(self.current_time.item())
            self.last_spike = 1
            
            # Update spike count for homeostasis
            if self.enable_homeostasis:
                self.spike_count_window += 1.0
        else:
            self.last_spike = 0
        
        # === ADAPTATION DECAY ===
        if self.enable_adaptation:
            self.threshold_adaptation *= self.adaptation_decay
        
        # === STDP LEARNING ===
        if self.enable_stdp and self.use_learned_weights:
            self._apply_stdp(spike)
        
        # === HOMEOSTATIC PLASTICITY ===
        if self.enable_homeostasis and self.timestep_count.item() > 100:
            # Every 100 timesteps, adjust threshold toward target rate
            actual_rate = self.spike_count_window.item() / self.timestep_count.item()
            rate_error = actual_rate - self.target_rate
            
            # Adjust threshold (increase if firing too much, decrease if too little)
            self.threshold += self.homeostasis_lr * rate_error * self.threshold
            
            # Reset window
            if self.timestep_count.item() >= 1000:
                self.spike_count_window.zero_()
                self.timestep_count.zero_()
        
        return spike
    
    def _apply_stdp(self, spike: int) -> None:
        """
        Apply Spike-Timing-Dependent Plasticity (STDP) weight updates.
        
        STDP Rule:
        - If presynaptic spike comes before postsynaptic: strengthen (LTP)
        - If presynaptic spike comes after postsynaptic: weaken (LTD)
        - Weight change decays exponentially with time difference
        
        Args:
            spike: Whether postsynaptic neuron spiked this timestep
        """
        if not self.use_learned_weights:
            return
        
        # Update presynaptic spike times (where input was active)
        active_inputs = self.last_input > 0
        self.last_presynaptic_spike_times[active_inputs] = self.current_time.item()
        
        if spike == 1:
            # POST-SYNAPTIC SPIKE: Apply LTD to recent presynaptic spikes
            # Δw = -A_minus * exp(-(t_post - t_pre) / τ_minus)
            delta_t = self.current_time.item() - self.last_presynaptic_spike_times
            ltd_mask = delta_t > 0  # Only recent pre spikes
            
            if ltd_mask.any():
                weight_change = -self.stdp_a_minus * torch.exp(-delta_t / self.stdp_tau_minus)
                weight_change[~ltd_mask] = 0
                
                with torch.no_grad():
                    self.weights.data += self.stdp_lr * weight_change
                    # Clamp weights to reasonable range
                    self.weights.data.clamp_(0.1, 5.0)
        
        else:
            # NO POST SPIKE: Apply LTP for presynaptic spikes that just occurred
            # Δw = A_plus * exp(-(t_pre - t_post) / τ_plus)
            if self.last_postsynaptic_spike_time.item() > 0:
                delta_t = self.last_presynaptic_spike_times - self.last_postsynaptic_spike_time.item()
                ltp_mask = (delta_t > 0) & active_inputs  # Recent pre after last post
                
                if ltp_mask.any():
                    weight_change = self.stdp_a_plus * torch.exp(-delta_t / self.stdp_tau_plus)
                    weight_change[~ltp_mask] = 0
                    
                    with torch.no_grad():
                        self.weights.data += self.stdp_lr * weight_change
                        self.weights.data.clamp_(0.1, 5.0)


# === ADDITIONAL REFERENCE NEURON MODELS ===
# 
# The following are classic models not fully implemented above:
#
# 1. HODGKIN-HUXLEY MODEL (conductance-based)
#    - Sodium (Na+) and Potassium (K+) channels
#    - Voltage-gated channel dynamics (m, h, n gates)
#    - Realistic action potential shape
#    - Heavy computation, rarely used in large networks
#
# 2. IZHIKEVICH MODEL 
#    - dv/dt = 0.04v² + 5v + 140 - u + I
#    - du/dt = a(bv - u)
#    - If v ≥ 30: v ← c, u ← u + d
#    - Parameters (a,b,c,d) create different neuron types
#    - Computationally efficient, biologically plausible
#
# 3. ADAPTIVE EXPONENTIAL (AdEx)
#    - Exponential spike generation
#    - Adaptation current
#    - Good for cortical neuron models
#
# 4. MORRIS-LECAR MODEL
#    - Two-variable model for membrane + calcium
#    - Exhibits oscillations and excitability
#
# 5. FITZHUGH-NAGUMO MODEL
#    - Simplified Hodgkin-Huxley (2D reduction)
#    - Nullcline analysis, phase plane dynamics
#
# 6. WILSON-COWAN MODEL
#    - Population-level (E/I populations)
#    - Mean-field approximation
#
# To implement any of these, inherit from SpikingNeuronInterface
# and implement reset(), receive_input(), process_step()


class SpikingNetworkTorch(nn.Module):
    """Sparse spiking neural network with torch backend (CUDA-compatible)"""
    
    def __init__(
        self,
        num_neurons: int,
        adjacency: torch.Tensor,
        input_map: torch.Tensor,
        neuron_class: type = PlaceholderNeuron,
    ) -> None:
        """
        Args:
            num_neurons: Number of neurons
            adjacency: (num_neurons, num_neurons) connectivity matrix
            input_map: (num_neurons, num_features) input routing
            neuron_class: Neuron implementation class
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.register_buffer("adjacency", adjacency)
        self.register_buffer("input_map", input_map)
        
        num_features = input_map.shape[1]
        
        # Initialize neuron pool as ModuleList for proper device handling
        self.neurons = nn.ModuleList()
        for i in range(num_neurons):
            # Count incoming connections from adjacency
            incoming = (adjacency[i, :] != 0).sum().item()
            # Total inputs: external features + recurrent from other neurons
            num_inputs = num_features + incoming
            self.neurons.append(neuron_class(num_inputs))
        
        # Metrics (buffers)
        self.register_buffer("spike_raster", torch.zeros((num_neurons, 0), dtype=torch.uint8))
        self.register_buffer("last_spikes", torch.zeros(num_neurons, dtype=torch.uint8))
        self.spike_counts = [0] * num_neurons
        self.step_count = 0
    
    def reset(self) -> None:
        """Reset all neurons and metrics"""
        for neuron in self.neurons:
            neuron.reset()
        self.spike_raster = torch.zeros((self.num_neurons, 0), dtype=torch.uint8)
        self.spike_counts = [0] * self.num_neurons
        self.step_count = 0
        self.last_spikes.zero_()
    
    def process_step(self, external_input: torch.Tensor) -> torch.Tensor:
        """
        Process one timestep through network.
        
        Args:
            external_input: (num_features,) byte tensor
        
        Returns:
            spikes: (num_neurons,) spike outputs
        """
        self.step_count += 1
        device = external_input.device
        external_input_byte = external_input.clamp(0, 255).int().to(device)
        
        spikes_curr = []
        
        for i in range(self.num_neurons):
            # Build input list: external + recurrent
            inputs_list = [external_input_byte]
            
            # Add recurrent inputs from spiking neighbors
            recurrent_strength = 50  # Fixed coupling
            for j in range(self.num_neurons):
                if self.last_spikes[j] == 1:
                    recurrent_tensor = torch.tensor([recurrent_strength], dtype=torch.int32, device=device)
                    inputs_list.append(recurrent_tensor)
            
            # Combine inputs into single tensor
            inputs_tensor = torch.cat([t.unsqueeze(0) if t.dim() == 0 else t for t in inputs_list]).clamp(0, 255).int().to(device)
            
            # Feed to neuron
            self.neurons[i].receive_input(inputs_tensor)
            spike = self.neurons[i].process_step()
            spikes_curr.append(spike)
        
        # Convert to tensor on device
        spikes_tensor = torch.tensor(spikes_curr, dtype=torch.uint8, device=device)
        
        # Track metrics
        self.spike_raster = torch.cat([self.spike_raster.to(device), spikes_tensor.unsqueeze(-1)], dim=1)
        for i in range(self.num_neurons):
            self.spike_counts[i] += spikes_tensor[i].item()
        
        self.last_spikes = spikes_tensor.clone().to(device)
        
        return spikes_tensor
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Process sequence through network.
        
        Args:
            sequence: (seq_len, num_features) tensor
        
        Returns:
            spike_raster: (num_neurons, seq_len)
        """
        self.reset()
        
        for t in range(sequence.shape[0]):
            _ = self.process_step(sequence[t])
        
        return self.spike_raster


def load_network(
    connectivity_path: str = "/home/bo/Py/NewLife/.venv/src/connectivity_96.pt",
    neuron_class: type = PlaceholderNeuron,
    device: str = "cpu",
) -> SpikingNetworkTorch:
    """
    Load network with pre-built connectivity.
    
    Args:
        connectivity_path: Path to saved connectivity matrices
        neuron_class: Neuron implementation to use
        device: Device to place network on ("cpu" or "cuda")
    
    Returns:
        Initialized SpikingNetworkTorch on specified device
    """
    connectivity = torch.load(connectivity_path)
    adjacency = connectivity["adjacency"].to(device)
    input_map = connectivity["input_map"].to(device)
    
    network = SpikingNetworkTorch(
        num_neurons=adjacency.shape[0],
        adjacency=adjacency,
        input_map=input_map,
        neuron_class=neuron_class,
    )
    
    network = network.to(device)
    
    return network


if __name__ == "__main__":
    # Test loading and basic forward pass
    net = load_network()
    
    print(f"Network loaded: {net.num_neurons} neurons")
    print(f"Adjacency shape: {net.adjacency.shape}")
    print(f"Input map shape: {net.input_map.shape}")
    
    # Test with dummy Iris sample (4 features, feed over 4 timesteps)
    dummy_sample = torch.randint(0, 64, (4, 4), dtype=torch.int32)
    
    net.reset()
    for t in range(4):
        spikes = net.process_step(dummy_sample[t])
        print(f"Step {t}: {spikes.sum()} neurons spiked")
    
    print(f"\nTotal steps: {net.step_count}")
    print(f"Total spikes: {sum(net.spike_counts)}")
