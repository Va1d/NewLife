"""Lazy RNN with Local Learning

Each neuron maintains internal patterns and learns cooperatively with neighbors
through energy minimization rather than backpropagation.

Key idea: Neurons are "lazy" — they minimize energy by:
1. Trying to match input with internal patterns
2. Cooperating with neighbors when neighbors handle the signal better
3. Learning locally via Hebbian-like rules modulated by energy budget
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn  # type: ignore[import-not-found]


@dataclass
class LazyRNNConfig:
    """Configuration for Lazy RNN layer"""
    hidden_size: int = 128
    input_size: int = 1
    num_neighbors: int = 4  # Each neuron connects to ~4 others
    energy_penalty: float = 0.01  # Cost per unit activity
    cooperation_strength: float = 0.05  # How much neighbor activity suppresses own
    learning_rate: float = 0.01
    hebbian_strength: float = 0.1
    device: str = "cpu"


class LazyRNNCell(nn.Module):
    """Single lazy RNN cell with local learning"""

    def __init__(self, config: LazyRNNConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_size = config.input_size

        # Per-neuron internal wave patterns (freq, amp, phase)
        self._freqs = nn.Parameter(
            torch.randn(config.hidden_size, device=config.device) * 0.01 + 0.05
        )
        self._amps = nn.Parameter(
            torch.ones(config.hidden_size, device=config.device) * 0.5
        )
        self._phases = nn.Parameter(
            torch.rand(config.hidden_size, device=config.device) * 2 * math.pi
        )

        # Input-to-hidden weights (learnable, local updates)
        self.W_ih = nn.Parameter(
            torch.randn(config.hidden_size, config.input_size, device=config.device) * 0.1
        )

        # Hidden-to-hidden weights (recurrent, for neighbor cooperation)
        self.W_hh = nn.Parameter(
            torch.randn(config.hidden_size, config.hidden_size, device=config.device) * 0.01
        )

        # Output bias
        self.bias = nn.Parameter(torch.zeros(config.hidden_size, device=config.device))

        # Hidden state and energy tracking
        self.register_buffer("_h_prev", torch.zeros(config.hidden_size, device=config.device))
        self.register_buffer("_energy_trace", torch.zeros(config.hidden_size, device=config.device))
        self.register_buffer("_t", torch.tensor(0, device=config.device))

    def forward(
        self, x: torch.Tensor, h_prev: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: Input (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size) or None

        Returns:
            h: New hidden state (batch_size, hidden_size)
            info: Dict with energy, cooperation, sparsity metrics
        """
        batch_size = x.shape[0]
        device = x.device

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size, device=device)

        # 1. Generate internal wave attempts (per neuron)
        t_step = float(self._t.item())
        internal_wave = self._amps.unsqueeze(0) * torch.sin(
            2 * math.pi * self._freqs.unsqueeze(0) * t_step + self._phases.unsqueeze(0)
        )  # (1, hidden_size) → broadcast to (batch, hidden_size)

        # 2. Input projection + recurrent input
        x_proj = torch.matmul(x, self.W_ih.T)  # (batch, hidden_size)
        h_rec = torch.matmul(h_prev, self.W_hh.T)  # (batch, hidden_size) - recurrent coupling

        # Combined input signal (what neuron needs to respond to)
        signal = x_proj + h_rec + self.bias  # (batch, hidden_size)

        # 3. Compute per-neuron energy cost
        # Each neuron must decide: activate to match signal, or cooperate (stay lazy)?
        activation_attempt = torch.sigmoid(signal)  # (batch, hidden_size)
        mismatch = torch.abs(signal - internal_wave)  # (batch, hidden_size) - how well internal wave matches

        # Energy from activation
        energy_from_activity = self.config.energy_penalty * torch.abs(activation_attempt)

        # Energy from mismatch (trying to explain input)
        energy_from_mismatch = 0.1 * mismatch

        # Cooperation signal: if neighbors are active, suppress own activity (save energy)
        neighbor_activity = torch.abs(h_rec)  # Recurrent input magnitude
        cooperation_suppression = self.config.cooperation_strength * neighbor_activity

        # Total energy for each neuron
        total_energy = energy_from_activity + energy_from_mismatch - cooperation_suppression
        total_energy = torch.relu(total_energy)  # Energy can't be negative

        # 4. Compute final activation (lazy neurons suppress if energy too high)
        # Soft gate: if energy is high, neuron stays sleepy
        laziness_gate = torch.exp(-total_energy)  # High energy → low gate (lazy)
        h_new = activation_attempt * laziness_gate  # (batch, hidden_size)

        # 5. Local Hebbian-like learning (optional, can disable for now)
        # Δw_ih ∝ post-synaptic (h_new) × pre-synaptic (x) × energy modulation
        with torch.no_grad():
            correlation = torch.matmul(h_new.T, x)  # (hidden_size, input_size)
            energy_factor = (1.0 - total_energy.mean(dim=0, keepdim=True).T)  # Boost learning when low energy
            self.W_ih.grad = -self.config.learning_rate * correlation * energy_factor

        # 6. Track metrics
        sparsity = (h_new < 0.1).float().mean()  # Fraction of neurons nearly inactive
        avg_energy = total_energy.mean()
        cooperation_rate = (cooperation_suppression > 0).float().mean()

        info = {
            "energy": avg_energy.item(),
            "sparsity": sparsity.item(),
            "cooperation_rate": cooperation_rate.item(),
            "activation_mean": h_new.mean().item(),
            "mismatch_mean": mismatch.mean().item(),
        }

        self._h_prev = h_new[0].detach()  # Store first sample for next step
        self._energy_trace = total_energy[0].detach()
        self._t += 1

        return h_new, info


class LazyRNN(nn.Module):
    """Lazy RNN for sequence classification"""

    def __init__(self, config: LazyRNNConfig, num_classes: int = 10) -> None:
        super().__init__()
        self.config = config
        self.rnn_cell = LazyRNNCell(config)

        # Output layer (can be local learning or standard)
        self.readout = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: Input sequences (batch_size, seq_len, input_size)

        Returns:
            logits: Class predictions (batch_size, num_classes)
            energy_history: List of energy dicts per timestep
        """
        batch_size, seq_len, input_size = x.shape
        device = x.device

        h = torch.zeros(batch_size, self.config.hidden_size, device=device)
        energy_history = []

        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            h, info = self.rnn_cell(x_t, h)
            energy_history.append(info)

        # Readout: take final hidden state, classify
        logits = self.readout(h)  # (batch_size, num_classes)

        return logits, energy_history


if __name__ == "__main__":
    # Quick test
    print("Lazy RNN Test")
    print("=" * 60)

    config = LazyRNNConfig(
        hidden_size=32,
        input_size=1,
        num_neighbors=4,
        energy_penalty=0.01,
    )
    model = LazyRNN(config, num_classes=10)

    # Dummy sequential input: (batch=4, seq_len=10, input_size=1)
    x = torch.randn(4, 10, 1)
    logits, energy_history = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"\nEnergy metrics (first 3 timesteps):")
    for t in range(min(3, len(energy_history))):
        info = energy_history[t]
        print(f"  t={t}: energy={info['energy']:.4f}, sparsity={info['sparsity']:.4f}, "
              f"cooperation={info['cooperation_rate']:.4f}")
    print(f"\nAverage sparsity over sequence: "
          f"{sum(e['sparsity'] for e in energy_history) / len(energy_history):.4f}")
    print("=" * 60)
