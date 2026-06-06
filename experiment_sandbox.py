from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import plotly.graph_objects as go  # type: ignore[import-not-found]
from plotly.subplots import make_subplots  # type: ignore[import-not-found]

StimulusGenerator = Callable[[int], torch.Tensor]


@dataclass
class Config:
    steps: int = 20000
    num_neurons: int = 1
    ext_dim: int = 8
    waves: int = 3
    device: str = "cuda:1"
    decay_steps_divisor: int = 4
    lr_w: float = 0.005
    lr_theta: float = 0.001
    freq_min: float = 0.001
    freq_max: float = 0.1
    error_ema_alpha: float = 0.03
    phase_lock_strength: float = 0.04
    phase_damping: float = 0.008
    amp_decay: float = 0.0008
    energy_osc_freq: float = 0.01
    energy_osc_amp: float = 0.2
    energy_osc_phase: float = 0.0
    competence_window: int = 50
    competence_threshold: float = 0.05
    competence_boost: float = 2.0
    hedonic_error_weight: float = 1.0  # pleasure from reducing error
    hedonic_stability_weight: float = 0.5  # pleasure from smooth learning
    hedonic_coherence_window: int = 10  # steps to measure output consistency


class Neuron(Protocol):
    def receive_signal(self, signal: torch.Tensor) -> None:
        ...

    def receive_communication(self, signal: torch.Tensor) -> None:
        ...

    def weight_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def activate(self, num_inputs: int) -> tuple[float,  float, float]:
        ...


class WaveMatchingNeuron:
    def __init__(
        self,
        num_inputs: int,
        device: torch.device,
        lr_w: float = 0.005,
        lr_theta: float = 0.001,
        freq_min: float = 0.001,
        freq_max: float = 0.1,
        decay_steps: int = 2000,
        error_ema_alpha: float = 0.03,
        phase_lock_strength: float = 0.04,
        phase_damping: float = 0.008,
        amp_decay: float = 0.0008,
        energy_osc_freq: float = 0.01,
        energy_osc_amp: float = 0.2,
        energy_osc_phase: float = 0.0,
        competence_window: int = 50,
        competence_threshold: float = 0.05,
        competence_boost: float = 2.0,
        hedonic_error_weight: float = 1.0,
        hedonic_stability_weight: float = 0.5,
        hedonic_coherence_window: int = 10,
    ) -> None:
        self.num_inputs = num_inputs
        self.device = device
        self.lr_w = lr_w
        self.lr_theta = lr_theta
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.decay_steps = max(1, decay_steps)
        self.error_ema_alpha = error_ema_alpha
        self.phase_lock_strength = phase_lock_strength
        self.phase_damping = phase_damping
        self.amp_decay = amp_decay
        self.energy_osc_freq = energy_osc_freq
        self.energy_osc_amp = energy_osc_amp
        self.energy_osc_phase = energy_osc_phase
        self.competence_window = competence_window
        self.competence_threshold = competence_threshold
        self.competence_boost = competence_boost
        self.hedonic_error_weight = hedonic_error_weight
        self.hedonic_stability_weight = hedonic_stability_weight
        self.hedonic_coherence_window = hedonic_coherence_window

        rng = torch.Generator(device=device).manual_seed(42)
        self._freqs: torch.Tensor = torch.rand((num_inputs,), generator=rng, device=device) * (freq_max - freq_min) + freq_min
        self._amps: torch.Tensor = torch.ones((num_inputs,), device=device) * (1.0 / max(1, num_inputs))
        self._phases: torch.Tensor = torch.rand((num_inputs,), generator=rng, device=device) * 2 * math.pi

        self._theta: torch.Tensor = torch.zeros((), device=device)
        self._error_ema: torch.Tensor = torch.zeros((num_inputs,), device=device)
        self._signal: torch.Tensor = torch.zeros((0,), device=device)
        self._comm: torch.Tensor = torch.zeros((0,), device=device)
        self._t = 0
        self._error_history: list[float] = []
        self._activity_history: list[float] = []  # for coherence measurement
        self._hedonic_state: float = 0.0  # current valence (pleasure/pain)

    def receive_signal(self, signal: torch.Tensor) -> None:
        self._signal = signal.to(self.device)

    def receive_communication(self, signal: torch.Tensor) -> None:
        self._comm = signal.to(self.device)

    def weight_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._freqs, self._phases

    def activate(self, num_inputs: int) -> tuple[float, float, float]:
        if num_inputs != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, got {num_inputs}.")

        signal = self._signal
        comm = self._comm
        if signal.numel() == 0:
            signal = torch.zeros((0,), device=self.device)
        if comm.numel() == 0:
            comm = torch.zeros((0,), device=self.device)
        total_input = torch.cat([signal, comm])
        if total_input.numel() != self.num_inputs:
            raise ValueError("Total input size does not match num_inputs.")

        t = float(self._t)
        internal_wave = self._amps * torch.sin(2 * math.pi * self._freqs * t + self._phases)
        mismatch = total_input - internal_wave

        self._error_ema = (1.0 - self.error_ema_alpha) * self._error_ema + self.error_ema_alpha * mismatch
        
        # Track error history for competence-driven learning boost
        current_error = float((self._error_ema ** 2).mean().item())
        self._error_history.append(current_error)
        if len(self._error_history) > self.competence_window:
            self._error_history.pop(0)
        
        # Compute hedonic signal: pain/pleasure based on learning state
        hedonic_value = 0.0
        
        # Pain/Pleasure from error trajectory
        if len(self._error_history) >= 2:
            error_delta = self._error_history[-2] - current_error  # positive = improvement = pleasure
            hedonic_value += self.hedonic_error_weight * error_delta
        
        # Pain/Pleasure from stability (smooth vs jerky updates)
        if len(self._error_history) >= 3:
            recent_volatility = abs(self._error_history[-1] - self._error_history[-2]) + abs(self._error_history[-2] - self._error_history[-3])
            stability_bonus = -recent_volatility  # negative volatility = positive pleasure
            hedonic_value += self.hedonic_stability_weight * stability_bonus
        
        # Track activity for coherence measurement
        net = mismatch.mean() - self._theta
        activity = float(torch.sigmoid(net).item())
        self._activity_history.append(activity)
        if len(self._activity_history) > self.hedonic_coherence_window:
            self._activity_history.pop(0)
        
        # Pain/Pleasure from coherence (consistency of behavior)
        if len(self._activity_history) >= 2:
            activity_variance = sum((a - sum(self._activity_history)/len(self._activity_history))**2 for a in self._activity_history) / len(self._activity_history)
            coherence_bonus = -activity_variance  # low variance = high coherence = pleasure
            hedonic_value += coherence_bonus * 0.3
        
        self._hedonic_state = 0.9 * self._hedonic_state + 0.1 * hedonic_value  # smooth hedonic state
        
        # Competence boost: if error has improved significantly, boost learning
        competence_boost_factor = 1.0
        if len(self._error_history) >= max(2, self.competence_window // 2):
            past_error = self._error_history[0]  # error from ~competence_window steps ago
            improvement = past_error - current_error
            if improvement > self.competence_threshold:
                # Error improved significantly: temporarily boost learning capacity
                competence_boost_factor = self.competence_boost
        
        lr_scale = 1.0 / (1.0 + (t / self.decay_steps))
        lr_budget = 1.0 + self.energy_osc_amp * math.sin(
            2 * math.pi * self.energy_osc_freq * t + self.energy_osc_phase
        )
        lr_effective = lr_scale * lr_budget * competence_boost_factor
        
        # Hedonic modulation: positive hedonic state boosts learning appetite
        hedonic_multiplier = 1.0 + max(0.0, self._hedonic_state) * 0.5
        lr_effective *= hedonic_multiplier

        d_amp = torch.sin(2 * math.pi * self._freqs * t + self._phases)
        self._amps += self.lr_w * lr_effective * self._error_ema * d_amp
        self._amps *= (1.0 - self.amp_decay)

        d_phase = self._amps * torch.cos(2 * math.pi * self._freqs * t + self._phases)
        self._phases += self.lr_w * lr_effective * self._error_ema * d_phase
        phase_error = -self._phases
        self._phases += lr_effective * self.phase_lock_strength * phase_error
        self._phases *= (1.0 - self.phase_damping)

        d_freq = self._amps * 2 * math.pi * t * torch.cos(2 * math.pi * self._freqs * t + self._phases)
        self._freqs += self.lr_w * 0.05 * lr_effective * self._error_ema * d_freq
        self._freqs.clamp_(self.freq_min, self.freq_max)
        self._amps.clamp_(0.0, 2.0)

        self._theta += self.lr_theta * (activity - 0.05)

        energy = float((self._error_ema ** 2).mean().item())
        self._t += 1
        return float(activity), float(energy), float(self._hedonic_state)  


def run_experiment(
    steps: int,
    external_generator: StimulusGenerator,
    neurons: list[Neuron],
    dendrites: torch.Tensor,
    axons: torch.Tensor,
    input_map: torch.Tensor,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[list[float]]]:
    if dendrites.ndim != 2 or axons.ndim != 2 or input_map.ndim != 2:
        raise ValueError("Connectivity matrices must be 2D tensors.")

    num_neurons = len(neurons)
    if dendrites.shape != (num_neurons, num_neurons):
        raise ValueError("Dendrites matrix must be [num_neurons, num_neurons].")
    if axons.shape != (num_neurons, num_neurons):
        raise ValueError("Axons matrix must be [num_neurons, num_neurons].")

    ext_dim = input_map.shape[1]
    if input_map.shape[0] != num_neurons:
        raise ValueError("Input map must be [num_neurons, external_dim].")

    device = input_map.device
    dendrites = dendrites.to(device)
    axons = axons.to(device)

    activities = torch.zeros((num_neurons,), device=device)
    spike_rate: list[list[float]] = [[] for _ in range(num_neurons)]
    energy_proxy: list[list[float]] = [[] for _ in range(num_neurons)]
    activity_energy: list[list[float]] = [[] for _ in range(num_neurons)]
    hedonic_trace: list[list[float]] = [[] for _ in range(num_neurons)]

    for t in range(steps):
        external_raw = external_generator(t).to(device)
        if external_raw.numel() != ext_dim:
            raise ValueError("External generator returned wrong input size.")

        external_per_neuron = input_map * external_raw
        outgoing = axons @ activities
        comm_matrix = dendrites * outgoing

        for idx, neuron in enumerate(neurons):
            neuron.receive_signal(external_per_neuron[idx])
            neuron.receive_communication(comm_matrix[idx])
            activity, energy, hedonic = neuron.activate(ext_dim + num_neurons)
            activities[idx] = activity
            spike_rate[idx].append(activity)
            energy_proxy[idx].append(energy)
            activity_energy[idx].append(activity ** 2)
            hedonic_trace[idx].append(hedonic)

    return spike_rate, energy_proxy, activity_energy, hedonic_trace


def simple_external_generator(waves: int, ext_dim: int, device: torch.device) -> StimulusGenerator:
    freqs = torch.linspace(0.005, 0.05, waves)
    phases = torch.linspace(0.0, 2 * math.pi, waves)

    def generator(t: int) -> torch.Tensor:
        base = torch.zeros((ext_dim,), device=device)
        for k in range(waves):
            base += torch.sin(2 * math.pi * freqs[k] * t + phases[k])
        return base / max(1, waves)

    return generator


if __name__ == "__main__":
    cfg = Config()
    device = torch.device(cfg.device)
    steps = cfg.steps
    num_neurons = cfg.num_neurons
    ext_dim = cfg.ext_dim

    dendrites = torch.eye(num_neurons, device=device)
    axons = torch.eye(num_neurons, device=device)
    input_map = torch.ones((num_neurons, ext_dim), device=device)

    neurons: list[Neuron] = [
        WaveMatchingNeuron(
            num_inputs=ext_dim + num_neurons,
            device=device,
            decay_steps=max(1, steps // max(1, cfg.decay_steps_divisor)),
            lr_w=cfg.lr_w,
            lr_theta=cfg.lr_theta,
            freq_min=cfg.freq_min,
            freq_max=cfg.freq_max,
            error_ema_alpha=cfg.error_ema_alpha,
            phase_lock_strength=cfg.phase_lock_strength,
            phase_damping=cfg.phase_damping,
            amp_decay=cfg.amp_decay,
            energy_osc_freq=cfg.energy_osc_freq,
            energy_osc_amp=cfg.energy_osc_amp,
            energy_osc_phase=cfg.energy_osc_phase,
            competence_window=cfg.competence_window,
            competence_threshold=cfg.competence_threshold,
            competence_boost=cfg.competence_boost,
            hedonic_error_weight=cfg.hedonic_error_weight,
            hedonic_stability_weight=cfg.hedonic_stability_weight,
            hedonic_coherence_window=cfg.hedonic_coherence_window,
        )
    ]

    generator = simple_external_generator(waves=cfg.waves, ext_dim=ext_dim, device=device)
    spike_rate, energy_proxy, activity_energy, hedonic_trace = run_experiment(steps, generator, neurons, dendrites, axons, input_map)

    avg_spike = sum(series[-1] for series in spike_rate) / max(1, num_neurons)
    avg_energy = sum(series[-1] for series in energy_proxy) / max(1, num_neurons)
    avg_hedonic = sum(series[-1] for series in hedonic_trace) / max(1, num_neurons)
    print(f"Final spike rate: {avg_spike:.4f}")
    print(f"Final energy proxy: {avg_energy:.6f}")
    print(f"Final hedonic state: {avg_hedonic:.6f}")

    energy_min = min(
        min(min(series) for series in energy_proxy),
        min(min(series) for series in activity_energy),
    )
    energy_max = max(
        max(max(series) for series in energy_proxy),
        max(max(series) for series in activity_energy),
    )
    energy_pad = max(1e-8, (energy_max - energy_min) * 0.05)
    energy_range = [energy_min - energy_pad, energy_max + energy_pad]

    hedonic_min = min(min(series) for series in hedonic_trace)
    hedonic_max = max(max(series) for series in hedonic_trace)
    hedonic_pad = max(1e-8, (hedonic_max - hedonic_min) * 0.1)
    hedonic_range = [hedonic_min - hedonic_pad, hedonic_max + hedonic_pad]

    fig = make_subplots(  # type: ignore[misc]
        rows=3, cols=1,
        subplot_titles=("Spike rate over time", "Energy proxy over time", "Hedonic state over time"),
        vertical_spacing=0.10
    )

    fig.add_trace(  # type: ignore[misc]
        go.Scatter(  # type: ignore[misc]
            y=spike_rate[0],
            mode='lines',
            name='Spike rate',
            line=dict(color='blue', width=2)  # type: ignore[misc]
        ),
        row=1, col=1
    )

    fig.add_trace(  # type: ignore[misc]
        go.Scatter(  # type: ignore[misc]
            y=energy_proxy[0],
            mode='lines',
            name='Energy proxy',
            line=dict(color='red', width=2)  # type: ignore[misc]
        ),
        row=2, col=1
    )

    fig.add_trace(  # type: ignore[misc]
        go.Scatter(  # type: ignore[misc]
            y=activity_energy[0],
            mode='lines',
            name='Activity energy',
            line=dict(color='orange', width=2)  # type: ignore[misc]
        ),
        row=2, col=1
    )

    fig.add_trace(  # type: ignore[misc]
        go.Scatter(  # type: ignore[misc]
            y=hedonic_trace[0],
            mode='lines',
            name='Hedonic state',
            line=dict(color='green', width=2)  # type: ignore[misc]
        ),
        row=3, col=1
    )

    frames = []
    for idx in range(num_neurons):
        frames.append( # type: ignore
            go.Frame(  # type: ignore[misc]
                name=f"neuron_{idx + 1}",
                data=[
                    go.Scatter(y=spike_rate[idx], mode='lines', line=dict(color='blue', width=2)),  # type: ignore[misc]
                    go.Scatter(y=energy_proxy[idx], mode='lines', line=dict(color='red', width=2)),  # type: ignore[misc]
                    go.Scatter(y=activity_energy[idx], mode='lines', line=dict(color='orange', width=2)),  # type: ignore[misc]
                    go.Scatter(y=hedonic_trace[idx], mode='lines', line=dict(color='green', width=2)),  # type: ignore[misc]
                ],
            )
        )
    fig.frames = frames

    fig.update_xaxes(title_text="Time step", row=3, col=1)  # type: ignore[misc]
    fig.update_yaxes(title_text="Mean firing rate", row=1, col=1)  # type: ignore[misc]
    fig.update_yaxes(title_text="Energy proxy (mean a^2)", row=2, col=1, range=energy_range)  # type: ignore[misc]
    fig.update_yaxes(title_text="Valence (pleasure/pain)", row=3, col=1, range=hedonic_range)  # type: ignore[misc]

    fig.update_layout(  # type: ignore[misc]
        autosize=True,
        showlegend=False,
        title_text="Experiment Sandbox",
        sliders=[{
            "currentvalue": {"prefix": "Neuron: "},
            "steps": [
                {
                    "method": "animate",
                    "label": str(idx + 1),
                    "args": [[f"neuron_{idx + 1}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
                }
                for idx in range(num_neurons)
            ],
        }],
    )

    fig.show()  # type: ignore[misc]
