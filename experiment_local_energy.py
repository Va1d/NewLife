import math
import random
from dataclasses import dataclass
from itertools import product
import json
from typing import Any

import torch
import plotly.graph_objects as go  # type: ignore[import-not-found]
from plotly.subplots import make_subplots  # type: ignore[import-not-found]


@dataclass
class Config:
    neurons: int = 64
    steps: int = 10000
    lr_w: float = 0.005
    lr_theta: float = 0.001
    noise: float = 0.15
    waves: int = 3
    base_amp: float = 1.0
    osc_amp: float = 0.2
    osc_freq: float = 0.02
    proximity_sigma: float = 6.0
    sparsity: float = 0.15
    seed: int = 42
    use_oscillator: bool = True
    device: str = "cpu"
    sweep: bool = False
    sweep_noise: str = ""
    sweep_lr_w: str = ""
    sweep_sparsity: str = ""
    sweep_osc_amp: str = ""
    sweep_out: str = ""


def make_sparse_weights(n: int, sigma: float, sparsity: float, seed: int, device: str | torch.device) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    w = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Proximity probability in a ring
            dist = min((i - j) % n, (j - i) % n)
            p = math.exp(-(dist ** 2) / (2 * sigma ** 2))
            if torch.rand((), generator=rng).item() < p * sparsity:
                w[i, j] = torch.randn((), generator=rng).item() * 0.1
    return w.to(device)


def make_waves(n: int, waves: int, seed: int, device: str | torch.device) -> tuple[list[float], list[float], torch.Tensor]:
    rng = random.Random(seed)
    freqs = [rng.uniform(0.005, 0.05) for _ in range(waves)]
    phases = [rng.uniform(0, 2 * math.pi) for _ in range(waves)]
    per_neuron_phase = torch.tensor([rng.uniform(0, 2 * math.pi) for _ in range(n)], device=device)
    return freqs, phases, per_neuron_phase


def input_signal(t: int, n: int, waves: int, freqs: list[float], phases: list[float], per_neuron_phase: torch.Tensor, base_amp: float, noise: float, rng: torch.Generator, device: str | torch.device) -> torch.Tensor:
    sig = torch.zeros(n, device=device)
    for k in range(waves):
        sig += base_amp * torch.sin(2 * math.pi * freqs[k] * t + phases[k] + per_neuron_phase)
    sig /= max(1, waves)
    sig += noise * torch.randn(n, generator=rng, device=device)
    return sig


def run_experiment(cfg: Config) -> tuple[list[float], list[float]]:
    torch.manual_seed(cfg.seed)  # type: ignore[call-arg]
    device = torch.device(cfg.device)
    rng = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    # Each neuron has trainable wave parameters for creating internal patterns
    # Shape: [neurons, waves] for each parameter type
    neuron_freqs = torch.rand((cfg.neurons, cfg.waves), generator=rng, device=device) * 0.045 + 0.005
    neuron_amps = torch.ones((cfg.neurons, cfg.waves), device=device) * cfg.base_amp / cfg.waves
    neuron_phases = torch.rand((cfg.neurons, cfg.waves), generator=rng, device=device) * 2 * math.pi
    
    theta = torch.zeros(cfg.neurons, device=device)
    activity = torch.zeros(cfg.neurons, device=device)

    # Input signal parameters (external patterns neuron try to match/neutralize)
    freqs, phases, per_neuron_phase = make_waves(cfg.neurons, cfg.waves, cfg.seed, device)

    spike_rate: list[float] = []
    energy_proxy: list[float] = []

    # Smooth, laggish learning controls
    decay_steps = max(1, cfg.steps // 4)
    phase_lock_strength = 0.04
    phase_damping = 0.008
    error_ema = torch.zeros(cfg.neurons, device=device)
    error_ema_alpha = 0.03
    amp_decay = 0.0008

    for t in range(cfg.steps):
        # External input signal
        inp = input_signal(
            t,
            cfg.neurons,
            cfg.waves,
            freqs,
            phases,
            per_neuron_phase,
            cfg.base_amp,
            cfg.noise,
            rng,
            device,
        )

        # Each neuron generates its internal composite wave pattern
        internal_wave = torch.zeros(cfg.neurons, device=device)
        for k in range(cfg.waves):
            internal_wave += neuron_amps[:, k] * torch.sin(2 * math.pi * neuron_freqs[:, k] * t + neuron_phases[:, k])
        
        # Net input = external - internal (neuron tries to cancel/neutralize)
        mismatch = inp - internal_wave
        net = mismatch - theta
        activity = torch.sigmoid(net)

        # Update wave parameters to better match/neutralize input
        # Error signal: difference between input and internal pattern
        error = inp - internal_wave
        error_ema = (1.0 - error_ema_alpha) * error_ema + error_ema_alpha * error
        lr_scale = 1.0 / (1.0 + (t / decay_steps))
        
        for k in range(cfg.waves):
            # Gradient descent on amplitude
            d_amp = torch.sin(2 * math.pi * neuron_freqs[:, k] * t + neuron_phases[:, k])
            neuron_amps[:, k] += cfg.lr_w * lr_scale * error_ema * d_amp
            neuron_amps[:, k] *= (1.0 - amp_decay)
            
            # Gradient descent on phase
            d_phase = neuron_amps[:, k] * torch.cos(2 * math.pi * neuron_freqs[:, k] * t + neuron_phases[:, k])
            neuron_phases[:, k] += cfg.lr_w * lr_scale * error_ema * d_phase
            
            # Gradient descent on frequency (smaller learning rate)
            d_freq = neuron_amps[:, k] * 2 * math.pi * t * torch.cos(2 * math.pi * neuron_freqs[:, k] * t + neuron_phases[:, k])
            neuron_freqs[:, k] += cfg.lr_w * 0.05 * lr_scale * error_ema * d_freq

            # Phase locking toward external composite phase (smooth, laggy correction)
            target_phase = phases[k] + per_neuron_phase
            phase_error = torch.atan2(torch.sin(target_phase - neuron_phases[:, k]), torch.cos(target_phase - neuron_phases[:, k]))
            neuron_phases[:, k] += lr_scale * phase_lock_strength * phase_error
            neuron_phases[:, k] *= (1.0 - phase_damping)
        
        # Clamp frequency to reasonable range
        neuron_freqs.clamp_(0.001, 0.1)
        neuron_amps.clamp_(0.0, 2.0)  # Keep amplitudes positive and bounded
        
        # Threshold adaptation
        theta += cfg.lr_theta * (activity - 0.05)

        # Track stats
        spike_rate.append(activity.mean().item())
        energy_proxy.append((error_ema ** 2).mean().item())  # Energy is smoothed mismatch

    return spike_rate, energy_proxy


def parse_list(value: str) -> list[float]:
    return [float(v) for v in value.split(",") if v.strip() != ""]


def run_sweep(cfg: Config) -> None:
    sweep_params = {
        "noise": parse_list(cfg.sweep_noise),
        "lr_w": parse_list(cfg.sweep_lr_w),
        "sparsity": parse_list(cfg.sweep_sparsity),
        "osc_amp": parse_list(cfg.sweep_osc_amp),
    }

    keys = [k for k, v in sweep_params.items() if v]
    values = [sweep_params[k] for k in keys]
    if not keys:
        print("No sweep parameters provided.")
        return

    results: list[dict[str, Any]] = []
    for combo in product(*values):
        combo_cfg = Config(**cfg.__dict__)
        k: str
        v: float
        for k, v in zip(keys, combo):
            setattr(combo_cfg, k, v)

        spike_rate, energy_proxy = run_experiment(combo_cfg)
        results.append({
            **{k: getattr(combo_cfg, k) for k in keys},
            "final_spike": spike_rate[-1],
            "final_energy": energy_proxy[-1],
        })

        print("Sweep result:", results[-1])

    if cfg.sweep_out:
        import csv
        with open(cfg.sweep_out, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved sweep results to {cfg.sweep_out}")


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return Config(**data)


def main() -> None:
    # Set CONFIG_PATH to a JSON file if you want to override defaults.
    CONFIG_PATH = ""

    if CONFIG_PATH:
        cfg = load_config(CONFIG_PATH)
    else:
        cfg = Config()

    if cfg.sweep:
        run_sweep(cfg)
        return

    spike_rate, energy_proxy = run_experiment(cfg)

    print("Final spike rate:", spike_rate[-1])
    print("Final energy proxy:", energy_proxy[-1])

    energy_min = min(energy_proxy)
    energy_max = max(energy_proxy)
    energy_pad = max(1e-8, (energy_max - energy_min) * 0.05)
    energy_range = [energy_min - energy_pad, energy_max + energy_pad]

    # Create subplot figure with plotly
    fig = make_subplots(  # type: ignore[misc]
        rows=2, cols=1,
        subplot_titles=("Spike rate over time", "Energy proxy over time"),
        vertical_spacing=0.12
    )
    
    # Add spike rate trace
    fig.add_trace(  # type: ignore[misc]
        go.Scatter(  # type: ignore[misc]
            y=spike_rate,
            mode='lines',
            name='Spike rate',
            line=dict(color='blue', width=2)  # type: ignore[misc]
        ),
        row=1, col=1
    )
    
    # Add energy proxy trace
    fig.add_trace(  # type: ignore[misc]
        go.Scatter(  # type: ignore[misc]
            y=energy_proxy,
            mode='lines',
            name='Energy proxy',
            line=dict(color='red', width=2)  # type: ignore[misc]
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time step", row=2, col=1)  # type: ignore[misc]
    fig.update_yaxes(title_text="Mean firing rate", row=1, col=1)  # type: ignore[misc]
    fig.update_yaxes(title_text="Energy proxy (mean a²)", row=2, col=1, range=energy_range)  # type: ignore[misc]
    
    fig.update_layout(  # type: ignore[misc]
        height=600,
        width=1000,
        showlegend=False,
        title_text="Local Energy Minimization Experiment"
    )
    
    fig.show()  # type: ignore[misc]


if __name__ == "__main__":
    main()
