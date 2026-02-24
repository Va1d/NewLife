# Biological Neuron Features Reference

**Comprehensive feature set in PlaceholderNeuron**

This document catalogs all known biological neuron modeling techniques implemented in `PlaceholderNeuron`.

---

## 🧠 Feature Categories

### 1. MEMBRANE DYNAMICS

**Leaky Integration** (DEFAULT: ENABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    leak_factor=0.95,  # 5% decay per timestep
    resting_potential=0,
)
```
- **What:** Membrane potential passively decays toward resting value
- **Why:** Prevents unlimited integration, models ion channel leak
- **Papers:** Lapicque (1907), Abbott (1999)
- **Typical values:** 0.9-0.99 (higher = less leak)

---

### 2. SPIKING MECHANISMS

**Fixed Threshold** (DEFAULT: 100)
```python
n = PlaceholderNeuron(num_inputs=10, threshold=100)
```
- **What:** Spike when V ≥ threshold
- **Classic:** Integrate-and-Fire (LIF) model

**Adaptive Threshold** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    enable_adaptation=True,
    adaptation_increment=10.0,  # How much threshold increases per spike
    adaptation_decay=0.9,        # Decay rate back to baseline
)
```
- **What:** Threshold increases after each spike, making subsequent spikes harder
- **Why:** Models spike-frequency adaptation seen in real neurons
- **Papers:** Benda & Herz (2003), Brette & Gerstner (2005)
- **Use case:** Prevents runaway spiking, creates realistic firing patterns

**Reset After Spike**
```python
n = PlaceholderNeuron(
    num_inputs=10,
    reset_potential=0,  # Where V jumps to after spike
)
```
- **Classic:** After spike, V → reset_potential
- **Variants:** 
  - `reset_potential = resting_potential` (simple reset)
  - `reset_potential < resting_potential` (hyperpolarizing reset)

---

### 3. REFRACTORY PERIODS

**Absolute Refractory** (DEFAULT: 3 timesteps)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    refractory_period=3,  # Cannot spike for 3 timesteps
)
```
- **What:** After spike, neuron cannot fire
- **Why:** Models sodium channel inactivation
- **Papers:** Hodgkin & Huxley (1952)

**Relative Refractory** (DEFAULT: 5 timesteps)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    relative_refractory_steps=5,
    relative_refractory_threshold_mult=1.5,  # 50% higher threshold
)
```
- **What:** After absolute refractory, threshold temporarily elevated
- **Why:** Models potassium channel recovery
- **More realistic** than absolute alone

---

### 4. SYNAPTIC PROCESSING

**Learnable Weights** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    use_learned_weights=True,
    init_weight_mean=1.0,
    init_weight_std=0.2,
)
```
- **What:** Each input has independent weight (strength)
- **Why:** Enables learning, models synaptic efficacy
- **Combined with STDP:** Weights adjust during learning

**Inhibitory/Excitatory Inputs**
```python
# Not explicitly separated in current implementation
# But can be modeled by:
# 1. Negative weights for inhibitory synapses
# 2. Separate input channels with different processing
```
- **Papers:** Dale's principle (1935), Eccles (1964)

---

### 5. LEARNING RULES

**STDP (Spike-Timing-Dependent Plasticity)** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    use_learned_weights=True,   # Required
    enable_stdp=True,
    stdp_lr=0.01,               # Learning rate
    stdp_tau_plus=20.0,         # Potentiation time constant (ms)
    stdp_tau_minus=20.0,        # Depression time constant (ms)
    stdp_a_plus=0.01,           # LTP amplitude
    stdp_a_minus=0.01,          # LTD amplitude
)
```
- **What:** Weights strengthen if pre→post spike timing <20ms, weaken if reversed
- **Why:** Hebbian learning: "neurons that fire together, wire together"
- **Papers:** Markram et al. (1997), Bi & Poo (1998)
- **Δw = A * exp(-Δt / τ)**
  - Δt = t_post - t_pre
  - Positive Δt → LTP (Long-Term Potentiation)
  - Negative Δt → LTD (Long-Term Depression)

**Homeostatic Plasticity** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    enable_homeostasis=True,
    target_rate=0.05,           # Target: 5% of timesteps spike
    homeostasis_lr=0.001,       # Adjustment rate
)
```
- **What:** Neuron adjusts threshold to maintain target firing rate
- **Why:** Prevents runaway excitation/silence in learning networks
- **Papers:** Turrigiano & Nelson (2004)
- **Mechanism:** If firing too much → increase threshold, if too little → decrease

---

### 6. ADVANCED DYNAMICS

**Stochastic Noise** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    enable_noise=True,
    noise_std=2.0,  # Gaussian noise added to V each timestep
)
```
- **What:** Random membrane potential fluctuations
- **Why:** Models channel noise, synaptic noise, background activity
- **Papers:** Faisal et al. (2008)
- **Effect:** Creates variability in spike timing, prevents deterministic behavior

**Burst Firing** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    enable_bursts=True,
    burst_threshold=150,  # Secondary high threshold
    burst_isi=2,          # Inter-spike interval in burst
)
```
- **What:** When V exceeds high threshold → rapid sequence of spikes
- **Why:** Seen in cortical pyramidal neurons, thalamic neurons
- **Papers:** Lisman (1997), Izhikevich (2000)
- **Mechanism:** T-type calcium channels, dendritic calcium spikes

**After-Hyperpolarization (AHP)** (DEFAULT: DISABLED)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    enable_ahp=True,
    ahp_amplitude=-20.0,  # Hyperpolarization strength
    ahp_duration=10,      # How many timesteps
)
```
- **What:** After spike, V temporarily drops below resting
- **Why:** Calcium-activated potassium channels
- **Papers:** Storm (1990)
- **Effect:** Natural spike-frequency adaptation, limits firing rate

---

## 📚 Additional Classic Models (Reference Only)

### IZHIKEVICH MODEL
```
dv/dt = 0.04v² + 5v + 140 - u + I
du/dt = a(bv - u)
If v ≥ 30: v ← c, u ← u + d
```
- **Parameters:** (a, b, c, d) control neuron type
- **Types:** Regular spiking, fast spiking, chattering, bursting, etc.
- **Paper:** Izhikevich (2003) - "Simple Model of Spiking Neurons"
- **Advantage:** Computationally efficient, 20+ neuron behaviors
- **To implement:** Create `IzhikevichNeuron(SpikingNeuronInterface)`

### HODGKIN-HUXLEY MODEL
```
C dV/dt = I - g_Na*m³h(V - E_Na) - g_K*n⁴(V - E_K) - g_L(V - E_L)
dm/dt = α_m(1-m) - β_m*m
dh/dt = α_h(1-h) - β_h*h
dn/dt = α_n(1-n) - β_n*n
```
- **What:** Conductance-based, models ion channels explicitly
- **Why:** Most biologically realistic for single neuron
- **Paper:** Hodgkin & Huxley (1952) - Nobel Prize
- **Disadvantage:** Computationally expensive (4 ODEs per neuron)
- **Use:** Gold standard for detailed single neuron studies

### ADAPTIVE EXPONENTIAL (AdEx)
```
C dV/dt = -g_L(V - E_L) + g_L*Δ_T*exp((V - V_T)/Δ_T) - w + I
τ_w dw/dt = a(V - E_L) - w
```
- **What:** Exponential spike, adaptation current
- **Why:** Good balance of realism and efficiency
- **Paper:** Brette & Gerstner (2005)

### FITZHUGH-NAGUMO
```
dv/dt = v - v³/3 - w + I
dw/dt = ε(v + a - bw)
```
- **What:** 2D reduction of Hodgkin-Huxley
- **Why:** Easier mathematical analysis (phase plane)
- **Paper:** FitzHugh (1961), Nagumo et al. (1962)

---

## 🔬 Experimental Configurations

### Configuration 1: Minimal (Baseline)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    threshold=100,
    leak_factor=0.95,
)
# Just leaky integrate-and-fire
```

### Configuration 2: Realistic Cortical Neuron
```python
n = PlaceholderNeuron(
    num_inputs=10,
    threshold=80,
    leak_factor=0.98,
    refractory_period=3,
    relative_refractory_steps=5,
    enable_adaptation=True,
    adaptation_increment=5.0,
    adaptation_decay=0.95,
    enable_noise=True,
    noise_std=1.5,
)
# Models regular spiking pyramidal neuron
```

### Configuration 3: Learning Neuron (STDP)
```python
n = PlaceholderNeuron(
    num_inputs=10,
    threshold=100,
    use_learned_weights=True,
    enable_stdp=True,
    stdp_lr=0.005,
    stdp_tau_plus=20.0,
    stdp_tau_minus=20.0,
    enable_homeostasis=True,
    target_rate=0.05,
)
# Learns input patterns via spike timing
```

### Configuration 4: Bursting Thalamic Neuron
```python
n = PlaceholderNeuron(
    num_inputs=10,
    threshold=80,
    enable_bursts=True,
    burst_threshold=120,
    enable_ahp=True,
    ahp_amplitude=-15.0,
    ahp_duration=8,
)
# Models burst firing behavior
```

### Configuration 5: Maximum Features
```python
n = PlaceholderNeuron(
    num_inputs=10,
    threshold=100,
    leak_factor=0.97,
    refractory_period=3,
    relative_refractory_steps=5,
    enable_adaptation=True,
    adaptation_increment=8.0,
    adaptation_decay=0.92,
    use_learned_weights=True,
    enable_stdp=True,
    stdp_lr=0.01,
    enable_homeostasis=True,
    target_rate=0.05,
    enable_noise=True,
    noise_std=2.0,
    enable_bursts=True,
    enable_ahp=True,
)
# All features enabled (experimental)
```

---

## 📊 Feature Selection Guide

| Feature | Computational Cost | Biological Realism | Learning Impact | Recommended For |
|---------|-------------------|-------------------|-----------------|-----------------|
| **Leak** | Low | High | Medium | All models |
| **Adaptation** | Low | High | Medium | Realistic firing patterns |
| **Refractory** | Very Low | High | Low | All models |
| **Learned Weights** | Low | High | High | Any learning task |
| **STDP** | Medium | Very High | Very High | Unsupervised learning |
| **Homeostasis** | Low | High | High | Prevents runaway learning |
| **Noise** | Low | Medium | Medium | Robustness testing |
| **Bursts** | Low | Medium | Low | Specific neuron types |
| **AHP** | Low | High | Low | Adaptation alternative |

---

## 🎯 Recommended Starting Point for Iris Experiment

```python
from iris_network import PlaceholderNeuron, load_network

# Start simple, add features incrementally
network = load_network(
    neuron_class=lambda num_inputs: PlaceholderNeuron(
        num_inputs=num_inputs,
        threshold=80,           # Tune based on input range
        leak_factor=0.95,       # 5% decay
        refractory_period=2,    # Prevents double-spiking
        enable_adaptation=True, # Natural rate limiting
        adaptation_increment=5.0,
        adaptation_decay=0.9,
    )
)
```

**Next steps:**
1. Run with minimal features → establish baseline
2. Add STDP → enable learning
3. Add homeostasis → stabilize learning
4. Add noise → test robustness
5. Compare metrics vs benchmarks in LEARNING_SIGNALS_GUIDE.md

---

## 📖 Key Papers by Feature

**Leaky Integrate-and-Fire:**
- Lapicque (1907) - Original LIF model
- Abbott (1999) - Modern LIF review

**Adaptation:**
- Benda & Herz (2003) - "A universal model for spike-frequency adaptation"
- Brette & Gerstner (2005) - "Adaptive exponential integrate-and-fire model"

**STDP:**
- Markram et al. (1997) - "Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSPs"
- Bi & Poo (1998) - "Synaptic modifications in cultured hippocampal neurons"
- Song et al. (2000) - "Competitive Hebbian learning through STDP"

**Homeostasis:**
- Turrigiano & Nelson (2004) - "Homeostatic plasticity in the developing nervous system"

**Noise:**
- Faisal et al. (2008) - "Noise in the nervous system"

**Bursting:**
- Lisman (1997) - "Bursts as a unit of neural information"
- Izhikevich (2000) - "Neural excitability, spiking and bursting"

**General Models:**
- Gerstner & Kistler (2002) - "Spiking Neuron Models" (textbook)
- Izhikevich (2007) - "Dynamical Systems in Neuroscience" (textbook)

---

**All features are now in `PlaceholderNeuron` - enable/disable via constructor parameters!**
