# Early Learning Signals Guide for Iris Spiking Network

**Last Updated:** February 22, 2026

This guide explains what metrics to watch when implementing different `SpikingNeuronInterface` implementations to detect learning as early as possible.

---

## ✅ Architecture Ready for Experimentation

- **96 neurons** with sparse connectivity (24 connections/neuron, 75% sparse)
- **Iris dataset**: 150 samples, 4 features, 3 balanced classes, 6-bit Gray code quantization
- **Network capacity**: 192 patterns (96 neurons × 2 patterns/neuron) vs 150 samples = **128% coverage**
- **7 clustering metrics** + 2 early learning indicators
- **Full logging**: TensorBoard + Plotly dashboards + console output

### To Implement Your Neuron:

1. Create new class inheriting from `SpikingNeuronInterface` in [iris_network.py](/.venv/src/iris_network.py)
2. Implement: `reset()`, `receive_input(inputs: torch.Tensor)`, `process_step() -> int`
3. Pass your class to `load_network(neuron_class=YourNeuron)` in [iris_experiment.py](/.venv/src/iris_experiment.py)
4. Run and monitor metrics below

---

## 🎯 Earliest Learning Signals (Ordered by Speed)

### 1. **Top Neuron Overlap** ⚡ (Appears in ~10-50 samples)

**What it measures:** Jaccard similarity of top-5 responsive neurons across classes

```
Top Neuron Overlap: 0.750 (Emerging)
└─ Lower is better: <0.3 = strong specialization, >0.7 = no specialization
```

**Interpretation:**
- **1.000** = All classes use identical neurons (no learning yet) ← PlaceholderNeuron baseline
- **0.7-0.9** = High overlap (minimal learning)
- **0.3-0.7** = Emerging specialization (neurons starting to differentiate)
- **< 0.3** = Strong class-specific neuron preferences ✓

**Why it's earliest:** Neurons only need to fire differently for different classes, not form global clusters yet.

**Watch for:** Drop from 1.0 → 0.8 in first 50 samples, then → 0.5 within 150 samples

**TensorBoard:** `learning/top_neuron_overlap`

---

### 2. **Mean Discriminability** ⚡ (Appears in ~20-100 samples)

**What it measures:** Average variance of each neuron's spike counts across the 3 classes

```
Mean Discriminability: 2.450 (Early learning)
└─ Higher is better: >10 = excellent, >5 = good, >1 = early learning
```

**Interpretation:**
- **0.000** = All neurons fire identically for all classes (no learning) ← PlaceholderNeuron baseline
- **0.5-1.0** = Barely detectable differentiation
- **1.0-5.0** = Early learning (neurons developing preferences)
- **5.0-10.0** = Good learning (clear class-specific responses)
- **> 10.0** = Excellent discrimination ✓

**Why it's early:** Measures raw variance without requiring structured clusters.

**Watch for:** Rise from 0.0 → 1.0+ in first epoch

**TensorBoard:** `learning/mean_discriminability`

---

### 3. **Silhouette Score** ⚡⚡ (Appears in ~100-150 samples)

**What it measures:** How well spike patterns fit within their class vs. other classes

```
Silhouette Score: 0.180 (Weak clustering)
└─ Higher is better: >0.5 = excellent, >0.3 = good, >0.1 = emerging
```

**Interpretation:**
- **-1.0 to 0.0** = Wrong clustering or random
- **0.0 to 0.1** = No meaningful clustering yet ← PlaceholderNeuron baseline
- **0.1 to 0.3** = Emerging clusters (early learning signal!) 
- **0.3 to 0.5** = Good clustering
- **> 0.5** = Excellent separation ✓

**Why it's early:** Measures compactness within classes without requiring full separation.

**Watch for:** Rise from ~0.0 → 0.15+ indicates genuine learning

**TensorBoard:** `clustering/silhouette_score`

---

### 4. **Neuron Specialization** ⚡⚡ (Appears in ~150+ samples)

**What it measures:** Percentage of neurons with class-specific firing preferences (computed from variance)

```
Neuron Specialization: 0.220 (Early learning)
└─ 0-1 scale, higher is better
```

**Interpretation:**
- **0.0** = No neurons prefer any class ← PlaceholderNeuron baseline
- **0.1-0.3** = About 10-30% of neurons showing preferences
- **0.5+** = Majority of neurons specialized ✓
- **0.8+** = Strong learned structure

**Why it's mid-early:** Requires stable firing patterns across multiple samples per class.

**Watch for:** Rise above 0.15 in first epoch

**TensorBoard:** `clustering/neuron_specialization`

---

### 5. **Davies-Bouldin Index** ⚡⚡⚡ (Appears in ~200+ samples)

**What it measures:** Ratio of within-cluster to between-cluster distances

```
Davies-Bouldin Index: 1.430 (Improving)
└─ Lower is better: <1.0 = good, <0.5 = excellent
```

**Interpretation:**
- **> 2.0** = Poor separation
- **1.0-2.0** = Emerging clusters
- **0.5-1.0** = Good clusters
- **< 0.5** = Excellent separation ✓
- **0.0** = Perfect separation (rare) OR no spikes yet ← PlaceholderNeuron baseline

**Why it's slower:** Requires established cluster boundaries.

**Watch for:** Drop from high values → approaching 1.0

**TensorBoard:** `clustering/davies_bouldin_index`

---

## ⚠️ Metrics That Are NOT Early Signals

### Spike Rate Reduction
**Problem:** Could indicate dying neurons rather than learning. Many network bugs manifest as reduced spiking.

**Better alternative:** Watch discriminability and overlap instead.

### Calinski-Harabasz Index
**Problem:** Between/within variance ratio requires well-formed clusters. Late signal.

### Normalized Mutual Information (NMI)
**Problem:** Requires k-means clustering to converge, which needs clear structure.

---

## 📊 Recommended Monitoring Strategy

### Console Output (Real-time):
```
🎯 EARLY LEARNING SIGNALS:
  Top Neuron Overlap: 0.650 (Emerging)
  Mean Discriminability: 1.850 (Early learning)  ← WATCH THIS FIRST
  Silhouette Score: 0.120 (Weak clustering)     ← WATCH THIS SECOND
```

### TensorBoard (Across Runs):
```bash
tensorboard --logdir runs/
```

**Most important plots:**
1. `learning/mean_discriminability` - should rise from 0 → 1+ quickly
2. `learning/top_neuron_overlap` - should drop from 1.0 → 0.7-0.5
3. `clustering/silhouette_score` - should rise from 0 → 0.15+
4. `spikes/[class]_avg_total` - verify neurons are spiking (not dying)

### Plotly Dashboards (Post-run Analysis):
- `plots/activation_heatmap.html` - visualize which neurons respond to which class
- `plots/top_neurons_by_class.html` - verify different neurons for different classes

---

## 🚀 Quick Start Experimentation Workflow

1. **Implement your neuron** (e.g., `ThresholdNeuron`, `LeakyIntegrateFireNeuron`):
   ```python
   class MyNeuron(SpikingNeuronInterface):
       def __init__(self, num_inputs: int):
           super().__init__(num_inputs)
           # Your state variables here
       
       def reset(self): ...
       def receive_input(self, inputs: torch.Tensor): ...
       def process_step(self) -> int: ...
   ```

2. **Run experiment**:
   ```python
   # In iris_experiment.py main block:
   from iris_network import MyNeuron
   
   network = load_network(
       connectivity_path=config.connectivity_path,
       neuron_class=MyNeuron,  # <-- Replace PlaceholderNeuron
       device=config.device,
   )
   ```

3. **Watch for these milestones**:
   - **First 50 samples:** Mean discriminability > 0.5 (neurons differentiating)
   - **First 100 samples:** Top neuron overlap < 0.8 (specialization starting)
   - **First 150 samples:** Silhouette score > 0.1 (clusters forming)
   - **After full epoch (target):** 
     - Mean discriminability > 5.0 (strong neuron preferences)
     - Top neuron overlap < 0.4 (class-specific neurons)
     - Silhouette score > 0.5 (competitive with K-Means)
     - NMI > 0.6 (meaningful class structure)

4. **Compare implementations:** Run multiple neuron types, compare TensorBoard curves

---

## 🎓 Iris Dataset Benchmarks (Traditional Neural Networks)

### Classification Performance (Supervised Learning)

Traditional backpropagation-based neural networks on Iris typically achieve:

| Method | Accuracy | Notes |
|--------|----------|-------|
| **Logistic Regression** | 95-97% | Linear baseline |
| **2-Layer MLP** | 96-98% | 10-20 hidden units |
| **3-Layer MLP** | 97-100% | Well-tuned network |
| **SVM (RBF kernel)** | 96-98% | Standard benchmark |
| **Random Forest** | 94-97% | Ensemble baseline |

**Confusion Matrix Insights:**
- Setosa is **perfectly separable** (100% accuracy)
- Versicolor vs Virginica have **natural overlap** (~2-4 samples misclassified)
- Best achievable accuracy: ~97-98% due to inherent class overlap

### Clustering Performance (Unsupervised Learning)

Standard unsupervised methods (PCA + K-means, autoencoders) on Iris:

| Metric | K-Means (Raw Features) | K-Means (PCA) | Autoencoder Latent | Target for Your SNN |
|--------|------------------------|---------------|-------------------|---------------------|
| **Silhouette Score** | 0.45-0.55 | 0.50-0.60 | 0.55-0.70 | **> 0.50** |
| **Davies-Bouldin Index** | 0.60-0.80 | 0.55-0.70 | 0.40-0.60 | **< 0.60** |
| **Calinski-Harabasz** | 400-500 | 450-550 | 500-650 | **> 450** |
| **Adjusted Rand Index** | 0.55-0.65 | 0.65-0.75 | 0.70-0.85 | **> 0.65** |
| **Normalized Mutual Info** | 0.60-0.70 | 0.70-0.80 | 0.75-0.90 | **> 0.70** |

**Key Insight:** Your spiking network should aim for clustering metrics comparable to K-Means on raw features as a **minimum viable learning**, with autoencoder-level performance as the **success criterion**.

### Why Iris is a Good Benchmark

✅ **Advantages:**
- Small dataset (150 samples) = fast iteration
- Only 4 features = manageable input space
- Balanced classes (50 each) = no bias issues
- Well-studied = known benchmarks

⚠️ **Limitations:**
- Very easy dataset (linear separability except Versicolor/Virginica)
- Success here doesn't guarantee scaling to harder problems
- Gray code quantization may introduce new challenges not present in float-based methods

---

## 🔬 What Good Learning Looks Like (Expected Values After 1 Epoch)

**Your Spiking Network Goals:**
- **Minimum Viable Learning:** Match K-Means on raw features (Silhouette ~0.5)
- **Good Learning:** Approach supervised MLP performance (Silhouette 0.5-0.6, Discriminability > 5.0)
- **Excellent Learning:** Match autoencoder latent clustering (Silhouette > 0.6, NMI > 0.75)

| Metric | Baseline (PlaceholderNeuron) | Minimal Learning | Good Learning | Excellent | **🎯 Target (vs Traditional ML)** |
|--------|------------------------------|------------------|---------------|-----------|-----------------------------------|
| **Mean Discriminability** | 0.000 | 1.0-2.0 | 5.0-10.0 | > 10.0 | **> 5.0** (neuron specialization) |
| **Top Neuron Overlap** | 1.000 | 0.6-0.8 | 0.3-0.5 | < 0.3 | **< 0.4** (class-specific neurons) |
| **Silhouette Score** | 0.000 | 0.1-0.2 | 0.3-0.5 | > 0.5 | **> 0.50** (match K-Means) |
| **Neuron Specialization** | 0.000 | 0.1-0.2 | 0.5-0.7 | > 0.8 | **> 0.5** (majority specialized) |
| **Davies-Bouldin** | 0.000* | 1.5-2.0 | 0.7-1.0 | < 0.5 | **< 0.70** (match K-Means) |
| **Calinski-Harabasz** | 1.000* | 50-200 | 300-450 | > 450 | **> 400** (match K-Means) |
| **Normalized Mutual Info** | 0.000 | 0.2-0.4 | 0.5-0.7 | > 0.7 | **> 0.60** (meaningful clustering) |

\* PlaceholderNeuron shows 0.0/1.0 because all spike patterns are identical (all zeros), creating artificial "perfect separation".

**Success Criteria for Unsupervised Spiking Network:**
1. **Silhouette > 0.50** = Clusters as well as K-Means (competitive with traditional unsupervised)
2. **Mean Discriminability > 5.0** = Neurons develop strong class preferences
3. **Top Neuron Overlap < 0.40** = Different neurons handle different classes
4. **NMI > 0.60** = Spike patterns encode class structure

If you achieve these targets, your spiking network is learning as well as traditional unsupervised methods on Iris!

---

## 💡 Debugging Hints

### All metrics stay at baseline (0.000, 1.000):
- **Check:** Are neurons spiking at all? Look at `spikes/[class]_avg_total` in TensorBoard
- **Fix:** Lower spike threshold, verify `receive_input()` is being called, check membrane dynamics

### Metrics improve then degrade:
- **Check:** Dying neurons, runaway inhibition, or threshold adaptation
- **Fix:** Monitor spike counts over time, add homeostatic mechanisms

### One class dominates:
- **Check:** Dataset imbalance (Iris is balanced, so check preprocessing)
- **Fix:** Verify Gray code encoding is working, check input routing

### Metrics improve slowly:
- **Tune:** Learning rate, threshold values, connection weights
- **Try:** Different connectivity patterns, more neurons

### Plateau below benchmark targets (e.g., Silhouette stuck at 0.3):
- **Expected:** Gray code quantization may make clustering harder than float-based methods
- **Check:** Are you using all 96 neurons effectively? Monitor activation sparsity
- **Try:** Increase network capacity, adjust threshold to balance spike rate
- **Compare:** Run K-means on your 6-bit Gray-coded features to establish realistic upper bound

---

## 📝 Files Reference

- [iris_experiment.py](/.venv/src/iris_experiment.py) - Main experiment script, metrics computation
- [iris_network.py](/.venv/src/iris_network.py) - Network architecture, neuron interface
- [connectivity_96.pt](/.venv/src/connectivity_96.pt) - Pre-built sparse connectivity matrices
- `runs/` - TensorBoard logs
- `plots/` - Plotly HTML dashboards

---

**Ready to experiment! Replace `PlaceholderNeuron` with your implementation and watch the metrics.**
