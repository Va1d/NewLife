# NEAT: Neuroevolution of Augmenting Topologies

Welcome! This is a working implementation of NEAT that evolves **neural network topologies** (not just weights).

## What You Have

Four Python files:

1. **`neat_network.py`** - Core NEAT implementation
   - `NEATNode`: A neuron with ID, type, activation
   - `NEATConnection`: A link with weight, enabled status
   - `NEATGenome`: Complete network specification (all nodes + connections)
   - `NEATModule`: Converts genome → PyTorch executable network
   - `NEATNetworkBuilder`: Factory for building networks

2. **`neat_evolution.py`** - DEAP + NEAT integration
   - `InnovationTracker`: Global ID assignment for new structures
   - Mutation operators: `mutate_add_connection`, `mutate_add_neuron`, `mutate_weights`
   - `NEATEvolver`: Main controller that runs the evolution

3. **`neat_utils.py`** - Analysis & visualization
   - `genome_to_string()`: Print network structure
   - `compare_genomes()`: See how different two networks are
   - `network_statistics()`: Analyze entire population
   - `has_recurrent_connection()`: Detect feedback loops

4. **`test_neat.py`** - Working examples
   - TEST 1: Network construction
   - TEST 2: Quick evolution (3 gen)
   - TEST 3: Full evolution (8 gen, realistic)

## How to Run

### Test the network construction:
```bash
cd /home/bo/Py/NewLife
source .venv/bin/activate
python .venv/src/neat_network.py
```

Output shows:
- Genome structure
- Network topology
- Forward pass example

### Test the full suite:
```bash
# Quick tests (2 minutes)
python .venv/src/test_neat.py

# Full evolution test (5 minutes on GPU, 30+ on CPU)
python .venv/src/test_neat.py --full
```

## Key Concepts to Understand

### 1. Genomes Encode Networks

```python
genome = NEATGenome(num_inputs=5, num_outputs=1)

# Genome contains:
# - Nodes: [0,1,2,3,4] inputs + [5] output + hidden nodes
# - Connections: list of links with weights

genome.add_connection(0, 5, weight=0.5)  # input 0 → output, weight 0.5
h1 = genome.add_node(activation="tanh")  # Create hidden neuron
genome.add_connection(1, h1, weight=0.8) # input 1 → hidden
genome.add_connection(h1, 5, weight=0.4) # hidden → output
```

### 2. Mutation Operators Create Evolution

**Mutation 1: Add Connection**
```
Before: [Input] → [Output]
        [Input] → [Hidden] → [Output]

Mutation: Add connection between Input and Output
After:   [Input] → [Output]
        [Hidden] (new link added!)
```

**Mutation 2: Add Neuron** (most important!)
```
Before: Input[0] ──w=0.5──> Output

Mutation: Split connection by inserting neuron
After:   Input[0] ──w=1.0──> NewNode ──w=0.5──> Output
```

**Mutation 3: Mutate Weights**
```
Before: w = 0.5
After:  w = 0.5 + random_gaussian() = 0.47 or 0.63 etc
```

### 3. Networks Can Have Feedback (Recurrence)

```python
genome.add_connection(output_node, hidden_node, weight=0.1)
# Output feeds back to hidden layer
# Creates a recurrent loop
```

This is powerful! Recurrent networks can:
- Remember past inputs (temporal dynamics)
- Oscillate (useful for time series)
- Create strange attractors (chaotic but structured)

### 4. Innovation Tracking (Why Genomes Are Comparable)

```
Global Innovation History:
  Connection (0,5) → innovation #0
  Connection (1,3) → innovation #1
  Neuron creation → innovation #2

Genome A: [innov#0, innov#1, innov#5]
Genome B: [innov#0, innov#1, innov#7]

Same innovations (#0, #1) → similar networks → same species
```

This allows **speciation**: groups of similar networks evolve together.

## Code Walkthrough by Example

### Example 1: Create and Build a Network

```python
from neat_network import NEATGenome, NEATNetworkBuilder
import torch

# 1. Create genome (specification)
genome = NEATGenome(num_inputs=2, num_outputs=1)
genome.add_connection(0, 2, weight=0.5)   # input 0 → output
genome.add_connection(1, 2, weight=-0.3)  # input 1 → output

# 2. Build executable network
network = NEATNetworkBuilder.build_network(genome, device='cpu')

# 3. Use network
test_input = torch.tensor([[0.5, -0.3], [0.1, 0.9]], dtype=torch.float32)
output = network(test_input)

print(f"Input:\n{test_input}")
print(f"Output:\n{output}")  # (2, 1) shaped output
```

### Example 2: Evolve Networks

```python
from neat_evolution import NEATEvolver
from torch.utils.data import DataLoader

# 1. Create evolver
evolver = NEATEvolver(
    num_inputs=256,        # Your feature dimension
    num_outputs=1,         # Binary classification
    train_loader=train_dl, # Your training data
    val_loader=val_dl,     # Your validation data
    device='cuda:1'
)

# 2. Run evolution
population, stats = evolver.evolve(
    pop_size=20,      # 20 networks per generation
    generations=8     # Run 8 generations
)

# 3. Analyze
best = max(population, key=lambda x: x.fitness.values[0])
print(f"Best F1: {best.fitness.values[0]:.4f}")
print(best.get_topology_string())
```

### Example 3: Visualize Results

```python
from neat_utils import genome_to_string, network_statistics

# Show one network
print(genome_to_string(best_genome))

# Output:
# NEAT Genome
# ============================================================
# Network Size: 7 nodes, 5 active connections
#
# Input Nodes:
#   0: input
#   1: input
#
# Hidden Nodes (3):
#   3: tanh
#   4: relu
#   5: sigmoid
#
# Output Nodes:
#   2: output (tanh)
#
# Active Connections (5):
#   0 → 2: w=+0.523
#   0 → 3: w=+0.882
#   1 → 4: w=-0.341
#   3 → 2: w=+0.756
#   4 → 2: w=+0.234
#
# ⚠ Recurrent Connections (1):
#   2 → 3: Potential feedback loop

# Show population statistics
stats = network_statistics(population)
print(stats)
# {
#   'population_size': 20,
#   'avg_nodes': 5.3,
#   'max_nodes': 8,
#   'avg_complexity': 0.32,
#   'networks_with_recurrence': 3,
#   'recurrence_percentage': 15.0
# }
```

## Questions You'll Have (& Answers)

**Q: Why would a chaotic recurrent network work better than a clean feedforward network?**

A: Evolution explores different state spaces. Recurrence can:
   - Create implicit memory (network state depends on history)
   - Solve temporal patterns more elegantly
   - Accidentally find robust solutions that don't overfit
   - But also: sometimes it just gets lucky or overfits differently

**Q: How do I connect this to my bot activity detection?**

A: Replace the mock DataLoader in `test_neat.py` with your actual data:
```python
# Instead of:
train_loader, val_loader = create_mock_dataloaders(...)

# Do:
from loader import load_bot_activity_data
train_data, val_data, test_data = load_bot_activity_data()
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)
```

**Q: Will NEAT be faster than hyperparameter evolution?**

A: No, probably slower:
   - Hyperparameter evolution: 20 epochs × 8 gens ≈ 5 min on GPU
   - NEAT: 20 networks × 3 epochs × 8 gens ≈ 30 min on GPU
   - NEAT explores more complex space (topology + weights)
   - Trade-off: time vs. potentially better architecture

**Q: How do I extract a trained NEAT network to use in production?**

A: Save the best genome:
```python
import pickle

best = max(population, key=lambda x: x.fitness.values[0])
pickle.dump(best, open('best_neat_network.pkl', 'wb'))

# Later:
best = pickle.load(open('best_neat_network.pkl', 'rb'))
network = NEATNetworkBuilder.build_network(best, device='cuda:1')
network.load_state_dict(torch.load('weights.pt'))
```

## What Might Emerge? (The Cool Part)

Evolution might create:

1. **Sparse networks** - Only 3-4 active connections out of 20
   - Faster inference
   - Easier to understand information flow

2. **Recurrent structures** - Feedback loops
   - Remember patterns over time
   - Could find temporal patterns you didn't hardcode

3. **Mixed activations** - ReLU in some places, tanh in others
   - Optimize activation type per neuron
   - Might be more data-efficient

4. **Skip connections** - Input directly to output
   - Quick feature pass-through
   - Residual-like connections

5. **Completely weird topologies** that somehow work
   - Maybe 2 neurons with specific feedback that captures pattern
   - You can't explain why, but it works

## Integration with Your Project

When ready, integrate with your bot activity task:

```python
# Step 1: Load your data
from loader import load_bot_activity_data

# Step 2: Create NEAT evolver
evolver = NEATEvolver(
    num_inputs=256,        # Your transformer input size
    num_outputs=1,         # Binary: bot or not
    train_loader=train_dl,
    val_loader=val_dl,
    device='cuda:1'
)

# Step 3: Run evolution
pop, stats = evolver.evolve(pop_size=20, generations=15)

# Step 4: Train best network fully
best_genome = max(pop, key=lambda x: x.fitness.values[0])
best_network = NEATNetworkBuilder.build_network(best_genome, device='cuda:1')

# Train for 50 epochs (from scratch or with evolved weights)
# ... your training loop ...
```

## Next: Questions to Ask Me

Read through the code, run the tests, then ask:

1. "How does the forward pass handle recurrence?"
2. "Why does the innovation tracker matter?"
3. "How do I interpret a weird topology?"
4. "What's the exact mutation strategy?"
5. "How do I add other mutation operators?"
6. "Can I evolve activations too? (Not just topology)"

Enjoy! 🚀
