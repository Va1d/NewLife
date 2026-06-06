"""
Test NEAT Evolution with real bot activity data.

This shows how to:
1. Load your bot activity training data (Stock #10)
2. Create NEAT evolver
3. Run evolution
4. Analyze results
"""

import sys
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset

# Import NEAT modules
from neat_network import NEATGenome, NEATNetworkBuilder
from neat_evolution import NEATEvolver
from neat_utils import genome_to_string, network_statistics, calculate_complexity

# Import loader
from loader import TheSetGPU


class FlattenedBotActivityDataset(Dataset):
    """
    Wrapper around TheSetGPU that flattens time-series to flat features.
    
    TheSetGPU returns sequences [256 steps, 388 time_steps, n_features]
    We average over time to get [256 steps, n_features] for NEAT.
    """
    def __init__(self, the_set_gpu_dataset, device='cuda:1'):
        """Initialize with TheSetGPU dataset."""
        self.dataset = the_set_gpu_dataset
        self.device = device
        
        # Pre-compute flattened features and labels
        self.features = []
        self.labels = []
        
        print(f"Pre-processing {len(the_set_gpu_dataset)} sequences...")
        for i in range(len(the_set_gpu_dataset)):
            x_batch, y_batch, seq_lengths = the_set_gpu_dataset[i]
            
            # x_batch shape: [256 steps, 388 max_seq_len, n_features]
            # Average over time dimension: [256, n_features]
            x_flat = x_batch.mean(dim=1)  # Average across time steps
            
            self.features.append(x_flat)
            self.labels.append(y_batch)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1} sequences...")
        
        # Concatenate all
        self.features = torch.cat(self.features, dim=0)  # [256*N, n_features]
        self.labels = torch.cat(self.labels, dim=0)      # [256*N]
        
        print(f"Final shapes - features: {self.features.shape}, labels: {self.labels.shape}")
        print(f"  Bot activity (1): {self.labels.sum().item():.0f} / {len(self.labels)} ({100*self.labels.mean().item():.1f}%)")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def create_real_dataloaders(device: str = 'cuda:1',
                            target_stock_idx: int = 10,
                            batch_size: int = 32,
                            train_split: float = 0.8) -> tuple:
    """
    Load real bot activity data using TheSetGPU.
    
    Returns train and validation DataLoaders compatible with NEAT.
    """
    print("\n" + "="*70)
    print("Loading Real Bot Activity Data")
    print("="*70)
    
    # Load using TheSetGPU
    print(f"\nInitializing TheSetGPU (Stock #{target_stock_idx}, device={device})...")
    try:
        the_set_gpu = TheSetGPU(device=device, target_stock_idx=target_stock_idx)
        print(f"✓ Loaded {len(the_set_gpu)} sequences")
    except Exception as e:
        print(f"✗ Failed to load TheSetGPU: {e}")
        raise
    
    # Flatten to NEAT-compatible format
    print("\nFlattening sequences to feature vectors...")
    flat_dataset = FlattenedBotActivityDataset(the_set_gpu, device=device)
    
    # Split into train/val
    split_idx = int(len(flat_dataset) * train_split)
    train_ds = torch.utils.data.Subset(flat_dataset, range(split_idx))
    val_ds = torch.utils.data.Subset(flat_dataset, range(split_idx, len(flat_dataset)))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"\nData split:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Feature dimension: {flat_dataset.features.shape[1]}")
    
    return train_loader, val_loader


def create_mock_dataloaders(num_samples: int = 500,
                           num_features: int = 256,
                           batch_size: int = 32) -> tuple:
    """
    Create mock data loaders for testing.

    In real scenario, replace with your bot activity dataset.
    """
    # Random features
    X = torch.randn(num_samples, num_features)

    # Random labels (bot activity: 1, normal: 0)
    # Make ~17% positive (like your Stock #10 data)
    y = (torch.rand(num_samples) < 0.17).int()

    # Split into train/val
    split_point = int(0.8 * num_samples)

    train_ds = TensorDataset(X[:split_point], y[:split_point])
    val_ds = TensorDataset(X[split_point:], y[split_point:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def test_neat_basic():
    """Quick test: NEAT network construction."""
    print("\n" + "="*70)
    print("TEST 1: NEAT Network Construction")
    print("="*70)

    # Create a genome
    genome = NEATGenome(num_inputs=5, num_outputs=1)

    # Add some structure
    h1 = genome.add_node(activation="tanh")
    h2 = genome.add_node(activation="relu")

    genome.add_connection(0, h1, weight=0.5)
    genome.add_connection(1, h2, weight=-0.3)
    genome.add_connection(h1, 5, weight=0.8)  # h1 to output
    genome.add_connection(h2, 5, weight=0.4)  # h2 to output
    genome.add_connection(5, h1, weight=0.1)  # Recurrent feedback!

    print(genome_to_string(genome))

    # Build network
    network = NEATNetworkBuilder.build_network(genome, device='cpu')

    # Test forward pass
    test_input = torch.randn(4, 5)  # batch_size=4, num_inputs=5
    output = network(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values:\n{output}")
    print("\n✓ Network construction test passed!")

    return network, genome


def test_neat_evolution_quick():
    """
    Quick evolution test: 3 generations, 5 population.

    Shows the evolution process in action.
    Shows what "emerging structures" look like.
    """
    print("\n" + "="*70)
    print("TEST 2: Quick NEAT Evolution (3 gen, pop=5)")
    print("="*70)

    # Create minimal mock data
    print("\nLoading data...")
    train_loader, val_loader = create_mock_dataloaders(
        num_samples=100,  # Small for quick test
        num_features=20,
        batch_size=16
    )

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create evolver
    evolver = NEATEvolver(
        num_inputs=20,
        num_outputs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        seed=42
    )

    # Run evolution (very short for quick test)
    print("\nStarting evolution...")
    try:
        population, stats = evolver.evolve(
            pop_size=5,
            generations=3,
            cxpb=0.7,
            mutpb=0.3
        )

        print("\n" + "="*70)
        print("Evolution Complete!")
        print("="*70)

        # Analyze best individual
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        best_f1 = best_individual.fitness.values[0]
        best_size = best_individual.fitness.values[1]

        print(f"\nBest Individual F1: {best_f1:.4f}")
        print(f"Best Individual Size: {-best_size:.0f} nodes")

        print(genome_to_string(best_individual))

        # Show population statistics
        pop_stats = network_statistics(population)
        print("\nPopulation Statistics:")
        for key, val in pop_stats.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")

        print("\n✓ Evolution test passed!")

        return population, stats

    except Exception as e:
        print(f"\n✗ Evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_neat_full(use_real_data: bool = True):
    """
    Full evolution test: realistic settings.

    This is what you'd run for real bot activity detection.
    
    Args:
        use_real_data: If True, use Stock #10 bot activity. If False, use mock data.
    """
    print("\n" + "="*70)
    print("TEST 3: Full NEAT Evolution (8 gen, pop=20)")
    if use_real_data:
        print("Using REAL bot activity data (Stock #10)")
    else:
        print("Using MOCK random data")
    print("="*70)

    if use_real_data:
        print("⏱  This will take 5-15 minutes on GPU (training on real data)")
    else:
        print("⏱  This will take 1-2 seconds on GPU (training on mock data)")

    # Load data
    print("\nLoading data...")
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    
    if use_real_data:
        try:
            train_loader, val_loader = create_real_dataloaders(
                device=device,
                target_stock_idx=10,  # Stock #10 has cleanest bot signals
                batch_size=32
            )
            num_features = train_loader.dataset.dataset.features.shape[1]
        except Exception as e:
            print(f"\n✗ Failed to load real data: {e}")
            print("  Falling back to mock data...")
            train_loader, val_loader = create_mock_dataloaders(
                num_samples=500,
                num_features=256,
                batch_size=32
            )
            num_features = 256
            use_real_data = False
    else:
        train_loader, val_loader = create_mock_dataloaders(
            num_samples=500,
            num_features=256,
            batch_size=32
        )
        num_features = 256

    print(f"Using device: {device}")

    # Create evolver
    evolver = NEATEvolver(
        num_inputs=num_features,
        num_outputs=1,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        seed=42
    )

    # Run full evolution
    print("\nStarting evolution...")
    try:
        population, stats = evolver.evolve(
            pop_size=20,
            generations=8,
            cxpb=0.7,
            mutpb=0.3
        )

        print("\n" + "="*70)
        print("Evolution Complete!")
        if use_real_data:
            print("✓ Successfully evolved NEAT network on real bot activity data!")
        print("="*70)

        # Analyze best individual
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        best_f1 = best_individual.fitness.values[0]
        best_size = best_individual.fitness.values[1]

        print(f"\nBest Individual F1: {best_f1:.4f}")
        print(f"Best Individual Size: {-best_size:.0f} nodes")

        print(genome_to_string(best_individual))

        # Show all evolved structures
        print("\nAll Individuals (sorted by F1):")
        sorted_pop = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
        for i, ind in enumerate(sorted_pop[:5]):
            f1 = ind.fitness.values[0]
            size = -ind.fitness.values[1]
            complexity = calculate_complexity(ind)
            print(f"  {i+1}. F1={f1:.4f}, Nodes={size:.0f}, Complexity={complexity:.3f}")

        print("\n✓ Full evolution test passed!")

        return population, stats

    except Exception as e:
        print(f"\n✗ Evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NEAT Evolution Test Suite")
    print("="*70)

    # Test 1: Network construction
    try:
        test_neat_basic()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    # Test 2: Quick evolution
    try:
        pop, stats = test_neat_evolution_quick()
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    # Test 3: Full evolution (optional, takes time)
    use_real = "--real" in sys.argv or "--real-data" in sys.argv
    
    if "--full" in sys.argv:
        try:
            pop, stats = test_neat_full(use_real_data=use_real)
        except Exception as e:
            print(f"❌ Test 3 failed: {e}")
    else:
        print("\n" + "="*70)
        print("To run full evolution test with MOCK data (1-2 seconds):")
        print("  python test_neat.py --full")
        print("\nTo run with REAL bot activity data (5-15 minutes):")
        print("  python test_neat.py --full --real-data")
        print("="*70)
