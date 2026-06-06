"""
Test NEAT Evolution with Real Market Data Backtesting.

Evolves trading bot networks that:
- Receive raw normalized market features
- Output trading signals (position size)
- Get evaluated by backtesting returns
- Survive/die based on trading performance
"""

import sys
import torch
import time
from market_data import RawMarketDataProvider
from backtest import BacktestSimulator
from neat_network import NEATGenome, NEATNetworkBuilder
from neat_evolution import NEATEvolver
from neat_utils import genome_to_string, network_statistics, calculate_complexity


def test_backtest_simple():
    """Quick test: Single bot backtest."""
    print("\n" + "="*70)
    print("TEST 1: Simple Bot Backtest")
    print("="*70)
    
    # Load market data
    print("\nLoading market data...")
    market_data = RawMarketDataProvider(device='cuda:1', target_stock_idx=10, normalize=True)
    
    # Create simulator
    print("\nInitializing backtest simulator...")
    simulator = BacktestSimulator(
        market_data=market_data,
        initial_capital=100000.0,
        position_size_mult=0.1,
        early_exit_threshold=0.3,
        device='cuda:1'
    )
    
    # Test with a simple genome
    print("\nCreating test bot...")
    genome = NEATGenome(num_inputs=market_data.features.shape[1], num_outputs=1)
    
    # Add some simple structure
    h1 = genome.add_node(activation='tanh')
    for i in range(min(10, market_data.features.shape[1])):
        genome.add_connection(i, h1, weight=0.1 * (i % 3 - 1))
    genome.add_connection(h1, market_data.features.shape[1], weight=0.5)
    
    # Run backtest
    print("\nRunning backtest...")
    result = simulator.backtest(genome)
    
    print("\nBot Performance:")
    print(f"  Initial capital: $100,000")
    print(f"  Final balance: ${result.final_balance:,.0f}")
    print(f"  Total return: {result.total_return*100:.2f}%")
    print(f"  Sharpe ratio: {result.sharpe_ratio:.3f}")
    print(f"  Max drawdown: {result.max_drawdown*100:.2f}%")
    print(f"  Win rate: {result.win_rate*100:.1f}%")
    print(f"  Trades: {result.num_trades}")
    print(f"  Survived: {result.survived}")
    print(f"  Fitness: {result.fitness_score:.4f}")
    
    print("\n✓ Backtest test passed!")
    return simulator


def test_neat_evolution_quick(simulator: BacktestSimulator):
    """
    Quick evolution test: 3 generations, 5 population.
    
    Evolves trading bots directly on market data.
    """
    print("\n" + "="*70)
    print("TEST 2: Quick NEAT Evolution (3 gen, pop=5)")
    print("="*70)
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create evolver
    print("\nInitializing NEAT evolver...")
    evolver = NEATEvolver(
        backtest_simulator=simulator,
        num_inputs=simulator.market_data.features.shape[1],
        num_outputs=1,
        device=device,
        seed=42
    )
    
    # Run evolution
    print("\nStarting evolution...")
    start_time = time.time()
    try:
        population, stats = evolver.evolve(
            pop_size=5,
            generations=3,
            cxpb=0.7,
            mutpb=0.3
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("Evolution Complete!")
        print("="*70)
        
        # Analyze best individual
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        best_fitness = best_individual.fitness.values[0]
        
        print(f"\nBest Bot Fitness: {best_fitness:.4f}")
        print(f"Time elapsed: {elapsed:.1f} seconds")
        
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


def test_neat_full(simulator: BacktestSimulator):
    """
    Full evolution test: 8 generations, 20 population.
    
    This is realistic bot evolution on real market data.
    """
    print("\n" + "="*70)
    print("TEST 3: Full NEAT Bot Evolution (8 gen, pop=20)")
    print("="*70)
    print("⏱  This will take 15-30 minutes (backtesting is slow but realistic)")
    
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create evolver
    print("\nInitializing NEAT evolver...")
    evolver = NEATEvolver(
        backtest_simulator=simulator,
        num_inputs=simulator.market_data.features.shape[1],
        num_outputs=1,
        device=device,
        seed=42
    )
    
    # Run full evolution
    print("\nStarting evolution...")
    start_time = time.time()
    try:
        population, stats = evolver.evolve(
            pop_size=20,
            generations=8,
            cxpb=0.7,
            mutpb=0.3
        )
        
        elapsed = time.time() - start_time
        
        print("\n" + "="*70)
        print("Evolution Complete!")
        print("="*70)
        print(f"Total time: {elapsed/60:.1f} minutes")
        
        # Analyze best individual
        best_individual = max(population, key=lambda x: x.fitness.values[0])
        best_fitness = best_individual.fitness.values[0]
        
        print(f"\nBest Bot Fitness: {best_fitness:.4f}")
        
        print(genome_to_string(best_individual))
        
        # Show top bots
        print("\nTop 5 Evolved Trading Bots (by fitness):")
        sorted_pop = sorted(population, key=lambda x: x.fitness.values[0], reverse=True)
        for i, ind in enumerate(sorted_pop[:5]):
            fitness = ind.fitness.values[0]
            size = len(ind.nodes)
            complexity = calculate_complexity(ind)
            print(f"  {i+1}. Fitness={fitness:.4f}, Nodes={size}, Complexity={complexity:.3f}")
        
        print("\n✓ Full evolution test passed!")
        
        return population, stats
    
    except Exception as e:
        print(f"\n✗ Evolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    print("\n" + "="*70)
    print("NEAT Bot Evolution Test Suite")
    print("Evolving trading bots via backtesting")
    print("="*70)
    
    # Test 1: Simple backtest
    try:
        simulator = test_backtest_simple()
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Test 2: Quick evolution
    try:
        pop, stats = test_neat_evolution_quick(simulator)
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Full evolution
    if "--full" in sys.argv:
        try:
            pop, stats = test_neat_full(simulator)
        except Exception as e:
            print(f"❌ Test 3 failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*70)
        print("To run full evolution (15-30 minutes):")
        print("  python test_neat_bots.py --full")
        print("="*70)
