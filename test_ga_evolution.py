#!/usr/bin/env python3
"""
Quick test of GA Bot Evolution with Parallel Evaluation
Run from command line: python test_ga_evolution.py
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from ga_bot_evolution import BOTEvolver


if __name__ == "__main__":
    print("="*80)
    print("GA BOT EVOLUTION SANDBOX - PARALLEL EVALUATION")
    print("="*80)

    # Initialize evolver (loads data)
    print("\nInitializing evolver...")
    evolver = BOTEvolver(seed=42)

    # Small test run: 20 generations, 30 population
    print("\nStarting evolution (20 gen, 30 pop)...")
    print("This will use multiprocessing for parallel fitness evaluation")
    print("Expected speedup: ~8-15x on your 32 cores (leave 2 free)\n")

    start_time = time.time()
    best_bot, evolution_stats = evolver.evolve(
        pop_size=30,
        generations=20,
        cxpb=0.7,
        mutpb=0.3
    )
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("BEST BOT FOUND")
    print("="*80)
    print(f"\nGenome (11 genes):")
    print(f"  Weights (5): {[f'{x:.3f}' for x in best_bot[:5]]}")
    print(f"  Entry threshold: {best_bot[5]:.4f}")
    print(f"  Position size: {best_bot[6]:.4f}")
    print(f"  Stop loss: {best_bot[7]:.4f}")
    print(f"  Take profit: {best_bot[8]:.4f}")
    print(f"  Holding bars: {int(best_bot[9])}")
    print(f"  Max positions: {int(best_bot[10])}")

    print(f"\nValidation Fitness (Sharpe, -Drawdown): {best_bot.fitness.values}")

    # Test on hold-out data
    print("\nTesting on held-out TEST set...")
    test_metrics = evolver.evaluate_on_test(best_bot)

    print(f"\n{'Metric':<20} | {'Value':>10}")
    print("-" * 35)
    for k, v in sorted(test_metrics.items()):
        if k != 'trades' and k != 'num_trades':
            print(f"{k:<20} | {v:>10.4f}")

    print(f"Number of trades: {test_metrics.get('num_trades', 0)}")

    # Timing analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    total_eval_time = sum(s.get('eval_time', 0) for s in evolution_stats)
    total_gen_time = sum(s.get('gen_time', 0) for s in evolution_stats)

    print(f"Total evolution time: {total_time:.1f}s")
    print(f"Total eval time (parallel): {total_eval_time:.1f}s ({100*total_eval_time/total_time:.1f}%)")
    print(f"Total gen time: {total_gen_time:.1f}s")
    print(f"Speedup with {evolver.num_workers} cores: ~{(30*20*0.020) / total_eval_time:.1f}x")

    print("\n" + "="*80)
    print("Evolution complete! You can now:")
    print(f"  1. Increase pop_size (was 30) for better diversity")
    print(f"  2. Increase generations (was 20) for longer search")
    print(f"  3. Modify bot strategy in trading_bot.py")
    print(f"  4. Try different fitness functions in ga_bot_evolution.py")
    print("="*80)

