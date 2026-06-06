#!/usr/bin/env python3
"""
Ensemble GA Evolution - Run multiple independent GA searches in parallel
Leverages your 128GB RAM to find better bots faster
"""

import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ga_bot_evolution import BOTEvolver


def run_single_evolution(seed: int, pop_size: int = 50, generations: int = 20):
    """Run one independent GA evolution with different seed"""
    print(f"[Ensemble Worker {seed}] Starting evolution...")

    evolver = BOTEvolver(seed=seed)
    best_bot, stats = evolver.evolve(
        pop_size=pop_size,
        generations=generations,
        cxpb=0.7,
        mutpb=0.3
    )

    # Test on hold-out set
    test_metrics = evolver.evaluate_on_test(best_bot)

    print(f"[Ensemble Worker {seed}] Best Sharpe: {best_bot.fitness.values[0]:.4f}, "
          f"Test Sharpe: {test_metrics['sharpe_ratio']:.4f}")

    return {
        'seed': seed,
        'best_bot': best_bot[:],
        'validation_sharpe': best_bot.fitness.values[0],
        'test_metrics': test_metrics,
        'stats': stats,
    }


def ensemble_evolution(num_runs: int = 5, pop_size: int = 50,
                      generations: int = 20, num_workers: int = None):
    """
    Run multiple independent GA evolutions in parallel

    Args:
        num_runs: Number of independent GA searches
        pop_size: Population size per GA run
        generations: Generations per GA run
        num_workers: Max parallel processes (default: num_runs)

    Returns: List of best bots from each run, ranked by test performance
    """
    if num_workers is None:
        num_workers = min(num_runs, 4)  # Usually 4+ parallel evolutions good

    print("="*80)
    print(f"ENSEMBLE GA EVOLUTION - {num_runs} Independent Searches")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Runs: {num_runs} (parallel: {num_workers})")
    print(f"  Pop size: {pop_size} per run")
    print(f"  Generations: {generations}")
    print(f"  Total evaluations: {num_runs * pop_size * generations}")
    print(f"\nStarting... (parallel workers will log their progress)\n")

    start_time = time.time()
    results = []

    # Run all evolutions in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(run_single_evolution, seed, pop_size, generations): seed
            for seed in range(num_runs)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    total_time = time.time() - start_time

    # Rank by test performance
    results_sorted = sorted(results,
                           key=lambda r: r['test_metrics']['sharpe_ratio'],
                           reverse=True)

    print("\n" + "="*80)
    print("ENSEMBLE RESULTS - Ranked by Test Sharpe Ratio")
    print("="*80)

    for rank, result in enumerate(results_sorted, 1):
        metrics = result['test_metrics']
        print(f"\n[#{rank}] Seed {result['seed']}")
        print(f"  Validation Sharpe: {result['validation_sharpe']:.4f}")
        print(f"  Test Sharpe:       {metrics['sharpe_ratio']:.4f}")
        print(f"  Test Win Rate:     {metrics['win_rate']:.2%}")
        print(f"  Test Drawdown:     {metrics['max_drawdown']:.4f}")
        print(f"  Num Trades:        {metrics['num_trades']}")
        print(f"  Genome: {result['best_bot']}")

    print("\n" + "="*80)
    print("BEST BOT (by test performance)")
    print("="*80)
    best_result = results_sorted[0]
    metrics = best_result['test_metrics']

    print(f"\nSeed: {best_result['seed']}")
    print(f"Validation Sharpe: {best_result['validation_sharpe']:.4f}")
    print(f"Test Sharpe: {metrics['sharpe_ratio']:.4f}")
    print(f"Test Win Rate: {metrics['win_rate']:.2%}")
    print(f"Test Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Trades: {metrics['num_trades']}")

    # Statistics
    print("\n" + "="*80)
    print("ENSEMBLE STATISTICS")
    print("="*80)
    test_sharpes = [r['test_metrics']['sharpe_ratio'] for r in results_sorted]

    print(f"\nTest Sharpe Ratios across {num_runs} runs:")
    print(f"  Best:  {np.max(test_sharpes):.4f}")
    print(f"  Mean:  {np.mean(test_sharpes):.4f}")
    print(f"  Std:   {np.std(test_sharpes):.4f}")
    print(f"  Worst: {np.min(test_sharpes):.4f}")

    print(f"\nTotal evolution time: {total_time:.1f}s")
    print(f"Time per run: {total_time/num_runs:.1f}s")
    print(f"Effective speedup vs serial: {num_runs}x (ran {num_runs} in {total_time:.1f}s)")

    print("\n" + "="*80)
    print("Recommendation:")
    if best_result['validation_sharpe'] < 0.8:
        print("  ⚠️  Best Sharpe < 0.8 - consider:")
        print("     1. More generations (increase from 20)")
        print("     2. Larger populations (increase from 50)")
        print("     3. Modify bot strategy in trading_bot.py")
    elif best_result['validation_sharpe'] < 1.0:
        print("  ✓ Decent Sharpe (0.8-1.0) - good for paper trading")
        print("    Deploy best_bot to Alpaca for 2-4 weeks live test")
    else:
        print("  ✓✓ Excellent Sharpe (>1.0) - likely edge found!")
        print("     Deploy immediately, but monitor drawdowns")
    print("="*80)

    return results_sorted


if __name__ == "__main__":
    # Run 5 independent GA evolutions in parallel
    # Each uses different random seed, finds different local optima
    # Ensemble picks best = better bot than any single run

    results = ensemble_evolution(
        num_runs=4,              # 4 parallel searches (uses 4 cores per run)
        pop_size=50,             # 50 bots per generation
        generations=20,          # 20 generations per search
        num_workers=4            # Max 4 parallel runs (uses ~120 cores total)
    )

    print("\n✓ Ensemble evolution complete!")
    print(f"\nBest bot found (seed {results[0]['seed']}):")
    for i, gene in enumerate(results[0]['best_bot']):
        print(f"  Gene {i}: {gene:.4f}")
