"""
GA Bot Evolution - Main DEAP Framework
Evolves trading bots using genetic algorithms on your Stock #10 data
"""

import numpy as np
from typing import Tuple, List
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# DEAP imports
from deap import base, creator, tools, algorithms
import random

sys.path.insert(0, str(Path(__file__).parent))

from ga_utils import (
    load_bot_activity_data, calculate_metrics, format_genome,
    print_evolution_stats, calculate_sharpe_ratio
)
from trading_bot import TradingBot, BotParameters


class BOTEvolver:
    """
    Genetic Algorithm framework for evolving trading bots
    """

    def __init__(self, seed: int = 42, num_workers: int = None):
        """Initialize DEAP framework and data

        Args:
            seed: Random seed for reproducibility
            num_workers: Number of worker processes (default: use all cores - 2)
        """
        random.seed(seed)
        np.random.seed(seed)

        # Set up parallel evaluation
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)  # Leave 2 cores free
        self.num_workers = num_workers

        # Load dataset
        self.data = load_bot_activity_data()
        self.prices_train = self.data['prices'][self.data['train_mask']]
        self.volumes_train = self.data['volumes'][self.data['train_mask']]
        self.signals_train = self.data['signals'][self.data['train_mask']]

        self.prices_val = self.data['prices'][self.data['val_mask']]
        self.volumes_val = self.data['volumes'][self.data['val_mask']]
        self.signals_val = self.data['signals'][self.data['val_mask']]

        self.prices_test = self.data['prices'][self.data['test_mask']]
        self.volumes_test = self.data['volumes'][self.data['test_mask']]
        self.signals_test = self.data['signals'][self.data['test_mask']]

        print(f"[GA] Train: {len(self.prices_train)} bars")
        print(f"[GA] Val:   {len(self.prices_val)} bars")
        print(f"[GA] Test:  {len(self.prices_test)} bars")
        print(f"[GA] Using {self.num_workers} worker processes for parallel evaluation")

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Configure DEAP framework"""
        # Define fitness (2 objectives: maximize Sharpe, minimize drawdown)
        if 'FitnessMulti' in dir(creator):
            del creator.FitnessMulti
        if 'Individual' in dir(creator):
            del creator.Individual

        creator.create("FitnessMulti", base.Fitness,
                      weights=(1.0, -1.0))  # max Sharpe, min drawdown
        creator.create("Individual", list,
                      fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()

        # Genome definition: [weight1, weight2, weight3, weight4, weight5,
        #                     entry_threshold, position_size, stop_loss,
        #                     take_profit, holding_bars, max_positions]
        self.toolbox.register("weight", random.uniform, 0, 1)  # 5 weights
        self.toolbox.register("entry_threshold", random.uniform, 0.3, 0.7)
        self.toolbox.register("position_size", random.uniform, 0.01, 0.1)
        self.toolbox.register("stop_loss", random.uniform, 0.01, 0.05)
        self.toolbox.register("take_profit", random.uniform, 0.02, 0.10)
        self.toolbox.register("holding_bars", random.randint, 5, 50)
        self.toolbox.register("max_positions", random.randint, 1, 5)

        # Individual: 5 weights + 6 other params = 11 genes
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat,
                            list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_bot)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.3)
        self.toolbox.register("mutate", self._mutate_bot)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Bounds checking
        self.toolbox.decorate("mate",
                            tools.DeltaPenality(self._bounds_check,
                                              (0,) * 11))
        self.toolbox.decorate("mutate",
                            tools.DeltaPenality(self._bounds_check,
                                              (0,) * 11))

    def _create_individual(self):
        """Create one random individual (bot)"""
        individual = creator.Individual([
            self.toolbox.weight(),
            self.toolbox.weight(),
            self.toolbox.weight(),
            self.toolbox.weight(),
            self.toolbox.weight(),
            self.toolbox.entry_threshold(),
            self.toolbox.position_size(),
            self.toolbox.stop_loss(),
            self.toolbox.take_profit(),
            self.toolbox.holding_bars(),
            self.toolbox.max_positions(),
        ])
        return individual

    def _mutate_bot(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Mutate a bot's genes"""
        # Gaussian mutation with 20% probability per gene
        for i in range(len(individual)):
            if random.random() < 0.2:
                if i < 5:  # Weights
                    individual[i] += random.gauss(0, 0.1)
                elif i == 5:  # entry_threshold
                    individual[i] += random.gauss(0, 0.05)
                elif i == 6:  # position_size
                    individual[i] += random.gauss(0, 0.01)
                elif i == 7:  # stop_loss
                    individual[i] += random.gauss(0, 0.005)
                elif i == 8:  # take_profit
                    individual[i] += random.gauss(0, 0.01)
                else:  # discrete params
                    individual[i] = int(individual[i] + random.gauss(0, 2))

        return (individual,)

    def _bounds_check(self, individual: creator.Individual) -> bool:
        """Check if individual is within bounds"""
        # Weights 0-1
        for i in range(5):
            if not (0 <= individual[i] <= 1):
                return False

        # entry_threshold 0.3-0.7
        if not (0.3 <= individual[5] <= 0.7):
            return False

        # position_size 0.01-0.1
        if not (0.01 <= individual[6] <= 0.1):
            return False

        # stop_loss 0.01-0.05
        if not (0.01 <= individual[7] <= 0.05):
            return False

        # take_profit 0.02-0.10
        if not (0.02 <= individual[8] <= 0.10):
            return False

        # holding_bars 5-50
        if not (5 <= individual[9] <= 50):
            return False

        # max_positions 1-5
        if not (1 <= individual[10] <= 5):
            return False

        return True

    def _evaluate_batch(self, genomes: List[List[float]]) -> List[Tuple[float, float]]:
        """Evaluate a batch of genomes in parallel using multiprocessing"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            fitnesses = list(executor.map(self.evaluate_bot, genomes))
        return fitnesses

    def evaluate_bot(self, genome: List[float]) -> Tuple[float, float]:
        """
        Fitness function: TRAIN on train set, EVALUATE on val set

        Returns: (sharpe_ratio, -max_drawdown)
        Higher is better for both
        """
        params = BotParameters(
            entry_weights=np.array(genome[:5]),
            entry_threshold=genome[5],
            position_size=genome[6],
            stop_loss_pct=genome[7],
            take_profit_pct=genome[8],
            holding_bars=int(genome[9]),
            max_concurrent_positions=int(genome[10]),
        )

        # Simulate on TRAIN set
        bot_train = TradingBot(params, self.prices_train,
                              self.volumes_train, self.signals_train)
        train_returns = bot_train.simulate()

        if len(train_returns) == 0:
            return (0.0, 0.0)

        train_sharpe = calculate_sharpe_ratio(np.array(train_returns))

        # Simulate on VALIDATION set (fitness evaluation)
        bot_val = TradingBot(params, self.prices_val,
                            self.volumes_val, self.signals_val)
        val_returns = bot_val.simulate()

        if len(val_returns) == 0:
            return (0.0, 0.0)

        val_metrics = calculate_metrics(np.array(val_returns))
        val_sharpe = val_metrics['sharpe_ratio']
        val_drawdown = val_metrics['max_drawdown']

        # Fitness: sharpe on val, penalize max drawdown
        fitness = (val_sharpe, -val_drawdown)

        return fitness

    def evolve(self, pop_size: int = 50, generations: int = 20,
               cxpb: float = 0.7, mutpb: float = 0.3) -> Tuple[List, List]:
        """
        Run genetic algorithm for bot evolution

        Args:
            pop_size: Population size per generation
            generations: Number of generations to evolve
            cxpb: Crossover probability
            mutpb: Mutation probability

        Returns: (best_individuals, stats)
        """
        print(f"\n[GA] Starting evolution: pop_size={pop_size}, gen={generations}")
        print(f"[GA] Parallel evaluation with {self.num_workers} workers")

        # Create initial population
        pop = self.toolbox.population(n=pop_size)

        # Evaluate initial population (parallel)
        start_time = time.time()
        fitnesses = self._evaluate_batch(pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        init_eval_time = time.time() - start_time

        print(f"[GA] Initial population evaluated ({pop_size} bots) in {init_eval_time:.1f}s")

        stats = []

        # Evolution loop
        for gen in range(generations):
            gen_start = time.time()

            # Select next generation
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Apply crossover
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i-1], offspring[i] = self.toolbox.mate(
                        offspring[i-1], offspring[i]
                    )
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # Apply mutation
            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate individuals with invalid fitness (parallel)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                eval_start = time.time()
                fitnesses = self._evaluate_batch(invalid_ind)
                eval_time = time.time() - eval_start
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
            else:
                eval_time = 0

            # Replace population
            pop[:] = offspring

            # Track stats
            fits = [ind.fitness.values[0] for ind in pop]  # First objective (Sharpe)
            best_idx = np.argmax(fits)
            best_ind = pop[best_idx]

            gen_time = time.time() - gen_start

            stats.append({
                'gen': gen,
                'best_sharpe': fits[best_idx],
                'avg_sharpe': np.mean(fits),
                'std_sharpe': np.std(fits),
                'best_individual': best_ind[:],
                'eval_time': eval_time,
                'gen_time': gen_time,
            })

            print(f"[Gen {gen:3d}] Sharpe: best={fits[best_idx]:6.3f}, "
                  f"avg={np.mean(fits):6.3f}, std={np.std(fits):6.3f} "
                  f"| Eval: {eval_time:5.1f}s, Gen: {gen_time:5.1f}s")

        # Return best overall
        best_idx = np.argmax([ind.fitness.values[0] for ind in pop])
        best_ind = pop[best_idx]

        return best_ind, stats

    def evaluate_on_test(self, genome: List[float]):
        """Evaluate a genome on held-out TEST set"""
        params = BotParameters(
            entry_weights=np.array(genome[:5]),
            entry_threshold=genome[5],
            position_size=genome[6],
            stop_loss_pct=genome[7],
            take_profit_pct=genome[8],
            holding_bars=int(genome[9]),
            max_concurrent_positions=int(genome[10]),
        )

        bot = TradingBot(params, self.prices_test,
                        self.volumes_test, self.signals_test)
        returns = bot.simulate()

        if len(returns) == 0:
            return {'error': 'No trades'}

        metrics = calculate_metrics(np.array(returns))
        metrics['trades'] = bot.trades
        metrics['num_trades'] = len(bot.trades)

        return metrics


def main():
    """Example: Evolve bots"""
    evolver = BOTEvolver(seed=42)

    # Evolve for 20 generations with population 50
    best_bot, evolution_stats = evolver.evolve(
        pop_size=50,
        generations=20,
        cxpb=0.7,
        mutpb=0.3
    )

    print("\n" + "="*80)
    print("BEST BOT FOUND")
    print("="*80)
    print(f"Genome: {best_bot}")
    print(f"Fitness (Sharpe, -Drawdown): {best_bot.fitness.values}")

    # Test on held-out data
    print("\nTesting on hold-out set...")
    test_metrics = evolver.evaluate_on_test(best_bot)
    print(f"\nTest Metrics:")
    for k, v in test_metrics.items():
        if k != 'trades':
            print(f"  {k}: {v:.4f}")

    return best_bot, evolution_stats


if __name__ == "__main__":
    best_bot, stats = main()
