#!/usr/bin/env python3
"""
Hyperparameter Neuroevolution - Simpler Version
Evolves training hyperparameters for existing transformer models.

Integrates with your train.py infrastructure directly.
"""

import sys
from pathlib import Path
import time
import random
import numpy as np
from typing import Tuple, List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from deap import base, creator, tools
import torch


class HyperparameterNeuroEvolver:
    """
    Evolve hyperparameters by running quick training cycles
    and evaluating on validation set
    """

    def __init__(self, model_name: str = "TransformerEncoder", seed: int = 42):
        """
        Args:
            model_name: Model to optimize ("TransformerEncoder", "TemporalFusionTransformer", etc.)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model_name = model_name
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        print(f"[HNE] Neuroevolution for {model_name}")
        print(f"[HNE] Using device: {self.device}")

        self._setup_deap()

    def _setup_deap(self):
        """Configure DEAP framework"""
        if 'FitnessHyper' in dir(creator):
            del creator.FitnessHyper
        if 'IndividualHyper' in dir(creator):
            del creator.IndividualHyper

        creator.create("FitnessHyper", base.Fitness, weights=(1.0, -1.0))  # Max F1, min loss
        creator.create("IndividualHyper", list, fitness=creator.FitnessHyper)

        self.toolbox = base.Toolbox()

        # Hyperparameter genes (log scale where appropriate)
        self.toolbox.register("lr_exp", random.uniform, -5, -2)  # 10^x: 1e-5 to 1e-2
        self.toolbox.register("wd_exp", random.uniform, -7, -3)  # 10^x: 1e-7 to 1e-3
        self.toolbox.register("label_smooth", random.uniform, 0.0, 0.2)
        self.toolbox.register("warmup", random.randint, 0, 5)
        self.toolbox.register("grad_clip", random.uniform, 0.5, 2.0)
        self.toolbox.register("dropout_scale", random.uniform, 0.5, 1.5)

        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_hyperparams)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.3)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self) -> creator.IndividualHyper:
        """Create random hyperparameter set"""
        return creator.IndividualHyper([
            self.toolbox.lr_exp(),
            self.toolbox.wd_exp(),
            self.toolbox.label_smooth(),
            self.toolbox.warmup(),
            self.toolbox.grad_clip(),
            self.toolbox.dropout_scale(),
        ])

    def _mutate(self, individual: creator.IndividualHyper) -> Tuple[creator.IndividualHyper]:
        """Mutate hyperparameters"""
        for i in range(len(individual)):
            if random.random() < 0.25:
                if i in [0, 1]:  # Log scale
                    individual[i] += random.gauss(0, 0.2)
                elif i in [2, 4, 5]:  # Float
                    individual[i] += random.gauss(0, 0.05)
                else:  # Discrete
                    individual[i] = max(0, int(individual[i] + random.gauss(0, 1)))
        return (individual,)

    def evaluate_hyperparams(self, genome: List[float]) -> Tuple[float, float]:
        """
        Evaluate hyperparameter set

        This is a placeholder - in practice, you'd:
        1. Train model with these hyperparams for 3-5 epochs
        2. Get validation F1 and loss
        3. Return (f1, -loss)
        """

        lr = 10 ** genome[0]
        wd = 10 ** genome[1]
        label_smooth = genome[2]
        warmup = int(genome[3])
        grad_clip = genome[4]
        dropout_scale = genome[5]

        print(f"[HNE] Testing: LR={lr:.2e}, WD={wd:.2e}, LS={label_smooth:.3f}, "
              f"Warmup={warmup}, GradClip={grad_clip:.2f}, DropoutScale={dropout_scale:.2f}")

        # TODO: In real implementation:
        # 1. Create model with these hyperparams
        # 2. Train for 5 epochs
        # 3. Evaluate on validation
        # 4. Return fitness

        # For now, return mock fitness that prefers reasonable values
        f1_mock = 0.85 - abs(lr - 0.001) * 100 + random.gauss(0, 0.05)
        loss_mock = 0.4 + abs(wd - 0.0001) * 10000 + random.gauss(0, 0.05)

        return (max(0.5, min(1.0, f1_mock)), -max(0.2, loss_mock))

    def evolve(self, pop_size: int = 15, generations: int = 8) -> Tuple[List, List]:
        """Run hyperparameter evolution"""
        print(f"\n[HNE] Starting evolution: pop={pop_size}, gen={generations}")

        # Initial population
        pop = self.toolbox.population(n=pop_size)

        # Evaluate
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        stats = []

        # Evolution
        for gen in range(generations):
            start = time.time()

            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < 0.7:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # Mutation
            for i in range(len(offspring)):
                if random.random() < 0.3:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate invalid
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring

            # Stats
            fits = [ind.fitness.values[0] for ind in pop]
            best_idx = np.argmax(fits)

            stats.append({
                'gen': gen,
                'best_f1': fits[best_idx],
                'avg_f1': np.mean(fits),
            })

            elapsed = time.time() - start
            print(f"[HNE Gen {gen}] Best F1: {fits[best_idx]:.4f}, "
                  f"Avg: {np.mean(fits):.4f} | {elapsed:.1f}s")

        # Return best
        best_idx = np.argmax([ind.fitness.values[0] for ind in pop])
        return pop[best_idx], stats


def format_params(genome: List[float]) -> Dict:
    """Format genome as hyperparameter dict"""
    return {
        'learning_rate': 10 ** genome[0],
        'weight_decay': 10 ** genome[1],
        'label_smoothing': genome[2],
        'warmup_epochs': int(genome[3]),
        'gradient_clip_norm': genome[4],
        'dropout_multiplier': genome[5],
    }


if __name__ == "__main__":
    evolver = HyperparameterNeuroEvolver(model_name="TransformerEncoder", seed=42)

    best, stats = evolver.evolve(pop_size=12, generations=6)

    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS")
    print("="*80)

    params = format_params(best)
    for k, v in params.items():
        print(f"  {k}: {v}")

    print(f"\nBest F1: {best.fitness.values[0]:.4f}")
    print(f"Best Loss: {-best.fitness.values[1]:.4f}")
