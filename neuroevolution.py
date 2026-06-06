"""
Neuroevolution - Evolve Neural Network Hyperparameters using DEAP

Evolves training hyperparameters for your bot activity detection transformers.
Architecture stays fixed, but finds optimal learning recipe.

Genome (8 genes):
  - learning_rate (1e-5 to 1e-2, log scale)
  - weight_decay (1e-7 to 1e-3, log scale)
  - label_smoothing (0.0 to 0.3)
  - warmup_epochs (0 to 5, discrete)
  - gradient_clip_norm (0.5 to 2.0)
  - dropout_multiplier (0.5 to 2.0, scales model dropout)
  - scheduler_type (0=linear, 1=cosine, 2=step)
  - batch_size (discrete: 16, 32, 64, 128)
"""

import numpy as np
from typing import Tuple, List, Dict
import sys
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random
import tempfile
import os

# DEAP imports
from deap import base, creator, tools
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, StepLR

sys.path.insert(0, str(Path(__file__).parent))

from loader import TheSetGPU
from model import (
    TransformerEncoder, TemporalFusionTransformer, BayesianTransformer,
    MCDropoutTransformer, MambaEncoder, xLSTMEncoder
)


class HyperparameterEvolver:
    """
    Evolve neural network training hyperparameters
    """

    def __init__(self, model_class: str = "TransformerEncoder", num_workers: int = None, seed: int = 42):
        """
        Args:
            model_class: Which model to evolve ("TransformerEncoder", "TemporalFusionTransformer", etc.)
            num_workers: Number of parallel workers (default: CPU count - 2)
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.model_class_name = model_class
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        # Setup workers
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 2)
        self.num_workers = num_workers

        # Load dataset
        print("[NE] Loading bot activity dataset...")
        self.train_dataset = TheSetGPU(split='train', use_cache=False)
        self.val_dataset = TheSetGPU(split='val', use_cache=False)

        print(f"[NE] Train: {len(self.train_dataset)} sessions")
        print(f"[NE] Val:   {len(self.val_dataset)} sessions")
        print(f"[NE] Using {self.num_workers} workers for parallel evaluation")

        # Model config
        self.model_config = self._get_model_config()

        # Setup DEAP
        self._setup_deap()

    def _get_model_config(self) -> Dict:
        """Get base model configuration from train.py"""
        configs = {
            "TransformerEncoder": {
                "d_model": 128,
                "d_ff": 512,
                "num_layers": 3,
                "num_heads": 8,
                "dropout": 0.1,
            },
            "TemporalFusionTransformer": {
                "d_model": 96,
                "d_ff": 384,
                "num_layers": 2,
                "dropout": 0.1,
            },
            "BayesianTransformer": {
                "d_model": 96,
                "d_ff": 384,
                "num_layers": 2,
                "dropout": 0.05,
            },
            "MCDropoutTransformer": {
                "d_model": 128,
                "d_ff": 512,
                "num_layers": 2,
                "dropout": 0.2,
            },
            "MambaEncoder": {
                "d_model": 128,
                "num_layers": 3,
                "dropout": 0.1,
            },
            "xLSTMEncoder": {
                "d_model": 128,
                "num_layers": 3,
                "dropout": 0.1,
            },
        }
        return configs.get(self.model_class_name, configs["TransformerEncoder"])

    def _setup_deap(self):
        """Configure DEAP framework"""
        if 'FitnessNeuro' in dir(creator):
            del creator.FitnessNeuro
        if 'Individual' in dir(creator):
            del creator.Individual

        # Fitness: maximize F1, minimize loss
        creator.create("FitnessNeuro", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessNeuro)

        self.toolbox = base.Toolbox()

        # Gene definitions (hyperparameter ranges)
        self.toolbox.register("learning_rate", random.uniform, -5, -2)  # 10^x scale
        self.toolbox.register("weight_decay", random.uniform, -7, -3)   # 10^x scale
        self.toolbox.register("label_smoothing", random.uniform, 0.0, 0.3)
        self.toolbox.register("warmup_epochs", random.randint, 0, 5)
        self.toolbox.register("gradient_clip", random.uniform, 0.5, 2.0)
        self.toolbox.register("dropout_multiplier", random.uniform, 0.5, 2.0)
        self.toolbox.register("scheduler_type", random.randint, 0, 2)  # 0=linear, 1=cosine, 2=step
        self.toolbox.register("batch_size", random.choice, [16, 32, 64, 128])

        # Individual
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_hyperparams)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.3)
        self.toolbox.register("mutate", self._mutate_hyperparams)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self) -> creator.Individual:
        """Create one random individual (hyperparameter set)"""
        individual = creator.Individual([
            self.toolbox.learning_rate(),      # 0: log scale LR
            self.toolbox.weight_decay(),       # 1: log scale WD
            self.toolbox.label_smoothing(),    # 2: label smoothing
            self.toolbox.warmup_epochs(),      # 3: warmup epochs
            self.toolbox.gradient_clip(),      # 4: gradient clip norm
            self.toolbox.dropout_multiplier(), # 5: dropout multiplier
            self.toolbox.scheduler_type(),     # 6: scheduler type
            self.toolbox.batch_size(),         # 7: batch size
        ])
        return individual

    def _mutate_hyperparams(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Mutate hyperparameters"""
        for i in range(len(individual)):
            if random.random() < 0.2:  # 20% mutation probability per gene
                if i in [0, 1]:  # Log scale params
                    individual[i] += random.gauss(0, 0.3)
                elif i in [2, 4, 5]:  # Float params
                    individual[i] += random.gauss(0, 0.05)
                elif i == 3:  # Warmup epochs (discrete)
                    individual[i] = max(0, int(individual[i] + random.gauss(0, 1)))
                elif i == 6:  # Scheduler type
                    if random.random() < 0.1:
                        individual[i] = random.randint(0, 2)
                elif i == 7:  # Batch size
                    if random.random() < 0.1:
                        individual[i] = random.choice([16, 32, 64, 128])

        return (individual,)

    def evaluate_hyperparams(self, genome: List[float]) -> Tuple[float, float]:
        """
        Evaluate hyperparameter set by training and validating

        Args:
            genome: [lr, weight_decay, label_smoothing, warmup_epochs, gradient_clip, dropout_mult, scheduler, batch_size]

        Returns: (F1_score, -loss)
        """
        # Parse genome
        lr = 10 ** genome[0]
        weight_decay = 10 ** genome[1]
        label_smoothing = genome[2]
        warmup_epochs = int(genome[3])
        gradient_clip = genome[4]
        dropout_mult = genome[5]
        scheduler_type = int(genome[6])
        batch_size = int(genome[7])

        print(f"[NE] Evaluating: LR={lr:.2e}, WD={weight_decay:.2e}, LS={label_smoothing:.2f}, "
              f"Warmup={warmup_epochs}, GradClip={gradient_clip:.2f}, DO_mult={dropout_mult:.2f}")

        try:
            # Create model
            model_config = self.model_config.copy()
            if 'dropout' in model_config:
                model_config['dropout'] *= dropout_mult

            model_class = globals()[self.model_class_name]
            model = model_class(**model_config).to(self.device)

            # Setup training
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            # Setup scheduler
            if scheduler_type == 0:  # Linear
                scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=10)
            elif scheduler_type == 1:  # Cosine
                scheduler = CosineAnnealingLR(optimizer, T_max=10)
            else:  # Step
                scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

            loss_fn = nn.BCEWithLogitsLoss(label_smoothing=label_smoothing)

            # Train for 5 epochs (quick evaluation)
            best_val_f1 = 0.0
            best_val_loss = float('inf')

            for epoch in range(5):
                # Warmup
                if epoch < warmup_epochs:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * (epoch + 1) / warmup_epochs

                # Train
                model.train()
                train_loss = 0.0
                for sample in self.train_dataset:
                    optimizer.zero_grad()
                    market_data = sample['market_data'].to(self.device)
                    target = sample['target'].unsqueeze(-1).to(self.device).float()

                    logits = model(market_data)
                    loss = loss_fn(logits, target)

                    loss.backward()
                    if gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

                    train_loss += loss.item()

                scheduler.step()

                # Validate
                model.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    val_preds = []
                    val_targets = []

                    for sample in self.val_dataset:
                        market_data = sample['market_data'].to(self.device)
                        target = sample['target'].unsqueeze(-1).to(self.device).float()

                        logits = model(market_data)
                        loss = loss_fn(logits, target)
                        val_loss += loss.item()

                        preds = torch.sigmoid(logits).cpu().numpy()
                        targets = target.cpu().numpy()

                        val_preds.extend((preds > 0.5).astype(int))
                        val_targets.extend(targets.astype(int))

                    val_loss /= len(self.val_dataset)

                    # Calculate F1
                    val_preds = np.array(val_preds).flatten()
                    val_targets = np.array(val_targets).flatten()

                    tp = np.sum((val_preds == 1) & (val_targets == 1))
                    fp = np.sum((val_preds == 1) & (val_targets == 0))
                    fn = np.sum((val_preds == 0) & (val_targets == 1))

                    precision = tp / (tp + fp + 1e-6)
                    recall = tp / (tp + fn + 1e-6)
                    f1 = 2 * precision * recall / (precision + recall + 1e-6)

                    if f1 > best_val_f1:
                        best_val_f1 = f1
                        best_val_loss = val_loss

            return (best_val_f1, -best_val_loss)

        except Exception as e:
            print(f"[NE] Error in evaluation: {e}")
            return (0.0, 0.0)

    def _evaluate_batch(self, genomes: List[List[float]]) -> List[Tuple[float, float]]:
        """Evaluate batch of hyperparameter sets in parallel"""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            fitnesses = list(executor.map(self.evaluate_hyperparams, genomes))
        return fitnesses

    def evolve(self, pop_size: int = 20, generations: int = 10,
               cxpb: float = 0.7, mutpb: float = 0.3) -> Tuple[List, List]:
        """
        Run neuroevolution for hyperparameter optimization

        Args:
            pop_size: Population size
            generations: Number of generations
            cxpb: Crossover probability
            mutpb: Mutation probability

        Returns: (best_hyperparams, stats)
        """
        print(f"\n[NE] Starting neuroevolution: model={self.model_class_name}, "
              f"pop_size={pop_size}, gen={generations}")

        # Create initial population
        pop = self.toolbox.population(n=pop_size)

        # Evaluate initial population (sequential for now, can parallelize)
        print("[NE] Evaluating initial population...")
        fitnesses = [self.evaluate_hyperparams(ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        stats = []

        # Evolution loop
        for gen in range(generations):
            gen_start = time.time()

            # Select
            offspring = self.toolbox.select(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values

            # Evaluate invalid
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = [self.evaluate_hyperparams(ind) for ind in invalid_ind]
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            pop[:] = offspring

            # Track stats
            fits = [ind.fitness.values[0] for ind in pop]
            best_idx = np.argmax(fits)
            best_ind = pop[best_idx]

            stats.append({
                'gen': gen,
                'best_f1': fits[best_idx],
                'avg_f1': np.mean(fits),
                'std_f1': np.std(fits),
            })

            gen_time = time.time() - gen_start
            print(f"[NE Gen {gen}] Best F1: {fits[best_idx]:.4f}, Avg: {np.mean(fits):.4f}, "
                  f"Std: {np.std(fits):.4f} | Time: {gen_time:.1f}s")

        # Return best
        best_idx = np.argmax([ind.fitness.values[0] for ind in pop])
        best_ind = pop[best_idx]

        return best_ind, stats


def format_hyperparams(genome: List[float]) -> Dict[str, float]:
    """Convert genome to readable hyperparameter dict"""
    scheduler_map = {0: "linear", 1: "cosine", 2: "step"}

    return {
        'learning_rate': 10 ** genome[0],
        'weight_decay': 10 ** genome[1],
        'label_smoothing': genome[2],
        'warmup_epochs': int(genome[3]),
        'gradient_clip_norm': genome[4],
        'dropout_multiplier': genome[5],
        'scheduler_type': scheduler_map[int(genome[6])],
        'batch_size': int(genome[7]),
    }


if __name__ == "__main__":
    from model import TransformerEncoder

    # Test neuroevolution
    evolver = HyperparameterEvolver(model_class="TransformerEncoder", seed=42)

    print("\nStarting neuroevolution (this will take a while)...")
    best_hyperparams, stats = evolver.evolve(
        pop_size=10,
        generations=5,
        cxpb=0.7,
        mutpb=0.3
    )

    print("\n" + "="*80)
    print("BEST HYPERPARAMETERS FOUND")
    print("="*80)

    params = format_hyperparams(best_hyperparams)
    for k, v in params.items():
        print(f"  {k}: {v}")

    print(f"\nBest F1: {best_hyperparams.fitness.values[0]:.4f}")
    print(f"Best Loss: {-best_hyperparams.fitness.values[1]:.4f}")
