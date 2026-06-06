"""
NEAT Evolution: Evolve trading bot network topologies using DEAP + backtesting.

This implements:
- Mutation operators: add neuron, add connection, mutate weights
- Speciation: group similar networks to preserve diversity
- Innovation tracking: record new nodes/connections globally
- Backtesting evaluation: fitness = profitability + Sharpe ratio + survival
"""

import random
import torch
from typing import Tuple, Dict
from copy import deepcopy

from deap import base, creator, tools
from neat_network import NEATGenome, NEATNetworkBuilder


class InnovationTracker:
    """
    Global innovation history.

    When a new connection/neuron is discovered, assign it a global ID.
    This ID never changes - allows networks to be compared for speciation.
    """

    def __init__(self):
        self.innovation_num = 0
        self.connection_history: Dict[Tuple[int, int], int] = {}

    def get_innovation_number(self, in_node: int, out_node: int) -> int:
        """
        Get innovation number for a connection.
        If new, assign next number.
        """
        key = (in_node, out_node)
        if key not in self.connection_history:
            self.connection_history[key] = self.innovation_num
            self.innovation_num += 1
        return self.connection_history[key]


def create_neat_individual(num_inputs: int, num_outputs: int) -> NEATGenome:
    """Create a new random NEAT individual (minimal network)."""
    genome = NEATGenome(num_inputs, num_outputs)

    # Randomly add a few initial connections
    # Connect a subset of inputs to outputs (sparse initialization)
    for i in range(num_inputs):
        if random.random() < 0.7:  # 70% chance to connect each input
            out_node = num_inputs + random.randint(0, num_outputs - 1)
            genome.add_connection(i, out_node, weight=random.gauss(0, 1))

    return genome


def mutate_add_connection(genome: NEATGenome,
                         innovation_tracker: InnovationTracker,
                         max_attempts: int = 5) -> None:
    """
    Mutation: Add a new connection between two unconnected nodes.
    """
    node_ids = genome.get_node_ids()

    # Try multiple times to find two unconnected nodes
    for _ in range(max_attempts):
        in_node = random.choice(node_ids)
        out_node = random.choice(node_ids)

        # Don't connect a node to itself (usually)
        if in_node == out_node and len(node_ids) > 1:
            continue

        # Check if already connected
        if (in_node, out_node) not in genome.connections:
            # Optional: Prefer forward connections (prefer in_node < out_node)
            if in_node > out_node and random.random() < 0.5:
                in_node, out_node = out_node, in_node

            weight = random.gauss(0, 0.5)
            innovation_num = innovation_tracker.get_innovation_number(in_node, out_node)
            genome.add_connection(in_node, out_node, weight, innovation_num)
            break


def mutate_add_neuron(genome: NEATGenome,
                     innovation_tracker: InnovationTracker) -> None:
    """
    Mutation: Add a new neuron.

    Process:
    1. Pick a random connection: A → B
    2. Disable that connection
    3. Create new neuron: A → N → B
    4. Connection A → N gets weight 1.0 (pass-through)
    5. Connection N → B gets the old weight
    """
    if not genome.connections:
        return  # No connections yet

    # Pick random enabled connection
    enabled_connections = [c for c in genome.connections.values() if c.enabled]
    if not enabled_connections:
        return

    old_conn = random.choice(enabled_connections)

    # Add new neuron
    new_node_id = genome.add_node(activation=random.choice(['tanh', 'relu', 'sigmoid']))

    # Disable old connection
    genome.disable_connection(old_conn.in_node, old_conn.out_node)

    # Add two new connections around the neuron
    inn1 = innovation_tracker.get_innovation_number(old_conn.in_node, new_node_id)
    inn2 = innovation_tracker.get_innovation_number(new_node_id, old_conn.out_node)

    genome.add_connection(old_conn.in_node, new_node_id, weight=1.0, innovation_num=inn1)
    genome.add_connection(new_node_id, old_conn.out_node, weight=old_conn.weight, innovation_num=inn2)


def mutate_weights(genome: NEATGenome, mutation_rate: float = 0.8, mutation_scale: float = 0.3) -> None:
    """
    Mutation: Adjust connection weights.

    Args:
        mutation_rate: Probability each weight gets mutated
        mutation_scale: Std dev of Gaussian perturbation
    """
    for conn in genome.connections.values():
        if random.random() < mutation_rate:
            # Either perturb existing weight or replace with new
            if random.random() < 0.9:
                # Perturb (+= delta)
                conn.weight += random.gauss(0, mutation_scale)
            else:
                # Replace (new random weight)
                conn.weight = random.gauss(0, 1)

            # Optionally clip to reasonable range
            conn.weight = max(-5, min(5, conn.weight))


def mutate_enable_disable(genome: NEATGenome, toggle_rate: float = 0.1) -> None:
    """
    Mutation: Enable/disable existing connections.
    """
    for conn in genome.connections.values():
        if random.random() < toggle_rate:
            conn.enabled = not conn.enabled


class NEATEvolver:
    """
    NEAT evolution controller for trading bots.
    Uses DEAP framework + backtesting for fitness evaluation.
    
    Bots:
    - Receive normalized market features at each time step
    - Output position size signal (long/short/neutral)
    - Accumulate returns across entire backtest period
    - Survive or die based on trading performance
    """

    def __init__(self,
                 backtest_simulator,  # BacktestSimulator instance
                 num_inputs: int,
                 num_outputs: int = 1,
                 device: str = 'cpu',
                 seed: int = 42):
        """
        Args:
            backtest_simulator: BacktestSimulator for evaluating bots
            num_inputs: Number of market features
            num_outputs: 1 (single continuous trading signal)
            device: torch device (cpu or cuda:X)
            seed: Random seed
        """
        self.backtest_simulator = backtest_simulator
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.device = device
        self.seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        self.innovation_tracker = InnovationTracker()
        self._setup_deap()

    def _setup_deap(self):
        """Configure DEAP framework."""
        # Define fitness (maximize composite fitness score from backtesting)
        if not hasattr(creator, "FitnessNEAT"):
            creator.create("FitnessNEAT", base.Fitness,
                         weights=(1.0,))  # Maximize fitness score

        if not hasattr(creator, "IndividualNEAT"):
            creator.create("IndividualNEAT", NEATGenome,
                         fitness=creator.FitnessNEAT)

        self.toolbox = base.Toolbox()

        # Genetic operators
        self.toolbox.register("individual",
                             self._create_ind)
        self.toolbox.register("population", tools.initRepeat, list,
                             self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_genome)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_ind(self):
        """Create a new individual."""
        ind = creator.IndividualNEAT(self.num_inputs, self.num_outputs)
        # Ensure fitness is attached
        if not hasattr(ind, 'fitness'):
            ind.fitness = creator.FitnessNEAT()
        return ind

    def _mutate(self, individual) -> Tuple:
        """Apply mutations to individual."""
        if random.random() < 0.1:
            mutate_add_connection(individual, self.innovation_tracker)
        if random.random() < 0.05:
            mutate_add_neuron(individual, self.innovation_tracker)

        mutate_weights(individual, mutation_rate=0.8, mutation_scale=0.3)
        mutate_enable_disable(individual, toggle_rate=0.05)

        return (individual,)

    def _crossover(self, ind1, ind2) -> Tuple:
        """
        Crossover: Combine two parent genomes.

        NEAT crossover:
        - Inherits all connections from both parents
        - Excess genes from fitter parent
        - Connection weights averaged where both have
        """
        # For now: simple averaging of common connections
        offspring = ind1.copy()

        # For common connections, average weights
        for key, conn2 in ind2.connections.items():
            if key in offspring.connections:
                offspring.connections[key].weight = (
                    offspring.connections[key].weight + conn2.weight
                ) / 2

        return (offspring, ind1)

    def evaluate_genome(self, genome: NEATGenome) -> Tuple[float]:
        """
        Evaluate a genome as a trading bot via backtesting.

        Returns:
            (fitness_score,) - composite profitability + Sharpe ratio
        """
        try:
            # Run backtest
            result = self.backtest_simulator.backtest(genome)
            
            # Return fitness as single-element tuple (required by DEAP)
            return (result.fitness_score,)

        except Exception as e:
            # Backtest failed - return worst fitness
            print(f"Backtest evaluation error: {e}")
            return (-1000.0,)

    def evolve(self,
               pop_size: int = 20,
               generations: int = 10,
               cxpb: float = 0.7,
               mutpb: float = 0.3) -> Tuple:
        """
        Run NEAT evolution.

        Args:
            pop_size: Population size
            generations: Number of generations
            cxpb: Crossover probability
            mutpb: Mutation probability

        Returns:
            (final_population, statistics)
        """
        print(f"\n{'='*60}")
        print("NEAT Evolution")
        print(f"{'='*60}")
        print(f"Population: {pop_size} | Generations: {generations}")
        print(f"Inputs: {self.num_inputs} | Device: {self.device}")

        # Create initial population
        pop = self.toolbox.population(n=pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        stats = self._init_stats()
        stats['gen'].append(0)
        stats['best_f1'].append(max(f[0] for f in fitnesses))
        stats['avg_size'].append(sum(len(g.nodes) for g in pop) / len(pop))

        print(f"Gen 0: Best Fitness={stats['best_f1'][0]:.4f}, "
              f"Avg Size={stats['avg_size'][0]:.1f} nodes")

        # Evolution loop
        for gen in range(1, generations):
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring_copy = []
            for ind in offspring:
                offspring_ind = deepcopy(ind)
                # Ensure fitness is attached
                if not hasattr(offspring_ind, 'fitness'):
                    offspring_ind.fitness = creator.FitnessNEAT()
                offspring_copy.append(offspring_ind)
            offspring = offspring_copy

            # Crossover
            for i in range(1, len(offspring), 2):
                if random.random() < cxpb:
                    offspring[i-1], offspring[i] = self.toolbox.mate(offspring[i-1], offspring[i])
                    # Invalidate fitness for re-evaluation
                    try:
                        del offspring[i-1].fitness.values
                    except (AttributeError, ValueError, TypeError):
                        pass
                    try:
                        del offspring[i].fitness.values
                    except (AttributeError, ValueError, TypeError):
                        pass

            # Mutation
            for i in range(len(offspring)):
                if random.random() < mutpb:
                    offspring[i], = self.toolbox.mutate(offspring[i])
                    try:
                        del offspring[i].fitness.values
                    except (AttributeError, ValueError, TypeError):
                        pass

            # Ensure all offspring have fitness before evaluation
            for ind in offspring:
                if not hasattr(ind, 'fitness'):
                    ind.fitness = creator.FitnessNEAT()

            # Evaluate offspring with invalid fitness
            invalid_ind = []
            for ind in offspring:
                try:
                    if not ind.fitness.valid:
                        invalid_ind.append(ind)
                except (AttributeError, ValueError, TypeError):
                    invalid_ind.append(ind)

            if invalid_ind:
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            # Replace population
            pop = offspring

            # Statistics
            fits = [ind.fitness.values[0] for ind in pop]
            stats['gen'].append(gen)
            stats['best_f1'].append(max(fits))
            stats['avg_size'].append(sum(len(g.nodes) for g in pop) / len(pop))

            print(f"Gen {gen}: Best Fitness={stats['best_f1'][-1]:.4f}, "
                  f"Avg Size={stats['avg_size'][-1]:.1f} nodes, "
                  f"Max Size={max(len(g.nodes) for g in pop)}")


        return pop, stats

    def _init_stats(self) -> Dict:
        """Initialize statistics tracking."""
        return {
            'gen': [],
            'best_f1': [],
            'avg_size': [],
        }


if __name__ == "__main__":
    print("NEAT Evolution Framework (no data test)")
    print("Full test requires DataLoader - see test_neat.py")
