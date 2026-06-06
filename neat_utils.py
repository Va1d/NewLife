"""
NEAT Utilities: Visualize, compare, and analyze evolved networks.
"""

from typing import Dict
from neat_network import NEATGenome


def genome_to_string(genome: NEATGenome) -> str:
    """Human-readable genome visualization."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append("NEAT Genome")
    lines.append(f"{'='*60}")

    lines.append(f"\nNetwork Size: {len(genome.nodes)} nodes, "
                f"{sum(1 for c in genome.connections.values() if c.enabled)} active connections")

    # Organize nodes by type
    input_nodes = [n for n in genome.nodes.values() if n.node_type == "input"]
    hidden_nodes = [n for n in genome.nodes.values() if n.node_type == "hidden"]
    output_nodes = [n for n in genome.nodes.values() if n.node_type == "output"]

    if input_nodes:
        lines.append("\nInput Nodes:")
        for n in sorted(input_nodes, key=lambda x: x.id):
            lines.append(f"  {n.id}: {n.node_type}")

    if hidden_nodes:
        lines.append(f"\nHidden Nodes ({len(hidden_nodes)}):")
        for n in sorted(hidden_nodes, key=lambda x: x.id):
            lines.append(f"  {n.id}: {n.activation}")

    if output_nodes:
        lines.append("\nOutput Nodes:")
        for n in sorted(output_nodes, key=lambda x: x.id):
            lines.append(f"  {n.id}: {n.node_type} ({n.activation})")

    # List connections
    enabled_conns = [c for c in genome.connections.values() if c.enabled]
    disabled_conns = [c for c in genome.connections.values() if not c.enabled]

    if enabled_conns:
        lines.append(f"\nActive Connections ({len(enabled_conns)}):")
        for conn in sorted(enabled_conns, key=lambda c: (c.in_node, c.out_node)):
            lines.append(f"  {conn.in_node} → {conn.out_node}: w={conn.weight:+.3f}")

    if disabled_conns:
        lines.append(f"\nDisabled Connections ({len(disabled_conns)}):")
        for conn in sorted(disabled_conns, key=lambda c: (c.in_node, c.out_node)):
            lines.append(f"  {conn.in_node} → {conn.out_node}: w={conn.weight:+.3f} [disabled]")

    # Check for recurrent connections
    recurrent = []
    for conn in enabled_conns:
        if conn.in_node > conn.out_node:  # Some heuristic for recurrence
            recurrent.append(conn)

    if recurrent:
        lines.append(f"\n⚠ Recurrent Connections ({len(recurrent)}):")
        for conn in recurrent:
            lines.append(f"  {conn.in_node} → {conn.out_node}: Potential feedback loop")

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def compare_genomes(g1: NEATGenome, g2: NEATGenome) -> Dict:
    """
    Compare two genomes.

    Returns dict with:
    - size_difference: node count difference
    - connection_diff: connections only in g1, only in g2, shared
    - weight_correlation: correlation of shared connection weights
    """
    # Size comparison
    size_diff = len(g2.nodes) - len(g1.nodes)

    # Connection comparison
    g1_keys = set(g1.connections.keys())
    g2_keys = set(g2.connections.keys())

    only_g1 = g1_keys - g2_keys
    only_g2 = g2_keys - g1_keys
    shared = g1_keys & g2_keys

    # Weight correlation for shared connections
    shared_weights_1 = [g1.connections[key].weight for key in shared]
    shared_weights_2 = [g2.connections[key].weight for key in shared]

    if shared_weights_1 and shared_weights_2:
        import numpy as np
        correlation = np.corrcoef(shared_weights_1, shared_weights_2)[0, 1]
    else:
        correlation = 0.0

    return {
        'size_diff': size_diff,
        'unique_to_g1': len(only_g1),
        'unique_to_g2': len(only_g2),
        'shared_connections': len(shared),
        'weight_correlation': correlation,
    }


def calculate_complexity(genome: NEATGenome) -> float:
    """
    Simple complexity metric: (nodes * connections) / (# of possible connections)

    Values: 0.0 (minimal) to 1.0 (fully connected)
    """
    n = len(genome.nodes)
    active = sum(1 for c in genome.connections.values() if c.enabled)
    max_possible = n * (n - 1)  # Directed acyclic (roughly)

    if max_possible == 0:
        return 0.0

    return active / max_possible


def has_recurrent_connection(genome: NEATGenome) -> bool:
    """Check if network has any recurrent (feedback) connections."""
    # Simple heuristic: check if any connection goes "backwards"
    # (higher ID to lower ID, roughly indicating feedback)
    for conn in genome.connections.values():
        if conn.enabled and conn.in_node > conn.out_node:
            return True
    return False


def network_statistics(genomes: list) -> Dict:
    """
    Analyze a population of genomes.

    Returns statistics about the population.
    """
    if not genomes:
        return {}

    sizes = [len(g.nodes) for g in genomes]
    active_conns = [sum(1 for c in g.connections.values() if c.enabled) for g in genomes]
    complexities = [calculate_complexity(g) for g in genomes]
    recurrent_count = sum(1 for g in genomes if has_recurrent_connection(g))

    return {
        'population_size': len(genomes),
        'avg_nodes': sum(sizes) / len(sizes),
        'max_nodes': max(sizes),
        'min_nodes': min(sizes),
        'avg_connections': sum(active_conns) / len(active_conns),
        'avg_complexity': sum(complexities) / len(complexities),
        'networks_with_recurrence': recurrent_count,
        'recurrence_percentage': 100 * recurrent_count / len(genomes),
    }


if __name__ == "__main__":
    from neat_network import NEATGenome

    # Test: Create a genome and show visualization
    print("NEAT Utilities Demo")
    print("=" * 60)

    genome = NEATGenome(num_inputs=2, num_outputs=1)

    # Add connections and nodes
    genome.add_connection(0, 2, weight=0.5)
    h1 = genome.add_node(activation="tanh")
    genome.add_connection(1, h1, weight=0.8)
    genome.add_connection(h1, 2, weight=0.4)
    genome.add_connection(2, h1, weight=0.1)  # Recurrent!

    print(genome_to_string(genome))
    print(f"Complexity: {calculate_complexity(genome):.3f}")
    print(f"Has recurrent: {has_recurrent_connection(genome)}")
