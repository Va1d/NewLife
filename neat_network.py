"""
NEAT Network: Encode and build neural networks from evolved genomes.

NEAT = Neuroevolution of Augmenting Topologies
- Networks evolve topology (connections, neurons)
- Not fixed layers like traditional architectures
- Topology can be sparse, recurrent, messy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class NEATNode:
    """Represents a neuron in the network."""

    def __init__(self, node_id: int, node_type: str, activation: str = "relu"):
        """
        Args:
            node_id: Unique identifier for this neuron
            node_type: 'input', 'hidden', or 'output'
            activation: Activation function type
                        ('relu', 'tanh', 'sigmoid', 'linear', 'elu', 'swish')
        """
        self.id = node_id
        self.node_type = node_type
        self.activation = activation
        self.bias = 0.0

    def __repr__(self):
        return f"Node({self.id}, {self.node_type}, {self.activation})"


class NEATConnection:
    """Represents a connection between two neurons."""

    def __init__(self,
                 in_node: int,
                 out_node: int,
                 weight: float = 0.0,
                 enabled: bool = True,
                 innovation_number: int = 0):
        """
        Args:
            in_node: Source neuron ID
            out_node: Target neuron ID
            weight: Connection strength
            enabled: Whether this connection is active
            innovation_number: Global innovation ID (for speciation)
        """
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number

    def key(self) -> Tuple[int, int]:
        """Unique identifier for this connection (order matters!)."""
        return (self.in_node, self.out_node)

    def __repr__(self):
        status = "✓" if self.enabled else "✗"
        return f"Conn({self.in_node}→{self.out_node}, w={self.weight:.2f}) {status}"


class NEATGenome:
    """
    Genome = Complete specification of a network.

    Encodes:
    - Neurons (nodes): type, activation function
    - Connections: source, target, weight, enabled status
    - Global IDs: for comparing networks (speciation)
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        self.nodes: Dict[int, NEATNode] = {}
        self.connections: Dict[Tuple[int, int], NEATConnection] = {}

        # Initialize with input and output nodes only
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.next_node_id = num_inputs + num_outputs

        # Create input nodes
        for i in range(num_inputs):
            self.nodes[i] = NEATNode(i, "input", "linear")

        # Create output nodes
        for i in range(num_outputs):
            node_id = num_inputs + i
            self.nodes[node_id] = NEATNode(node_id, "output", "tanh")

    def get_node_ids(self) -> List[int]:
        """Get all node IDs in order."""
        return sorted(self.nodes.keys())

    def get_connection_genes(self) -> List[NEATConnection]:
        """Get all connections (sorted for consistency)."""
        return sorted(self.connections.values(),
                     key=lambda c: (c.in_node, c.out_node))

    def add_connection(self,
                      in_node: int,
                      out_node: int,
                      weight: float = 0.0,
                      innovation_num: int = 0) -> bool:
        """
        Add a connection between two nodes.

        Returns:
            True if added, False if already exists
        """
        key = (in_node, out_node)
        if key in self.connections:
            return False

        self.connections[key] = NEATConnection(
            in_node, out_node, weight, True, innovation_num
        )
        return True

    def add_node(self, activation: str = "tanh") -> int:
        """
        Add a new hidden node.

        Returns:
            ID of new node
        """
        node_id = self.next_node_id
        self.nodes[node_id] = NEATNode(node_id, "hidden", activation)
        self.next_node_id += 1
        return node_id

    def enable_connection(self, in_node: int, out_node: int):
        """Re-enable a disabled connection."""
        key = (in_node, out_node)
        if key in self.connections:
            self.connections[key].enabled = True

    def disable_connection(self, in_node: int, out_node: int):
        """Disable (but don't delete) a connection."""
        key = (in_node, out_node)
        if key in self.connections:
            self.connections[key].enabled = False

    def copy(self) -> "NEATGenome":
        """Create a deep copy of this genome."""
        new_genome = NEATGenome(self.num_inputs, self.num_outputs)

        # Copy nodes
        for node_id, node in self.nodes.items():
            if node.node_type == "hidden":
                new_genome.nodes[node_id] = NEATNode(
                    node.id, node.node_type, node.activation
                )
        new_genome.next_node_id = self.next_node_id

        # Copy connections
        for (in_id, out_id), conn in self.connections.items():
            new_conn = NEATConnection(
                conn.in_node,
                conn.out_node,
                conn.weight,
                conn.enabled,
                conn.innovation_number
            )
            new_genome.connections[(in_id, out_id)] = new_conn

        return new_genome

    def get_network_size(self) -> Tuple[int, int]:
        """Return (num_nodes, num_enabled_connections)."""
        num_enabled = sum(1 for c in self.connections.values() if c.enabled)
        return (len(self.nodes), num_enabled)


class NEATNetworkBuilder:
    """Convert a NEAT genome to a PyTorch neural network."""

    ACTIVATIONS = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'linear': nn.Identity(),
        'elu': nn.ELU(),
        'swish': nn.SiLU(),
    }

    @staticmethod
    def build_network(genome: NEATGenome, device: str = 'cpu') -> nn.Module:
        """
        Build a PyTorch model from a NEAT genome.

        The network is dynamically constructed based on the genome topology.

        Args:
            genome: NEAT genome specification
            device: 'cpu' or 'cuda:X'

        Returns:
            PyTorch nn.Module that can be used for training
        """
        return NEATModule(genome, device)


class NEATModule(nn.Module):
    """Dynamic PyTorch module built from NEAT genome."""

    def __init__(self, genome: NEATGenome, device: str = 'cpu'):
        super().__init__()
        self.genome = genome
        self.device = device

        # Build weight matrices and biases dynamically
        self._build_connections()
        self._build_activations()

        self.to(device)

    def _build_connections(self):
        """Create weight matrix for enabled connections."""
        # Map node IDs to indices for matrix operations
        self.node_indices = {nid: i for i, nid in enumerate(self.genome.get_node_ids())}
        num_nodes = len(self.node_indices)

        # Initialize weight matrix as a parameter (learnable)
        weight_data = torch.zeros(num_nodes, num_nodes, device=self.device)

        # Populate weights from enabled connections
        for conn in self.genome.get_connection_genes():
            if conn.enabled:
                in_idx = self.node_indices[conn.in_node]
                out_idx = self.node_indices[conn.out_node]
                weight_data[in_idx, out_idx] = conn.weight

        self.weight_matrix = nn.Parameter(weight_data, requires_grad=True)

        # Biases for each node
        bias_data = torch.zeros(num_nodes, device=self.device)
        self.bias_vector = nn.Parameter(bias_data, requires_grad=True)

    def _build_activations(self):
        """Store activation function for each node."""
        self.activations = {}
        for node in self.genome.nodes.values():
            if node.activation in NEATNetworkBuilder.ACTIVATIONS:
                self.activations[node.id] = NEATNetworkBuilder.ACTIVATIONS[node.activation]
            else:
                self.activations[node.id] = nn.ReLU()  # Default fallback

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the NEAT network.

        Implements: output = activation(input @ weight_matrix + bias)

        For recurrent networks, we do multiple iterations to let signals propagate.

        Args:
            x: Input tensor (batch_size, num_inputs)

        Returns:
            Output tensor (batch_size, num_outputs)
        """
        batch_size = x.shape[0]
        num_nodes = len(self.node_indices)

        # Initialize neuron states (one per batch element)
        # states[i] = output value of node i
        states = torch.zeros(batch_size, num_nodes, device=self.device)

        # Set input node values
        for i in range(self.genome.num_inputs):
            idx = self.node_indices[i]
            states[:, idx] = x[:, i]

        # Forward propagation with 3 iterations for recurrent networks
        # (allows feedback loops to stabilize)
        num_iterations = 3
        for _ in range(num_iterations):
            new_states = states.clone()

            # For each node, compute: activation(sum of inputs)
            for node_id, node in self.genome.nodes.items():
                if node.node_type == "input":
                    continue  # Inputs are fixed

                node_idx = self.node_indices[node_id]

                # Gather inputs to this node
                node_input = torch.zeros(batch_size, device=self.device)
                for _, in_idx in self.node_indices.items():
                    node_input += states[:, in_idx] * self.weight_matrix[in_idx, node_idx]

                # Apply activation function
                activation_fn = self.activations.get(node_id, nn.ReLU())
                new_states[:, node_idx] = activation_fn(node_input + self.bias_vector[node_idx])

            states = new_states

        # Extract output nodes
        output_indices = [
            self.node_indices[self.genome.num_inputs + i]
            for i in range(self.genome.num_outputs)
        ]
        output = states[:, output_indices]

        return output

    def get_topology_string(self) -> str:
        """Human-readable network topology."""
        lines = []
        lines.append(f"NEAT Network: {len(self.genome.nodes)} nodes, "
                    f"{sum(1 for c in self.genome.connections.values() if c.enabled)} active connections")

        for node_id in sorted(self.node_indices.keys()):
            node = self.genome.nodes[node_id]
            lines.append(f"  Node {node_id}: {node.node_type} ({node.activation})")

        lines.append("  Connections:")
        for conn in self.genome.get_connection_genes():
            if conn.enabled:
                lines.append(f"    {conn.in_node} → {conn.out_node}: w={conn.weight:.3f}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test: Create a simple NEAT genome and build network
    print("=" * 60)
    print("NEAT Network Test")
    print("=" * 60)

    # Create genome: 2 inputs, 1 output
    genome = NEATGenome(num_inputs=2, num_outputs=1)

    # Add some connections
    genome.add_connection(0, 2, weight=0.5)    # input 0 → output
    genome.add_connection(1, 2, weight=-0.3)   # input 1 → output

    # Add hidden node
    h1 = genome.add_node(activation="tanh")
    genome.add_connection(0, h1, weight=0.8)   # input 0 → hidden
    genome.add_connection(h1, 2, weight=0.4)   # hidden → output

    # Add recurrent connection (feedback loop!)
    genome.add_connection(2, h1, weight=0.1)   # output → hidden (recurrent!)

    print("\nGenome Structure:")
    print(f"Nodes: {genome.get_node_ids()}")
    print(f"Connections: {len(genome.connections)}")

    # Build network
    network = NEATNetworkBuilder.build_network(genome, device='cpu')
    print("\nNetwork Topology:")
    print(network.get_topology_string())

    # Test forward pass
    test_input = torch.tensor([[0.5, -0.3], [0.1, 0.9]], dtype=torch.float32)
    output = network(test_input)
    print(f"\nTest Input:\n{test_input}")
    print(f"Network Output:\n{output}")
    print("\n✓ NEAT network working!")
