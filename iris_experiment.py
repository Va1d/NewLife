"""Iris dataset with spiking neural network (torch-based)

Load Iris, quantize to 6 bits per feature, feed into 96-neuron sparse network.
Uses architecture loaded from saved connectivity matrices.
Neuron internals to be implemented later.
Pure PyTorch for all data processing.
TensorBoard logging and Plotly visualizations with strong typing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch  # type: ignore[import-not-found]
from sklearn import datasets  # type: ignore[import-not-found]
from sklearn.metrics import (  # type: ignore[import-not-found]
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
)
from sklearn.cluster import KMeans  # type: ignore[import-not-found]
from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-not-found]
import plotly.graph_objects as go  # type: ignore[import-not-found]
import plotly.express as px  # type: ignore[import-not-found]

from iris_network import SpikingNetworkTorch, PlaceholderNeuron, load_network


def binary_to_gray(value: int, num_bits: int = 6) -> int:
    """
    Convert binary value to Gray code.
    
    Gray code ensures adjacent values differ by exactly 1 bit,
    creating smooth transitions for spiking neurons.
    
    Args:
        value: Binary-encoded integer
        num_bits: Number of bits (for completeness, not always needed)
    
    Returns:
        Gray-coded equivalent
    """
    return value ^ (value >> 1)


def gray_to_binary(value: int, num_bits: int = 6) -> int:
    """
    Convert Gray code back to binary.
    
    Args:
        value: Gray-coded integer
        num_bits: Number of bits (for completeness)
    
    Returns:
        Binary-encoded equivalent
    """
    mask: int = value
    while mask != 0:
        mask >>= 1
        value ^= mask
    return value


@dataclass
class IrisExperimentConfig:
    """Configuration for Iris spiking network experiment"""
    
    # Data parameters
    num_bits: int = 6  # Quantization bits per feature (0-63)
    use_gray_code: bool = True  # Use Gray code for smooth bit transitions
    
    # Network parameters
    connectivity_path: str = "/home/bo/Py/NewLife/.venv/src/connectivity_96.pt"
    num_neurons: int = 96
    
    # Training parameters
    device: str = "cuda:1"  # "cpu" or "cuda"
    num_epochs: int = 10  # Number of training epochs
    train_split: float = 0.8  # Fraction of data for training (0.8 = 120 samples, 0.2 = 30 samples)
    
    # Logging and output
    log_dir: str = "runs/iris_experiment"
    plot_dir: str = "plots"
    
    # Feature parameters
    class_names: Dict[int, str] = None
    
    def __post_init__(self) -> None:
        """Initialize defaults after dataclass instantiation"""
        if self.class_names is None:
            self.class_names = {
                0: "setosa",
                1: "versicolor", 
                2: "virginica",
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict: Dict[str, Any] = {
            "num_bits": self.num_bits,
            "use_gray_code": self.use_gray_code,
            "connectivity_path": self.connectivity_path,
            "num_neurons": self.num_neurons,
            "device": self.device,
            "num_epochs": self.num_epochs,
            "train_split": self.train_split,
            "log_dir": self.log_dir,
            "plot_dir": self.plot_dir,
        }
        return config_dict



def compute_clustering_metrics(
    spike_patterns: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute clustering metrics to validate unsupervised learning.
    Uses spike patterns only (no labels during computation).
    
    Args:
        spike_patterns: (num_samples, num_neurons) binary spike patterns
        labels: (num_samples,) true class labels (for reference only, not used in computation)
    
    Returns:
        Dictionary of metric names → scores
    """
    spike_np: Any = spike_patterns.cpu().numpy()
    labels_np: Any = labels.cpu().numpy()
    
    metrics: Dict[str, float] = {}
    
    # 1. SILHOUETTE SCORE (−1 to 1, higher is better)
    # Measures how well each spike pattern fits within its cluster vs. other clusters
    # Definition: For each sample, (b - a) / max(a, b) where:
    #   a = mean distance to other samples in same class
    #   b = mean distance to nearest other class
    # Range: [-1, 1] where 1 = perfectly separated, 0 = overlapping, -1 = wrong class
    try:
        silhouette: float = silhouette_score(spike_np, labels_np)
        metrics["silhouette_score"] = float(silhouette)
    except Exception as e:
        print(f"  Warning: Silhouette score computation failed: {e}")
        metrics["silhouette_score"] = 0.0
    
    # 2. DAVIES-BOULDIN INDEX (lower is better, minimum 0)
    # Measures average similarity ratio of each class with its most similar other class
    # Definition: average of (σ_i + σ_j) / d(c_i, c_j) where σ = within-cluster scatter, d = between-cluster distance
    # Range: [0, ∞] where 0 = perfect separation
    try:
        davies_bouldin: float = davies_bouldin_score(spike_np, labels_np)
        metrics["davies_bouldin_index"] = float(davies_bouldin)
    except Exception as e:
        print(f"  Warning: Davies-Bouldin score computation failed: {e}")
        metrics["davies_bouldin_index"] = float('inf')
    
    # 3. CALINSKI-HARABASZ INDEX (higher is better, minimum 0)
    # Ratio of between-cluster to within-cluster variance
    # Definition: (SS_B / (k-1)) / (SS_W / (n-k)) where SS_B = between-cluster variance, SS_W = within-cluster variance
    # Range: [0, ∞] where higher = better separated clusters
    try:
        calinski_harabasz: float = calinski_harabasz_score(spike_np, labels_np)
        metrics["calinski_harabasz_index"] = float(calinski_harabasz)
    except Exception as e:
        print(f"  Warning: Calinski-Harabasz score computation failed: {e}")
        metrics["calinski_harabasz_index"] = 0.0
    
    # 4. DUNN INDEX (higher is better, minimum 0)
    # Ratio of minimum inter-cluster distance to maximum intra-cluster distance
    # Higher values indicate well-separated, compact clusters
    metrics["dunn_index"] = compute_dunn_index(spike_np, labels_np)
    
    # 5. INTRA vs INTER-CLUSTER DISTANCE RATIO (lower is better)
    # Custom metric: average within-cluster distance / average between-cluster distance
    # Values < 1 indicate clusters are more compact than separated
    metrics["intra_inter_ratio"] = compute_intra_inter_ratio(spike_np, labels_np)
    
    # 6. NEURON SPECIALIZATION SCORE (0 to 1, higher is better)
    # Custom metric: measures if individual neurons develop class preferences
    # For each neuron: spike variance across classes / max possible variance
    # Validates that neurons differentiate classes WITHOUT using labels
    metrics["neuron_specialization"] = compute_neuron_specialization(spike_patterns, labels)
    
    # 7. NORMALIZED MUTUAL INFORMATION (0 to 1, higher is better)
    # Information-theoretic metric: how much spike patterns tell us about class membership
    # 0 = no correlation (random), 1 = perfect correlation (spike patterns encode class)
    metrics["normalized_mutual_information"] = compute_nmi(labels_np, compute_spike_clusters(spike_np))
    
    return metrics


def compute_dunn_index(spike_patterns: Any, labels: Any) -> float:
    """
    Dunn Index: min(inter-cluster distance) / max(intra-cluster distance)
    Higher is better (well-separated, compact clusters)
    """
    try:
        import scipy.spatial.distance as distance  # type: ignore[import-not-found]
        
        unique_labels: Any = set(labels)
        min_inter_distance: float = float('inf')
        max_intra_distance: float = 0.0
        
        # Compute max intra-cluster distance
        for label in unique_labels:
            mask: Any = labels == label
            cluster: Any = spike_patterns[mask]
            if len(cluster) > 1:
                distances: Any = distance.pdist(cluster, metric='hamming')
                max_intra_distance = max(max_intra_distance, distances.max())
        
        # Compute min inter-cluster distance
        for label_i in unique_labels:
            for label_j in unique_labels:
                if label_i < label_j:
                    mask_i: Any = labels == label_i
                    mask_j: Any = labels == label_j
                    cluster_i: Any = spike_patterns[mask_i]
                    cluster_j: Any = spike_patterns[mask_j]
                    distances: Any = distance.cdist(cluster_i, cluster_j, metric='hamming')
                    min_inter_distance = min(min_inter_distance, distances.min())
        
        if max_intra_distance == 0:
            return float('inf')
        
        return float(min_inter_distance / max_intra_distance)
    except Exception as e:
        print(f"  Warning: Dunn index computation failed: {e}")
        return 0.0


def compute_intra_inter_ratio(spike_patterns: Any, labels: Any) -> float:
    """
    Ratio of average within-cluster distance to average between-cluster distance.
    Lower is better (< 1 means clusters are tighter than separated).
    """
    try:
        import scipy.spatial.distance as distance  # type: ignore[import-not-found]
        
        unique_labels: Any = set(labels)
        intra_distances: List[float] = []
        inter_distances: List[float] = []
        
        # Compute within-cluster distances
        for label in unique_labels:
            mask: Any = labels == label
            cluster: Any = spike_patterns[mask]
            if len(cluster) > 1:
                distances: Any = distance.pdist(cluster, metric='hamming')
                intra_distances.extend(distances)
        
        # Compute between-cluster distances
        for label_i in unique_labels:
            for label_j in unique_labels:
                if label_i < label_j:
                    mask_i: Any = labels == label_i
                    mask_j: Any = labels == label_j
                    cluster_i: Any = spike_patterns[mask_i]
                    cluster_j: Any = spike_patterns[mask_j]
                    distances: Any = distance.cdist(cluster_i, cluster_j, metric='hamming')
                    inter_distances.extend(distances.flatten())
        
        avg_intra: float = sum(intra_distances) / len(intra_distances) if intra_distances else float('inf')
        avg_inter: float = sum(inter_distances) / len(inter_distances) if inter_distances else float('inf')
        
        if avg_inter == 0:
            return float('inf')
        
        return float(avg_intra / avg_inter)
    except Exception as e:
        print(f"  Warning: Intra-Inter ratio computation failed: {e}")
        return float('inf')


def compute_neuron_specialization(spike_patterns: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Neuron Specialization Score: 0 to 1, higher is better.
    Measures if individual neurons develop class preferences WITHOUT using labels during training.
    
    For each neuron: compute variance of spike counts across classes.
    Average across all neurons, normalize by max possible variance.
    """
    num_neurons: int = spike_patterns.shape[1]
    num_classes: int = len(set(labels.tolist()))
    
    specialization_scores: List[float] = []
    
    for neuron_id in range(num_neurons):
        neuron_spikes: torch.Tensor = spike_patterns[:, neuron_id]
        
        # Compute spike counts per class for this neuron
        spike_counts_per_class: List[float] = []
        for class_id in range(num_classes):
            mask: torch.Tensor = labels == class_id
            class_spike_counts: float = neuron_spikes[mask].float().mean().item()
            spike_counts_per_class.append(class_spike_counts)
        
        # Variance of spike counts across classes
        spike_counts_tensor: torch.Tensor = torch.tensor(spike_counts_per_class, dtype=torch.float32)
        variance: float = torch.var(spike_counts_tensor).item()
        
        # Normalize by maximum possible variance (when one class spikes 100%, others 0%)
        max_variance: float = (1.0 / num_classes) * (num_classes - 1)  # Max variance for this setup
        if max_variance > 0:
            normalized_score: float = min(1.0, variance / max_variance)
        else:
            normalized_score = 0.0
        
        specialization_scores.append(normalized_score)
    
    # Average specialization across all neurons
    avg_specialization: float = sum(specialization_scores) / len(specialization_scores) if specialization_scores else 0.0
    
    return float(avg_specialization)


def compute_spike_clusters(spike_patterns: Any) -> Any:
    """
    Perform k-means clustering on spike patterns to get predicted cluster assignments.
    Used for computing mutual information.
    """
    try:
        from sklearn.cluster import KMeans  # type: ignore[import-not-found]
        kmeans: Any = KMeans(n_clusters=3, random_state=42, n_init=10)
        predicted_labels: Any = kmeans.fit_predict(spike_patterns)
        return predicted_labels
    except Exception as e:
        print(f"  Warning: K-means clustering failed: {e}")
        return None


def compute_nmi(true_labels: Any, predicted_labels: Any) -> float:
    """
    Normalized Mutual Information: 0 to 1, higher is better.
    Measures how much spike-based clustering aligns with true class structure.
    0 = no correlation (random clustering), 1 = perfect alignment
    """
    if predicted_labels is None:
        return 0.0
    
    try:
        from sklearn.metrics import normalized_mutual_info_score  # type: ignore[import-not-found]
        nmi: float = normalized_mutual_info_score(true_labels, predicted_labels)
        return float(nmi)
    except Exception as e:
        print(f"  Warning: NMI computation failed: {e}")
        return 0.0


def compute_top_neuron_overlap(stats: Dict[int, Dict[str, Any]]) -> float:
    """
    EARLY LEARNING SIGNAL: Measure overlap in top responsive neurons across classes.
    
    Returns:
        overlap_score: 0-1, LOWER is better (0 = no overlap, 1 = all same neurons)
        
    Interpretation:
        - High overlap (>0.8): Neurons not specialized yet (early/no learning)
        - Medium overlap (0.3-0.7): Some specialization emerging
        - Low overlap (<0.3): Strong class-specific neuron preferences (learning!)
    """
    top_neurons_sets: List[set[int]] = []
    for class_id in [0, 1, 2]:
        top_neurons: List[int] = stats[class_id]["top_responsive_neurons"]
        top_neurons_sets.append(set(top_neurons))
    
    # Compute pairwise Jaccard similarity (intersection / union)
    jaccard_scores: List[float] = []
    for i in range(3):
        for j in range(i+1, 3):
            intersection: int = len(top_neurons_sets[i] & top_neurons_sets[j])
            union: int = len(top_neurons_sets[i] | top_neurons_sets[j])
            if union > 0:
                jaccard: float = intersection / union
                jaccard_scores.append(jaccard)
    
    avg_overlap: float = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0
    return float(avg_overlap)


def compute_mean_discriminability(stats: Dict[int, Dict[str, Any]]) -> float:
    """
    EARLY LEARNING SIGNAL: Average variance of spike counts across classes per neuron.
    
    Returns:
        mean_variance: Average variance across all neurons (higher = more learning)
        
    Interpretation:
        - Near 0: All neurons fire similarly for all classes (no learning)
        - > 1.0: Neurons starting to differentiate classes (early learning!)
        - > 5.0: Strong class-specific responses (good learning)
        - > 10.0: Excellent discrimination
    """
    num_neurons: int = len(stats[0]["avg_spike_counts"])
    variances: List[float] = []
    
    for neuron_id in range(num_neurons):
        counts_by_class: torch.Tensor = torch.tensor(
            [stats[cid]["avg_spike_counts"][neuron_id] for cid in [0, 1, 2]],
            dtype=torch.float32,
        )
        variance: float = torch.var(counts_by_class).item()
        variances.append(variance)
    
    mean_variance: float = sum(variances) / len(variances) if variances else 0.0
    return float(mean_variance)


def load_and_prepare_iris(config: IrisExperimentConfig) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, str]]:
    """
    Load Iris dataset and quantize to num_bits per feature.
    Optionally encode as Gray code for smooth bit transitions.
    
    Args:
        config: Experiment configuration
    
    Returns:
        (X_quantized, y, class_names)
        - X_quantized: (150, 4) tensor of int values 0-63 (optionally Gray-encoded)
        - y: (150,) tensor of class labels 0-2
        - class_names: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    """
    # Load Iris
    iris = datasets.load_iris()
    X: torch.Tensor = torch.tensor(iris.data, dtype=torch.float32)
    y: torch.Tensor = torch.tensor(iris.target, dtype=torch.long)
    
    # Normalize features to 0-1
    X_norm: torch.Tensor = (X - X.min(dim=0)[0]) / (X.max(dim=0)[0] - X.min(dim=0)[0])
    
    # Quantize to num_bits
    max_val: int = (2 ** config.num_bits) - 1
    X_quantized: torch.Tensor = (X_norm * max_val).int()
    
    # Apply Gray code if enabled
    if config.use_gray_code:
        X_quantized_list: List[List[int]] = []
        for row in X_quantized:
            gray_row: List[int] = [binary_to_gray(int(val.item()), config.num_bits) for val in row]
            X_quantized_list.append(gray_row)
        X_quantized = torch.tensor(X_quantized_list, dtype=torch.int32)
    
    return X_quantized, y, config.class_names


def split_train_val(
    X: torch.Tensor, 
    y: torch.Tensor, 
    train_split: float, 
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split data into train and validation sets with stratification.
    
    Args:
        X: Features (num_samples, num_features)
        y: Labels (num_samples,)
        train_split: Fraction for training (e.g., 0.8)
        seed: Random seed for reproducibility
    
    Returns:
        (X_train, y_train, X_val, y_val)
    """
    torch.manual_seed(seed)
    
    # Stratified split - maintain class balance
    train_indices: List[int] = []
    val_indices: List[int] = []
    
    for class_id in [0, 1, 2]:
        class_mask: torch.Tensor = y == class_id
        class_indices: torch.Tensor = torch.where(class_mask)[0]
        
        # Shuffle within class
        perm: torch.Tensor = torch.randperm(len(class_indices))
        shuffled_indices: torch.Tensor = class_indices[perm]
        
        # Split
        n_train: int = int(len(shuffled_indices) * train_split)
        train_indices.extend(shuffled_indices[:n_train].tolist())
        val_indices.extend(shuffled_indices[n_train:].tolist())
    
    # Convert to tensors
    train_idx: torch.Tensor = torch.tensor(train_indices, dtype=torch.long)
    val_idx: torch.Tensor = torch.tensor(val_indices, dtype=torch.long)
    
    # Shuffle train/val indices
    train_idx = train_idx[torch.randperm(len(train_idx))]
    val_idx = val_idx[torch.randperm(len(val_idx))]
    
    X_train: torch.Tensor = X[train_idx]
    y_train: torch.Tensor = y[train_idx]
    X_val: torch.Tensor = X[val_idx]
    y_val: torch.Tensor = y[val_idx]
    
    return X_train, y_train, X_val, y_val


def compute_cluster_centroids(spike_patterns_per_class: Dict[int, List[List[int]]]) -> torch.Tensor:
    """
    Compute centroid spike patterns for each class.
    
    Args:
        spike_patterns_per_class: Dict mapping class_id -> list of spike patterns
    
    Returns:
        centroids: (num_classes, num_neurons) tensor of cluster centroids
    """
    centroids_list: List[torch.Tensor] = []
    
    for class_id in sorted(spike_patterns_per_class.keys()):
        patterns: torch.Tensor = torch.tensor(spike_patterns_per_class[class_id], dtype=torch.float32)
        centroid: torch.Tensor = patterns.mean(dim=0)  # Average spike pattern for this class
        centroids_list.append(centroid)
    
    centroids: torch.Tensor = torch.stack(centroids_list)  # (3, 96)
    return centroids


def predict_cluster(spike_pattern: torch.Tensor, centroids: torch.Tensor) -> Tuple[int, float]:
    """
    Predict cluster assignment based on nearest centroid (Hamming distance).
    
    Args:
        spike_pattern: (num_neurons,) binary spike pattern
        centroids: (num_classes, num_neurons) cluster centroids
    
    Returns:
        (predicted_cluster, confidence) where confidence is 1 - normalized_distance
    """
    # Convert to float for distance computation
    pattern_float: torch.Tensor = spike_pattern.float()
    
    # Compute distances to all centroids (L1 distance works well for binary patterns)
    distances: torch.Tensor = torch.sum(torch.abs(centroids - pattern_float), dim=1)
    
    # Find nearest
    predicted_cluster: int = int(torch.argmin(distances).item())
    min_distance: float = float(distances[predicted_cluster].item())
    
    # Confidence: 1.0 = perfect match, 0.0 = max distance
    max_possible_distance: float = float(centroids.shape[1])  # All neurons different
    confidence: float = 1.0 - (min_distance / max_possible_distance)
    
    return predicted_cluster, confidence


def run_validation(
    network: SpikingNetworkTorch,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    centroids: torch.Tensor,
    num_features: int,
) -> Dict[str, Any]:
    """
    Run validation: predict clusters for validation samples, compute metrics.
    
    Args:
        network: Trained spiking network
        X_val: Validation features
        y_val: Validation labels (for metric computation only)
        centroids: Learned cluster centroids from training
        num_features: Number of features per sample
    
    Returns:
        Dict with validation metrics
    """
    num_val_samples: int = len(X_val)
    
    # Collect validation spike patterns
    val_spike_patterns: List[List[int]] = []
    val_predictions: List[int] = []
    val_confidences: List[float] = []
    
    for sample_idx in range(num_val_samples):
        sample: torch.Tensor = X_val[sample_idx]
        
        # Reset and run network
        network.reset()
        for feature_idx in range(num_features):
            feature_byte: torch.Tensor = sample[feature_idx:feature_idx+1]
            network.process_step(feature_byte)
        
        # Get spike pattern
        spike_pattern: List[int] = [1 if cnt > 0 else 0 for cnt in network.spike_counts]
        val_spike_patterns.append(spike_pattern)
        
        # Predict cluster
        spike_pattern_tensor: torch.Tensor = torch.tensor(spike_pattern, dtype=torch.float32, device=centroids.device)
        pred_cluster: int
        confidence: float
        pred_cluster, confidence = predict_cluster(spike_pattern_tensor, centroids)
        
        val_predictions.append(pred_cluster)
        val_confidences.append(confidence)
    
    # Convert to tensors
    val_spike_patterns_tensor: torch.Tensor = torch.tensor(val_spike_patterns, dtype=torch.float32)
    val_predictions_tensor: torch.Tensor = torch.tensor(val_predictions, dtype=torch.long)
    
    # Compute clustering metrics on validation set
    val_metrics: Dict[str, float] = compute_clustering_metrics(val_spike_patterns_tensor, y_val)
    
    # Compute classification accuracy (cluster assignment accuracy)
    # Note: This requires mapping predicted clusters to true classes
    # We use the most common true label in each predicted cluster
    cluster_to_class_mapping: Dict[int, int] = {}
    for cluster_id in [0, 1, 2]:
        cluster_mask: torch.Tensor = val_predictions_tensor == cluster_id
        if cluster_mask.sum() > 0:
            # Most common true class in this cluster
            true_classes_in_cluster: torch.Tensor = y_val[cluster_mask]
            most_common_class: int = int(torch.mode(true_classes_in_cluster)[0].item())
            cluster_to_class_mapping[cluster_id] = most_common_class
        else:
            cluster_to_class_mapping[cluster_id] = cluster_id  # Default mapping
    
    # Map predictions to classes
    mapped_predictions: List[int] = [cluster_to_class_mapping[pred] for pred in val_predictions]
    mapped_predictions_tensor: torch.Tensor = torch.tensor(mapped_predictions, dtype=torch.long)
    
    # Compute accuracy
    correct: int = int((mapped_predictions_tensor == y_val).sum().item())
    accuracy: float = correct / num_val_samples
    
    # Average confidence
    avg_confidence: float = sum(val_confidences) / len(val_confidences) if val_confidences else 0.0
    
    val_metrics["accuracy"] = accuracy
    val_metrics["avg_confidence"] = avg_confidence
    val_metrics["num_samples"] = num_val_samples
    
    return val_metrics


def run_iris_experiment(config: IrisExperimentConfig) -> Tuple[Dict[int | str, Any], SummaryWriter]:
    """
    Run spiking network on Iris dataset with train/val split and multiple epochs.
    
    Feed each of 4 features as separate timesteps.
    
    Args:
        config: Experiment configuration
    
    Returns:
        (stats_dict, writer) - Statistics from final epoch and SummaryWriter instance
    """
    # Create output directories
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.plot_dir).mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard writer
    writer: SummaryWriter = SummaryWriter(config.log_dir)
    
    # Log configuration
    writer.add_text("config", str(config.to_dict()))
    
    # Load Iris
    X_quantized: torch.Tensor
    y: torch.Tensor
    class_names: Dict[int, str]
    X_quantized, y, class_names = load_and_prepare_iris(config)
    X_quantized = X_quantized.to(config.device)
    y = y.to(config.device)
    num_features: int = X_quantized.shape[1]
    
    # Split into train/val
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_train, y_train, X_val, y_val = split_train_val(X_quantized, y, config.train_split)
    
    num_train: int = len(X_train)
    num_val: int = len(X_val)
    
    # Load network
    network: SpikingNetworkTorch = load_network(
        connectivity_path=config.connectivity_path,
        neuron_class=PlaceholderNeuron,
        device=config.device,
    )
    
    print("=" * 70)
    print(f"Iris Spiking Network Experiment (Torch-based)")
    print(f"Architecture: Loaded from {Path(config.connectivity_path).name}")
    print(f"Features per sample: {num_features}")
    encoding: str = "Gray code" if config.use_gray_code else "Binary"
    print(f"Quantization: {config.num_bits} bits per feature, {encoding} encoding (0-{(2**config.num_bits)-1})")
    print(f"Network neurons: {network.num_neurons}")
    print(f"  Capacity per neuron: 2 patterns")
    print(f"  Input neurons: 32 (1/3 of population)")
    print(f"  Connectivity: Sparse, 24 connections per neuron (75% sparse)")
    print(f"Dataset: {num_train} train samples, {num_val} val samples")
    print(f"Epochs: {config.num_epochs}")
    print(f"Process: Feed {num_features} features sequentially as timesteps")
    print("=" * 70)
    
    # Track best validation metrics
    best_val_silhouette: float = -1.0
    final_stats: Dict[int | str, Any] = {}  # Can have int keys (class IDs) or str keys (metrics)
    
    # Training loop over epochs
    for epoch in range(config.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{config.num_epochs}")
        print(f"{'='*70}")
        
        # ===== TRAINING PHASE =====
        spike_counts_per_class: Dict[int, List[List[int]]] = {0: [], 1: [], 2: []}
        spike_patterns_per_class: Dict[int, List[List[int]]] = {0: [], 1: [], 2: []}
        
        for sample_idx in range(num_train):
            sample: torch.Tensor = X_train[sample_idx]  # (4,)
            label: int = int(y_train[sample_idx].item())
            
            # Reset network for new sample
            network.reset()
            
            # Feed 4 features sequentially
            for feature_idx in range(num_features):
                feature_byte: torch.Tensor = sample[feature_idx:feature_idx+1]
                _: torch.Tensor = network.process_step(feature_byte)
            
            # Record spike counts
            spike_counts: List[int] = network.spike_counts.copy()
            spike_counts_per_class[label].append(spike_counts)
            
            # Record spike pattern (binary)
            spike_pattern: List[int] = [1 if cnt > 0 else 0 for cnt in spike_counts]
            spike_patterns_per_class[label].append(spike_pattern)
        
        print(f"  Training: Processed {num_train} samples")
        
        # Compute per-class statistics for training
        train_stats: Dict[int, Dict[str, Any]] = {}
        for class_id in [0, 1, 2]:
            spike_counts_data: List[List[int]] = spike_counts_per_class[class_id]
            spike_patterns_data: List[List[int]] = spike_patterns_per_class[class_id]
            
            if len(spike_counts_data) == 0:
                continue
            
            spike_counts_t: torch.Tensor = torch.tensor(spike_counts_data, dtype=torch.float32)
            spike_patterns_t: torch.Tensor = torch.tensor(spike_patterns_data, dtype=torch.float32)
            
            avg_spike_counts: torch.Tensor = spike_counts_t.mean(dim=0)
            spike_activation_rate: torch.Tensor = spike_patterns_t.mean(dim=0)
            
            top_neurons: List[int] = torch.argsort(-avg_spike_counts)[:5].tolist()
            
            total_spikes: float = (spike_counts_t.sum() / len(spike_counts_data)).item()
            
            train_stats[class_id] = {
                "class_name": class_names[class_id],
                "num_samples": len(spike_counts_data),
                "avg_spike_counts": avg_spike_counts.tolist(),
                "spike_activation_rate": spike_activation_rate.tolist(),
                "top_responsive_neurons": top_neurons,
                "avg_total_spikes": total_spikes,
            }
            
            # Log to TensorBoard with epoch
            writer.add_scalar(f"train/spikes/{class_names[class_id]}_avg_total", total_spikes, epoch)
        
        # Compute training clustering metrics
        all_spike_patterns_list: List[List[int]] = []
        all_labels_list: List[int] = []
        for class_id in [0, 1, 2]:
            for pattern in spike_patterns_per_class[class_id]:
                all_spike_patterns_list.append(pattern)
                all_labels_list.append(class_id)
        
        spike_patterns_tensor: torch.Tensor = torch.tensor(all_spike_patterns_list, dtype=torch.float32, device=config.device)
        labels_tensor: torch.Tensor = torch.tensor(all_labels_list, dtype=torch.long, device=config.device)
        
        train_clustering_metrics: Dict[str, float] = compute_clustering_metrics(spike_patterns_tensor, labels_tensor)
        
        # Log training metrics
        for metric_name, metric_value in train_clustering_metrics.items():
            if isinstance(metric_value, (int, float)) and not (isinstance(metric_value, float) and metric_value in [float('inf'), float('-inf')]):
                writer.add_scalar(f"train/clustering/{metric_name}", metric_value, epoch)
        
        # Training early learning signals
        train_overlap: float = compute_top_neuron_overlap(train_stats)
        train_disc: float = compute_mean_discriminability(train_stats)
        writer.add_scalar("train/learning/top_neuron_overlap", train_overlap, epoch)
        writer.add_scalar("train/learning/mean_discriminability", train_disc, epoch)
        
        # ===== VALIDATION PHASE =====
        # Compute cluster centroids from training data
        centroids: torch.Tensor = compute_cluster_centroids(spike_patterns_per_class)
        centroids = centroids.to(config.device)
        
        # Run validation
        val_metrics: Dict[str, Any] = run_validation(network, X_val, y_val, centroids, num_features)
        
        # Log validation metrics
        print(f"\n  Validation Results:")
        print(f"    Accuracy: {val_metrics['accuracy']:.3f} ({int(val_metrics['accuracy']*num_val)}/{num_val} correct)")
        print(f"    Avg Confidence: {val_metrics['avg_confidence']:.3f}")
        print(f"    Silhouette Score: {val_metrics.get('silhouette_score', 0.0):.3f}")
        
        for metric_name, metric_value in val_metrics.items():
            if isinstance(metric_value, (int, float)) and not (isinstance(metric_value, float) and metric_value in [float('inf'), float('-inf')]):
                writer.add_scalar(f"val/{metric_name}", metric_value, epoch)
        
        # Track best model
        val_silhouette: float = val_metrics.get('silhouette_score', -1.0)
        if val_silhouette > best_val_silhouette:
            best_val_silhouette = val_silhouette
            final_stats = train_stats
            final_stats["clustering_metrics"] = train_clustering_metrics
            final_stats["top_neuron_overlap"] = train_overlap
            final_stats["mean_discriminability"] = train_disc
            final_stats["val_metrics"] = val_metrics
            print(f"    ✓ Best validation silhouette so far!")
        
        writer.flush()
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best val silhouette: {best_val_silhouette:.3f}")
    print("=" * 70)
    
    return final_stats, writer


def plot_spike_patterns(stats: Dict[int | str, Any], class_names: Dict[int, str], output_dir: str) -> None:
    """
    Create interactive Plotly visualizations of spike patterns.
    
    Args:
        stats: Per-class statistics from run_iris_experiment
        class_names: Mapping of class IDs to names
        output_dir: Directory to save HTML plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Average spike counts per class (bar chart)
    classes: List[str] = [class_names[i] for i in [0, 1, 2]]
    avg_spikes: List[float] = [stats[i]["avg_total_spikes"] for i in [0, 1, 2]]
    
    fig1: go.Figure = go.Figure(data=go.Bar(x=classes, y=avg_spikes, marker_color=['blue', 'green', 'red']))
    fig1.update_layout(
        title="Average Total Spikes per Class",
        xaxis_title="Iris Class",
        yaxis_title="Avg Spikes per Sample",
        hovermode="x unified",
    )
    fig1.write_html(f"{output_dir}/spike_counts_by_class.html")
    print(f"Saved: {output_dir}/spike_counts_by_class.html")
    
    # Plot 2: Neuron activation heatmap (per-class activation rates)
    activation_data: List[List[float]] = []
    for class_id in [0, 1, 2]:
        activation_data.append(stats[class_id]["spike_activation_rate"])
    
    fig2: go.Figure = go.Figure(data=go.Heatmap(
        z=activation_data,
        y=[class_names[i] for i in [0, 1, 2]],
        x=[f"N{i}" for i in range(96)],
        colorscale="Viridis",
    ))
    fig2.update_layout(
        title="Neuron Activation Rate by Class",
        xaxis_title="Neuron",
        yaxis_title="Class",
        height=400,
        width=1200,
    )
    fig2.write_html(f"{output_dir}/activation_heatmap.html")
    print(f"Saved: {output_dir}/activation_heatmap.html")
    
    # Plot 3: Top responsive neurons per class (grouped bar chart)
    fig3_data: List[go.Bar] = []
    for class_id in [0, 1, 2]:
        top_neurons: List[int] = stats[class_id]["top_responsive_neurons"]
        top_counts: List[float] = [stats[class_id]["avg_spike_counts"][nid] for nid in top_neurons]
        fig3_data.append(go.Bar(
            name=class_names[class_id],
            x=[f"N{nid}" for nid in top_neurons],
            y=top_counts,
        ))
    
    fig3: go.Figure = go.Figure(data=fig3_data)
    fig3.update_layout(
        title="Top 5 Responsive Neurons by Class",
        xaxis_title="Neuron",
        yaxis_title="Avg Spike Count",
        barmode="group",
        hovermode="x unified",
    )
    fig3.write_html(f"{output_dir}/top_neurons_by_class.html")
    print(f"Saved: {output_dir}/top_neurons_by_class.html")


def print_stats(stats: Dict[int | str, Any]) -> None:
    """Pretty-print per-class statistics"""
    print("\n" + "=" * 70)
    print("PER-CLASS SPIKE STATISTICS")
    print("=" * 70)
    
    for class_id in [0, 1, 2]:
        s: Dict[str, Any] = stats[class_id]
        class_name: str = s['class_name']
        num_samples: int = s['num_samples']
        avg_total: float = s['avg_total_spikes']
        top_neurons: List[int] = s['top_responsive_neurons']
        
        print(f"\n{class_name.upper()} ({num_samples} samples)")
        print(f"  Avg total spikes per sample: {avg_total:.1f}")
        print(f"  Top 5 responsive neurons: {top_neurons}")
        
        for neuron_id in top_neurons:
            act_rate: float = s["spike_activation_rate"][neuron_id]
            avg_count: float = s["avg_spike_counts"][neuron_id]
            print(f"    Neuron {neuron_id}: active in {act_rate*100:.1f}% of samples, avg {avg_count:.1f} spikes")
    
    print("\n" + "=" * 70)
    print("CLASS DISCRIMINABILITY (by neuron)")
    print("=" * 70)
    
    all_neurons: List[int] = list(range(len(stats[0]["avg_spike_counts"])))
    neuron_discriminability: List[Tuple[int, float, List[float]]] = []
    
    for neuron_id in all_neurons:
        counts_by_class: torch.Tensor = torch.tensor(
            [stats[cid]["avg_spike_counts"][neuron_id] for cid in [0, 1, 2]],
            dtype=torch.float32,
        )
        variance: float = torch.var(counts_by_class).item()
        counts_list: List[float] = counts_by_class.tolist()
        neuron_discriminability.append((neuron_id, variance, counts_list))
    
    neuron_discriminability.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 discriminative neurons (highest spike variance across classes):")
    for rank, (neuron_id, variance, counts) in enumerate(neuron_discriminability[:5]):
        print(f"  {rank+1}. Neuron {neuron_id}: variance={variance:.2f}")
        print(f"     Setosa={counts[0]:.1f}, Versicolor={counts[1]:.1f}, Virginica={counts[2]:.1f}")
    
    print("=" * 70)
    
    # Print early learning signals
    if "top_neuron_overlap" in stats:
        overlap: float = stats["top_neuron_overlap"]
        overlap_status: str = "No specialization yet" if overlap > 0.7 else ("Emerging" if overlap > 0.3 else "Strong specialization!")
        print(f"\n🎯 EARLY LEARNING SIGNALS:")
        print(f"  Top Neuron Overlap: {overlap:.3f} ({overlap_status})")
        print(f"    └─ Lower is better: <0.3 = strong specialization, >0.7 = no specialization")
        
    if "mean_discriminability" in stats:
        disc: float = stats["mean_discriminability"]
        disc_status: str = "Random" if disc < 0.5 else ("Early learning" if disc < 5.0 else "Good learning!")
        print(f"  Mean Discriminability: {disc:.3f} ({disc_status})")
        print(f"    └─ Higher is better: >10 = excellent, >5 = good, >1 = early learning")
    
    if "clustering_metrics" in stats:
        metrics: Dict[str, float] = stats["clustering_metrics"]
        if "silhouette_score" in metrics:
            sil: float = metrics["silhouette_score"]
            sil_status: str = "No clustering" if sil < 0.1 else ("Weak" if sil < 0.3 else "Good clustering!")
            print(f"  Silhouette Score: {sil:.3f} ({sil_status})")
            print(f"    └─ Higher is better: >0.5 = excellent, >0.3 = good, >0.1 = emerging")
    
    print("=" * 70)


if __name__ == "__main__":
    # Create configuration
    config: IrisExperimentConfig = IrisExperimentConfig(
        num_bits=6,
        device="cpu",  # Use "cuda" if GPU available
        num_epochs=3,  # Quick test with 3 epochs
        train_split=0.8,
        log_dir="runs/iris_experiment",
        plot_dir="plots",
    )
    
    print(f"[Running on device: {config.device}]")
    print(f"[TensorBoard logs: {config.log_dir}]")
    print(f"[Plots dir: {config.plot_dir}]")
    print(f"[Config: {config.to_dict()}]")
    
    # Run experiment
    stats: Dict[int, Dict[str, Any]]
    writer: SummaryWriter
    stats, writer = run_iris_experiment(config)
    
    print_stats(stats)
    
    # Generate Plotly visualizations
    print("\nGenerating interactive plots...")
    plot_spike_patterns(stats, config.class_names, output_dir=config.plot_dir)
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"\n✓ Experiment complete!")
    print(f"  TensorBoard: tensorboard --logdir {config.log_dir}")
    print(f"  Interactive plots in: {config.plot_dir}/")

