"""Sequential MNIST Dataloader

Converts standard MNIST images (28x28) into sequences (784 timesteps, 1 input dim per step).
Each pixel value becomes a sequential input.
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms  # type: ignore[import-not-found]


class SequentialMNIST(Dataset):
    """MNIST converted to sequences: 28x28 image → 784-step sequence"""

    def __init__(self, train: bool = True, root: str = "./data") -> None:
        """
        Args:
            train: Load training (True) or test (False) split
            root: Directory to download/store MNIST
        """
        transform = transforms.Compose([
            transforms.ToTensor(),  # type: ignore[misc]
        ])
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.mnist[idx]
        # img shape: (1, 28, 28) from ToTensor
        # Flatten to (784,) and normalize to [0, 1]
        seq = img.flatten().unsqueeze(-1)  # (784, 1)
        return seq, label


def get_sequential_mnist_loaders(
    batch_size: int = 32,
    train_split: float = 1.0,
    root: str = "./data",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for Sequential MNIST.

    Args:
        batch_size: Batch size for loader
        train_split: Fraction of training data to use (for quick testing use <1.0)
        root: Directory for data
        num_workers: Number of loader workers

    Returns:
        (train_loader, test_loader)
        
    Data shape from loader:
        - input: (batch_size, 784, 1)  — 784 timesteps, 1 dim each
        - label: (batch_size,)  — digit 0-9
    """
    train_dataset = SequentialMNIST(train=True, root=root)
    test_dataset = SequentialMNIST(train=False, root=root)

    # Optionally subsample training data
    if train_split < 1.0:
        train_size = int(len(train_dataset) * train_split)
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset,
            [train_size, len(train_dataset) - train_size]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test
    train_loader, test_loader = get_sequential_mnist_loaders(batch_size=16)

    print("Sequential MNIST Dataloader Test")
    print("=" * 50)

    for batch_idx, (seq, label) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Input shape: {seq.shape}  (batch, timesteps, input_dim)")
        print(f"  Label shape: {label.shape}")
        print(f"  Labels in batch: {label[:5].tolist()}")
        print(f"  Input range: [{seq.min():.3f}, {seq.max():.3f}]")

        if batch_idx == 0:
            print(f"\nFirst sample details:")
            print(f"  Sequence length: {seq.shape[1]}")
            print(f"  Input dimension: {seq.shape[2]}")
            print(f"  First 10 timestep values: {seq[0, :10, 0].tolist()}")
            break

    print("\n" + "=" * 50)
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Total test samples: {len(test_loader.dataset)}")
    print(f"Batches per epoch (train): {len(train_loader)}")
