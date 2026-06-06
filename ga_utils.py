"""
GA Bot Evolution - Utility Functions
Load data, calculate metrics, visualization
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent))


def load_bot_activity_data():
    """
    Load Stock #10 bot activity data from loader.py
    Returns: (prices, volumes, bot_signals, test_mask)
    """
    from loader import TheSetGPU

    print("[GA] Loading bot activity dataset...")
    dataset = TheSetGPU(split='train', use_cache=False)

    # Extract Stock #10 data
    all_prices = []
    all_volumes = []
    all_bot_signals = []

    for i in range(len(dataset)):
        sample = dataset[i]
        # sample = {
        #     'market_data': (B, 256, features),
        #     'target': binary label,
        #     'target_bot_activity': (256,) binary signal
        # }

        market_data = sample['market_data'].numpy()  # (B, 256, features)
        bot_activity = sample['target_bot_activity'].numpy()  # (256,)

        # Extract OHLCV (assuming standard format)
        # Typically: [open, high, low, close, volume, ...]
        all_prices.append(market_data[:, :, 3])  # close price (feature 3)
        all_volumes.append(market_data[:, :, 4])  # volume (feature 4)
        all_bot_signals.append(bot_activity)

    # Concatenate all
    prices = np.concatenate(all_prices, axis=1)  # (B, total_steps)
    volumes = np.concatenate(all_volumes, axis=1)  # (B, total_steps)
    signals = np.concatenate(all_bot_signals, axis=0)  # (total_steps,)

    # Use first stock (B[0])
    prices = prices[0]
    volumes = volumes[0]

    print(f"[GA] Loaded: {len(prices)} price bars, {len(signals)} bot signals")

    # Split: 60% train, 20% val, 20% test
    n = len(prices)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_mask = np.arange(n) < train_end
    val_mask = (np.arange(n) >= train_end) & (np.arange(n) < val_end)
    test_mask = np.arange(n) >= val_end

    return {
        'prices': prices,
        'volumes': volumes,
        'signals': signals,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
    }


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio of returns"""
    excess_returns = returns - risk_free_rate
    if len(excess_returns) < 2 or np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # annualized


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from cumulative returns"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown) if len(drawdown) > 0 else 0.0


def calculate_win_rate(returns: np.ndarray) -> float:
    """Percentage of profitable trades"""
    if len(returns) == 0:
        return 0.0
    return np.sum(returns > 0) / len(returns)


def calculate_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate all relevant metrics for a bot"""
    return {
        'total_return': np.sum(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': calculate_win_rate(returns),
        'num_trades': len(returns),
        'avg_trade': np.mean(returns) if len(returns) > 0 else 0.0,
    }


def format_genome(genome: List[float]) -> Dict[str, float]:
    """
    Convert genome list to parameter dict
    Genome structure: [weights...threshold, pos_size, stop_loss, take_profit, holding_bars, max_pos]
    """
    # Assuming 5 signal sources
    num_weights = 5

    return {
        'entry_weights': np.array(genome[:num_weights]),  # 5 weights
        'entry_threshold': genome[num_weights],  # 0.3-0.7
        'position_size': genome[num_weights + 1],  # 0.01-0.1
        'stop_loss_pct': genome[num_weights + 2],  # 0.01-0.05
        'take_profit_pct': genome[num_weights + 3],  # 0.02-0.10
        'holding_bars': int(genome[num_weights + 4]),  # 5-50
        'max_concurrent_positions': int(genome[num_weights + 5]),  # 1-5
    }


def print_evolution_stats(gen: int, pop_fitnesses: List[Tuple[float, ...]],
                         best_individual: List[float], best_fitness: Tuple[float, ...]):
    """Pretty print generation statistics"""
    fitnesses = [f[0] for f in pop_fitnesses]  # First fitness objective
    print(f"\n[Gen {gen:3d}] Sharpe: best={best_fitness[0]:6.3f}, "
          f"avg={np.mean(fitnesses):6.3f}, std={np.std(fitnesses):6.3f}, "
          f"max={np.max(fitnesses):6.3f}")
