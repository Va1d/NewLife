"""
Backtesting simulation for NEAT-evolved trading bots.

Simulates a trading bot across historical market data, tracking:
- Account balance and capital changes
- Trade returns and Sharpe ratio
- Drawdowns and survival
- Composite fitness score for evolution
"""

import torch
import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass
from neat_network import NEATGenome, NEATNetworkBuilder
from market_data import RawMarketDataProvider


@dataclass
class BacktestResult:
    """Results from backtesting a single bot."""
    total_return: float          # (final_balance - initial) / initial
    sharpe_ratio: float           # Risk-adjusted return
    max_drawdown: float           # Worst peak-to-trough decline
    win_rate: float               # Fraction of winning trades
    num_trades: int               # Total number of trades
    final_balance: float          # Final account balance ($)
    survived: bool                # Did bot avoid total ruin?
    fitness_score: float          # Composite fitness for evolution
    
    def __repr__(self) -> str:
        return (f"BacktestResult(\n"
                f"  Return: {self.total_return*100:.2f}%\n"
                f"  Sharpe: {self.sharpe_ratio:.3f}\n"
                f"  Max DD: {self.max_drawdown*100:.2f}%\n"
                f"  Win Rate: {self.win_rate*100:.1f}%\n"
                f"  Trades: {self.num_trades}\n"
                f"  Balance: ${self.final_balance:,.0f}\n"
                f"  Survived: {self.survived}\n"
                f"  Fitness: {self.fitness_score:.4f}\n)")


class BacktestSimulator:
    """
    Simulates trading bot performance on historical market data.
    
    Bot lifecycle:
    1. Start with $100,000 initial capital
    2. Receive market features at each time step
    3. Output trading signal (position size in [-1, 1])
    4. Realize returns based on next-step price move
    5. Continue until data ends or account depleted
    
    Fitness = weighted combination of Sharpe ratio and final balance
    """
    
    def __init__(self, 
                 market_data: RawMarketDataProvider,
                 initial_capital: float = 100000.0,
                 position_size_mult: float = 0.1,
                 early_exit_threshold: float = 0.3,
                 device: str = 'cuda:1'):
        """
        Initialize backtest simulator.
        
        Args:
            market_data: RawMarketDataProvider with features and returns
            initial_capital: Starting account balance ($100K)
            position_size_mult: Multiplier for output signal -> actual position
                                (output in [-1,1] × this = fraction of capital risked)
            early_exit_threshold: Stop trading if balance drops below this % of initial
            device: GPU device for computation
        """
        self.market_data = market_data
        self.initial_capital = initial_capital
        self.position_size_mult = position_size_mult
        self.early_exit_threshold = early_exit_threshold
        self.device = device
        
        print(f"BacktestSimulator initialized:")
        print(f"  Initial capital: ${initial_capital:,.0f}")
        print(f"  Position multiplier: {position_size_mult:.2%}")
        print(f"  Early exit at: {early_exit_threshold:.0%} of initial balance")
    
    def backtest(self, genome: NEATGenome) -> BacktestResult:
        """
        Run full backtest of a NEAT genome as a trading bot.
        
        Args:
            genome: NEAT network genome to test
        
        Returns:
            BacktestResult with all performance metrics
        """
        try:
            # Build network
            network = NEATNetworkBuilder.build_network(genome, device=self.device)
            network.eval()
            
            # Get all market data
            features, returns = self.market_data.get_all_data()
            
            # Trading state
            balance = self.initial_capital
            position = 0.0  # Current position size in [-1, 1]
            trades = []     # List of (entry_price, exit_price, return)
            balances = [balance]
            positions = []
            
            max_balance = balance
            early_exit = False
            
            # Simulate trading across all time steps
            with torch.no_grad():
                for step in range(len(features) - 1):
                    # Get market features at this step
                    feature_vec = features[step:step+1].to(self.device)  # [1, n_features]
                    
                    # Bot outputs a position signal
                    signal = network(feature_vec)  # [1, 1]
                    signal_value = signal.squeeze().cpu().item()
                    
                    # Clamp to [-1, 1] and scale to position size
                    signal_value = np.clip(signal_value, -1.0, 1.0)
                    new_position = signal_value * self.position_size_mult
                    
                    # Close previous position and open new position
                    # (simplified: doesn't track fractional shares, just P&L on balance)
                    if position != 0:
                        # P&L from holding position through this return
                        pnl = balance * position * returns[step].item()
                        balance += pnl
                        
                        trade_return = returns[step].item() * position
                        trades.append((position, new_position, trade_return))
                    
                    # Update position
                    position = new_position
                    positions.append(position)
                    
                    # Track balance
                    balances.append(balance)
                    max_balance = max(max_balance, balance)
                    
                    # Check early exit (ran out of funds)
                    if balance < self.initial_capital * self.early_exit_threshold:
                        early_exit = True
                        break
            
            # Calculate metrics
            total_return = (balance - self.initial_capital) / self.initial_capital
            
            # Sharpe ratio: return per unit volatility
            if len(trades) > 1:
                trade_returns = np.array([t[2] for t in trades])
                mean_return = trade_returns.mean()
                std_return = trade_returns.std() + 1e-8
                sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
                win_rate = (trade_returns > 0).sum() / len(trade_returns)
            else:
                sharpe_ratio = 0.0
                win_rate = 0.5
            
            # Max drawdown
            balances = np.array(balances)
            running_max = np.maximum.accumulate(balances)
            drawdown = (balances - running_max) / (running_max + 1e-8)
            max_drawdown = drawdown.min()
            
            # Survival
            survived = balance > 0 and not early_exit
            
            # Composite fitness score
            # Reward: positive return + high Sharpe + survival
            # Penalize: negative return + low Sharpe + early exit
            fitness = self._calculate_fitness(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                final_balance=balance,
                survived=survived,
                early_exit=early_exit
            )
            
            return BacktestResult(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=abs(max_drawdown),
                win_rate=win_rate,
                num_trades=len(trades),
                final_balance=balance,
                survived=survived,
                fitness_score=fitness
            )
        
        except Exception as e:
            # Network failed or other error - return worst fitness
            print(f"Backtest error: {e}")
            return BacktestResult(
                total_return=-1.0,
                sharpe_ratio=-10.0,
                max_drawdown=1.0,
                win_rate=0.0,
                num_trades=0,
                final_balance=0.0,
                survived=False,
                fitness_score=-1000.0
            )
    
    def _calculate_fitness(self,
                          total_return: float,
                          sharpe_ratio: float,
                          max_drawdown: float,
                          final_balance: float,
                          survived: bool,
                          early_exit: bool) -> float:
        """
        Calculate composite fitness score for evolution.
        
        Combines:
        - Sharpe ratio (consistency, risk-adjusted)
        - Final balance (absolute profitability)
        - Survival penalty (avoid ruin, but gentle)
        
        Returns:
            fitness_score: Higher is better (used by DEAP)
        """
        # Base fitness from Sharpe ratio (smooth, 0 is neutral)
        sharpe_component = np.clip(sharpe_ratio, -5, 5)  # Clip outliers
        
        # Return component: reward positive, penalize negative but not harshly
        return_component = total_return * 10  # Scale up for visibility
        
        # Survival bonus (not too important due to random data)
        survival_bonus = 0.5 if survived else -0.1
        early_exit_penalty = -1.0 if early_exit else 0.0
        
        # Drawdown penalty (minor)
        drawdown_penalty = -max_drawdown * 0.5
        
        # Composite
        fitness = (
            sharpe_component * 2.0 +      # High weight on consistency
            return_component * 1.0 +       # Medium weight on profit
            survival_bonus +               # Small bonus for not blowing up
            early_exit_penalty +           # Small penalty if exited early
            drawdown_penalty               # Small penalty for volatility
        )
        
        return fitness


if __name__ == "__main__":
    # Test backtest simulator
    print("Testing BacktestSimulator...\n")
    
    # Load market data
    print("Loading market data...")
    market_data = RawMarketDataProvider(device='cuda:1', target_stock_idx=10, normalize=True)
    
    # Create simulator
    print("\nInitializing simulator...")
    simulator = BacktestSimulator(
        market_data=market_data,
        initial_capital=100000.0,
        position_size_mult=0.1,
        early_exit_threshold=0.3,
        device='cuda:1'
    )
    
    # Test with a simple NEAT genome
    print("\nCreating test genome...")
    from neat_network import NEATGenome
    
    genome = NEATGenome(num_inputs=market_data.features.shape[1], num_outputs=1)
    h1 = genome.add_node(activation='tanh')
    h2 = genome.add_node(activation='relu')
    
    # Add some connections
    for i in range(min(5, market_data.features.shape[1])):
        genome.add_connection(i, h1, weight=0.1)
        genome.add_connection(i, h2, weight=-0.1)
    
    genome.add_connection(h1, market_data.features.shape[1], weight=0.5)
    genome.add_connection(h2, market_data.features.shape[1], weight=-0.5)
    
    # Run backtest
    print("\nRunning backtest...")
    result = simulator.backtest(genome)
    
    print("\nBacktest Results:")
    print(result)
