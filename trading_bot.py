"""
GA Bot Strategy - Defines the trading bot logic for evolution
Each bot instance = different genome (different parameters)
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class BotParameters:
    """Parameters that define a trading bot's behavior"""
    entry_weights: np.ndarray      # Shape (5,) - weights for 5 signals
    entry_threshold: float          # 0.3-0.7 - confidence threshold for entry
    position_size: float            # 0.01-0.1 - % of capital per trade
    stop_loss_pct: float            # 0.01-0.05 - stop loss %
    take_profit_pct: float          # 0.02-0.10 - profit target %
    holding_bars: int               # 5-50 - bars to hold position
    max_concurrent_positions: int   # 1-5 - max open trades


class TradingBot:
    """
    Simulates a trading bot given parameters and price/volume data
    """

    def __init__(self, params: BotParameters,
                 prices: np.ndarray,
                 volumes: np.ndarray,
                 bot_signals: np.ndarray,
                 initial_capital: float = 10000.0,
                 commission: float = 0.001):
        """
        Args:
            params: BotParameters with evolution genes
            prices: (n_bars,) close prices
            volumes: (n_bars,) volumes
            bot_signals: (n_bars,) binary bot activity signals
            initial_capital: Starting capital
            commission: Commission per trade (0.1%)
        """
        self.params = params
        self.prices = prices
        self.volumes = volumes
        self.bot_signals = bot_signals
        self.initial_capital = initial_capital
        self.commission = commission

        self.n_bars = len(prices)
        self.trades = []  # List of (entry_price, exit_price, bars_held, pnl)
        self.trade_returns = []

    def _calculate_entry_signal(self, bar: int) -> float:
        """
        Calculate composite entry signal from multiple sources
        Returns confidence score 0-1
        """
        lookback = min(10, bar)
        if lookback < 2:
            return 0.0

        # Signal 1: Bot activity (direct)
        bot_signal = float(self.bot_signals[bar])

        # Signal 2: Recent price momentum (% change last 5 bars)
        price_change = (self.prices[bar] - self.prices[bar-lookback]) / self.prices[bar-lookback]
        momentum = max(0, min(1, (price_change + 0.05) / 0.10))  # Normalize to 0-1

        # Signal 3: Volume spike
        avg_vol = np.mean(self.volumes[max(0, bar-lookback):bar])
        volume_spike = min(1, self.volumes[bar] / (avg_vol + 1e-6) / 2)

        # Signal 4: Volatility (low volatility = better for trading)
        price_std = np.std(self.prices[max(0, bar-lookback):bar])
        volatility = max(0, 1 - (price_std / self.prices[bar]))

        # Signal 5: Price stability (small recent moves)
        recent_range = np.max(self.prices[max(0, bar-5):bar]) - np.min(self.prices[max(0, bar-5):bar])
        stability = max(0, 1 - (recent_range / self.prices[bar]))

        # Composite signal using learned weights
        signals = np.array([bot_signal, momentum, volume_spike, volatility, stability])
        weighted_signal = np.dot(self.params.entry_weights, signals)

        # Normalize to 0-1
        return max(0, min(1, weighted_signal))

    def simulate(self) -> List[float]:
        """
        Simulate bot trading for entire price series
        Returns: list of trade returns (percentage gains)
        """
        open_positions = []  # List of (entry_price, entry_bar, target_profit, stop_loss)

        for bar in range(1, self.n_bars):
            current_price = self.prices[bar]

            # ===== EXIT LOGIC =====
            closed_positions = []
            for i, (entry_price, entry_bar, target, stop) in enumerate(open_positions):
                bars_held = bar - entry_bar
                should_exit = False
                pnl_pct = 0.0

                # Exit 1: Profit target hit
                if current_price >= entry_price * (1 + target):
                    pnl_pct = target
                    should_exit = True

                # Exit 2: Stop loss hit
                elif current_price <= entry_price * (1 - stop):
                    pnl_pct = -stop
                    should_exit = True

                # Exit 3: Holding period exceeded
                elif bars_held >= self.params.holding_bars:
                    pnl_pct = (current_price - entry_price) / entry_price
                    should_exit = True

                if should_exit:
                    # Apply commission on both entry and exit
                    net_return = pnl_pct * (1 - 2 * self.commission)
                    self.trade_returns.append(net_return)
                    self.trades.append((entry_price, current_price, bars_held, net_return))
                    closed_positions.append(i)

            # Remove closed positions
            open_positions = [open_positions[i] for i in range(len(open_positions))
                            if i not in closed_positions]

            # ===== ENTRY LOGIC =====
            if len(open_positions) < self.params.max_concurrent_positions:
                signal_strength = self._calculate_entry_signal(bar)

                if signal_strength >= self.params.entry_threshold:
                    entry_price = current_price
                    open_positions.append((
                        entry_price,
                        bar,
                        self.params.take_profit_pct,
                        self.params.stop_loss_pct
                    ))

        # Close all remaining open positions at last bar
        for entry_price, entry_bar, target, stop in open_positions:
            current_price = self.prices[-1]
            pnl_pct = (current_price - entry_price) / entry_price
            net_return = pnl_pct * (1 - 2 * self.commission)
            self.trade_returns.append(net_return)
            self.trades.append((entry_price, current_price, self.n_bars - entry_bar, net_return))

        return self.trade_returns
