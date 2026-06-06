# GA Bot Evolution Sandbox

A modern genetic algorithm framework for evolving trading bots using DEAP and your Stock #10 bot activity dataset.

## 📁 Files

- **`ga_bot_evolution.py`** - Main DEAP framework with genetic algorithm
- **`trading_bot.py`** - Trading bot strategy & simulation engine
- **`ga_utils.py`** - Utility functions (data loading, metrics, visualization)
- **`test_ga_evolution.py`** - Quick test script to see it working

## 🚀 Quick Start

```bash
cd /home/bo/Py/NewLife
source .venv/bin/activate
python .venv/src/test_ga_evolution.py
```

This will:
1. Load Stock #10 bot activity data (split 60/20/20)
2. Evolve 30 bots for 20 generations
3. Show best bot parameters & test set performance
4. Takes ~5-10 minutes

## 🧬 How It Works

### The Genome (11 Genes)

Each bot is defined by 11 parameters:

```
[weight1, weight2, weight3, weight4, weight5,     # 5 signal weights
 entry_threshold,                                  # 0.3-0.7
 position_size,                                    # 0.01-0.1 (% per trade)
 stop_loss_pct,                                    # 0.01-0.05
 take_profit_pct,                                  # 0.02-0.10
 holding_bars,                                     # 5-50 bars
 max_concurrent_positions]                         # 1-5 open trades
```

### The Trading Strategy

Each bot generates signals from 5 sources:

1. **Bot Activity Signal** - Direct signal from your model
2. **Momentum** - Recent price direction (5-bar change)
3. **Volume Spike** - Recent volume relative to average
4. **Volatility** - Low volatility better for trading
5. **Price Stability** - Recent range size

Composite signal: `weighted_sum(signals) >= entry_threshold`

### Train/Val/Test Split

```
60% TRAIN  |  20% VAL   |  20% TEST
-----------|------------|----------
GA trains  | GA evaluates | Final evaluation
on this    | fitness here | (no overfitting)
```

**Critical**: GA evaluates fitness on VAL set (never sees TEST set)

## 🔧 Configuration

### Change Evolution Parameters

In `test_ga_evolution.py`:

```python
best_bot, stats = evolver.evolve(
    pop_size=50,        # More = better diversity (slower)
    generations=50,     # More = longer evolution
    cxpb=0.7,          # Crossover probability
    mutpb=0.3,         # Mutation probability
)
```

### Change Bot Strategy

Edit `trading_bot.py` - `_calculate_entry_signal()`:

```python
def _calculate_entry_signal(self, bar: int) -> float:
    # Add more signals, change weightings, etc.
```

### Change Fitness Function

Edit `ga_bot_evolution.py` - `evaluate_bot()`:

```python
def evaluate_bot(self, genome):
    # Current: Sharpe on val set, penalize drawdown
    # You can add: win rate, Sortino, profit factor, etc.
```

## 📊 Output Interpretation

```
[Gen  0] Sharpe: best=0.523, avg=0.102, std=0.234, max=0.523
[Gen  1] Sharpe: best=0.587, avg=0.195, std=0.301, max=0.651
...
[Gen 19] Sharpe: best=1.245, avg=0.687, std=0.412, max=1.503
```

- **best**: Best Sharpe ratio in generation
- **avg**: Population average
- **std**: Diversity (low = converged)
- **max**: Best ever found

### Final Output

```
Test Metrics:
  total_return: 0.1240      (12.4% gain on unseen data)
  sharpe_ratio: 1.050       (good risk-adjusted return)
  max_drawdown: -0.0845     (8.45% max loss)
  win_rate: 0.5823          (58% profitable trades)
  num_trades: 87            (reasonable frequency)
```

**Golden metric**: Win rate > 55% + Sharpe > 0.8 = maybe worth trading

## 🎯 Next Steps

### Phase 1: Optimize Current Strategy
```bash
# Increase evolution intensity
pop_size = 100
generations = 50
# Should find better bots
```

### Phase 2: Test Other Stocks
```python
# In ga_utils.py, modify load_bot_activity_data()
# to load different stocks
# Goal: Do bots generalize across symbols?
```

### Phase 3: Add More Signals
```python
# In trading_bot.py, add signals:
# - RSI (overbought/oversold)
# - Moving averages
# - Bollinger bands
# - Custom features from your transformer model
```

### Phase 4: Paper Trading
```python
# Deploy best bot to Alpaca paper trading
# Compare: backtest vs live returns
# Expect: 50-70% of backtest performance
```

## 📈 What to Expect

| Metric | Typical | Great | Unrealistic |
|--------|---------|-------|------------|
| Sharpe | 0.5-1.0 | 1.0-2.0 | >3.0 |
| Win Rate | 50-60% | 60-70% | >75% |
| Drawdown | -5% to -15% | -3% to -5% | <-1% |
| # Trades | 50-200/eval period | 100-300 | >500 |

**Rule of thumb**: If backtest Sharpe > 1.5, expect ~0.7-1.0 on paper trading

## ⚠️ Common Pitfalls

1. **Overfitting** - GA optimizes on VAL set (not TEST) to prevent this
2. **Look-ahead bias** - TradingBot only uses price at current bar (correct)
3. **Survivorship bias** - Stock #10 only (add other stocks later)
4. **Market regime change** - 5.5 years data is minimum
5. **Slippage** - Modeled at 0.1% per trade (add more if needed)

## 🔬 Science Background (For You)

DEAP uses **tournament selection** (like natural selection):
- Random tournament (size 3), winner survives
- Repeated = pressure toward best genes

**Crossover** (like sexual reproduction):
- Blend genes from 2 parents → 2 children
- Alpha=0.3 means small perturbation

**Mutation** (like random variation):
- Each gene 20% chance Gaussian perturbation
- Discrete genes: small integer changes

**Multi-objective**: Sharpe AND Drawdown (Pareto front)

---

**Questions?** Look at the docstrings in each file - they're detailed!
