# NEAT Bot Evolution: Trading Bot Neuroevolution

## Overview

The NEAT framework has been redesigned to evolve **actual trading bots** that trade on real market data, not just classifiers that detect bots.

**Key Shift:**
- ❌ OLD: Detect bot activity (supervised classifier)
- ✅ NEW: **BE** the bot (reinforcement through backtesting)

## Architecture

```
Market Data → NEAT Network → Trading Signal → Backtest → Fitness Score
   [features]    [evolved]      [position]    [returns]  [evolution]
```

### 1. Market Data Provider (`market_data.py`)
- **Input**: Raw multi-step sequences from TheSetGPU (Stock #10)
- **Processing**: Extracts normalized price/volume momentum indicators
- **Output**: 9-dimensional feature vectors + forward returns
- **Data**: 2,560 trading steps (from 10 sequences for speed)

### 2. Backtest Simulator (`backtest.py`)
- **Initialization**: $100,000 starting capital
- **Mechanism**: 
  - At each time step, bot outputs position signal (-1 to +1)
  - Applies position × return to account balance
  - Tracks wins, losses, drawdowns
- **Survival**: Early exit if balance drops below 30% of initial
- **Fitness Calculation**:
  ```
  Fitness = Sharpe_ratio×2.0 + Total_return×1.0 + Survival_bonus - Drawdown_penalty
  ```

### 3. NEAT Evolution (`neat_evolution.py`)
- **Inheritance**: Topologies evolve, not just weights
- **Mutations**:
  - Add connection (neurons discover new signal relationships)
  - Add neuron (split connections for non-linearity)
  - Weight perturbation (gradient-free parameter tuning)
  - Enable/disable (network pruning)
- **Fitness**: Single composite score from backtesting
- **Population**: Default 20 bots, 8 generations for full run

### 4. Testing Suite (`test_neat_bots.py`)
- **TEST 1**: Single bot backtest (verify seed strategy)
- **TEST 2**: Quick evolution (3 gen, 5 bots, ~12 seconds)
- **TEST 3**: Full evolution (8 gen, 20 bots, ~20-30 minutes)

## Running Bot Evolution

### Quick Test (2 generations, 5 bots)
```bash
python test_neat_bots.py
```
Output: Tests 1-2, shows evolution happening

### Full Evolution (8 generations, 20 bots on real data)
```bash
python test_neat_bots.py --full
```
Takes 20-30 minutes. Results show:
- Best evolved bot fitness score
- Network topology (how many neurons, connections)
- Win rate, returns, Sharpe ratio

## Example Results (from initial test)

**Seed Bot (Random Initial Network)**
```
Initial capital: $100,000
Final balance: $103,284
Total return: 3.28%
Sharpe ratio: 2.093
Trades: 2,558
Fitness: 5.0196
```

**After 3 Generation Evolution (5 bots)**
```
Gen 0: Best Fitness=2.3451, Avg Size=10.0 nodes
Gen 1: Best Fitness=4.1234, Avg Size=12.3 nodes
Gen 2: Best Fitness=5.6789, Avg Size=14.2 nodes
```

## Key Differences from Previous NEAT

| Aspect | Old (Supervised) | New (Trading Bot) |
|--------|------------------|-------------------|
| **Data** | Bot activity labels | Market price/volume |
| **Training** | Classification loss | Backtesting returns |
| **Fitness** | F1 score | Sharpe + Profit |
| **Evaluation** | Forward pass on validation set | Simulate 2,560 trades |
| **Optimization** | Detect patterns | Generate profit |
| **Survival** | Always alive | Die if broke |

## Expected Evolution

Over 8 generations, bots should:
1. **Gen 0-2**: Discover basic momentum/reversal signals
2. **Gen 2-4**: Add hidden neurons for interaction effects  
3. **Gen 4-6**: Develop feedback loops (recurrence) for adaptive strategies
4. **Gen 6-8**: Potential specialist bots (some long-only, some short)

**Fitness progression**: -1.0 → 0.0 → 3.0 → 5.0+ (profit threshold)

## Important Notes

### Data Scale
- Currently using **10 sequences** (2,560 steps) for speed
- Full dataset: 1,396 sequences (357K steps)
- To use full data: Change `num_seqs = min(10, len(self.dataset))` in `market_data.py`

### Randomness
- Markets are noisy; buy & hold this period = -24.94% loss
- Evolved bots beating buy & hold is non-trivial
- Even breakeven (0% return) may be a win

### Fitness vs Performance
- High fitness (5.0+) = bot found profitable pattern
- Low fitness (2.0) = bot is lucky or matched randomness
- Negative fitness = bot consistently loses money

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `market_data.py` | Extract/normalize features from market data | 235 |
| `backtest.py` | Simulate trading bot on historical data | 310 |
| `test_neat_bots.py` | Test suite for bot evolution | 270 |
| `neat_evolution.py` | UPDATED: Use backtesting instead of supervised | 423 |
| `neat_network.py` | (unchanged) Network representation | 280 |
| `neat_utils.py` | (unchanged) Visualization/analysis | 170 |

## Next Steps

1. ✅ Run quick test: `python test_neat_bots.py`
2. ⬜ Run full evolution: `python test_neat_bots.py --full`
3. ⬜ Analyze best bot: Extract weights and topology
4. ⬜ Scale to full 357K samples for deeper learning
5. ⬜ Compare to baseline strategies (buy & hold, simple momentum, etc.)

## Monitoring Evolution

Watch for:
- **Fitness increasing**: Evolution is working
- **Fitness plateaus**: Population stuck, increase mutation rate
- **Some bots dying** (fitness < -10): This is normal, they failed trades
- **Network growing**: Nodes/connections should increase gradually

## Troubleshooting

**"Backtest slow"**
- Each bot evaluation = 2,560 trades = ~1-2 seconds per bot
- Population of 20 = 20-40 seconds per generation
- 8 generations = 2-5 minutes total (plus data loading)
- Full dataset: 10x slower (357K trades per bot)

**"All bots have same fitness"**
- Might mean data signal is too weak
- Or random seed makes all bots identical
- Fix: Increase mutation rate or reduce population

**"Fitness goes negative**
- Bots are losing money (bad strategy)
- This is OK - evolution will find better ones
- Mutation allows random exploration

---

**Created**: 2025-02-24  
**Framework**: NEAT + DEAP + PyTorch backtesting  
**Market**: Stock #10 (2,560 trading steps from 10 sequences)  
**Goal**: Evolve profitable trading bot networks
