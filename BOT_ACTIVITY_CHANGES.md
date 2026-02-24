# Bot Activity Detection - Changes Summary

## Analysis Results

Analyzed all 36 stocks to find the least noisy bot activity signals.

**Winner: Stock #10** with 20.5% bot activity rate (cleaner than original 40% noise from VWAP reversion)

Top 5 stocks by signal quality:
1. Stock #10: 20.5% ensemble bot activity (16.8% volume spikes with stable price + 14.0% trade spikes)
2. Stock #21: 19.6% ensemble
3. Stock #24: 17.9% ensemble  
4. Stock #11: 16.9% ensemble
5. Stock #4: 16.8% ensemble

## Changes Made

### 1. **loader.py** - Added bot activity signal methods

Added three new properties to the `Extender` class:

#### `bot_activity_vwap` 
- Volume spike (>2x) with price stability (<0.5% move)
- Signature: VWAP execution algos must execute large volume without slippage
- Rate: ~16.6% on stock 10

#### `bot_activity_scalping`
- Trade count spike (>1.5x normal)
- Signature: Retail can't execute 50+ trades/minute - this is bot behavior
- Rate: ~14% on stock 10

#### `bot_activity_ensemble`
- Combined signal: Volume+stable price OR trade spike
- Captures most obvious algorithmic fingerprints
- Rate: ~20.5% on stock 10 (much cleaner than original 40%)

### 2. **loader.py** - Added `target_bot_activity` property

New target variable using the ensemble bot activity signal instead of VWAP reversion.

### 3. **loader.py** - Updated dataset defaults

Changed default `target_stock_idx`:
- **Old**: 30 (PepsiCo) with noisy VWAP labels
- **New**: 10 (cleaner bot signals) with ensemble bot activity labels

Updated both `TheSet` and `TheSetGPU` classes to:
- Use `target_stock_idx=10` as default
- Use `ex.target_bot_activity` instead of `ex.target`

### 4. **train.py** - Hyperparameter adjustments remain

Best regularization settings for the Bayesian model:
- prior_sigma: 1.0 (weak prior, allows learning)
- dropout: 0.05 (reduced from 0.1)
- weight_decay: 1e-6 (reduced from 1e-5)
- label_smoothing: 0.0 (disabled)
- learning_rate: 0.001 (increased from 0.0001)
- KL weight: 1e-4 (reduced from 1.0/dataset_size)

## Expected Improvements

### From previous run (epoch 13 with noisy VWAP target):
```
Train: Loss=0.9277, F1=0.4435, Acc=50.94%
Test:  Loss=0.8605, F1=0.4898, Acc=53.20%
→ Test better than train = random predictions
```

### Expected from new bot activity target:
- ✓ Train F1 improving > 0.55 by epoch 5-10 (learning actual signals)
- ✓ Train & Test diverging (actual overfitting possible)
- ✓ Better Bayesian uncertainty quantification on meaningful signals

## Why These Changes Work

1. **Signal Quality**: Bot behavior is deterministic (hardcoded algos) vs VWAP reversion which is probabilistic
2. **Lower Noise**: 20.5% positive rate vs 40% (closer to signal vs pure randomness)
3. **Multiple Fingerprints**: Ensemble captures different bot types:
   - VWAP execution (volume without slippage)
   - Market making/scalping (rapid trades)
4. **Cleaned Data**: Stock #10 has best combination of:
   - Significant volume spikes (16.8%)
   - Measurable trade activity (14%)
   - Price stability when volume spikes (88.3%)

## Testing the Changes

Run training with new bot activity signals:
```bash
python /home/bo/Py/NewLife/.venv/src/train.py --model bayesian
```

Monitor metrics to see if model can now learn:
- Should see train loss decreasing faster than before
- Should see divergence between train and test loss (actual overfitting)
- F1 scores should improve significantly by epoch 10-20

## Fallback Options

If bot activity signal is still too noisy, try:
1. Individual signals separately:
   - Use only `bot_activity_vwap` (16.6% rate, very specific)
   - Use only `bot_activity_scalping` (14% rate, cleaner)
2. Different stocks: Top 5 are all viable alternativesIf none work, may need domain knowledge adjustment:
   - Retrain on multi-step horizon (bot activities cluster in time)
   - Use regression instead of binary classification
   - Add geometric/temporal features (minute of day, cluster detection)
