# Bayesian Transformer Diagnostics & Findings

## DIAGNOSIS SUMMARY

Your model is **not failing due to model size or regularization**. The **root cause is fundamental data signal weakness**.

### Evidence:

**Diagnostic 1 - Tiny model (32 dims, 1 layer):**
- Train Accuracy: 59.8% (random guessing at best)
- Test Accuracy: 59.4% 
- **Conclusion**: Model can only learn the class prior, not features

**Diagnostic 2 - Large model with strong regularization removed:**
- Train Loss: 5.6 (stuck)
- Test Loss: 5.5 (better than train - random!)
- **Conclusion**: Even without regularization, KL divergence blocks learning

**Diagnostic 3 - Large model without KL divergence:**
- Train Loss: 0.77, Test Loss: 0.84
- Cannot clearly overfit
- **Conclusion**: Model tries but features don't correlate with labels

---

## ROOT CAUSES

### 1. **KL Divergence Weight Too Strong**
   - Prior sigma=0.1 created huge KL penalty (5-6 loss)
   - Model couldn't overcome Bayesian prior to fit data
   - **Fix**: Reduced to prior_sigma=1.0 and KL weight=1e-4

### 2. **VWAP Mean Reversion Label is Too Noisy**
   - This target (price moves toward VWAP next step) is ~50% random
   - Stock price doesn't reliably revert to VWAP at 1-step horizon
   - Your features don't capture this signal
   - **This is the main problem**

### 3. **Regularization Was Compounding the Issue**
   - weight_decay=1e-5, label_smoothing=0.1, dropout=0.1
   - Made it impossible for model to even try to learn
   - **Fixed**: Reduced to weight_decay=1e-6, label_smoothing=0.0, dropout=0.05

---

## CHANGES MADE TO train.py

✅ Bayesian model config:
- prior_sigma: 0.1 → 1.0 (weaker prior)
- dropout: 0.1 → 0.05

✅ Hyperparameter config:
- learning_rate: 0.0001 → 0.001
- weight_decay: 1e-5 → 1e-6
- label_smoothing: 0.1 → 0.0

✅ KL weight:
- Changed from 1.0/len(train_indices) to fixed 1e-4

---

## WHAT YOU SHOULD DO NEXT

### Option 1: Debug Your Labels (Recommended)
```python
# Add this to check label quality:
from loader import TheSetGPU
ds = TheSetGPU()

# Inspect a session
x_batch, y_batch, seq_lengths = ds[0]

# What % are positive?
pos_rate = (y_batch == 1).sum().item() / len(y_batch)
print(f"Positive rate: {pos_rate:.1%}")  # Likely ~40%

# Are they clustered or random?
print(f"Positive indices: {torch.where(y_batch==1)[0].tolist()}")

# Check if there's ANY pattern
from scipy.stats import entropy
print(f"Entropy of labels: {entropy([pos_rate, 1-pos_rate]):.3f}")  
# Random = 0.693, deterministic = 0
```

**If entropy is >0.69**: Your labels are basically random. The model can't learn randomness.

### Option 2: Change Target Definition
Instead of VWAP mean reversion at 1-step horizon, try:
- **Raw direction**: Did price go up or down? (Binary, clearer signal)
- **Volatility increase**: Did volatility increase next step?
- **Volume change**: Did volume surge? (Bot activity indicator)
- **Longer horizon**: VWAP reversion over 5-10 steps (allows time to revert)

### Option 3: Use Regression Instead
- Predict **how much** price moved toward VWAP (continuous)
- Gives model finer gradients to learn from
- Change output_dim to match actual price levels

### Option 4: Feature Investigation
- **Do your features even correlate with the target?**
```python
import torch
import numpy as np

x_batch, y_batch, seq_lengths = ds[0]
correlations = []

for feature_idx in range(x_batch.shape[-1]):
    feature = x_batch[:, -1, feature_idx]  # Last timestep
    corr = np.corrcoef(feature.cpu().numpy(), y_batch.cpu().numpy())[0, 1]
    correlations.append(corr)

top_features = np.argsort(np.abs(correlations))[-5:]
print(f"Top correlated features: {top_features}")
print(f"Correlations: {[correlations[i] for i in top_features]}")

# If all correlations < 0.1: your features don't predict your target
```

---

## ROW 13/70 Analysis

Your reported metrics after training:
- Train F1: 0.4435
- Test F1: 0.4898  
- **Test > Train** = Not learning, model is random

With the changes above, you should see:
- ✓ Train F1 improving toward 0.6-0.7 (if signal exists)
- ✓ Gap between train/test (actual overfitting)
- ✓ Or staying flat ~0.5 (confirms no signal)

---

## TO RUN THE NEW CODE

```bash
python /home/bo/Py/NewLife/.venv/src/train.py --model bayesian
```

Should show different behavior by epoch 5-10 if the underlying data has any signal.

---

**My Assessment**: You've been debugging model architecture/training technique, but the real issue is **your target variable doesn't align with your features**. Focus on that first.
