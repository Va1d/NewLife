import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from lb import drank, load_panda, DB
from torch import tensor
from typing import *
from progressbar import progressbar
from os import path
from collections import defaultdict
import functools
import torch.nn.functional as F

MIN_DAY = 1148
MAX_DAY = 2546 
MISS_DAYS = [2269, 2309]
DAY_PATH = "/home/bo/Py/DAYS"
codes = drank()

def pull_day(day:int)->Tuple[tensor,tensor]:
    res = {code:{} for code in codes}
    with DB() as db:
        for r in db("SELECT R.code, R.minute, R.open, R.close, R.high, R.low, R.volume, R.trade_count, R.vwap FROM r16.red R INNER JOIN r16.codes C ON R.code = C.code AND C.cosher WHERE R.day=%(day)s",day=day):
            res[r[0]][r[1]] = list(r[2:])
    d, m = [[res[code][i] if i in res[code] else [0]*7 for i in range(390)] for code in codes],[[1 if i in res[code] else 0 for i in range(390)] for code in codes]
    return tensor(d), tensor(m)

def pull_days():
    for day in progressbar(range(MIN_DAY, MAX_DAY+1)):
        if day in MISS_DAYS:
            continue
        day_data, day_mask = pull_day(day)
        torch.save(day_data, path.join(DAY_PATH,f"{day}_data.pt"))
        torch.save(day_mask, path.join(DAY_PATH,f"{day}_mask.pt"))


def aggregate():
    data, mask = [],[]
    for day in progressbar(range(MIN_DAY, MAX_DAY+1)):
        if day in MISS_DAYS:
            continue
        dd = torch.load(path.join(DAY_PATH,f"{day}_data.pt"))
        dm = torch.load(path.join(DAY_PATH,f"{day}_mask.pt"))
        data.append(dd.permute(1,0,2))
        mask.append(dm.permute(1,0).unsqueeze(-1))
    d, m= torch.stack(data), torch.stack(mask)    
    torch.save(d, path.join(DAY_PATH,f"aggregate_{MAX_DAY}_data.pt"))
    torch.save(m, path.join(DAY_PATH,f"aggregate_{MAX_DAY}_mask.pt"))

class Extender:
    def __init__(self, input_stock_idx=None, target_stock_idx=30)->None:
        data = torch.load(path.join(DAY_PATH,f"aggregate_{MAX_DAY}_data.pt"))
        mask = torch.load(path.join(DAY_PATH,f"aggregate_{MAX_DAY}_mask.pt"))

        # If input_stock_idx specified, slice input features to single stock
        if input_stock_idx is not None:
            self.data = data[:, :, input_stock_idx:input_stock_idx+1, :]
            self.mask = mask[:, :, input_stock_idx:input_stock_idx+1, :]
        else:
            self.data = data
            self.mask = mask

        self.target_stock_idx = target_stock_idx
        self.eps = 1e-8

    def prev(self, t:tensor)->tensor:
        degap = torch.gather(t, 1, torch.cummax(torch.arange(390).reshape((1,390,1,1))*self.mask,dim=1).values)
        return torch.cat((degap[:,0:1], degap[:,:-1]),1)        
    def exp(self, t:tensor)->tensor:
        m = self.mask == 0        
        xmin = torch.cummin(t.masked_fill(m, float('inf')),dim=1).values
        xmax = torch.cummax(t.masked_fill(m, -float('inf')),dim=1).values
        denom = (xmax - xmin).clamp(min=self.eps)
        out = (t - xmin) / denom
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out[:,6:]
        
    def prev_day_mean(self, t:tensor)->tensor:
        return torch.nanmean(t.masked_fill(self.mask == 0, float('nan')),dim=1,keepdim=True)     
    def zs(self,t:tensor)->tensor:  
        xmean = self.prev_day_mean(t)  
        diffs = (t - xmean).pow(2)
        var_sum = torch.where(self.mask==1, diffs, 0.0).sum(dim=1, keepdim=True)
        count = (self.mask.sum(dim=1, keepdim=True)-1).clamp(min=1)
        xstd = (var_sum/count).sqrt()
        denom = xstd.clamp(min=self.eps)
        zscores = (t[1:] - xmean[:-1]) / denom[:-1]
        zscores = zscores.masked_fill(self.mask[1:] == 0, 0)
        # Clip to reasonable range (-5, 5 is 99.9999% of normal distribution)
        out = torch.clamp(zscores, -5.0, 5.0)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    @functools.cached_property
    def open(self)->tensor:
        return self.data[:,:,:,0:1]
    @functools.cached_property
    def close(self)->tensor:
        return self.data[:,:,:,1:2]
    @functools.cached_property
    def high(self)->tensor:
        return self.data[:,:,:,2:3]
    @functools.cached_property
    def low(self)->tensor:
        return self.data[:,:,:,3:4]    
    @functools.cached_property
    def volume(self)->tensor:
        return self.data[:,:,:,4:5]    
    @functools.cached_property
    def trade_count(self)->tensor:
        return self.data[:,:,:,5:6] 
    @functools.cached_property
    def vwap(self)->tensor:
        return self.data[:,:,:,6:7] 
    @functools.cached_property
    def prev_close(self)->tensor:
        return self.prev(self.close)
    @functools.cached_property
    def typical_price(self)->tensor:        
        return (self.close + self.high + self.low)/3
    @functools.cached_property
    def ha_close(self)->tensor:        
        return (self.open + self.close + self.high + self.low)/4
    @functools.cached_property
    def true_range(self)->tensor:
        return torch.max(self.high-self.low, torch.max(torch.abs(self.high - self.prev_close), torch.abs(self.low - self.prev_close)))
    
    @functools.cached_property
    def log_return(self)->tensor:
        denom = self.prev_close.clamp(min=self.eps)
        ratio = (self.close / denom).clamp(min=self.eps)
        log_ret = torch.log(ratio)
        # Clip to reasonable range (±0.5 = ±65% price change)
        out = torch.clamp(log_ret, -0.5, 0.5)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    @functools.cached_property
    def flow(self)->tensor:   
        denom = (self.high - self.low).clamp(min=self.eps)
        out = (((self.close - self.low) - (self.high - self.close)) / denom)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    @functools.cached_property
    def open_zs(self)->tensor:
        return self.zs(self.open)
    @functools.cached_property
    def close_zs(self)->tensor:
        return self.zs(self.close)
    @functools.cached_property
    def high_zs(self)->tensor:
        return self.zs(self.high)
    @functools.cached_property
    def low_zs(self)->tensor:
        return self.zs(self.low)
    @functools.cached_property
    def volume_zs(self)->tensor:
        return self.zs(self.volume)
    @functools.cached_property
    def trade_count_zs(self)->tensor:
        return self.zs(self.trade_count)    
    @functools.cached_property
    def vwap_zs(self)->tensor:
        return self.zs(self.vwap)     
    @functools.cached_property
    def true_range_zs(self)->tensor:
        return self.zs(self.true_range)            
    @functools.cached_property
    def trade_size(self)->tensor:
        denom = (self.trade_count * self.typical_price).clamp(min=self.eps)
        out = (self.volume / denom)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Bot-focused indicators
    @functools.cached_property
    def vwap_deviation(self)->tensor:
        """Deviation from VWAP - key for VWAP execution algos"""
        denom = self.vwap.clamp(min=self.eps)
        out = ((self.close - self.vwap) / denom)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    @functools.cached_property
    def volume_velocity(self)->tensor:
        """Rate of change in volume - detects bot execution bursts"""
        prev_vol = self.prev(self.volume)
        denom = prev_vol.clamp(min=self.eps)
        velocity = (self.volume - prev_vol) / denom
        # Clip extreme velocities (prevent explosion from near-zero prev values)
        out = torch.clamp(velocity, -10.0, 10.0)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    @functools.cached_property
    def trade_velocity(self)->tensor:
        """Rate of change in trade count - bot activity indicator"""
        prev_tc = self.prev(self.trade_count)
        denom = prev_tc.clamp(min=self.eps)
        velocity = (self.trade_count - prev_tc) / denom
        # Clip extreme velocities (prevent explosion from near-zero prev values)
        out = torch.clamp(velocity, -10.0, 10.0)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    @functools.cached_property
    def rsi(self)->tensor:
        """RSI momentum - simple returns proxy to avoid dimension issues"""
        # Use recent returns as momentum proxy (simple and stable)
        returns = self.close / self.prev_close.clamp(min=self.eps) - 1
        
        # Clip extreme returns and normalize to 0-1 range
        returns_clipped = torch.clamp(returns, -0.1, 0.1)  # -10% to +10%
        rsi_proxy = (returns_clipped + 0.1) / 0.2  # Map to 0-1
        
        out = rsi_proxy[1:]
        return torch.nan_to_num(out, nan=0.5, posinf=1.0, neginf=0.0)
    
    @functools.cached_property
    def spread_ratio(self)->tensor:
        """High-Low spread relative to close - liquidity/volatility indicator"""
        denom = self.close.clamp(min=self.eps)
        out = ((self.high - self.low) / denom)[1:]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    
    @functools.cached_property
    def masking(self)->tensor:
        return self.mask[1:].float()
    
    @functools.cached_property
    def minute(self)->tensor:
        return torch.arange(390).reshape((1,390,1,1)).float().expand_as(self.mask)[1:]
    
    @functools.cached_property
    def bot_activity_vwap(self)->tensor:
        """Bot activity: volume spike with price stability (VWAP execution signature)
        
        VWAP algorithms must execute large volume without moving price.
        This is the mechanical fingerprint of algorithmic execution.
        """
        volume_spike = self.volume_velocity > 2.0
        price_stable = torch.abs(self.log_return) < 0.005
        return (volume_spike & price_stable).float()
    
    @functools.cached_property
    def bot_activity_scalping(self)->tensor:
        """Bot activity: rapid-fire trade execution (scalping/market making)
        
        Retail traders can't execute 50+ trades per minute - this is bot behavior.
        """
        return (self.trade_velocity > 1.5).float()
    
    @functools.cached_property
    def bot_activity_ensemble(self)->tensor:
        """Combined bot activity signal: multiple algorithmic fingerprints
        
        Label = 1 if:
        - Volume spike with stable price (VWAP execution), OR
        - Trade count spike (scalping/market making)
        
        This captures the most obvious bot behaviors.
        
        Returns: [days, seq_len] - averaged across stocks
        """
        v_spike = self.volume_velocity > 2.0
        p_stable = torch.abs(self.log_return) < 0.005
        t_spike = self.trade_velocity > 1.5
        
        # Combine signals: (volume spike with stable price) OR  (trade spike)
        combined = (v_spike & p_stable) | t_spike  # [days, seq_len, 36, 1]
        
        # Average across stocks to get single time series per day
        # This makes sense: if ANY stock is showing bot activity, mark that time
        result = combined.squeeze(-1).max(dim=2)[0].float()  # [days, seq_len]
        
        return result
    
    @functools.cached_property
    def expanded(self)->tensor:
        """Bot-focused features: volume, trade patterns, VWAP, momentum"""
        return torch.flatten(torch.concat((
            self.volume_zs,         # Volume patterns (bot execution)
            self.trade_count_zs,    # Trade frequency (scalping indicator)
            self.vwap_zs,           # VWAP level
            self.vwap_deviation,    # Distance from VWAP (mean reversion signal)
            self.volume_velocity,   # Volume rate of change
            self.trade_velocity,    # Trade count rate of change
            self.trade_size,        # Average trade size (retail vs institutional)
            self.rsi,               # Momentum (trend-following bots)
            self.true_range_zs,     # Volatility
            self.spread_ratio,      # Bid-ask spread proxy
            self.log_return,        # Price momentum
            self.flow,              # Money flow
            self.minute             # Time of day (opening/closing patterns)
        ), dim=3)[:,1:],2)
    @functools.cached_property
    def target(self)->tensor:
        """Predict VWAP mean reversion - key bot behavior pattern
        
        Target = 1 if price moves toward VWAP (bot execution signal)
        Target = 0 if price moves away or stays (no bot pattern)
        
        This captures VWAP execution algos and mean-reversion bots
        """
        # Current VWAP deviation
        curr_vwap_dev = (self.close - self.vwap) / self.vwap.clamp(min=self.eps)
        # Next day's VWAP deviation  
        next_vwap_dev = (self.close[1:] - self.vwap[1:]) / self.vwap[1:].clamp(min=self.eps)
        
        # Mean reversion: absolute deviation decreases
        curr_dev_abs = torch.abs(curr_vwap_dev[:-1])
        next_dev_abs = torch.abs(next_vwap_dev)
        
        # 1 = moved toward VWAP (reversion), 0 = moved away or stayed
        reversion = (next_dev_abs < curr_dev_abs).float()
        
        # Get last 256 values for target stock
        if self.data.shape[2] == 1:
            stock_idx = 0
        else:
            stock_idx = int(self.target_stock_idx) if self.target_stock_idx is not None else 0
            if stock_idx < 0 or stock_idx >= self.data.shape[2]:
                stock_idx = 0

        binary_target = reversion[:,-256:,stock_idx,0]
        
        return torch.nan_to_num(binary_target, nan=0.0, posinf=0.0, neginf=0.0)

    @functools.cached_property
    def target_bot_activity(self)->tensor:
        """Bot activity target: predict when algorithmic trading is happening on target stock
        
        Much cleaner signal than VWAP reversion - ~20% positive rate
        vs ~40% for VWAP (which is near-random)
        
        Returns: [days, 256] with bot activity for the target stock
        """
        # Get target stock index
        if self.data.shape[2] == 1:
            stock_idx = 0
        else:
            stock_idx = int(self.target_stock_idx) if self.target_stock_idx is not None else 0
            if stock_idx < 0 or stock_idx >= self.data.shape[2]:
                stock_idx = 0
        
        # Ensemble bot activity signal
        v_spike = self.volume_velocity > 2.0  # [days, seq_len, 36, 1]
        p_stable = torch.abs(self.log_return) < 0.005  # [days, seq_len, 36, 1]
        t_spike = self.trade_velocity > 1.5  # [days, seq_len, 36, 1]
        
        # Create signal for target stock only
        combined = (v_spike & p_stable) | t_spike  # [days, seq_len, 36, 1]
        
        # Extract target stock and squeeze
        bot_signal = combined[:, :, stock_idx, 0].float()  # [days, seq_len]
        
        # Take last 256 values
        binary_target = bot_signal[:, -256:]  # [days, 256]
        
        return torch.nan_to_num(binary_target, nan=0.0, posinf=0.0, neginf=0.0)

class TheSet(Dataset):
    def __init__(self, target_stock_idx=10, input_stock_idx=None)->None:
        super().__init__()
        ex = Extender(input_stock_idx=input_stock_idx, target_stock_idx=target_stock_idx)
        self.x = ex.expanded
        self.y = ex.target_bot_activity  # Changed from ex.target (VWAP) to bot activity
    def __len__(self)->int:
        return self.x.shape[0]
    def __getitem__(self, index):
        return self.sequence(self.x[index], self.y[index])
    def sequence(self, x:tensor, y:tensor)->Iterable[Tuple[tensor,tensor]]:
        # Predict 3 steps ahead, so stop at 253 (253+3=256)
        for i in range(253):
            yield x[:133+i], torch.stack([y[i+1], y[i+2], y[i+3]], dim=0)


class TheSetGPU(Dataset):
    """GPU-based dataset for bot behavior prediction"""
    def __init__(self, device='cuda:1', target_stock_idx=10, input_stock_idx=None)->None:
        super().__init__()
        # Default target is Stock #10 (cleanest bot activity signals); inputs use all stocks unless specified
        ex = Extender(input_stock_idx=input_stock_idx, target_stock_idx=target_stock_idx)
        self.x = ex.expanded.to(device)
        # Target: Bot activity detection (ensemble signal)
        self.y = ex.target_bot_activity.to(device)
        self.device = device
    def __len__(self)->int:
        return self.x.shape[0]
    
    def __getitem__(self, index):
        """Return entire session as parallel sequences for batched processing
        
        Returns:
            x_batch: [256, max_seq_len, n_features] - All input sequences padded to max length
            y_batch: [256] - Binary target values (1=VWAP reversion, 0=no reversion)
            seq_lengths: [256] - Actual sequence length for each step (133, 134, ..., 388)
            
        Note: Predicting bot behavior (VWAP mean reversion) rather than raw price direction
        """
        x = self.x[index]  # [seq_len, n_features]
        y = self.y[index]  # [256]
        # target_valid = self.target_valid_mask[index]  # [256]  # Old mask - not using anymore
        
        num_steps = 256
        max_seq_len = 133 + num_steps - 1  # 388
        n_features = x.shape[1]  # Dynamic feature dimension
        
        # Prepare batched sequences
        x_batch = torch.zeros(num_steps, max_seq_len, n_features, device=self.device)
        y_batch = torch.zeros(num_steps, device=self.device)
        # target_valid_mask = torch.zeros(num_steps, device=self.device)  # Old mask - not using
        seq_lengths = torch.zeros(num_steps, dtype=torch.long, device=self.device)
        
        for i in range(num_steps):
            seq_len = 133 + i
            # Copy actual sequence data (padding handled by transformer attention masks)
            x_batch[i, :seq_len] = x[:seq_len]
            # Target: VWAP mean reversion (1=price moved toward VWAP, 0=away or stayed)
            y_batch[i] = y[i]
            # target_valid_mask[i] = target_valid[i]  # Old mask - commented out
            seq_lengths[i] = seq_len
        
        return x_batch, y_batch, seq_lengths  # Removed target_valid_mask from return





if __name__=="__main__":
    ds = TheSetGPU()
    for j in range(len(ds)):
        x, y, seq_lengths = ds[j]  # No target_valid_mask anymore
        print(f"x.shape: {x.shape}, y.shape: {y.shape}, seq_lengths.shape: {seq_lengths.shape}")
        print(f"y (binary targets): {y}")
        print(f"Positive samples (price up): {y.sum().item()}/{len(y)}")
        break
        





    
        



        





