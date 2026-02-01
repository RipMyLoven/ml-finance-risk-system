"""
QUICK FIXES - Apply these patches to fix the major issues

Run this BEFORE training to patch the existing pipeline.
Or use training/train_fixed.py directly.
"""

# ============================================================
# DECISION TREE: DIAGNOSING CV >> ACCURACY
# ============================================================
"""
IF CV Score (0.90-0.99) >> Real Accuracy (0.57):

├── Check 1: Is 'future_return' in your feature columns?
│   └── YES → 🔴 CRITICAL LEAKAGE! Remove it immediately
│
├── Check 2: Are you mixing multiple symbols before CV split?
│   └── YES → 🔴 CRITICAL! Use per-symbol TimeSeriesSplit
│
├── Check 3: Is StandardScaler fitted on full dataset?
│   └── YES → 🟡 LEAKAGE! Fit only on train split
│
├── Check 4: Is there a gap between train and val?
│   └── NO  → 🟡 Add purge_gap >= prediction_horizon
│
├── Check 5: Are rolling features computed correctly?
│   └── Check: cumsum() or cummax() without proper windowing
│   └── Fix: Use rolling(window) only

EXPECTED AFTER FIXES:
- CV Score: 0.35-0.45 (realistic for 3-class financial prediction)
- Real Accuracy: 0.33-0.42
- Gap: < 5%
"""

# ============================================================
# RULES OF THUMB
# ============================================================
"""
1. IF CV Score > 0.70 for financial 3-class prediction
   → SOMETHING IS WRONG (market is not 70% predictable)

2. IF Train Accuracy >> Val Accuracy (gap > 15%)
   → Overfitting. Reduce complexity.

3. IF Val Accuracy ≈ 33% (random)
   → Features have no signal, or target is pure noise

4. IF Reversed-time accuracy > 40%
   → Features contain future information

5. Financial markets baseline:
   - 2-class (up/down): random = 50%, achievable = 52-55%
   - 3-class (up/flat/down): random = 33%, achievable = 36-42%
   - Anything higher is suspicious without exceptional alpha

6. CV-to-Production gap:
   - < 3%: Excellent pipeline
   - 3-5%: Acceptable
   - 5-10%: Minor leakage, investigate
   - > 10%: Major leakage, stop and fix
"""

# ============================================================
# METRIC ALIGNMENT CHECKLIST
# ============================================================
"""
Your current setup:
- Loss: multi_logloss (probability calibration)
- CV Metric: accuracy or log_loss
- Real Metric: probably win_rate or PnL

PROBLEM: These optimize different things!

SOLUTION - Choose ONE consistent objective:

Option A: Directional Accuracy
- Loss: multiclass cross-entropy
- CV Metric: accuracy
- Real Metric: directional win rate
- Note: Ignores magnitude, all moves equal

Option B: Expected Return Optimization
- Loss: custom loss weighting by magnitude
- CV Metric: expected PnL
- Real Metric: actual PnL
- Note: Hard to implement correctly

Option C: Confidence-Based Trading
- Loss: multi_logloss
- CV Metric: log_loss + accuracy at confidence threshold
- Real Metric: accuracy only on high-confidence trades
- Note: You trade only when P(up) > 0.6 or P(down) > 0.6

RECOMMENDED: Option C with confidence threshold = 0.55
- Filters out noisy predictions
- Aligns CV evaluation with trading logic
"""

# ============================================================
# SAMPLE PATCH CODE
# ============================================================

def patch_time_series_split_with_purge():
    """
    Replace sklearn TimeSeriesSplit with purged version.
    Copy this class into your training code.
    """
    import numpy as np
    
    class PurgedTimeSeriesSplit:
        def __init__(self, n_splits=5, purge_gap=6, embargo_gap=0):
            self.n_splits = n_splits
            self.purge_gap = purge_gap
            self.embargo_gap = embargo_gap
        
        def split(self, X, y=None, groups=None):
            n = len(X)
            fold_size = n // (self.n_splits + 1)
            
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                val_start = train_end + self.purge_gap + self.embargo_gap
                val_end = val_start + fold_size
                
                if val_end > n:
                    val_end = n
                if val_start >= n:
                    break
                
                yield np.arange(train_end), np.arange(val_start, val_end)
        
        def get_n_splits(self, X=None, y=None, groups=None):
            return self.n_splits
    
    return PurgedTimeSeriesSplit


def get_confidence_filtered_accuracy(y_true, y_pred_proba, threshold=0.55):
    """
    Calculate accuracy only on high-confidence predictions.
    This should match your trading logic.
    """
    import numpy as np
    
    max_proba = np.max(y_pred_proba, axis=1)
    confident_mask = max_proba >= threshold
    
    if confident_mask.sum() == 0:
        return 0.0, 0
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = np.mean(y_true[confident_mask] == y_pred[confident_mask])
    n_trades = confident_mask.sum()
    
    return accuracy, n_trades


# ============================================================
# IMMEDIATE ACTION ITEMS
# ============================================================
"""
TODAY (30 min):
1. Run: python training/diagnose_leakage.py
   → Get exact breakdown of where leakage comes from

2. Run: python training/train_fixed.py
   → Train with corrected pipeline

3. Compare metrics:
   - Old CV: 0.90-0.99
   - New CV: should be 0.35-0.45
   - If new CV is still > 0.60, there's more leakage to find

THIS WEEK:
1. Implement confidence-based trading threshold
2. Add purge gap to all CV loops
3. Backtest with realistic slippage/commission

VALIDATION:
Your model is working correctly when:
- CV accuracy ≈ Holdout accuracy ≈ Live accuracy (within 5%)
- All three are in 35-45% range for 3-class
- Train accuracy - Val accuracy < 15%
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nRun: python training/diagnose_leakage.py")
    print("Then: python training/train_fixed.py")
