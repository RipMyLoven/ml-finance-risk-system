"""
Advanced Risk Features Builder

Builds comprehensive risk features for the Risk Model including:
- Volatility regime features
- Correlation risk features
- Signal degradation detection
- Liquidity risk features
- Tail risk indicators

These features are used by the Risk Model to make
trade approval/rejection and position sizing decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


def build_advanced_risk_features(
    df: pd.DataFrame,
    market_predictions: Optional[Dict[str, np.ndarray]] = None,
    include_derivatives: bool = True,
    lookback_windows: List[int] = [5, 10, 20, 50, 100]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build comprehensive risk features for Risk Model
    
    Args:
        df: DataFrame with OHLCV data
        market_predictions: Optional dict with trading model predictions
        include_derivatives: Include funding/OI features if available
        lookback_windows: Windows for rolling calculations
        
    Returns:
        df: DataFrame with all features
        feature_names: List of feature column names
        
    Justification:
        Risk features capture different dimensions of risk:
        - Volatility: Price uncertainty
        - Correlation: Diversification breakdown risk
        - Liquidity: Execution risk
        - Tail risk: Extreme event probability
        - Signal degradation: Model reliability
    """
    print("Building Advanced Risk Features...")
    
    feature_cols = []
    
    # =======================================================================
    # 1. VOLATILITY REGIME FEATURES
    # =======================================================================
    df, vol_features = _add_volatility_features(df, lookback_windows)
    feature_cols.extend(vol_features)
    
    # =======================================================================
    # 2. TAIL RISK INDICATORS  
    # =======================================================================
    df, tail_features = _add_tail_risk_features(df, lookback_windows)
    feature_cols.extend(tail_features)
    
    # =======================================================================
    # 3. LIQUIDITY RISK FEATURES
    # =======================================================================
    df, liq_features = _add_liquidity_features(df, lookback_windows)
    feature_cols.extend(liq_features)
    
    # =======================================================================
    # 4. MARKET REGIME FEATURES
    # =======================================================================
    df, regime_features = _add_regime_features(df, lookback_windows)
    feature_cols.extend(regime_features)
    
    # =======================================================================
    # 5. CORRELATION RISK FEATURES
    # =======================================================================
    df, corr_features = _add_correlation_features(df, lookback_windows)
    feature_cols.extend(corr_features)
    
    # =======================================================================
    # 6. SIGNAL DEGRADATION FEATURES
    # =======================================================================
    df, signal_features = _add_signal_degradation_features(df, lookback_windows)
    feature_cols.extend(signal_features)
    
    # =======================================================================
    # 7. DRAWDOWN FEATURES
    # =======================================================================
    df, dd_features = _add_drawdown_features(df, lookback_windows)
    feature_cols.extend(dd_features)
    
    # =======================================================================
    # 8. DERIVATIVES RISK FEATURES (if available)
    # =======================================================================
    if include_derivatives:
        df, deriv_features = _add_derivatives_risk_features(df)
        feature_cols.extend(deriv_features)
    
    # =======================================================================
    # 9. MARKET MODEL FEATURES (if available)
    # =======================================================================
    if market_predictions is not None:
        df, model_features = _add_model_prediction_features(df, market_predictions)
        feature_cols.extend(model_features)
    else:
        # Add placeholder features
        df['model_confidence'] = 0.5
        df['model_uncertainty'] = 0.5
        df['model_signal_strength'] = 0.0
        feature_cols.extend(['model_confidence', 'model_uncertainty', 'model_signal_strength'])
    
    # =======================================================================
    # 10. RISK SCORE TARGET (for training)
    # =======================================================================
    df = _calculate_risk_target(df)
    
    # Clean up NaN and inf
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Filter only feature columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"  Total risk features: {len(feature_cols)}")
    
    return df, feature_cols


def _add_volatility_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Volatility regime features
    
    Captures:
    - Current volatility level
    - Volatility regime (high/low/normal)
    - Volatility clustering
    - Volatility term structure
    """
    features = []
    returns = df['close'].pct_change()
    
    # Historical volatility at multiple windows
    for w in windows:
        col_name = f'volatility_{w}'
        df[col_name] = returns.rolling(w).std()
        features.append(col_name)
    
    # Volatility ratios (term structure)
    df['vol_ratio_5_20'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['vol_ratio_10_50'] = df['volatility_10'] / (df['volatility_50'] + 1e-10)
    df['vol_ratio_20_100'] = df['volatility_20'] / (df['volatility_100'] + 1e-10)
    features.extend(['vol_ratio_5_20', 'vol_ratio_10_50', 'vol_ratio_20_100'])
    
    # Volatility regime classification
    vol_20 = df['volatility_20']
    vol_mean = vol_20.rolling(100).mean()
    vol_std = vol_20.rolling(100).std()
    
    df['vol_zscore'] = (vol_20 - vol_mean) / (vol_std + 1e-10)
    df['vol_regime_high'] = (df['vol_zscore'] > 1).astype(int)
    df['vol_regime_low'] = (df['vol_zscore'] < -1).astype(int)
    features.extend(['vol_zscore', 'vol_regime_high', 'vol_regime_low'])
    
    # Volatility persistence (GARCH-like)
    df['vol_change'] = df['volatility_5'].pct_change()
    df['vol_momentum'] = df['volatility_5'].diff(5)
    df['vol_clustering'] = (abs(returns) > returns.rolling(20).std() * 2).rolling(10).sum()
    features.extend(['vol_change', 'vol_momentum', 'vol_clustering'])
    
    # ATR-based volatility
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_50'] = tr.rolling(50).mean()
    df['atr_norm'] = df['atr_14'] / df['close']
    df['atr_ratio'] = df['atr_14'] / (df['atr_50'] + 1e-10)
    features.extend(['atr_14', 'atr_50', 'atr_norm', 'atr_ratio'])
    
    return df, features


def _add_tail_risk_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tail risk indicator features
    
    Captures:
    - Kurtosis (fat tails)
    - Skewness (directional tail risk)
    - Extreme value frequency
    - Historical VaR/CVaR estimates
    """
    features = []
    returns = df['close'].pct_change()
    
    # Rolling kurtosis (fat tails indicator)
    for w in [20, 50, 100]:
        col_name = f'kurtosis_{w}'
        df[col_name] = returns.rolling(w).kurt()
        features.append(col_name)
    
    # Rolling skewness (directional tail risk)
    for w in [20, 50, 100]:
        col_name = f'skewness_{w}'
        df[col_name] = returns.rolling(w).skew()
        features.append(col_name)
    
    # Extreme value frequency
    returns_std = returns.rolling(50).std()
    df['extreme_positive'] = ((returns > returns_std * 2.5).rolling(20).sum())
    df['extreme_negative'] = ((returns < -returns_std * 2.5).rolling(20).sum())
    df['extreme_ratio'] = df['extreme_negative'] / (df['extreme_positive'] + 1)
    features.extend(['extreme_positive', 'extreme_negative', 'extreme_ratio'])
    
    # Rolling VaR (95%)
    def rolling_var(x, q=0.05):
        return np.percentile(x, q * 100) if len(x) > 0 else 0
    
    df['var_95_50'] = returns.rolling(50).apply(lambda x: rolling_var(x, 0.05))
    df['var_99_50'] = returns.rolling(50).apply(lambda x: rolling_var(x, 0.01))
    features.extend(['var_95_50', 'var_99_50'])
    
    # Rolling CVaR (Expected Shortfall)
    def rolling_cvar(x, q=0.05):
        threshold = np.percentile(x, q * 100)
        tail = x[x <= threshold]
        return np.mean(tail) if len(tail) > 0 else threshold
    
    df['cvar_95_50'] = returns.rolling(50).apply(lambda x: rolling_cvar(x, 0.05))
    features.append('cvar_95_50')
    
    # Max drawdown in window
    for w in [20, 50, 100]:
        rolling_max = df['close'].rolling(w).max()
        dd = (df['close'] - rolling_max) / rolling_max
        df[f'max_dd_{w}'] = dd.rolling(w).min()
        features.append(f'max_dd_{w}')
    
    return df, features


def _add_liquidity_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Liquidity risk features
    
    Captures:
    - Volume patterns
    - Bid-ask spread proxy
    - Price impact estimates
    """
    features = []
    
    if 'volume' not in df.columns:
        return df, features
    
    # Volume ratios
    for w in [5, 10, 20, 50]:
        vol_sma = df['volume'].rolling(w).mean()
        df[f'volume_ratio_{w}'] = df['volume'] / (vol_sma + 1e-10)
        features.append(f'volume_ratio_{w}')
    
    # Volume trend
    vol_sma_5 = df['volume'].rolling(5).mean()
    vol_sma_20 = df['volume'].rolling(20).mean()
    vol_sma_50 = df['volume'].rolling(50).mean()
    
    df['volume_trend'] = vol_sma_5 / (vol_sma_20 + 1e-10) - 1
    df['volume_trend_long'] = vol_sma_20 / (vol_sma_50 + 1e-10) - 1
    features.extend(['volume_trend', 'volume_trend_long'])
    
    # Low liquidity indicator
    df['low_liquidity'] = (df['volume_ratio_20'] < 0.5).astype(int)
    df['very_low_liquidity'] = (df['volume_ratio_20'] < 0.25).astype(int)
    features.extend(['low_liquidity', 'very_low_liquidity'])
    
    # Volume volatility
    df['volume_volatility'] = df['volume'].pct_change().rolling(20).std()
    features.append('volume_volatility')
    
    # Amihud illiquidity measure proxy
    returns = df['close'].pct_change()
    df['illiquidity'] = abs(returns) / (df['volume'] * df['close'] + 1e-10) * 1e9
    df['illiquidity_sma'] = df['illiquidity'].rolling(20).mean()
    features.extend(['illiquidity', 'illiquidity_sma'])
    
    # Bid-ask spread proxy (using high-low)
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_proxy_sma'] = df['spread_proxy'].rolling(20).mean()
    df['spread_expansion'] = df['spread_proxy'] / (df['spread_proxy_sma'] + 1e-10)
    features.extend(['spread_proxy', 'spread_proxy_sma', 'spread_expansion'])
    
    return df, features


def _add_regime_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Market regime detection features
    
    Captures:
    - Trend regime
    - Mean reversion regime
    - Breakout/breakdown regime
    """
    features = []
    
    # Price vs moving averages
    for w in [20, 50, 100, 200]:
        if len(df) > w:
            sma = df['close'].rolling(w).mean()
            df[f'price_vs_sma_{w}'] = (df['close'] - sma) / sma
            features.append(f'price_vs_sma_{w}')
    
    # Trend strength
    sma_20 = df['close'].rolling(20).mean()
    sma_50 = df['close'].rolling(50).mean()
    df['trend_strength'] = (sma_20 - sma_50) / sma_50
    df['trend_direction'] = np.sign(df['trend_strength'])
    features.extend(['trend_strength', 'trend_direction'])
    
    # Regime classification
    returns_20 = df['close'].pct_change(20)
    df['regime_bull'] = (returns_20 > 0.05).astype(int)
    df['regime_bear'] = (returns_20 < -0.05).astype(int)
    df['regime_flat'] = ((returns_20 >= -0.05) & (returns_20 <= 0.05)).astype(int)
    features.extend(['regime_bull', 'regime_bear', 'regime_flat'])
    
    # Mean reversion indicator (price distance from MA)
    df['mean_reversion_potential'] = abs(df['price_vs_sma_20']) if 'price_vs_sma_20' in df.columns else 0
    features.append('mean_reversion_potential')
    
    # Breakout indicators
    high_20 = df['high'].rolling(20).max()
    low_20 = df['low'].rolling(20).min()
    df['near_high'] = (df['close'] >= high_20 * 0.98).astype(int)
    df['near_low'] = (df['close'] <= low_20 * 1.02).astype(int)
    df['range_position'] = (df['close'] - low_20) / (high_20 - low_20 + 1e-10)
    features.extend(['near_high', 'near_low', 'range_position'])
    
    return df, features


def _add_correlation_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Correlation risk features
    
    Captures:
    - Return autocorrelation
    - Volume-price correlation
    - Cross-timeframe correlation
    """
    features = []
    returns = df['close'].pct_change()
    
    # Return autocorrelation
    for lag in [1, 5, 10]:
        df[f'return_autocorr_{lag}'] = returns.rolling(50).apply(
            lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0
        )
        features.append(f'return_autocorr_{lag}')
    
    # Volume-return correlation
    if 'volume' in df.columns:
        vol_change = df['volume'].pct_change()
        df['vol_return_corr'] = returns.rolling(20).corr(vol_change)
        features.append('vol_return_corr')
    
    # High-low correlation with returns
    hl_range = (df['high'] - df['low']) / df['close']
    df['range_return_corr'] = returns.rolling(20).corr(hl_range)
    features.append('range_return_corr')
    
    # Momentum consistency
    mom_5 = df['close'].pct_change(5)
    mom_20 = df['close'].pct_change(20)
    df['momentum_consistency'] = np.sign(mom_5) == np.sign(mom_20)
    df['momentum_consistency'] = df['momentum_consistency'].astype(int)
    features.append('momentum_consistency')
    
    return df, features


def _add_signal_degradation_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Signal degradation detection features
    
    Captures:
    - Pattern breakdown indicators
    - Regime shift detection
    - Model reliability proxies
    """
    features = []
    returns = df['close'].pct_change()
    
    # Rolling Sharpe ratio as signal quality proxy
    for w in [20, 50, 100]:
        ret_mean = returns.rolling(w).mean()
        ret_std = returns.rolling(w).std()
        df[f'sharpe_{w}'] = ret_mean / (ret_std + 1e-10) * np.sqrt(252)
        features.append(f'sharpe_{w}')
    
    # Sharpe degradation
    df['sharpe_degradation'] = df['sharpe_20'] - df['sharpe_100']
    features.append('sharpe_degradation')
    
    # Information ratio stability
    df['ir_stability'] = df['sharpe_20'].rolling(20).std()
    features.append('ir_stability')
    
    # Regime change detection (structural break indicator)
    returns_zscore = (returns - returns.rolling(100).mean()) / (returns.rolling(100).std() + 1e-10)
    df['regime_break_indicator'] = (abs(returns_zscore) > 3).rolling(10).sum()
    features.append('regime_break_indicator')
    
    # Hit rate consistency (proxy for model degradation)
    positive_returns = (returns > 0).astype(int)
    df['hit_rate_20'] = positive_returns.rolling(20).mean()
    df['hit_rate_50'] = positive_returns.rolling(50).mean()
    df['hit_rate_drift'] = df['hit_rate_20'] - df['hit_rate_50']
    features.extend(['hit_rate_20', 'hit_rate_50', 'hit_rate_drift'])
    
    return df, features


def _add_drawdown_features(
    df: pd.DataFrame,
    windows: List[int]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drawdown-related features
    """
    features = []
    
    # Rolling max and drawdown
    rolling_max = df['close'].expanding().max()
    df['drawdown'] = (df['close'] - rolling_max) / rolling_max
    df['drawdown_depth'] = abs(df['drawdown'])
    features.extend(['drawdown', 'drawdown_depth'])
    
    # Drawdown duration
    in_dd = df['drawdown'] < 0
    df['dd_duration'] = in_dd.groupby((~in_dd).cumsum()).cumsum()
    features.append('dd_duration')
    
    # Recovery speed
    df['recovery_speed'] = df['drawdown'].diff()
    features.append('recovery_speed')
    
    # Underwater equity curve
    df['underwater'] = (df['drawdown'] < -0.05).astype(int)
    df['deep_underwater'] = (df['drawdown'] < -0.10).astype(int)
    features.extend(['underwater', 'deep_underwater'])
    
    return df, features


def _add_derivatives_risk_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derivatives market risk features (funding, OI, long/short ratio)
    """
    features = []
    
    # Funding rate features
    if 'funding_rate' in df.columns:
        df['funding_rate_norm'] = df['funding_rate'] * 1000
        df['funding_rate_sma'] = df['funding_rate'].rolling(8).mean() * 1000
        df['funding_rate_extreme'] = (abs(df['funding_rate']) > 0.001).astype(int)
        df['funding_rate_zscore'] = (
            (df['funding_rate'] - df['funding_rate'].rolling(24).mean()) /
            (df['funding_rate'].rolling(24).std() + 1e-10)
        )
        features.extend([
            'funding_rate_norm', 'funding_rate_sma', 
            'funding_rate_extreme', 'funding_rate_zscore'
        ])
    
    # Open interest features
    if 'open_interest' in df.columns:
        oi_change = df['open_interest'].pct_change()
        df['oi_change'] = oi_change
        df['oi_change_sma'] = oi_change.rolling(12).mean()
        df['oi_momentum'] = df['open_interest'].pct_change(24)
        features.extend(['oi_change', 'oi_change_sma', 'oi_momentum'])
    
    # Long/short ratio features
    if 'long_short_ratio' in df.columns:
        df['ls_ratio_norm'] = df['long_short_ratio'] - 1
        df['ls_ratio_extreme_long'] = (df['long_short_ratio'] > 2).astype(int)
        df['ls_ratio_extreme_short'] = (df['long_short_ratio'] < 0.5).astype(int)
        features.extend(['ls_ratio_norm', 'ls_ratio_extreme_long', 'ls_ratio_extreme_short'])
    
    return df, features


def _add_model_prediction_features(
    df: pd.DataFrame,
    predictions: Dict[str, np.ndarray]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Features from trading model predictions
    """
    features = []
    
    # Model confidence
    if 'confidence' in predictions:
        conf = predictions['confidence']
        if len(conf) == len(df):
            df['model_confidence'] = conf
        else:
            df['model_confidence'] = 0.5
        features.append('model_confidence')
    
    # Model uncertainty
    if 'P_up' in predictions and 'P_down' in predictions:
        p_up = predictions['P_up']
        p_down = predictions['P_down']
        if len(p_up) == len(df):
            df['model_P_up'] = p_up
            df['model_P_down'] = p_down
            df['model_uncertainty'] = 1 - np.maximum(p_up, p_down)
            df['model_signal_strength'] = p_up - p_down
        else:
            df['model_P_up'] = 0.33
            df['model_P_down'] = 0.33
            df['model_uncertainty'] = 0.5
            df['model_signal_strength'] = 0.0
        features.extend(['model_P_up', 'model_P_down', 'model_uncertainty', 'model_signal_strength'])
    
    # Expected return
    if 'expected_return' in predictions:
        exp_ret = predictions['expected_return']
        if len(exp_ret) == len(df):
            df['model_expected_return'] = exp_ret
        else:
            df['model_expected_return'] = 0.0
        features.append('model_expected_return')
    
    return df, features


def _calculate_risk_target(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """
    Calculate risk score target for supervised training
    
    Target components:
    - Maximum adverse excursion
    - Stop loss hit rate
    - Volatility spike
    """
    # Max adverse excursion for long
    future_lows = df['low'].rolling(window=horizon).min().shift(-horizon)
    df['mae_long'] = (df['close'] - future_lows) / df['close']
    
    # Max adverse excursion for short
    future_highs = df['high'].rolling(window=horizon).max().shift(-horizon)
    df['mae_short'] = (future_highs - df['close']) / df['close']
    
    # Combined MAE
    df['max_adverse_excursion'] = df[['mae_long', 'mae_short']].max(axis=1)
    
    # Stop loss hit at 2%
    df['stop_loss_hit'] = (df['max_adverse_excursion'] > 0.02).astype(int)
    
    # Volatility spike
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(20).std()
    future_vol = returns.shift(-horizon).rolling(horizon).std()
    df['volatility_spike'] = (future_vol / (rolling_vol + 1e-10) > 1.5).astype(int)
    
    # Combined risk score target (0-1)
    df['risk_target'] = (
        0.4 * df['max_adverse_excursion'].clip(0, 0.1) / 0.1 +
        0.3 * df['stop_loss_hit'] +
        0.3 * df['volatility_spike']
    ).clip(0, 1)
    
    return df
