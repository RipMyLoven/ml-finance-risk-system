"""
═══════════════════════════════════════════════════════════════════════════════
 2️⃣ FEATURE ENGINEERING VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Feature validation and quality checks:
✅ Split feature space: Scalp, Intraday, Swing, Risk
✅ Check correlations
✅ Check multicollinearity (VIF)
✅ Remove unstable features
✅ Fix feature list for each model
✅ Save feature configs
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class FeatureValidationResult:
    """Results from feature validation"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    feature_configs: Dict[str, List[str]] = field(default_factory=dict)
    correlation_matrix: Optional[pd.DataFrame] = None
    removed_features: Dict[str, List[str]] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


class FeatureValidator:
    """
    Feature Engineering Validator
    
    Implements all checks from checkpoint 2:
    - Feature space splitting by model type
    - Correlation analysis
    - Multicollinearity detection (VIF)
    - Stability analysis
    - Feature config generation
    """
    
    # Feature prefixes for each model type
    SCALP_PREFIXES = [
        'log_return_', 'rsi_', 'micro_vol_', 'vol_ratio_', 'atr_',
        'volume_ratio_', 'volume_spike', 'volume_trend', 'spread_',
        'price_level_', 'momentum_', 'cum_return_'
    ]
    
    INTRADAY_PREFIXES = [
        'sma_', 'ema_', 'macd_', 'bb_', 'adx_', 'obv_', 
        'vwap_', 'regime_', 'trend_', 'support_', 'resistance_'
    ]
    
    SWING_PREFIXES = [
        'weekly_', 'monthly_', 'quarterly_', 'long_term_',
        'correlation_btc_', 'beta_', 'volatility_regime_',
        'funding_', 'open_interest_', 'sentiment_'
    ]
    
    RISK_PREFIXES = [
        'volatility_', 'drawdown_', 'var_', 'cvar_', 'sharpe_',
        'sortino_', 'max_dd_', 'correlation_', 'beta_', 'stress_'
    ]
    
    # Correlation thresholds
    HIGH_CORRELATION_THRESHOLD = 0.95  # Features too similar
    MODERATE_CORRELATION_THRESHOLD = 0.80
    
    # Variance threshold
    MIN_VARIANCE = 0.01
    
    # VIF threshold
    MAX_VIF = 10.0
    
    def __init__(self, data_dir: str = None, log_to_console: bool = True):
        self.data_dir = Path(data_dir) if data_dir else None
        self.log_to_console = log_to_console
        self.results = FeatureValidationResult()
        
        # Feature storage
        self.all_features: List[str] = []
        self.scalp_features: List[str] = []
        self.intraday_features: List[str] = []
        self.swing_features: List[str] = []
        self.risk_features: List[str] = []
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_checks(self, df: pd.DataFrame = None) -> FeatureValidationResult:
        """Run all feature validation checks"""
        print("\n" + "="*80)
        print("  2️⃣  FEATURE ENGINEERING VALIDATION")
        print("="*80 + "\n")
        
        # Load sample data if not provided
        if df is None:
            df = self._load_sample_data()
        
        if df is None or len(df) == 0:
            self.results.errors.append("No data available for validation")
            self.results.passed = False
            return self.results
        
        # 1. Identify and split feature space
        self.log("Step 1: Splitting feature space by model type...")
        self._split_feature_space(df)
        
        # 2. Check correlations
        self.log("Step 2: Checking feature correlations...")
        self._check_correlations(df)
        
        # 3. Check multicollinearity
        self.log("Step 3: Checking multicollinearity (VIF)...")
        self._check_multicollinearity(df)
        
        # 4. Check stability
        self.log("Step 4: Checking feature stability over time...")
        self._check_stability(df)
        
        # 5. Remove unstable/problematic features
        self.log("Step 5: Removing unstable features...")
        self._remove_unstable_features(df)
        
        # 6. Generate and save feature configs
        self.log("Step 6: Generating feature configs...")
        self._generate_feature_configs()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _load_sample_data(self) -> Optional[pd.DataFrame]:
        """Load sample data with features"""
        # Try to load from project data folder
        project_dir = Path(__file__).parent.parent
        
        # Try to find existing features
        feature_files = list((project_dir / 'data').glob('*features*.parquet'))
        if not feature_files:
            feature_files = list((project_dir / 'data').glob('*features*.csv'))
        
        if feature_files:
            try:
                if feature_files[0].suffix == '.parquet':
                    return pd.read_parquet(feature_files[0])
                else:
                    return pd.read_csv(feature_files[0])
            except Exception as e:
                self.log(f"Error loading features: {e}", "WARNING")
        
        # Try to build features from raw data
        if self.data_dir:
            klines = list(self.data_dir.glob('klines_*_1h.csv'))
            if klines:
                try:
                    df = pd.read_csv(klines[0])
                    df = self._build_basic_features(df)
                    return df
                except Exception as e:
                    self.log(f"Error building features: {e}", "WARNING")
        
        return None
    
    def _build_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build basic features for validation"""
        # Returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1))
        df['log_return_5'] = np.log(df['close'] / df['close'].shift(5))
        df['log_return_10'] = np.log(df['close'] / df['close'].shift(10))
        
        # RSI
        for period in [7, 14, 21]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # Volatility
        df['micro_vol_10'] = df['log_return_1'].rolling(10).std()
        df['micro_vol_20'] = df['log_return_1'].rolling(20).std()
        df['volatility_20'] = df['log_return_1'].rolling(20).std() * np.sqrt(252)
        
        # Volume
        df['volume_ratio_10'] = df['volume'] / df['volume'].rolling(10).mean()
        df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Moving averages
        for period in [10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df['macd_line'].ewm(span=9).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_norm'] = df['atr_14'] / df['close']
        
        # Risk features
        df['drawdown_20'] = (df['close'] / df['close'].rolling(20).max() - 1)
        df['max_dd_60'] = df['drawdown_20'].rolling(60).min()
        
        return df.dropna()
    
    def _split_feature_space(self, df: pd.DataFrame):
        """Split features into model-specific groups"""
        # Get all numeric columns except price/time
        exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'close_time', 'quote_volume', 'trades', 'symbol', 'interval',
                       'market_type', 'target', 'taker_buy_volume', 'taker_buy_quote_volume']
        
        self.all_features = [col for col in df.columns 
                           if col not in exclude_cols and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Classify features
        for feat in self.all_features:
            feat_lower = feat.lower()
            
            # Check scalp prefixes
            if any(prefix in feat_lower for prefix in ['log_return', 'rsi_', 'micro_vol', 
                   'volume_ratio', 'volume_spike', 'atr_', 'spread_', 'cum_return']):
                self.scalp_features.append(feat)
            
            # Check intraday prefixes
            if any(prefix in feat_lower for prefix in ['sma_', 'ema_', 'macd', 'bb_', 
                   'adx', 'obv', 'vwap', 'regime', 'trend']):
                self.intraday_features.append(feat)
            
            # Check swing prefixes
            if any(prefix in feat_lower for prefix in ['weekly', 'monthly', 'long_term',
                   'correlation_btc', 'beta', 'funding', 'open_interest', 'sentiment']):
                self.swing_features.append(feat)
            
            # Check risk prefixes
            if any(prefix in feat_lower for prefix in ['volatility', 'drawdown', 'var_', 
                   'cvar', 'sharpe', 'sortino', 'max_dd', 'stress']):
                self.risk_features.append(feat)
        
        # Remove duplicates
        self.scalp_features = list(set(self.scalp_features))
        self.intraday_features = list(set(self.intraday_features))
        self.swing_features = list(set(self.swing_features))
        self.risk_features = list(set(self.risk_features))
        
        # Ensure each model has minimum features
        if len(self.scalp_features) < 5:
            self.scalp_features.extend([f for f in self.all_features[:10] if f not in self.scalp_features])
        if len(self.intraday_features) < 5:
            self.intraday_features.extend([f for f in self.all_features if f not in self.intraday_features][:10])
        if len(self.swing_features) < 5:
            self.swing_features.extend([f for f in self.all_features if f not in self.swing_features][:10])
        if len(self.risk_features) < 5:
            self.risk_features.extend([f for f in self.all_features if f not in self.risk_features][:10])
        
        # Log results
        print(f"\n  📊 FEATURE SPACE SPLIT:")
        print(f"    Total features: {len(self.all_features)}")
        print(f"    Scalp features: {len(self.scalp_features)}")
        print(f"    Intraday features: {len(self.intraday_features)}")
        print(f"    Swing features: {len(self.swing_features)}")
        print(f"    Risk features: {len(self.risk_features)}")
        
        self.results.stats['total_features'] = len(self.all_features)
        self.results.stats['scalp_count'] = len(self.scalp_features)
        self.results.stats['intraday_count'] = len(self.intraday_features)
        self.results.stats['swing_count'] = len(self.swing_features)
        self.results.stats['risk_count'] = len(self.risk_features)
        
        self.log("Feature space split complete", "SUCCESS")
    
    def _check_correlations(self, df: pd.DataFrame):
        """Check feature correlations"""
        feature_df = df[self.all_features].dropna()
        
        if len(feature_df) < 100:
            self.log("Not enough data for correlation analysis", "WARNING")
            return
        
        # Calculate correlation matrix
        corr_matrix = feature_df.corr()
        self.results.correlation_matrix = corr_matrix
        
        # Find highly correlated pairs
        high_corr_pairs = []
        moderate_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_val)
                
                if corr_val > self.HIGH_CORRELATION_THRESHOLD:
                    high_corr_pairs.append(pair)
                elif corr_val > self.MODERATE_CORRELATION_THRESHOLD:
                    moderate_corr_pairs.append(pair)
        
        # Sort by correlation
        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        moderate_corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n  🔗 CORRELATION ANALYSIS:")
        
        if high_corr_pairs:
            print(f"    ❌ High correlation (>{self.HIGH_CORRELATION_THRESHOLD}):")
            for f1, f2, corr in high_corr_pairs[:10]:
                print(f"       {f1} <-> {f2}: {corr:.3f}")
            self.results.warnings.append(f"Found {len(high_corr_pairs)} highly correlated feature pairs")
        else:
            print(f"    ✅ No highly correlated pairs found")
        
        if moderate_corr_pairs:
            print(f"\n    ⚠️  Moderate correlation ({self.MODERATE_CORRELATION_THRESHOLD}-{self.HIGH_CORRELATION_THRESHOLD}):")
            for f1, f2, corr in moderate_corr_pairs[:5]:
                print(f"       {f1} <-> {f2}: {corr:.3f}")
        
        self.results.stats['high_corr_pairs'] = len(high_corr_pairs)
        self.results.stats['moderate_corr_pairs'] = len(moderate_corr_pairs)
        
        # Store for removal
        self.results.removed_features['high_correlation'] = [p[1] for p in high_corr_pairs]
    
    def _check_multicollinearity(self, df: pd.DataFrame):
        """Check multicollinearity using VIF"""
        feature_df = df[self.all_features].dropna()
        
        if len(feature_df) < 100 or len(self.all_features) < 3:
            self.log("Not enough data/features for VIF analysis", "WARNING")
            return
        
        # Sample if too large
        if len(feature_df) > 10000:
            feature_df = feature_df.sample(10000, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        scaled_df = pd.DataFrame(scaled_features, columns=self.all_features)
        
        # Calculate VIF for each feature
        vif_data = []
        high_vif_features = []
        
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            
            for i, col in enumerate(scaled_df.columns[:20]):  # Limit for speed
                try:
                    vif = variance_inflation_factor(scaled_df.values, i)
                    vif_data.append({'feature': col, 'vif': vif})
                    
                    if vif > self.MAX_VIF:
                        high_vif_features.append((col, vif))
                except:
                    pass
        except ImportError:
            # Fallback: simple correlation-based check
            self.log("statsmodels not available, using correlation proxy", "WARNING")
            
            for col in scaled_df.columns[:20]:
                max_corr = scaled_df.corr()[col].abs().drop(col).max()
                approx_vif = 1 / (1 - max_corr**2 + 1e-10)
                vif_data.append({'feature': col, 'vif': approx_vif})
                
                if approx_vif > self.MAX_VIF:
                    high_vif_features.append((col, approx_vif))
        
        print(f"\n  📈 MULTICOLLINEARITY (VIF) CHECK:")
        
        if high_vif_features:
            print(f"    ⚠️  Features with VIF > {self.MAX_VIF}:")
            for feat, vif in sorted(high_vif_features, key=lambda x: x[1], reverse=True)[:10]:
                print(f"       {feat}: VIF = {vif:.2f}")
            self.results.warnings.append(f"Found {len(high_vif_features)} features with high VIF")
        else:
            print(f"    ✅ No high multicollinearity detected")
        
        self.results.stats['high_vif_count'] = len(high_vif_features)
        self.results.removed_features['high_vif'] = [f[0] for f in high_vif_features]
    
    def _check_stability(self, df: pd.DataFrame):
        """Check feature stability over time"""
        unstable_features = []
        
        if 'timestamp' not in df.columns or len(df) < 1000:
            self.log("Cannot check stability without sufficient time series data", "WARNING")
            return
        
        # Split into time periods
        n_periods = 5
        period_size = len(df) // n_periods
        
        for feat in self.all_features[:30]:  # Sample features
            try:
                period_stats = []
                
                for i in range(n_periods):
                    start_idx = i * period_size
                    end_idx = (i + 1) * period_size
                    period_data = df[feat].iloc[start_idx:end_idx].dropna()
                    
                    if len(period_data) > 10:
                        period_stats.append({
                            'mean': period_data.mean(),
                            'std': period_data.std(),
                            'median': period_data.median()
                        })
                
                if len(period_stats) >= 3:
                    # Check if statistics vary significantly
                    means = [s['mean'] for s in period_stats]
                    stds = [s['std'] for s in period_stats]
                    
                    mean_cv = np.std(means) / (np.mean(np.abs(means)) + 1e-10)
                    std_cv = np.std(stds) / (np.mean(stds) + 1e-10)
                    
                    if mean_cv > 1.0 or std_cv > 1.0:
                        unstable_features.append((feat, mean_cv, std_cv))
                        
            except Exception as e:
                pass
        
        print(f"\n  ⚡ FEATURE STABILITY CHECK:")
        
        if unstable_features:
            print(f"    ⚠️  Unstable features detected:")
            for feat, mean_cv, std_cv in sorted(unstable_features, key=lambda x: x[1], reverse=True)[:10]:
                print(f"       {feat}: mean_cv={mean_cv:.2f}, std_cv={std_cv:.2f}")
            self.results.warnings.append(f"Found {len(unstable_features)} unstable features")
        else:
            print(f"    ✅ All checked features are stable")
        
        self.results.stats['unstable_features_count'] = len(unstable_features)
        self.results.removed_features['unstable'] = [f[0] for f in unstable_features]
    
    def _remove_unstable_features(self, df: pd.DataFrame):
        """Remove unstable features from each model's feature set"""
        features_to_remove = set()
        
        # Collect all features to remove
        for key, features in self.results.removed_features.items():
            features_to_remove.update(features)
        
        # Also remove low variance features
        try:
            feature_df = df[self.all_features].dropna()
            selector = VarianceThreshold(threshold=self.MIN_VARIANCE)
            selector.fit(feature_df)
            low_var_mask = ~selector.get_support()
            low_var_features = [self.all_features[i] for i in range(len(self.all_features)) if low_var_mask[i]]
            features_to_remove.update(low_var_features)
            self.results.removed_features['low_variance'] = low_var_features
        except Exception as e:
            self.log(f"Variance check failed: {e}", "WARNING")
        
        # Remove from each feature set
        self.scalp_features = [f for f in self.scalp_features if f not in features_to_remove]
        self.intraday_features = [f for f in self.intraday_features if f not in features_to_remove]
        self.swing_features = [f for f in self.swing_features if f not in features_to_remove]
        self.risk_features = [f for f in self.risk_features if f not in features_to_remove]
        
        print(f"\n  🧹 FEATURE REMOVAL:")
        print(f"    Total features removed: {len(features_to_remove)}")
        for reason, features in self.results.removed_features.items():
            if features:
                print(f"    - {reason}: {len(features)} features")
        
        print(f"\n    Features after cleanup:")
        print(f"    - Scalp: {len(self.scalp_features)}")
        print(f"    - Intraday: {len(self.intraday_features)}")
        print(f"    - Swing: {len(self.swing_features)}")
        print(f"    - Risk: {len(self.risk_features)}")
        
        self.log("Unstable features removed", "SUCCESS")
    
    def _generate_feature_configs(self):
        """Generate and save feature configurations"""
        self.results.feature_configs = {
            'scalp': sorted(self.scalp_features),
            'intraday': sorted(self.intraday_features),
            'swing': sorted(self.swing_features),
            'risk': sorted(self.risk_features),
            'all': sorted(self.all_features)
        }
        
        # Save configs
        config_dir = Path(__file__).parent / 'configs'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / 'feature_configs.json'
        
        config_data = {
            'version': datetime.now().isoformat(),
            'configs': self.results.feature_configs,
            'stats': {
                'total_original': len(self.all_features),
                'total_removed': sum(len(f) for f in self.results.removed_features.values()),
                'scalp_final': len(self.scalp_features),
                'intraday_final': len(self.intraday_features),
                'swing_final': len(self.swing_features),
                'risk_final': len(self.risk_features),
            },
            'removal_reasons': {k: len(v) for k, v in self.results.removed_features.items()}
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n  💾 FEATURE CONFIGS SAVED:")
        print(f"    Path: {config_file}")
        
        self.log("Feature configs generated and saved", "SUCCESS")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("  📋 FEATURE VALIDATION SUMMARY")
        print("="*80)
        
        print(f"\n  Total Features: {len(self.all_features)}")
        print(f"  Total Warnings: {len(self.results.warnings)}")
        print(f"  Total Errors: {len(self.results.errors)}")
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings[:10]:
                print(f"    ⚠️  {warn}")
        
        if self.results.errors:
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    ❌ {err}")
            self.results.passed = False
        
        if self.results.passed:
            print(f"\n  ✅ FEATURE VALIDATION: PASSED")
        else:
            print(f"\n  ❌ FEATURE VALIDATION: FAILED")
        
        print("\n" + "="*80 + "\n")
    
    def save_report(self, output_path: str = None):
        """Save validation report"""
        if output_path is None:
            output_path = Path(__file__).parent / 'feature_validation_report.json'
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'feature_configs': self.results.feature_configs,
            'removed_features': self.results.removed_features,
            'stats': self.results.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_feature_validation(df: pd.DataFrame = None, data_dir: str = None) -> FeatureValidationResult:
    """
    Run complete feature validation
    
    Args:
        df: DataFrame with features (optional)
        data_dir: Path to data directory
        
    Returns:
        FeatureValidationResult
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data'
    
    validator = FeatureValidator(data_dir)
    result = validator.run_all_checks(df)
    validator.save_report()
    
    return result


if __name__ == "__main__":
    result = run_feature_validation()
    sys.exit(0 if result.passed else 1)
