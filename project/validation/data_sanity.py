"""
═══════════════════════════════════════════════════════════════════════════════
 1️⃣ DATA & SANITY CHECKS
═══════════════════════════════════════════════════════════════════════════════

Comprehensive data validation:
✅ Check input file structure
✅ Identify timeframes, instruments, feature types  
✅ Check for NaN, missing values, outliers
✅ Verify temporal integrity
✅ Detect look-ahead bias
✅ Detect target leakage
✅ Log data versions and dataset sizes
"""

import os
import sys
import glob
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import warnings

warnings.filterwarnings('ignore')

# Project imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DataSanityResult:
    """Results from data sanity checks"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
        print(f"  ⚠️  WARNING: {msg}")
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False
        print(f"  ❌ ERROR: {msg}")
    
    def add_success(self, msg: str):
        print(f"  ✅ {msg}")


class DataSanityChecker:
    """
    Comprehensive Data Sanity Checker
    
    Implements all checks from checkpoint 1:
    - File structure validation
    - Timeframe/instrument identification
    - NaN/outlier detection  
    - Temporal integrity
    - Look-ahead bias detection
    - Target leakage detection
    """
    
    # Expected columns in klines data
    KLINE_COLUMNS = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'symbol', 'interval', 'market_type'
    ]
    
    # Valid timeframes
    TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '12h', '1d', '3d', '1w']
    
    # Timeframe to bars per day mapping
    TF_BARS_PER_DAY = {
        '1m': 1440, '5m': 288, '15m': 96, '1h': 24, 
        '4h': 6, '12h': 2, '1d': 1, '3d': 0.33, '1w': 0.143
    }
    
    def __init__(self, data_dir: str, log_to_console: bool = True):
        """
        Args:
            data_dir: Path to data directory
            log_to_console: Whether to print logs
        """
        self.data_dir = Path(data_dir)
        self.log_to_console = log_to_console
        self.results = DataSanityResult()
        
        # Data inventory
        self.files_by_type: Dict[str, List[Path]] = defaultdict(list)
        self.symbols: set = set()
        self.timeframes: set = set()
        self.data_types: set = set()
        
    def log(self, msg: str, level: str = "INFO"):
        """Log message to console"""
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_checks(self) -> DataSanityResult:
        """Run all sanity checks"""
        print("\n" + "="*80)
        print("  1️⃣  DATA & SANITY CHECKS")
        print("="*80 + "\n")
        
        # 1. Scan and categorize files
        self.log("Step 1: Scanning data files...")
        self._scan_files()
        
        # 2. Validate file structure
        self.log("Step 2: Validating file structures...")
        self._validate_structures()
        
        # 3. Identify data types
        self.log("Step 3: Identifying timeframes, instruments, feature types...")
        self._identify_data_types()
        
        # 4. Check for NaN and outliers
        self.log("Step 4: Checking for NaN, missing values, outliers...")
        self._check_nan_outliers()
        
        # 5. Temporal integrity
        self.log("Step 5: Checking temporal integrity...")
        self._check_temporal_integrity()
        
        # 6. Look-ahead bias detection
        self.log("Step 6: Detecting look-ahead bias...")
        self._detect_lookahead_bias()
        
        # 7. Target leakage detection
        self.log("Step 7: Detecting target leakage...")
        self._detect_target_leakage()
        
        # 8. Version and size logging
        self.log("Step 8: Logging data versions and sizes...")
        self._log_versions_and_sizes()
        
        # Final summary
        self._print_summary()
        
        return self.results
    
    def _scan_files(self):
        """Scan and categorize all data files"""
        if not self.data_dir.exists():
            self.results.add_error(f"Data directory not found: {self.data_dir}")
            return
            
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            self.results.add_error("No CSV files found in data directory")
            return
        
        self.results.stats['total_files'] = len(csv_files)
        
        # Categorize by type
        for f in csv_files:
            name = f.name
            if name.startswith('klines_'):
                self.files_by_type['klines'].append(f)
            elif name.startswith('funding_rate_'):
                self.files_by_type['funding_rate'].append(f)
            elif name.startswith('long_short_ratio_'):
                self.files_by_type['long_short_ratio'].append(f)
            elif name.startswith('mark_price_'):
                self.files_by_type['mark_price'].append(f)
            elif name.startswith('open_interest_'):
                self.files_by_type['open_interest'].append(f)
            elif name.startswith('premium_index_'):
                self.files_by_type['premium_index'].append(f)
            elif name.startswith('taker_volume_'):
                self.files_by_type['taker_volume'].append(f)
            else:
                self.files_by_type['other'].append(f)
        
        self.results.add_success(f"Found {len(csv_files)} CSV files")
        for dtype, files in self.files_by_type.items():
            print(f"    - {dtype}: {len(files)} files")
    
    def _validate_structures(self):
        """Validate file structures match expectations"""
        errors = 0
        validated = 0
        
        # Check klines files
        for f in self.files_by_type['klines'][:5]:  # Sample check
            try:
                df = pd.read_csv(f, nrows=5)
                missing_cols = set(self.KLINE_COLUMNS) - set(df.columns)
                if missing_cols:
                    self.results.add_warning(f"{f.name}: Missing columns {missing_cols}")
                    errors += 1
                else:
                    validated += 1
            except Exception as e:
                self.results.add_error(f"Cannot read {f.name}: {e}")
                errors += 1
        
        if errors == 0:
            self.results.add_success(f"File structure validated ({validated} files sampled)")
        else:
            self.results.add_warning(f"Structure issues in {errors} files")
        
        self.results.stats['structure_errors'] = errors
    
    def _identify_data_types(self):
        """Identify timeframes, symbols, and feature types"""
        # Extract from klines filenames: klines_usdt_m_SYMBOL_TIMEFRAME.csv
        for f in self.files_by_type['klines']:
            name = f.stem  # filename without extension
            parts = name.split('_')
            if len(parts) >= 5:
                symbol = parts[-2]
                timeframe = parts[-1]
                self.symbols.add(symbol)
                if timeframe in self.TIMEFRAMES:
                    self.timeframes.add(timeframe)
        
        # Extract symbols from other files
        for dtype in ['funding_rate', 'long_short_ratio', 'open_interest']:
            for f in self.files_by_type[dtype]:
                name = f.stem
                parts = name.split('_')
                if len(parts) >= 4:
                    symbol = parts[-1]
                    self.symbols.add(symbol)
        
        self.data_types = set(self.files_by_type.keys())
        
        # Log findings
        print(f"\n  📊 DATA IDENTIFICATION:")
        print(f"    Timeframes: {sorted(self.timeframes)}")
        print(f"    Instruments: {len(self.symbols)} symbols")
        print(f"    Top symbols: {sorted(list(self.symbols))[:10]}...")
        print(f"    Data types: {sorted(self.data_types)}")
        
        self.results.stats['timeframes'] = list(self.timeframes)
        self.results.stats['n_symbols'] = len(self.symbols)
        self.results.stats['symbols'] = sorted(list(self.symbols))
        self.results.stats['data_types'] = list(self.data_types)
        
        # Categorize features by model type
        feature_types = {
            'scalp': ['5m', '15m'],
            'intraday': ['1h', '4h', '12h'],
            'swing': ['1d', '3d', '1w'],
            'risk': list(self.timeframes)  # uses all
        }
        
        print(f"\n  🎯 FEATURE SPACE BY MODEL:")
        for model, tfs in feature_types.items():
            available_tfs = [tf for tf in tfs if tf in self.timeframes]
            print(f"    {model.upper()}: {available_tfs}")
        
        self.results.stats['feature_types'] = feature_types
        self.results.add_success("Data types identified successfully")
    
    def _check_nan_outliers(self):
        """Check for NaN values and outliers"""
        nan_report = {}
        outlier_report = {}
        
        # Sample klines files
        sample_files = self.files_by_type['klines'][:10]
        
        for f in sample_files:
            try:
                df = pd.read_csv(f)
                
                # NaN check
                nan_counts = df.isnull().sum()
                nan_pct = (nan_counts / len(df) * 100).round(2)
                nan_cols = nan_pct[nan_pct > 0].to_dict()
                if nan_cols:
                    nan_report[f.name] = nan_cols
                
                # Outlier check (using IQR for price columns)
                for col in ['close', 'high', 'low', 'volume']:
                    if col in df.columns:
                        q1 = df[col].quantile(0.01)
                        q3 = df[col].quantile(0.99)
                        iqr = q3 - q1
                        outliers = ((df[col] < q1 - 3*iqr) | (df[col] > q3 + 3*iqr)).sum()
                        if outliers > 0:
                            if f.name not in outlier_report:
                                outlier_report[f.name] = {}
                            outlier_report[f.name][col] = outliers
                            
            except Exception as e:
                self.results.add_warning(f"Error checking {f.name}: {e}")
        
        # Report findings
        print(f"\n  🔍 NaN/OUTLIER CHECK:")
        
        if nan_report:
            print(f"    NaN found in {len(nan_report)} files:")
            for fname, cols in list(nan_report.items())[:3]:
                print(f"      {fname}: {cols}")
            self.results.add_warning(f"NaN values found in {len(nan_report)} files")
        else:
            self.results.add_success("No significant NaN values found")
        
        if outlier_report:
            print(f"    Outliers found in {len(outlier_report)} files:")
            for fname, cols in list(outlier_report.items())[:3]:
                print(f"      {fname}: {cols}")
            self.results.add_warning(f"Outliers found in {len(outlier_report)} files")
        else:
            self.results.add_success("No extreme outliers detected")
        
        self.results.stats['nan_report'] = nan_report
        self.results.stats['outlier_report'] = outlier_report
    
    def _check_temporal_integrity(self):
        """Check temporal integrity of data"""
        issues = []
        
        for f in self.files_by_type['klines'][:10]:
            try:
                df = pd.read_csv(f)
                
                # Check timestamp ordering
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Check monotonic
                    if not df['timestamp'].is_monotonic_increasing:
                        issues.append(f"{f.name}: Timestamps not monotonic")
                    
                    # Check gaps
                    if 'interval' in df.columns and len(df) > 1:
                        interval = df['interval'].iloc[0]
                        expected_delta = self._get_timedelta(interval)
                        
                        if expected_delta:
                            actual_deltas = df['timestamp'].diff().dropna()
                            gaps = (actual_deltas > expected_delta * 1.5).sum()
                            
                            if gaps > len(df) * 0.01:  # More than 1% gaps
                                gap_pct = (gaps / len(df) * 100)
                                issues.append(f"{f.name}: {gap_pct:.1f}% time gaps")
                                
            except Exception as e:
                issues.append(f"{f.name}: Error - {e}")
        
        print(f"\n  ⏰ TEMPORAL INTEGRITY:")
        
        if issues:
            for issue in issues[:5]:
                print(f"    ⚠️  {issue}")
            self.results.add_warning(f"Temporal issues found: {len(issues)}")
        else:
            self.results.add_success("Temporal integrity verified")
        
        self.results.stats['temporal_issues'] = issues
    
    def _get_timedelta(self, interval: str) -> pd.Timedelta:
        """Convert interval string to timedelta"""
        mapping = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '12h': pd.Timedelta(hours=12),
            '1d': pd.Timedelta(days=1),
            '3d': pd.Timedelta(days=3),
            '1w': pd.Timedelta(weeks=1),
        }
        return mapping.get(interval)
    
    def _detect_lookahead_bias(self):
        """
        Detect potential look-ahead bias
        
        Checks:
        - Future data in features
        - Incorrect shift directions
        - Features computed with future values
        """
        print(f"\n  🔮 LOOK-AHEAD BIAS DETECTION:")
        
        issues = []
        
        # Check feature files if they exist
        feature_dir = self.data_dir.parent / 'project' / 'features'
        
        if feature_dir.exists():
            for py_file in feature_dir.glob("*.py"):
                try:
                    content = py_file.read_text()
                    
                    # Look for suspicious patterns
                    suspicious_patterns = [
                        (r'\.shift\s*\(\s*-', "Negative shift (future data)"),
                        (r'\.iloc\s*\[\s*[^:]+:\s*\]', "Forward-looking iloc"),
                        (r'future', "Variable named 'future'"),
                        (r'target.*=.*close', "Target using raw close"),
                    ]
                    
                    import re
                    for pattern, desc in suspicious_patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE)
                        if matches:
                            issues.append(f"{py_file.name}: {desc}")
                            
                except Exception as e:
                    pass
        
        # Practical test: check correlation between features and future returns
        test_file = self.files_by_type['klines'][0] if self.files_by_type['klines'] else None
        
        if test_file:
            try:
                df = pd.read_csv(test_file)
                
                # Calculate returns
                df['return_1'] = df['close'].pct_change(1)
                df['future_return'] = df['close'].pct_change(1).shift(-1)
                
                # Check volume correlation with future return
                if 'volume' in df.columns:
                    corr = df['volume'].corr(df['future_return'])
                    if abs(corr) > 0.3:
                        issues.append(f"Volume-Future return correlation: {corr:.3f} (suspicious)")
                        
            except Exception as e:
                pass
        
        if issues:
            for issue in issues[:5]:
                print(f"    ⚠️  {issue}")
            self.results.add_warning(f"Potential look-ahead bias: {len(issues)} issues")
        else:
            self.results.add_success("No obvious look-ahead bias detected")
        
        self.results.stats['lookahead_issues'] = issues
    
    def _detect_target_leakage(self):
        """
        Detect target leakage
        
        Checks:
        - Features that directly contain target information
        - High correlation between features and target
        """
        print(f"\n  🚨 TARGET LEAKAGE DETECTION:")
        
        leaky_features = []
        
        # Load sample file and create synthetic target
        test_file = self.files_by_type['klines'][0] if self.files_by_type['klines'] else None
        
        if test_file:
            try:
                df = pd.read_csv(test_file)
                
                # Create target (future return direction)
                df['target'] = np.where(
                    df['close'].pct_change(6).shift(-6) > 0.01, 2,  # Up
                    np.where(df['close'].pct_change(6).shift(-6) < -0.01, 0, 1)  # Down/Neutral
                )
                
                # Check correlations
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        corr = df[col].corr(df['target'])
                        if abs(corr) > 0.5:
                            leaky_features.append((col, corr))
                
                # Check if any column name suggests leakage
                suspicious_names = ['target', 'future', 'label', 'y_']
                for col in df.columns:
                    for sus in suspicious_names:
                        if sus in col.lower() and col != 'target':
                            leaky_features.append((col, "suspicious name"))
                            
            except Exception as e:
                self.results.add_warning(f"Leakage test error: {e}")
        
        if leaky_features:
            for feat, corr in leaky_features[:5]:
                print(f"    ⚠️  {feat}: {corr}")
            self.results.add_warning(f"Potential target leakage: {len(leaky_features)} features")
        else:
            self.results.add_success("No obvious target leakage detected")
        
        self.results.stats['leaky_features'] = leaky_features
    
    def _log_versions_and_sizes(self):
        """Log data versions and dataset sizes"""
        print(f"\n  📁 DATA VERSION & SIZE:")
        
        total_rows = 0
        total_size_mb = 0
        file_stats = {}
        
        for dtype, files in self.files_by_type.items():
            dtype_rows = 0
            dtype_size = 0
            
            for f in files:
                try:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    total_size_mb += size_mb
                    dtype_size += size_mb
                    
                    # Sample row count
                    if len(files) <= 20 or files.index(f) % (len(files) // 10) == 0:
                        df = pd.read_csv(f)
                        dtype_rows += len(df)
                        total_rows += len(df)
                except:
                    pass
            
            file_stats[dtype] = {
                'files': len(files),
                'estimated_rows': dtype_rows,
                'size_mb': round(dtype_size, 2)
            }
            print(f"    {dtype}: {len(files)} files, ~{dtype_size:.1f} MB")
        
        # Calculate hash for version control
        version_hash = hashlib.md5(
            str(sorted([f.name for files in self.files_by_type.values() for f in files])).encode()
        ).hexdigest()[:12]
        
        print(f"\n    📊 TOTALS:")
        print(f"       Total files: {self.results.stats['total_files']}")
        print(f"       Total size: {total_size_mb:.1f} MB")
        print(f"       Data version hash: {version_hash}")
        print(f"       Timestamp: {datetime.now().isoformat()}")
        
        self.results.stats['total_size_mb'] = round(total_size_mb, 2)
        self.results.stats['file_stats'] = file_stats
        self.results.stats['data_version'] = version_hash
        self.results.stats['check_timestamp'] = datetime.now().isoformat()
        
        self.results.add_success(f"Data versioned: {version_hash}")
    
    def _print_summary(self):
        """Print final summary"""
        print("\n" + "="*80)
        print("  📋 DATA SANITY CHECK SUMMARY")
        print("="*80)
        
        print(f"\n  Total Errors: {len(self.results.errors)}")
        print(f"  Total Warnings: {len(self.results.warnings)}")
        
        if self.results.passed:
            print(f"\n  ✅ DATA SANITY CHECK: PASSED")
        else:
            print(f"\n  ❌ DATA SANITY CHECK: FAILED")
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    - {err}")
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings[:10]:
                print(f"    - {warn}")
        
        print("\n" + "="*80 + "\n")
    
    def save_report(self, output_path: str = None):
        """Save sanity check report to JSON"""
        if output_path is None:
            output_path = self.data_dir.parent / 'project' / 'validation' / 'data_sanity_report.json'
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'passed': self.results.passed,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'stats': self.results.stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_data_sanity_check(data_dir: str = None) -> DataSanityResult:
    """
    Run complete data sanity check
    
    Args:
        data_dir: Path to data directory. If None, uses default.
        
    Returns:
        DataSanityResult with all findings
    """
    if data_dir is None:
        # Find data directory
        script_dir = Path(__file__).parent.parent
        data_dir = script_dir.parent / 'data'
    
    checker = DataSanityChecker(data_dir)
    result = checker.run_all_checks()
    checker.save_report()
    
    return result


if __name__ == "__main__":
    result = run_data_sanity_check()
    sys.exit(0 if result.passed else 1)
