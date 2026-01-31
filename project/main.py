"""
Main Orchestrator - главный модуль системы

Архитектура:
    Binance API
        ↓
    Data Collector
        ↓
    Feature Engineering (multi-TF)
        ↓
    TF Models (Scalp / Intraday / Swing)
        ↓
    Meta Decision Engine
        ↓
    Risk AI
        ↓
    Signal Generator
        ↓
    Optinus Backtest
"""

import os
import sys
import argparse
import pickle
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Количество доступных ядер CPU
N_CORES = mp.cpu_count()

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TOP_SYMBOLS, MODEL_PATHS, 
    SCALP_TIMEFRAMES, INTRADAY_TIMEFRAMES, SWING_TIMEFRAMES,
    PREDICTION_HORIZON, PRICE_MOVE_THRESHOLD
)
from data.data_collector import BinanceDataCollector
from features.scalp_features import build_scalp_features
from features.intraday_features import build_intraday_features
from features.swing_features import build_swing_features
from training.train_scalp import load_model as load_scalp_model, predict as predict_scalp
from training.train_intraday import load_model as load_intraday_model, predict as predict_intraday
from training.train_swing import load_model as load_swing_model, predict as predict_swing
from training.train_risk import load_model as load_risk_model, predict_risk, build_risk_features
from meta.meta_engine import MetaDecisionEngine
from meta.ranking import CoinRanker, calculate_correlation_matrix
from signals.signal_generator import SignalGenerator, TradingSignal


class TradingSystem:
    """
    Главный класс торговой системы
    
    Объединяет все компоненты:
    - Data collection
    - Feature engineering
    - Model prediction
    - Meta decision
    - Signal generation
    
    Использует все доступные ядра CPU для параллельной обработки.
    """
    
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        symbols: List[str] = None,
        n_workers: int = -1
    ):
        """
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbols: список торговых пар
            n_workers: количество worker'ов (-1 = все ядра CPU)
        """
        self.symbols = symbols or TOP_SYMBOLS
        self.n_workers = N_CORES if n_workers == -1 else min(n_workers, N_CORES)
        
        # Initialize components
        print("Initializing Trading System...")
        print(f"Using {self.n_workers} CPU cores for parallel processing")
        
        # Data collector
        self.data_collector = BinanceDataCollector(api_key, api_secret)
        
        # Models (lazy loading)
        self._scalp_model = None
        self._intraday_model = None
        self._swing_model = None
        self._risk_model = None
        
        # Meta components
        self.meta_engine = MetaDecisionEngine()
        self.ranker = CoinRanker()
        self.signal_generator = SignalGenerator()
        
        print("Trading System initialized!")
    
    @property
    def scalp_model(self):
        """Lazy load scalp model"""
        if self._scalp_model is None:
            try:
                self._scalp_model, self._scalp_features, _ = load_scalp_model()
                print("Scalp model loaded")
            except FileNotFoundError:
                print("Warning: Scalp model not found")
        return self._scalp_model
    
    @property
    def intraday_model(self):
        """Lazy load intraday model"""
        if self._intraday_model is None:
            try:
                self._intraday_model, self._intraday_features, _ = load_intraday_model()
                print("Intraday model loaded")
            except FileNotFoundError:
                print("Warning: Intraday model not found")
        return self._intraday_model
    
    @property
    def swing_model(self):
        """Lazy load swing model"""
        if self._swing_model is None:
            try:
                self._swing_model, self._swing_features, _ = load_swing_model()
                print("Swing model loaded")
            except FileNotFoundError:
                print("Warning: Swing model not found")
        return self._swing_model
    
    @property
    def risk_model(self):
        """Lazy load risk model"""
        if self._risk_model is None:
            try:
                self._risk_model, self._risk_features, _ = load_risk_model()
                print("Risk model loaded")
            except FileNotFoundError:
                print("Warning: Risk model not found")
        return self._risk_model
    
    def collect_data(self, days: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Собрать данные для всех символов
        
        Returns:
            Dict[symbol][timeframe] = DataFrame
        """
        print(f"\nCollecting data for {len(self.symbols)} symbols using {self.n_workers} workers...")
        return self.data_collector.collect_all_symbols_data(self.symbols, days)
    
    def _process_symbol_features(self, symbol: str, tf_data: Dict[str, pd.DataFrame]) -> tuple:
        """
        Обработка фичей для одного символа (для параллельного выполнения)
        """
        symbol_features = {}
        
        try:
            # Scalp features (5m)
            if '5m' in tf_data:
                df, feature_names = build_scalp_features(
                    tf_data['5m'].copy(),
                    horizon=PREDICTION_HORIZON['scalp'],
                    threshold=PRICE_MOVE_THRESHOLD['scalp'] / 100
                )
                if len(df) > 0:
                    symbol_features['scalp_df'] = df
                    symbol_features['scalp_features'] = df[feature_names].values[-1:].astype(np.float32)
            
            # Intraday features (1h)
            if '1h' in tf_data:
                df, feature_names = build_intraday_features(
                    tf_data['1h'].copy(),
                    horizon=PREDICTION_HORIZON['intraday'],
                    threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100
                )
                if len(df) > 0:
                    symbol_features['intraday_df'] = df
                    symbol_features['intraday_features'] = df[feature_names].values[-1:].astype(np.float32)
            
            # Swing features (1d)
            if '1d' in tf_data:
                df, feature_names = build_swing_features(
                    tf_data['1d'].copy(),
                    horizon=PREDICTION_HORIZON['swing'],
                    threshold=PRICE_MOVE_THRESHOLD['swing'] / 100
                )
                if len(df) > 0:
                    symbol_features['swing_df'] = df
                    symbol_features['swing_features'] = df[feature_names].values[-1:].astype(np.float32)
            
            # Current price and volatility
            if '5m' in tf_data and len(tf_data['5m']) > 0:
                symbol_features['current_price'] = tf_data['5m']['close'].iloc[-1]
                returns = tf_data['5m']['close'].pct_change()
                symbol_features['volatility'] = returns.rolling(20).std().iloc[-1]
                symbol_features['avg_volatility'] = returns.rolling(100).std().iloc[-1]
        
        except Exception as e:
            print(f"  Error processing {symbol}: {e}")
        
        return (symbol, symbol_features)
    
    def prepare_features(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict]:
        """
        Подготовить фичи для всех символов и таймфреймов
        ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА с использованием всех ядер CPU
        
        Returns:
            Dict[symbol] = {
                'scalp_features': array,
                'intraday_features': array,
                'swing_features': array,
                'scalp_df': DataFrame,
                'intraday_df': DataFrame,
                'swing_df': DataFrame,
            }
        """
        print(f"\nPreparing features using {self.n_workers} CPU cores...")
        
        all_features = {}
        
        if JOBLIB_AVAILABLE and self.n_workers > 1 and len(data) > 1:
            # Параллельная обработка с joblib
            results = Parallel(n_jobs=self.n_workers, verbose=1, backend='loky')(
                delayed(self._process_symbol_features)(symbol, tf_data)
                for symbol, tf_data in data.items()
            )
            
            for symbol, symbol_features in results:
                if symbol_features:
                    all_features[symbol] = symbol_features
        else:
            # Последовательная обработка
            for symbol, tf_data in data.items():
                print(f"  Processing {symbol}...")
                _, symbol_features = self._process_symbol_features(symbol, tf_data)
                if symbol_features:
                    all_features[symbol] = symbol_features
        
        print(f"  Processed {len(all_features)} symbols")
        return all_features
    
    def _predict_symbol(self, symbol: str, symbol_features: Dict) -> tuple:
        """
        Предсказание для одного символа (для параллельного выполнения)
        """
        preds = {}
        
        try:
            # Scalp prediction
            if self.scalp_model and 'scalp_features' in symbol_features:
                X = symbol_features['scalp_features']
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                pred = predict_scalp(self.scalp_model, X)
                preds['scalp'] = {
                    'P_up': float(pred['P_up'][0]),
                    'P_down': float(pred['P_down'][0]),
                    'P_flat': float(pred['P_flat'][0]),
                    'expected_return': float(pred['expected_return'][0])
                }
            else:
                preds['scalp'] = {'P_up': 0.33, 'P_down': 0.33, 'P_flat': 0.34, 'expected_return': 0}
            
            # Intraday prediction
            if self.intraday_model and 'intraday_features' in symbol_features:
                X = symbol_features['intraday_features']
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                pred = predict_intraday(self.intraday_model, X)
                preds['intraday'] = {
                    'P_up': float(pred['P_up'][0]),
                    'P_down': float(pred['P_down'][0]),
                    'P_flat': float(pred['P_flat'][0]),
                    'expected_return': float(pred['expected_return'][0])
                }
            else:
                preds['intraday'] = {'P_up': 0.33, 'P_down': 0.33, 'P_flat': 0.34, 'expected_return': 0}
            
            # Swing prediction
            if self.swing_model and 'swing_features' in symbol_features:
                X = symbol_features['swing_features']
                X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                pred = predict_swing(self.swing_model, X)
                preds['swing'] = {
                    'P_up': float(pred['P_up'][0]),
                    'P_down': float(pred['P_down'][0]),
                    'P_flat': float(pred['P_flat'][0]),
                    'expected_return': float(pred['expected_return'][0])
                }
            else:
                preds['swing'] = {'P_up': 0.33, 'P_down': 0.33, 'P_flat': 0.34, 'expected_return': 0}
            
            # Risk prediction
            if self.risk_model and 'scalp_df' in symbol_features:
                risk_feature_names = self._risk_features if hasattr(self, '_risk_features') else None
                
                if risk_feature_names:
                    df = symbol_features['scalp_df'].copy()
                    available_features = [f for f in risk_feature_names if f in df.columns]
                    
                    if len(available_features) == len(risk_feature_names):
                        X = df[available_features].values[-1:].astype(np.float32)
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        risk_pred = predict_risk(
                            self.risk_model, X,
                            current_price=symbol_features.get('current_price', 0),
                            volatility=symbol_features.get('volatility', 0.02)
                        )
                        preds['risk_score'] = float(risk_pred['risk_score'][0])
                    else:
                        preds['risk_score'] = 0.5
                else:
                    preds['risk_score'] = 0.5
            else:
                preds['risk_score'] = 0.5
                
        except Exception as e:
            print(f"  Prediction error for {symbol}: {e}")
            preds = {
                'scalp': {'P_up': 0.33, 'P_down': 0.33, 'P_flat': 0.34, 'expected_return': 0},
                'intraday': {'P_up': 0.33, 'P_down': 0.33, 'P_flat': 0.34, 'expected_return': 0},
                'swing': {'P_up': 0.33, 'P_down': 0.33, 'P_flat': 0.34, 'expected_return': 0},
                'risk_score': 0.5
            }
        
        return (symbol, preds)
    
    def predict_all(
        self,
        features: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Получить предсказания всех моделей
        ПАРАЛЛЕЛЬНАЯ ОБРАБОТКА с использованием ThreadPoolExecutor
        
        Returns:
            Dict[symbol] = {
                'scalp': {'P_up': float, 'P_down': float, 'expected_return': float},
                'intraday': {...},
                'swing': {...},
                'risk_score': float
            }
        """
        print(f"\nGetting predictions using {self.n_workers} workers...")
        
        predictions = {}
        
        # Используем ThreadPoolExecutor для параллельных предсказаний
        # (ThreadPool лучше для I/O-bound и light CPU tasks, модели LightGBM thread-safe)
        if self.n_workers > 1 and len(features) > 1:
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {
                    executor.submit(self._predict_symbol, symbol, symbol_features): symbol
                    for symbol, symbol_features in features.items()
                }
                
                for future in as_completed(futures):
                    symbol, preds = future.result()
                    predictions[symbol] = preds
        else:
            # Последовательное выполнение
            for symbol, symbol_features in features.items():
                _, preds = self._predict_symbol(symbol, symbol_features)
                predictions[symbol] = preds
        
        print(f"  Predictions complete for {len(predictions)} symbols")
        return predictions
    
    def generate_signals(
        self,
        predictions: Dict[str, Dict],
        features: Dict[str, Dict]
    ) -> List[TradingSignal]:
        """
        Сгенерировать торговые сигналы
        """
        print("\nGenerating signals...")
        
        # Meta decisions
        decisions = self.meta_engine.batch_decide(predictions)
        
        # Debug print decisions
        print("\n--- Meta Decisions Debug ---")
        for symbol, decision in decisions.items():
            print(f"{symbol}: {decision.direction}, conf={decision.confidence:.3f}, "
                  f"should_trade={decision.should_trade}, reason={decision.reason}")
        print("----------------------------\n")
        
        # Calculate correlation matrix
        price_returns = {}
        for symbol, symbol_features in features.items():
            if 'scalp_df' in symbol_features:
                returns = symbol_features['scalp_df']['close'].pct_change().dropna().values
                if len(returns) > 24:
                    price_returns[symbol] = returns
        
        correlation_matrix = calculate_correlation_matrix(price_returns) if price_returns else None
        
        # Rank coins
        selected_coins = self.ranker.select_top_coins(decisions, correlation_matrix)
        
        # Print ranking
        print(self.ranker.get_summary(selected_coins))
        
        # Extract prices and volatilities
        prices = {}
        volatilities = {}
        avg_volatilities = {}
        
        for symbol, symbol_features in features.items():
            if 'current_price' in symbol_features:
                prices[symbol] = symbol_features['current_price']
            if 'volatility' in symbol_features:
                volatilities[symbol] = symbol_features['volatility']
            if 'avg_volatility' in symbol_features:
                avg_volatilities[symbol] = symbol_features['avg_volatility']
        
        # Generate signals
        signals = self.signal_generator.generate_signals(
            selected_coins,
            decisions,
            prices,
            volatilities,
            avg_volatilities
        )
        
        return signals
    
    def run(
        self,
        collect_data: bool = True,
        data_days: int = 30,
        data: Dict = None
    ) -> List[TradingSignal]:
        """
        Запустить полный цикл системы
        
        Args:
            collect_data: собирать ли новые данные
            data_days: за сколько дней собирать данные
            data: предварительно собранные данные (если есть)
            
        Returns:
            List of TradingSignals
        """
        print("\n" + "="*60)
        print("RUNNING TRADING SYSTEM")
        print(f"Time: {datetime.now()}")
        print("="*60)
        
        # Step 1: Collect data
        if collect_data or data is None:
            data = self.collect_data(days=data_days)
        
        if not data:
            print("Error: No data collected!")
            return []
        
        # Step 2: Prepare features
        features = self.prepare_features(data)
        
        if not features:
            print("Error: No features generated!")
            return []
        
        # Step 3: Get predictions
        predictions = self.predict_all(features)
        
        # Step 4: Generate signals
        signals = self.generate_signals(predictions, features)
        
        # Step 5: Output
        print("\n" + "="*60)
        print("GENERATED SIGNALS")
        print("="*60)
        
        if signals:
            for signal in signals:
                print(signal.format())
        else:
            print("No signals generated at this time.")
        
        return signals
    
    def train_models(
        self,
        data_path: str = None,
        n_iterations: int = 500
    ):
        """
        Обучить все модели на данных из CSV файлов
        
        Args:
            data_path: путь к директории с CSV файлами
            n_iterations: количество итераций LightGBM
        """
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        from data.data_loader import prepare_training_data
        from features.scalp_features import build_scalp_features
        from features.intraday_features import build_intraday_features  
        from features.swing_features import build_swing_features
        from training.train_scalp import train_model as train_scalp, save_model as save_scalp
        from training.train_intraday import train_model as train_intraday, save_model as save_intraday
        from training.train_swing import train_model as train_swing, save_model as save_swing
        from training.train_risk import train_model as train_risk, save_model as save_risk, build_risk_features
        from config import MODEL_PATHS, PREDICTION_HORIZON, PRICE_MOVE_THRESHOLD
        
        # Determine data path
        if data_path is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(os.path.dirname(project_dir), 'data')
        
        # Load and prepare data
        print(f"\nLoading data from: {data_path}")
        data = prepare_training_data(
            data_dir=data_path,
            timeframes=['5m', '15m', '1h', '4h', '1d', '3d']
        )
        
        if not data:
            print("ERROR: No data loaded!")
            return
        
        print(f"\nLoaded data for {len(data)} symbols")
        
        # Combine data from all symbols for training
        all_scalp_features = []
        all_scalp_targets = []
        all_intraday_features = []
        all_intraday_targets = []
        all_swing_features = []
        all_swing_targets = []
        all_risk_features = []
        all_risk_targets = []
        
        scalp_feature_names = None
        intraday_feature_names = None
        swing_feature_names = None
        risk_feature_names = None
        
        for symbol, tf_data in data.items():
            print(f"\nProcessing {symbol}...")
            
            # Scalp features (5m)
            if '5m' in tf_data:
                try:
                    df = tf_data['5m'].copy()
                    df, feature_names = build_scalp_features(
                        df,
                        horizon=PREDICTION_HORIZON['scalp'],
                        threshold=PRICE_MOVE_THRESHOLD['scalp'] / 100
                    )
                    if len(df) > 100 and 'target' in df.columns:
                        scalp_feature_names = feature_names
                        X = df[feature_names].values
                        y = df['target'].values
                        valid_mask = ~np.isnan(y)
                        all_scalp_features.append(X[valid_mask])
                        all_scalp_targets.append(y[valid_mask])
                        print(f"  Scalp: {len(X[valid_mask])} samples")
                except Exception as e:
                    print(f"  Scalp error: {e}")
            
            # Intraday features (1h)
            if '1h' in tf_data:
                try:
                    df = tf_data['1h'].copy()
                    df, feature_names = build_intraday_features(
                        df,
                        horizon=PREDICTION_HORIZON['intraday'],
                        threshold=PRICE_MOVE_THRESHOLD['intraday'] / 100
                    )
                    if len(df) > 50 and 'target' in df.columns:
                        intraday_feature_names = feature_names
                        X = df[feature_names].values
                        y = df['target'].values
                        valid_mask = ~np.isnan(y)
                        all_intraday_features.append(X[valid_mask])
                        all_intraday_targets.append(y[valid_mask])
                        print(f"  Intraday: {len(X[valid_mask])} samples")
                except Exception as e:
                    print(f"  Intraday error: {e}")
            
            # Swing features (1d)
            if '1d' in tf_data:
                try:
                    df = tf_data['1d'].copy()
                    df, feature_names = build_swing_features(
                        df,
                        horizon=PREDICTION_HORIZON['swing'],
                        threshold=PRICE_MOVE_THRESHOLD['swing'] / 100
                    )
                    if len(df) > 30 and 'target' in df.columns:
                        swing_feature_names = feature_names
                        X = df[feature_names].values
                        y = df['target'].values
                        valid_mask = ~np.isnan(y)
                        all_swing_features.append(X[valid_mask])
                        all_swing_targets.append(y[valid_mask])
                        print(f"  Swing: {len(X[valid_mask])} samples")
                except Exception as e:
                    print(f"  Swing error: {e}")
            
            # Risk features (from 5m data)
            if '5m' in tf_data:
                try:
                    df = tf_data['5m'].copy()
                    # Build basic risk features
                    df, feature_names = build_risk_features(df)
                    if len(df) > 100:
                        risk_feature_names = feature_names
                        X = df[feature_names].values
                        # Simple risk target: future volatility
                        df['future_vol'] = df['close'].pct_change().rolling(20).std().shift(-20)
                        y = df['future_vol'].values
                        valid_mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
                        all_risk_features.append(X[valid_mask])
                        all_risk_targets.append(y[valid_mask])
                        print(f"  Risk: {len(X[valid_mask])} samples")
                except Exception as e:
                    print(f"  Risk error: {e}")
        
        # Train Scalp model
        print("\n" + "-"*40)
        print("TRAINING SCALP MODEL")
        print("-"*40)
        if all_scalp_features:
            X_scalp = np.vstack(all_scalp_features).astype(np.float32)
            y_scalp = np.concatenate(all_scalp_targets).astype(np.int32)
            X_scalp = np.nan_to_num(X_scalp, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Training data: {X_scalp.shape}")
            model, metrics = train_scalp(X_scalp, y_scalp, n_iterations=n_iterations)
            save_scalp(model, scalp_feature_names, metrics)
            print(f"Scalp model saved! Metrics: {metrics}")
        else:
            print("No scalp training data available")
        
        # Train Intraday model
        print("\n" + "-"*40)
        print("TRAINING INTRADAY MODEL")
        print("-"*40)
        if all_intraday_features:
            X_intraday = np.vstack(all_intraday_features).astype(np.float32)
            y_intraday = np.concatenate(all_intraday_targets).astype(np.int32)
            X_intraday = np.nan_to_num(X_intraday, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Training data: {X_intraday.shape}")
            model, metrics = train_intraday(X_intraday, y_intraday, n_iterations=n_iterations)
            save_intraday(model, intraday_feature_names, metrics)
            print(f"Intraday model saved! Metrics: {metrics}")
        else:
            print("No intraday training data available")
        
        # Train Swing model
        print("\n" + "-"*40)
        print("TRAINING SWING MODEL")
        print("-"*40)
        if all_swing_features:
            X_swing = np.vstack(all_swing_features).astype(np.float32)
            y_swing = np.concatenate(all_swing_targets).astype(np.int32)
            X_swing = np.nan_to_num(X_swing, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Training data: {X_swing.shape}")
            model, metrics = train_swing(X_swing, y_swing, n_iterations=n_iterations)
            save_swing(model, swing_feature_names, metrics)
            print(f"Swing model saved! Metrics: {metrics}")
        else:
            print("No swing training data available")
        
        # Train Risk model
        print("\n" + "-"*40)
        print("TRAINING RISK MODEL")
        print("-"*40)
        if all_risk_features:
            X_risk = np.vstack(all_risk_features).astype(np.float32)
            y_risk = np.concatenate(all_risk_targets).astype(np.float32)
            X_risk = np.nan_to_num(X_risk, nan=0.0, posinf=0.0, neginf=0.0)
            y_risk = np.nan_to_num(y_risk, nan=0.0, posinf=0.0, neginf=0.0)
            print(f"Training data: {X_risk.shape}")
            model, metrics = train_risk(X_risk, y_risk, n_iterations=n_iterations)
            save_risk(model, risk_feature_names, metrics)
            print(f"Risk model saved! Metrics: {metrics}")
        else:
            print("No risk training data available")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='AI Crypto Trading System')
    parser.add_argument('--mode', type=str, default='run',
                        choices=['run', 'train', 'backtest', 'predict'],
                        help='Operation mode')
    parser.add_argument('--symbols', type=str, nargs='+', default=None,
                        help='Symbols to trade')
    parser.add_argument('--days', type=int, default=30,
                        help='Days of data to collect')
    parser.add_argument('--no-collect', action='store_true',
                        help='Skip data collection')
    parser.add_argument('--use-csv', action='store_true',
                        help='Use CSV data instead of API')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to CSV data folder')
    parser.add_argument('--workers', type=int, default=-1,
                        help='Number of CPU workers (-1 = all cores)')
    
    args = parser.parse_args()
    
    # Количество worker'ов
    n_workers = N_CORES if args.workers == -1 else min(args.workers, N_CORES)
    
    print("\n" + "="*60)
    print("AI CRYPTO TRADING SYSTEM")
    print("="*60)
    print(f"CPU Cores Available: {N_CORES}")
    print(f"Using Workers: {n_workers}")
    print("="*60)
    
    # Initialize system
    system = TradingSystem(symbols=args.symbols, n_workers=n_workers)
    
    if args.mode == 'run':
        # Run trading system
        signals = system.run(
            collect_data=not args.no_collect,
            data_days=args.days
        )
        
        # Save signals
        if signals:
            signals_data = [s.to_dict() for s in signals]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            signals_file = f'signals/signals_{timestamp}.json'
            os.makedirs('signals', exist_ok=True)
            
            import json
            with open(signals_file, 'w') as f:
                json.dump(signals_data, f, indent=2)
            
            print(f"\nSignals saved to {signals_file}")
    
    elif args.mode == 'predict':
        # Predict using CSV data
        from data.data_loader import prepare_training_data
        
        data_path = args.data_path
        if data_path is None:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(os.path.dirname(project_dir), 'data')
        
        print(f"\nLoading CSV data from: {data_path}")
        data = prepare_training_data(data_path, timeframes=['5m', '15m', '1h', '4h', '1d'])
        
        if data:
            # Convert to expected format (rename open_time -> expected format)
            signals = system.run(collect_data=False, data=data)
            
            if signals:
                signals_data = [s.to_dict() for s in signals]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                signals_file = f'signals/signals_{timestamp}.json'
                os.makedirs('signals', exist_ok=True)
                
                import json
                with open(signals_file, 'w') as f:
                    json.dump(signals_data, f, indent=2)
                
                print(f"\nSignals saved to {signals_file}")
        else:
            print("No data loaded!")
    
    elif args.mode == 'train':
        system.train_models()
    
    elif args.mode == 'backtest':
        # Run backtest
        from backtest.optinus_runner import test_backtester
        test_backtester()


if __name__ == "__main__":
    main()
