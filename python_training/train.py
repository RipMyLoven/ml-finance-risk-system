"""
🚀 Crypto AI Model Training with Optuna

Обучение модели с автоматическим подбором гиперпараметров.

Использование:
    python train.py                     # Быстрое обучение (20 trials)
    python train.py --trials 100        # Полное обучение (100 trials)
    python train.py --trials 200 --gpu  # С GPU
    python train.py --fast              # Без Optuna, быстрые параметры
"""
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_trades
from features.multi_timeframe import build_multi_timeframe_features


# ═══════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════════

# Путь к данным
DATA_DIR = None
for path in ["data", "../data", os.path.join(os.path.dirname(__file__), "..", "data")]:
    if os.path.isdir(path):
        DATA_DIR = path
        break
if DATA_DIR is None:
    DATA_DIR = "../data"

MODELS_DIR = "models"
HORIZON = 5


def prepare_data(use_cache: bool = True):
    """Подготовка данных с кэшированием"""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    cache_file = os.path.join(MODELS_DIR, "training_data_cache.pkl")
    
    if use_cache and os.path.exists(cache_file):
        print("📦 Загружаю данные из кэша...")
        import pickle
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print("📂 Загружаю данные...")
    trades = load_trades(DATA_DIR)
    
    if trades is None or len(trades) == 0:
        raise ValueError(f"Нет данных в {DATA_DIR}")
    
    print(f"✅ Загружено {len(trades):,} trades")
    
    print("🔧 Строю фичи...")
    df, feature_cols = build_multi_timeframe_features(trades, base_freq='1min', horizon=HORIZON)
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int32)
    
    # Заменяем NaN и Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Кэшируем
    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump((X, y, feature_cols), f)
    
    print(f"✅ Данные готовы: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    return X, y, feature_cols


def optuna_objective(trial, X_train, y_train, X_val, y_val, use_gpu=False):
    """Optuna objective function"""
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    }
    
    if use_gpu:
        params['device'] = 'gpu'
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )
    
    y_pred = model.predict(X_val)
    return roc_auc_score(y_val, y_pred)


def train_with_optuna(X, y, feature_cols, n_trials=20, use_gpu=False):
    """Обучение с Optuna hyperparameter tuning"""
    
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("❌ Optuna не установлен! pip install optuna")
        return None, None
    
    print(f"\n🔧 OPTUNA TUNING ({n_trials} trials)")
    print("=" * 50)
    
    # Split данных
    split_idx = int(len(X) * 0.7)
    val_split = int(len(X) * 0.85)
    
    X_train, X_val, X_test = X[:split_idx], X[split_idx:val_split], X[val_split:]
    y_train, y_val, y_test = y[:split_idx], y[split_idx:val_split], y[val_split:]
    
    print(f"   Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val, use_gpu),
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    print(f"\n🏆 Best AUC: {study.best_value:.4f}")
    print(f"📋 Best params: {study.best_params}")
    
    # Сохраняем лучшие параметры
    params_file = os.path.join(MODELS_DIR, "best_params.json")
    with open(params_file, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    
    return study.best_params, (X_train, y_train, X_val, y_val, X_test, y_test)


def train_final_model(best_params, X, y, feature_cols, use_gpu=False):
    """Обучение финальной модели с лучшими параметрами"""
    
    print(f"\n🚀 ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ")
    print("=" * 50)
    
    # Используем 85% для train, 15% для test
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        **best_params
    }
    
    if use_gpu:
        params['device'] = 'gpu'
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Оценка
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred_binary)
    
    print(f"\n📊 РЕЗУЛЬТАТЫ:")
    print(f"   ROC AUC: {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Best iteration: {model.best_iteration}")
    
    return model, auc


def save_model(model, feature_cols, auc):
    """Сохранение модели в разных форматах"""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. LightGBM native
    lgb_file = os.path.join(MODELS_DIR, "universal_model.txt")
    model.save_model(lgb_file)
    print(f"💾 LightGBM: {lgb_file}")
    
    # 2. Feature names
    features_file = os.path.join(MODELS_DIR, "feature_names.txt")
    with open(features_file, 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"📋 Features: {features_file}")
    
    # 3. ONNX
    export_to_onnx(model, feature_cols)
    
    # 4. Rust code
    generate_rust_code(feature_cols)
    
    print(f"\n✅ Модель сохранена! AUC: {auc:.4f}")


def export_to_onnx(model, feature_cols):
    """Экспорт модели в ONNX формат"""
    
    try:
        import onnx
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        print("\n📦 Экспорт в ONNX...")
        
        # Определяем входной тип
        initial_types = [('input', FloatTensorType([None, len(feature_cols)]))]
        
        # Конвертируем
        onnx_model = convert_lightgbm(
            model,
            initial_types=initial_types,
            target_opset=12
        )
        
        # Сохраняем
        onnx_file = os.path.join(MODELS_DIR, "universal_model.onnx")
        onnx.save_model(onnx_model, onnx_file)
        
        # Размер файла
        size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
        print(f"✅ ONNX: {onnx_file} ({size_mb:.1f} MB)")
        
        # Проверяем модель
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(onnx_file)
            
            # Тестовый inference
            test_input = np.random.randn(1, len(feature_cols)).astype(np.float32)
            outputs = sess.run(None, {'input': test_input})
            print(f"✅ ONNX Runtime проверка пройдена!")
        except ImportError:
            print("⚠️ onnxruntime не установлен, пропускаю проверку")
        except Exception as e:
            print(f"⚠️ ONNX проверка: {e}")
            
    except ImportError as e:
        print(f"⚠️ Для ONNX экспорта установите: pip install onnxmltools onnx")
    except Exception as e:
        print(f"⚠️ ONNX экспорт ошибка: {e}")


def generate_rust_code(feature_cols):
    """Генерация Rust кода для inference"""
    
    rust_code = f'''//! ONNX Model Inference for Rust
//! 
//! Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
//! Features: {len(feature_cols)}
//!
//! Cargo.toml:
//! ```toml
//! [dependencies]
//! ort = "2.0"
//! ndarray = "0.16"
//! ```

use ort::{{GraphOptimizationLevel, Session}};
use ndarray::Array2;
use std::error::Error;

pub const NUM_FEATURES: usize = {len(feature_cols)};

/// Load ONNX model
pub fn load_model(path: &str) -> Result<Session, Box<dyn Error>> {{
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(path)?;
    Ok(session)
}}

/// Run inference
/// Returns probability of price going UP
pub fn predict(session: &Session, features: &[f32]) -> Result<f32, Box<dyn Error>> {{
    if features.len() != NUM_FEATURES {{
        return Err(format!(
            "Expected {{}} features, got {{}}", 
            NUM_FEATURES, 
            features.len()
        ).into());
    }}
    
    let input = Array2::from_shape_vec((1, NUM_FEATURES), features.to_vec())?;
    let outputs = session.run(ort::inputs!["input" => input.view()]?)?;
    
    // Output[1] contains probabilities [[prob_down, prob_up]]
    let probs = outputs[1].try_extract_tensor::<f32>()?;
    let prob_up = probs[[0, 1]];
    
    Ok(prob_up)
}}

/// Trading signal based on probability
pub fn get_signal(prob: f32) -> &'static str {{
    if prob > 0.65 {{
        "STRONG_BUY"
    }} else if prob > 0.55 {{
        "BUY"
    }} else if prob < 0.35 {{
        "STRONG_SELL"
    }} else if prob < 0.45 {{
        "SELL"
    }} else {{
        "HOLD"
    }}
}}

// Feature names (in order):
/*
{chr(10).join(f"  {i}: {name}" for i, name in enumerate(feature_cols[:30]))}
  ... and {len(feature_cols) - 30} more
*/
'''
    
    rust_file = os.path.join(MODELS_DIR, "inference.rs")
    with open(rust_file, 'w', encoding='utf-8') as f:
        f.write(rust_code)
    print(f"🦀 Rust: {rust_file}")


def get_default_params():
    """Параметры по умолчанию (без Optuna)"""
    return {
        'num_leaves': 64,
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 50,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'feature_fraction': 0.8,
    }


def main():
    parser = argparse.ArgumentParser(description='Train Crypto AI Model')
    parser.add_argument('--trials', type=int, default=20, help='Optuna trials (default: 20)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--fast', action='store_true', help='Skip Optuna, use default params')
    parser.add_argument('--no-cache', action='store_true', help='Rebuild data cache')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 CRYPTO AI MODEL TRAINING")
    print("=" * 60)
    
    # Подготовка данных
    X, y, feature_cols = prepare_data(use_cache=not args.no_cache)
    
    # Выбор параметров
    if args.fast:
        print("\n⚡ Быстрый режим (без Optuna)")
        best_params = get_default_params()
    else:
        best_params, _ = train_with_optuna(X, y, feature_cols, args.trials, args.gpu)
        if best_params is None:
            print("⚠️ Optuna не сработал, использую default параметры")
            best_params = get_default_params()
    
    # Финальное обучение
    model, auc = train_final_model(best_params, X, y, feature_cols, args.gpu)
    
    # Сохранение
    save_model(model, feature_cols, auc)
    
    print("\n" + "=" * 60)
    print("✅ ГОТОВО!")
    print("=" * 60)
    print(f"\nИспользование:")
    print(f"  python live_predict.py btc        # Live предсказание")
    print(f"  python backtest_multi.py          # Бэктест")


if __name__ == "__main__":
    main()
