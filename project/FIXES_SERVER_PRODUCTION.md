# Исправления train_with_risk.py (Server Production Branch)

## Дата: 2026-02-06

Этот документ описывает все критические исправления, примененные к файлу `train_with_risk.py` для продакшн-сервера.

---

## 1. Binary Classification → 3-Class Classification

### Проблема
Модель использовала бинарную классификацию (Up/Down), что приводит к:
- Потере информации о "боковом" движении (Flat)
- Ложным сигналам при малых изменениях цены
- Завышенной уверенности в направлении

### Решение
Изменено на 3-class классификацию:
- **0 = Down** (ретурн < -threshold)
- **1 = Flat** (ретурн в пределах ±threshold)
- **2 = Up** (ретурн > +threshold)

### Файлы и строки изменены:

#### `_get_training_data_with_features()` (~строка 1943)
```python
# БЫЛО:
df = df.with_columns([
    (pl.col('future_return') > 0).cast(pl.Int32).alias('target')
])

# СТАЛО:
flat_threshold = {
    ModelType.SCALP: 0.001,    # 0.1%
    ModelType.INTRADAY: 0.002,  # 0.2%
    ModelType.SWING: 0.005      # 0.5%
}.get(model_type, 0.002)

df = df.with_columns([
    pl.when(pl.col('future_return') < -flat_threshold)
      .then(pl.lit(0))  # Down
      .when(pl.col('future_return') > flat_threshold)
      .then(pl.lit(2))  # Up
      .otherwise(pl.lit(1))  # Flat
      .cast(pl.Int32).alias('target')
])
```

#### `get_training_data()` (~строка 1114)
Аналогичное изменение.

---

## 2. LightGBM: Binary → Multiclass

### Проблема
LightGBM был настроен для бинарной классификации.

### Решение

#### `LightGBMTradingModel.__init__()` (~строка 1443)
```python
# БЫЛО:
self.lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    ...
}

# СТАЛО:
self.lgb_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    ...
}
```

#### Добавлен метод `predict_class()` (~строка 1537)
```python
def predict_class(self, X: np.ndarray) -> np.ndarray:
    """Predict class labels (0=Down, 1=Flat, 2=Up)."""
    proba = self.predict_proba(X)
    return np.argmax(proba, axis=1)
```

---

## 3. Data Leakage: Simple Split → TimeSeriesSplit

### Проблема
Использовался простой процентный split:
```python
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
```

Это приводит к **data leakage** - модель "видит будущее" при валидации.

### Решение
Использование `TimeSeriesSplit` из sklearn для правильного временного разделения.

#### `train_trading_models()` (~строка 2085)
```python
# БЫЛО:
split_idx = int(len(X) * (1 - self.config.test_size))
X_train, X_test = X[:split_idx], X[split_idx:]

# СТАЛО:
tscv = TimeSeriesSplit(n_splits=5)
fold_aucs = []

for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    ...
```

#### `train_risk_model()` (~строка 2161)
Аналогичное изменение с `TimeSeriesSplit(n_splits=3)`.

---

## 4. Fake Returns → Real Returns

### Проблема
Risk model использовала случайные числа вместо реальных ретурнов:
```python
returns = np.random.randn(len(X)) * 0.01  # PLACEHOLDER!
```

Это делало весь Risk Engine бесполезным.

### Решение

#### `_get_training_data_with_features()` теперь возвращает returns
```python
# БЫЛО:
return X, y, feature_cols

# СТАЛО:
returns = df.select('future_return').to_numpy().flatten().astype(np.float32)
return X, y, returns, feature_cols
```

#### `train_risk_model()` использует реальные returns
```python
# БЫЛО:
X, y, feature_names = self._get_training_data_with_features(...)
returns = np.random.randn(len(X)) * 0.01

# СТАЛО:
X, y, returns, feature_names = self._get_training_data_with_features(...)
self.logger.info(f"Returns stats: mean={returns.mean()*100:.3f}%, std={returns.std()*100:.3f}%")
```

---

## 5. Multiclass AUC Evaluation

### Проблема
`roc_auc_score(y_test, y_pred)` не работает для multiclass без параметров.

### Решение

```python
# БЫЛО:
auc = roc_auc_score(y_test, y_pred)

# СТАЛО:
auc = roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted')
```

---

## Сводная таблица изменений

| # | Файл/Метод | Строки | Проблема | Исправление |
|---|------------|--------|----------|-------------|
| 1 | `_get_training_data_with_features()` | ~1943-2010 | Binary target | 3-class target |
| 2 | `get_training_data()` | ~1114-1130 | Binary target | 3-class target |
| 3 | `LightGBMTradingModel.__init__()` | ~1443-1450 | Binary objective | Multiclass + num_class=3 |
| 4 | `LightGBMTradingModel` | ~1537 | Нет predict_class | Добавлен метод |
| 5 | `train_trading_models()` | ~2085-2130 | Simple split | TimeSeriesSplit(5) |
| 6 | `train_risk_model()` | ~2140-2175 | Random returns | Real returns |
| 7 | `train_risk_model()` | ~2161 | Simple split | TimeSeriesSplit(3) |
| 8 | `train_trading_models()` | ~2115 | Binary AUC | Multiclass OVR AUC |

---

## Как запустить

```bash
cd /home/ai/NogutiAI/aiTrainCrypto/project
python train_with_risk.py
```

---

## Зависимости

Убедитесь что установлены:
```bash
pip install polars lightgbm scikit-learn numpy scipy optuna pyyaml
```

---

## Примечания

- **Linux-only**: Этот код использует `os.sysconf()` для определения RAM, что работает только на Linux
- **config.yaml**: Настройки ресурсов загружаются из `config.yaml`
- **Пути**: По умолчанию используются пути `/home/ai/NogutiAI/aiTrainCrypto/...`
