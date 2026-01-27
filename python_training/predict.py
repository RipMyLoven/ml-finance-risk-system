"""
🤖 AI Predictor - Простой интерфейс для общения с моделью

Загружаешь CSV → Получаешь прогноз в понятном виде
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Добавляем путь к модулям
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features.multi_timeframe import (
    resample_to_timeframe,
    build_single_timeframe_features
)


class CryptoPredictor:
    """
    Простой класс для предсказаний
    """
    
    def __init__(self, model_dir: str = 'models'):
        """Загрузка модели"""
        print("🔄 Загружаю модель...")
        
        # Пробуем разные форматы модели
        txt_path = os.path.join(model_dir, 'universal_model.txt')
        joblib_path = os.path.join(model_dir, 'universal_model.joblib')
        
        if os.path.exists(txt_path):
            # LightGBM Booster из текстового файла
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=txt_path)
            self._is_booster = True
            print("   Используется LightGBM Booster (txt)")
        elif os.path.exists(joblib_path):
            import joblib
            self.model = joblib.load(joblib_path)
            self._is_booster = False
            print("   Используется joblib модель")
        else:
            raise FileNotFoundError(f"Модель не найдена в {model_dir}")
        
        # Загружаем список фичей
        features_path = os.path.join(model_dir, 'feature_names.txt')
        if not os.path.exists(features_path):
            features_path = os.path.join(model_dir, 'universal_features.txt')
            
        with open(features_path, 'r') as f:
            self.feature_names = [line.strip() for line in f.readlines()]
        
        print(f"✅ Модель загружена! ({len(self.feature_names)} фичей)")
    
    def _predict_proba(self, X):
        """Внутренний метод для получения вероятностей"""
        if self._is_booster:
            # LightGBM Booster возвращает raw scores, нужно применить sigmoid
            raw = self.model.predict(X)
            # sigmoid для бинарной классификации
            proba_1 = 1 / (1 + np.exp(-raw))
            proba_0 = 1 - proba_1
            return np.column_stack([proba_0, proba_1])
        else:
            return self.model.predict_proba(X)
    
    def predict_from_csv(self, csv_path: str) -> dict:
        """
        Предсказание из CSV файла с trades
        
        Returns:
            dict с прогнозом и объяснением
        """
        print(f"\n📊 Анализирую: {os.path.basename(csv_path)}")
        
        # Загружаем данные
        from utils import load_trades
        trades = load_trades(csv_path, verbose=False)
        
        return self.predict_from_trades(trades)
    
    def predict_from_trades(self, trades: pd.DataFrame) -> dict:
        """
        Предсказание из DataFrame с trades
        """
        # Строим бары на разных таймфреймах
        bars_1m = resample_to_timeframe(trades, '1min')
        bars_5m = resample_to_timeframe(trades, '5min')
        bars_15m = resample_to_timeframe(trades, '15min')
        bars_1h = resample_to_timeframe(trades, '1h')
        bars_4h = resample_to_timeframe(trades, '4h')
        bars_1d = resample_to_timeframe(trades, '1d')
        
        # Строим фичи
        df_1m = build_single_timeframe_features(bars_1m, suffix='')
        df_5m = build_single_timeframe_features(bars_5m, suffix='_5min')
        df_15m = build_single_timeframe_features(bars_15m, suffix='_15min')
        df_1h = build_single_timeframe_features(bars_1h, suffix='_1h')
        df_4h = build_single_timeframe_features(bars_4h, suffix='_4h')
        df_1d = build_single_timeframe_features(bars_1d, suffix='_1d')
        
        # Берём последние данные
        df = df_1m.copy()
        df = df.set_index('datetime')
        
        for tf_df, suffix in [(df_5m, '_5min'), (df_15m, '_15min'), (df_1h, '_1h'), (df_4h, '_4h'), (df_1d, '_1d')]:
            tf_features = [col for col in tf_df.columns if col.endswith(suffix)]
            tf_subset = tf_df[['datetime'] + tf_features].set_index('datetime')
            for col in tf_features:
                df[col] = tf_subset[col].reindex(df.index, method='ffill')
        
        df = df.reset_index()
        df = df.dropna()
        
        if len(df) == 0:
            return {"error": "Недостаточно данных для анализа"}
        
        # Берём последнюю строку (самые свежие данные)
        last_row = df.iloc[-1]
        
        # Собираем фичи
        features = []
        missing = []
        for f in self.feature_names:
            if f in last_row:
                features.append(float(last_row[f]))
            else:
                features.append(0.0)
                missing.append(f)
        
        if missing:
            print(f"⚠️ Отсутствуют фичи: {missing[:5]}...")
        
        # Предсказание
        features_array = np.array([features])
        probability = self._predict_proba(features_array)[0][1]
        
        # Формируем ответ
        return self._format_prediction(
            probability=probability,
            last_price=last_row['close'],
            last_time=last_row['datetime'],
            features=last_row
        )
    
    def predict_from_ohlcv(
        self,
        bars_1m: pd.DataFrame,
        bars_5m: pd.DataFrame = None,
        bars_15m: pd.DataFrame = None,
        bars_1h: pd.DataFrame = None
    ) -> dict:
        """
        Предсказание из готовых OHLCV баров
        
        Args:
            bars_1m: DataFrame с колонками [datetime, open, high, low, close, volume]
            bars_5m, bars_15m, bars_1h: опционально, для multi-timeframe
        """
        # Строим фичи
        df = build_single_timeframe_features(bars_1m.copy(), suffix='')
        
        if bars_5m is not None:
            df_5m = build_single_timeframe_features(bars_5m.copy(), suffix='_5min')
            # merge...
        
        df = df.dropna()
        
        if len(df) == 0:
            return {"error": "Недостаточно данных"}
        
        last_row = df.iloc[-1]
        
        features = []
        for f in self.feature_names:
            if f in last_row:
                features.append(float(last_row[f]))
            else:
                features.append(0.0)
        
        features_array = np.array([features])
        probability = self._predict_proba(features_array)[0][1]
        
        return self._format_prediction(
            probability=probability,
            last_price=last_row['close'],
            last_time=last_row.get('datetime', datetime.now()),
            features=last_row
        )
    
    def _format_prediction(self, probability: float, last_price: float, 
                          last_time, features: pd.Series) -> dict:
        """
        Форматирование предсказания в понятный вид с детальным анализом
        """
        # Определяем сигнал
        if probability > 0.70:
            signal = "🚀 STRONG BUY"
            signal_ru = "Сильный сигнал на покупку"
            confidence = "высокая"
        elif probability > 0.60:
            signal = "📈 BUY"
            signal_ru = "Сигнал на покупку"
            confidence = "средняя"
        elif probability > 0.55:
            signal = "↗️ WEAK BUY"
            signal_ru = "Слабый сигнал на покупку"
            confidence = "низкая"
        elif probability < 0.30:
            signal = "💥 STRONG SELL"
            signal_ru = "Сильный сигнал на продажу"
            confidence = "высокая"
        elif probability < 0.40:
            signal = "📉 SELL"
            signal_ru = "Сигнал на продажу"
            confidence = "средняя"
        elif probability < 0.45:
            signal = "↘️ WEAK SELL"
            signal_ru = "Слабый сигнал на продажу"
            confidence = "низкая"
        else:
            signal = "➡️ NEUTRAL"
            signal_ru = "Нейтрально, лучше подождать"
            confidence = "неопределённо"
        
        # Детальный анализ всех факторов
        analysis = self._analyze_factors(features)
        
        # Рассчитываем уровни входа/выхода
        trade_levels = self._calculate_trade_levels(probability, last_price, features, analysis)
        
        result = {
            "signal": signal,
            "signal_ru": signal_ru,
            "probability": probability,
            "probability_pct": f"{probability * 100:.1f}%",
            "confidence": confidence,
            "last_price": last_price,
            "last_time": str(last_time),
            "analysis": analysis,
            "trade_levels": trade_levels,
            "summary": self._generate_summary(probability, analysis),
            "recommendation": self._get_recommendation(probability, analysis, trade_levels, last_price)
        }
        
        return result
    
    def _analyze_factors(self, features: pd.Series) -> dict:
        """Детальный анализ всех факторов"""
        analysis = {
            "trend": {},
            "momentum": {},
            "volume": {},
            "volatility": {},
            "multi_tf": {}
        }
        
        # ═══════════════════════════════════════════════════════════
        # ТРЕНД
        # ═══════════════════════════════════════════════════════════
        
        # Позиция цены относительно SMA
        bullish_sma = 0
        bearish_sma = 0
        
        for period in [5, 10, 20, 50]:
            key = f'price_vs_sma{period}'
            if key in features and not pd.isna(features[key]):
                val = features[key]
                if val > 0.005:
                    bullish_sma += 1
                elif val < -0.005:
                    bearish_sma += 1
        
        if bullish_sma >= 3:
            analysis["trend"]["direction"] = "🟢 Восходящий"
            analysis["trend"]["description"] = f"Цена выше {bullish_sma} из 4 скользящих средних"
            analysis["trend"]["strength"] = "сильный"
        elif bearish_sma >= 3:
            analysis["trend"]["direction"] = "🔴 Нисходящий"
            analysis["trend"]["description"] = f"Цена ниже {bearish_sma} из 4 скользящих средних"
            analysis["trend"]["strength"] = "сильный"
        elif bullish_sma > bearish_sma:
            analysis["trend"]["direction"] = "🟡 Слабо восходящий"
            analysis["trend"]["description"] = "Смешанные сигналы, небольшой перевес быков"
            analysis["trend"]["strength"] = "слабый"
        elif bearish_sma > bullish_sma:
            analysis["trend"]["direction"] = "🟡 Слабо нисходящий"
            analysis["trend"]["description"] = "Смешанные сигналы, небольшой перевес медведей"
            analysis["trend"]["strength"] = "слабый"
        else:
            analysis["trend"]["direction"] = "⚪ Боковик"
            analysis["trend"]["description"] = "Нет явного тренда, цена в диапазоне"
            analysis["trend"]["strength"] = "нет"
        
        # EMA crossover
        if 'ema_cross' in features and not pd.isna(features['ema_cross']):
            ema = features['ema_cross']
            if ema > 0.01:
                analysis["trend"]["ema"] = "Быстрая EMA выше медленной (бычий сигнал)"
            elif ema < -0.01:
                analysis["trend"]["ema"] = "Быстрая EMA ниже медленной (медвежий сигнал)"
        
        # ═══════════════════════════════════════════════════════════
        # МОМЕНТУМ
        # ═══════════════════════════════════════════════════════════
        
        # RSI
        if 'rsi' in features and not pd.isna(features['rsi']):
            rsi = features['rsi'] * 100 + 50  # denormalize
            analysis["momentum"]["rsi_value"] = rsi
            
            if rsi > 80:
                analysis["momentum"]["rsi"] = f"🔴 Сильно перекуплен ({rsi:.0f})"
                analysis["momentum"]["rsi_signal"] = "Высокий риск отката вниз"
            elif rsi > 70:
                analysis["momentum"]["rsi"] = f"🟠 Перекуплен ({rsi:.0f})"
                analysis["momentum"]["rsi_signal"] = "Возможен откат, но тренд может продолжиться"
            elif rsi < 20:
                analysis["momentum"]["rsi"] = f"🟢 Сильно перепродан ({rsi:.0f})"
                analysis["momentum"]["rsi_signal"] = "Высокая вероятность отскока вверх"
            elif rsi < 30:
                analysis["momentum"]["rsi"] = f"🟡 Перепродан ({rsi:.0f})"
                analysis["momentum"]["rsi_signal"] = "Возможен отскок"
            else:
                analysis["momentum"]["rsi"] = f"⚪ Нейтральный ({rsi:.0f})"
                analysis["momentum"]["rsi_signal"] = "Нет экстремальных значений"
        
        # Stochastic
        if 'stoch_k' in features and not pd.isna(features['stoch_k']):
            stoch = (features['stoch_k'] + 0.5) * 100  # denormalize
            if stoch > 80:
                analysis["momentum"]["stochastic"] = f"Перекуплен ({stoch:.0f})"
            elif stoch < 20:
                analysis["momentum"]["stochastic"] = f"Перепродан ({stoch:.0f})"
        
        # Returns
        if 'return_1' in features and not pd.isna(features['return_1']):
            ret1 = features['return_1'] * 100
            analysis["momentum"]["last_candle"] = f"{ret1:+.2f}%"
        
        if 'return_5' in features and not pd.isna(features['return_5']):
            ret5 = features['return_5'] * 100
            analysis["momentum"]["last_5_candles"] = f"{ret5:+.2f}%"
        
        if 'return_20' in features and not pd.isna(features['return_20']):
            ret20 = features['return_20'] * 100
            analysis["momentum"]["last_20_candles"] = f"{ret20:+.2f}%"
        
        # ═══════════════════════════════════════════════════════════
        # ОБЪЁМ
        # ═══════════════════════════════════════════════════════════
        
        if 'volume_ratio' in features and not pd.isna(features['volume_ratio']):
            vol = features['volume_ratio']
            analysis["volume"]["ratio"] = vol
            
            if vol > 3:
                analysis["volume"]["level"] = "🔥 Очень высокий"
                analysis["volume"]["description"] = f"Объём в {vol:.1f}x выше среднего — сильный интерес"
            elif vol > 2:
                analysis["volume"]["level"] = "📈 Высокий"
                analysis["volume"]["description"] = f"Объём в {vol:.1f}x выше среднего"
            elif vol > 1.2:
                analysis["volume"]["level"] = "📊 Выше среднего"
                analysis["volume"]["description"] = f"Объём в {vol:.1f}x от среднего"
            elif vol < 0.5:
                analysis["volume"]["level"] = "📉 Низкий"
                analysis["volume"]["description"] = "Объём ниже среднего — слабый интерес, движение ненадёжно"
            else:
                analysis["volume"]["level"] = "⚪ Средний"
                analysis["volume"]["description"] = "Нормальный объём"
        
        # Buy pressure
        if 'buy_pressure' in features and not pd.isna(features['buy_pressure']):
            bp = features['buy_pressure']
            buy_pct = (bp + 0.5) * 100
            analysis["volume"]["buy_percent"] = buy_pct
            
            if bp > 0.15:
                analysis["volume"]["pressure"] = f"Сильное давление покупателей ({buy_pct:.0f}% покупок)"
                analysis["volume"]["pressure_signal"] = "Покупатели агрессивно скупают"
            elif bp > 0.05:
                analysis["volume"]["pressure"] = f"Лёгкое давление покупателей ({buy_pct:.0f}% покупок)"
                analysis["volume"]["pressure_signal"] = "Небольшой перевес покупателей"
            elif bp < -0.15:
                analysis["volume"]["pressure"] = f"Сильное давление продавцов ({buy_pct:.0f}% покупок)"
                analysis["volume"]["pressure_signal"] = "Продавцы агрессивно сливают"
            elif bp < -0.05:
                analysis["volume"]["pressure"] = f"Лёгкое давление продавцов ({buy_pct:.0f}% покупок)"
                analysis["volume"]["pressure_signal"] = "Небольшой перевес продавцов"
            else:
                analysis["volume"]["pressure"] = f"Баланс ({buy_pct:.0f}% покупок)"
                analysis["volume"]["pressure_signal"] = "Покупатели и продавцы в равновесии"
        
        # ═══════════════════════════════════════════════════════════
        # ВОЛАТИЛЬНОСТЬ
        # ═══════════════════════════════════════════════════════════
        
        if 'vol_ratio' in features and not pd.isna(features['vol_ratio']):
            vr = features['vol_ratio']
            analysis["volatility"]["ratio"] = vr
            
            if vr > 2:
                analysis["volatility"]["level"] = "🔥Очень высокая"
                analysis["volatility"]["description"] = "Волатильность резко выросла — возможен сильный импульс"
                analysis["volatility"]["risk"] = "высокий"
            elif vr > 1.3:
                analysis["volatility"]["level"] = "📈 Повышенная"
                analysis["volatility"]["description"] = "Волатильность выше нормы"
                analysis["volatility"]["risk"] = "средний"
            elif vr < 0.5:
                analysis["volatility"]["level"] = "📉 Низкая"
                analysis["volatility"]["description"] = "Рынок затих, возможен прорыв"
                analysis["volatility"]["risk"] = "низкий"
            else:
                analysis["volatility"]["level"] = "⚪ Нормальная"
                analysis["volatility"]["description"] = "Стандартная волатильность"
                analysis["volatility"]["risk"] = "средний"
        
        # Bollinger Bands
        if 'bb_position' in features and not pd.isna(features['bb_position']):
            bb = features['bb_position']  # -0.5 to 0.5
            bb_pct = (bb + 0.5) * 100
            
            if bb > 0.4:
                analysis["volatility"]["bollinger"] = f"У верхней границы ({bb_pct:.0f}%)"
            elif bb < -0.4:
                analysis["volatility"]["bollinger"] = f"У нижней границы ({bb_pct:.0f}%)"
            else:
                analysis["volatility"]["bollinger"] = f"В середине канала ({bb_pct:.0f}%)"
        
        # ═══════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME
        # ═══════════════════════════════════════════════════════════
        
        tf_signals = {"bullish": 0, "bearish": 0, "neutral": 0}
        
        for tf in ['_5min', '_15min', '_1h', '_4h', '_1d']:
            tf_name = tf.replace('_', '')
            
            ret_key = f'return_1{tf}'
            bp_key = f'buy_pressure{tf}'
            
            tf_signal = "neutral"
            
            if ret_key in features and not pd.isna(features[ret_key]):
                ret = features[ret_key]
                if ret > 0.002:
                    tf_signal = "bullish"
                elif ret < -0.002:
                    tf_signal = "bearish"
            
            if bp_key in features and not pd.isna(features[bp_key]):
                bp = features[bp_key]
                if bp > 0.1 and tf_signal != "bearish":
                    tf_signal = "bullish"
                elif bp < -0.1 and tf_signal != "bullish":
                    tf_signal = "bearish"
            
            tf_signals[tf_signal] += 1
            
            if tf_signal == "bullish":
                analysis["multi_tf"][tf_name] = "🟢 Бычий"
            elif tf_signal == "bearish":
                analysis["multi_tf"][tf_name] = "🔴 Медвежий"
            else:
                analysis["multi_tf"][tf_name] = "⚪ Нейтральный"
        
        # Общая оценка по таймфреймам (теперь 5 ТФ)
        if tf_signals["bullish"] >= 3:
            analysis["multi_tf"]["consensus"] = "Большинство таймфреймов бычьи"
        elif tf_signals["bearish"] >= 3:
            analysis["multi_tf"]["consensus"] = "Большинство таймфреймов медвежьи"
        else:
            analysis["multi_tf"]["consensus"] = "Таймфреймы расходятся"
        
        return analysis
    
    def _generate_summary(self, prob: float, analysis: dict) -> str:
        """Генерация текстового резюме"""
        parts = []
        
        # Тренд
        if "direction" in analysis["trend"]:
            parts.append(f"Тренд: {analysis['trend']['direction']}")
        
        # Моментум
        if "rsi" in analysis["momentum"]:
            parts.append(f"RSI: {analysis['momentum']['rsi']}")
        
        # Объём
        if "level" in analysis["volume"]:
            parts.append(f"Объём: {analysis['volume']['level']}")
        
        if "pressure" in analysis["volume"]:
            parts.append(analysis["volume"]["pressure_signal"])
        
        # Волатильность
        if "level" in analysis["volatility"]:
            parts.append(f"Волатильность: {analysis['volatility']['level']}")
        
        # Multi-TF
        if "consensus" in analysis["multi_tf"]:
            parts.append(analysis["multi_tf"]["consensus"])
        
        return " | ".join(parts)
    
    def _calculate_trade_levels(self, prob: float, price: float, 
                                  features: pd.Series, analysis: dict) -> dict:
        """
        Рассчитывает конкретные уровни для входа в сделку:
        - Entry (цена входа)
        - Stop Loss (где закрыть с убытком)
        - Take Profit (где закрыть с прибылью)
        """
        
        # Определяем волатильность для расчёта уровней
        atr_pct = 0.01  # По умолчанию 1%
        
        # Пытаемся получить ATR из фичей
        if 'atr_pct' in features and not pd.isna(features['atr_pct']):
            atr_pct = abs(features['atr_pct'])
            if atr_pct < 0.001:  # Минимум 0.1%
                atr_pct = 0.001
            if atr_pct > 0.1:  # Максимум 10%
                atr_pct = 0.1
        
        # Или из волатильности
        vol_ratio = analysis.get('volatility', {}).get('ratio', 1.0)
        if vol_ratio and vol_ratio > 0:
            # Корректируем ATR на волатильность
            atr_pct = atr_pct * min(vol_ratio, 2.0)
        
        # Определяем направление
        is_long = prob > 0.5
        
        # Уровень уверенности влияет на Risk/Reward
        if prob > 0.65 or prob < 0.35:
            confidence_mult = 1.0  # Высокая уверенность
            rr_ratio = 2.5  # Risk/Reward 1:2.5
        elif prob > 0.58 or prob < 0.42:
            confidence_mult = 1.2  # Средняя уверенность - чуть больше стоп
            rr_ratio = 2.0  # Risk/Reward 1:2
        else:
            confidence_mult = 1.5  # Низкая уверенность - широкий стоп
            rr_ratio = 1.5  # Risk/Reward 1:1.5
        
        # Базовое расстояние до стопа (1.5 ATR)
        stop_distance_pct = atr_pct * 1.5 * confidence_mult
        
        # Take profit = stop * R/R ratio
        tp_distance_pct = stop_distance_pct * rr_ratio
        
        # Рассчитываем уровни
        if is_long:
            # LONG: вход по рынку, стоп ниже, тейк выше
            entry_price = price
            stop_loss = price * (1 - stop_distance_pct)
            take_profit = price * (1 + tp_distance_pct)
            
            # Альтернативный вход - на откате
            entry_limit = price * (1 - atr_pct * 0.3)  # Чуть ниже текущей
            
            # Дополнительные тейки
            tp1 = price * (1 + tp_distance_pct * 0.5)  # 50% позиции
            tp2 = price * (1 + tp_distance_pct)        # 30% позиции
            tp3 = price * (1 + tp_distance_pct * 1.5)  # 20% позиции (трейлинг)
        else:
            # SHORT: вход по рынку, стоп выше, тейк ниже
            entry_price = price
            stop_loss = price * (1 + stop_distance_pct)
            take_profit = price * (1 - tp_distance_pct)
            
            # Альтернативный вход - на откате вверх
            entry_limit = price * (1 + atr_pct * 0.3)
            
            # Дополнительные тейки
            tp1 = price * (1 - tp_distance_pct * 0.5)
            tp2 = price * (1 - tp_distance_pct)
            tp3 = price * (1 - tp_distance_pct * 1.5)
        
        # Определяем количество знаков после запятой
        if price > 1000:
            decimals = 2
        elif price > 1:
            decimals = 4
        else:
            decimals = 6
        
        # ═══════════════════════════════════════════════════════════
        # РАСЧЁТ ПЛЕЧА (LEVERAGE)
        # ═══════════════════════════════════════════════════════════
        # Формула: Плечо = Допустимый_риск% / Стоп%
        # При стопе теряем не более 1-2% депозита
        
        # Базовый риск на сделку (% от депозита)
        if prob > 0.65 or prob < 0.35:
            max_risk_pct = 2.0  # Высокая уверенность - рискуем 2%
            confidence_text = "высокая"
        elif prob > 0.58 or prob < 0.42:
            max_risk_pct = 1.5  # Средняя - 1.5%
            confidence_text = "средняя"
        else:
            max_risk_pct = 1.0  # Низкая - только 1%
            confidence_text = "низкая"
        
        # Расчёт плеча: риск / стоп
        calculated_leverage = max_risk_pct / (stop_distance_pct * 100)
        
        # Ограничения по волатильности
        if vol_ratio and vol_ratio > 2:
            max_leverage = 5  # Высокая волатильность - max 5x
            vol_warning = "высокая волатильность"
        elif vol_ratio and vol_ratio > 1.3:
            max_leverage = 10  # Средняя - max 10x
            vol_warning = None
        else:
            max_leverage = 20  # Низкая - max 20x
            vol_warning = None
        
        # Финальное плечо (округляем вниз до целого)
        recommended_leverage = min(int(calculated_leverage), max_leverage)
        recommended_leverage = max(1, recommended_leverage)  # Минимум 1x
        
        # Консервативное плечо (50% от рекомендуемого)
        conservative_leverage = max(1, recommended_leverage // 2)
        
        # Агрессивное плечо (если очень уверен)
        if prob > 0.70 or prob < 0.30:
            aggressive_leverage = min(recommended_leverage + 5, max_leverage)
        else:
            aggressive_leverage = recommended_leverage
        
        # Расчёт размера позиции
        # Пример: депозит $1000, плечо 10x, риск 2% = можем открыть на $10000
        # При стопе 0.2% потеряем $20 (2% от $1000)
        
        # ═══════════════════════════════════════════════════════════
        # ОПРЕДЕЛЕНИЕ ТИПА СДЕЛКИ (СРОК)
        # ═══════════════════════════════════════════════════════════
        
        # Факторы для определения срока:
        # 1. Волатильность - высокая = быстрые сделки
        # 2. Согласованность ТФ - все согласны = можно держать дольше
        # 3. Сила тренда
        # 4. Размер тейка
        
        # Проверяем согласованность таймфреймов
        multi_tf = analysis.get('multi_tf', {})
        tf_aligned = 'бычьи' in multi_tf.get('consensus', '') or 'медвежьи' in multi_tf.get('consensus', '')
        
        # Сила тренда
        trend_strength = analysis.get('trend', {}).get('strength', 'нет')
        strong_trend = trend_strength == 'сильный'
        
        # Определяем тип сделки
        if vol_ratio and vol_ratio > 2:
            # Высокая волатильность = скальпинг
            trade_type = 'scalp'
            trade_type_ru = '⚡ СКАЛЬПИНГ'
            hold_time = '5-30 мин'
            trade_advice = 'Быстрый вход-выход, не жадничай'
        elif vol_ratio and vol_ratio > 1.3:
            # Средняя волатильность
            if tf_aligned and strong_trend:
                # Все ТФ согласованы + сильный тренд = интрадей/свинг
                trade_type = 'intraday'
                trade_type_ru = '📊 ИНТРАДЕЙ'
                hold_time = '1-8 часов'
                trade_advice = 'Можно держать до конца дня'
            else:
                trade_type = 'scalp'
                trade_type_ru = '⚡ СКАЛЬПИНГ'
                hold_time = '15-60 мин'
                trade_advice = 'Фиксируй прибыль быстро'
        else:
            # Низкая волатильность
            if tf_aligned and strong_trend:
                # Тренд на всех ТФ = можно держать долго
                trade_type = 'swing'
                trade_type_ru = '🌙 СВИНГ'
                hold_time = '1-3 дня'
                trade_advice = 'Можно держать с трейлинг-стопом'
            elif tf_aligned:
                trade_type = 'intraday'
                trade_type_ru = '📊 ИНТРАДЕЙ'
                hold_time = '2-8 часов'
                trade_advice = 'Закрой до конца дня'
            else:
                trade_type = 'scalp'
                trade_type_ru = '⚡ СКАЛЬПИНГ'
                hold_time = '30-90 мин'
                trade_advice = 'ТФ не согласованы, не держи долго'
        
        return {
            'direction': 'LONG' if is_long else 'SHORT',
            'entry_market': round(entry_price, decimals),
            'entry_limit': round(entry_limit, decimals),
            'stop_loss': round(stop_loss, decimals),
            'take_profit': round(take_profit, decimals),
            'tp1': round(tp1, decimals),
            'tp2': round(tp2, decimals),
            'tp3': round(tp3, decimals),
            'risk_pct': round(stop_distance_pct * 100, 2),
            'reward_pct': round(tp_distance_pct * 100, 2),
            'rr_ratio': rr_ratio,
            'atr_pct': round(atr_pct * 100, 3),
            # Leverage
            'leverage': recommended_leverage,
            'leverage_conservative': conservative_leverage,
            'leverage_aggressive': aggressive_leverage,
            'leverage_max': max_leverage,
            'position_risk_pct': max_risk_pct,
            'confidence': confidence_text,
            'vol_warning': vol_warning,
            # Trade type (срок)
            'trade_type': trade_type,
            'trade_type_ru': trade_type_ru,
            'hold_time': hold_time,
            'trade_advice': trade_advice
        }
    
    def _get_recommendation(self, prob: float, analysis: dict, 
                            trade_levels: dict = None, current_price: float = None) -> str:
        """Детальная рекомендация LONG/SHORT с причинами и уровнями"""
        
        # ═══════════════════════════════════════════════════════════
        # СОБИРАЕМ ФАКТОРЫ
        # ═══════════════════════════════════════════════════════════
        
        bullish_factors = []
        bearish_factors = []
        warnings = []
        
        # Тренд
        trend_dir = analysis["trend"].get("direction", "")
        if "Восходящий" in trend_dir:
            bullish_factors.append("📈 Восходящий тренд")
        elif "Нисходящий" in trend_dir:
            bearish_factors.append("📉 Нисходящий тренд")
        
        # RSI
        if "rsi_value" in analysis["momentum"]:
            rsi = analysis["momentum"]["rsi_value"]
            if rsi > 75:
                bearish_factors.append(f"🔴 RSI перекуплен ({rsi:.0f}) — скоро откат")
                warnings.append("RSI в зоне перекупленности!")
            elif rsi > 65:
                warnings.append(f"RSI высоковат ({rsi:.0f})")
            elif rsi < 25:
                bullish_factors.append(f"🟢 RSI перепродан ({rsi:.0f}) — ждём отскок")
            elif rsi < 35:
                bullish_factors.append(f"RSI низкий ({rsi:.0f}) — потенциал роста")
        
        # Buy pressure
        if "buy_percent" in analysis["volume"]:
            bp = analysis["volume"]["buy_percent"]
            if bp > 65:
                bullish_factors.append(f"🟢 Покупатели доминируют ({bp:.0f}%)")
            elif bp > 55:
                bullish_factors.append(f"Перевес покупателей ({bp:.0f}%)")
            elif bp < 35:
                bearish_factors.append(f"🔴 Продавцы доминируют ({bp:.0f}% покупок)")
            elif bp < 45:
                bearish_factors.append(f"Перевес продавцов ({bp:.0f}% покупок)")
        
        # Объём
        vol_ratio = analysis["volume"].get("ratio", 1)
        if vol_ratio > 2:
            # Объём подтверждает движение
            if prob > 0.5:
                bullish_factors.append(f"🔥 Высокий объём подтверждает рост")
            else:
                bearish_factors.append(f"🔥 Высокий объём подтверждает падение")
        elif vol_ratio < 0.5:
            warnings.append("⚠️ Низкий объём — сигнал ненадёжен")
        
        # Волатильность
        vol_risk = analysis["volatility"].get("risk", "средний")
        if vol_risk == "высокий":
            warnings.append("⚠️ Высокая волатильность — ставь стоп-лосс!")
        
        # Multi-TF
        consensus = analysis["multi_tf"].get("consensus", "")
        if "бычьи" in consensus:
            bullish_factors.append("🕐 Старшие ТФ бычьи")
        elif "медвежьи" in consensus:
            bearish_factors.append("🕐 Старшие ТФ медвежьи")
        else:
            warnings.append("Таймфреймы расходятся")
        
        # ═══════════════════════════════════════════════════════════
        # ПРИНИМАЕМ РЕШЕНИЕ
        # ═══════════════════════════════════════════════════════════
        
        bull_score = len(bullish_factors)
        bear_score = len(bearish_factors)
        
        # Определяем позицию
        if prob > 0.65 and bull_score >= 2 and bull_score > bear_score:
            position = "LONG"
            confidence = "ВЫСОКАЯ"
            emoji = "🟢"
        elif prob > 0.58 and bull_score > bear_score:
            position = "LONG"
            confidence = "СРЕДНЯЯ"
            emoji = "🟡"
        elif prob > 0.52 and bull_score > bear_score + 1:
            position = "LONG (осторожно)"
            confidence = "НИЗКАЯ"
            emoji = "🟡"
        elif prob < 0.35 and bear_score >= 2 and bear_score > bull_score:
            position = "SHORT"
            confidence = "ВЫСОКАЯ"
            emoji = "🔴"
        elif prob < 0.42 and bear_score > bull_score:
            position = "SHORT"
            confidence = "СРЕДНЯЯ"
            emoji = "🟠"
        elif prob < 0.48 and bear_score > bull_score + 1:
            position = "SHORT (осторожно)"
            confidence = "НИЗКАЯ"
            emoji = "🟠"
        else:
            position = "ЖДАТЬ"
            confidence = "—"
            emoji = "⏸️"
        
        # ═══════════════════════════════════════════════════════════
        # ФОРМИРУЕМ ОТВЕТ
        # ═══════════════════════════════════════════════════════════
        
        lines = []
        
        # Главная рекомендация
        lines.append("="*50)
        if position == "LONG":
            lines.append(f"{emoji} РЕКОМЕНДАЦИЯ: ОТКРЫТЬ LONG")
            lines.append(f"   Уверенность: {confidence}")
        elif position == "LONG (осторожно)":
            lines.append(f"{emoji} РЕКОМЕНДАЦИЯ: LONG (с осторожностью)")
            lines.append(f"   Уверенность: {confidence}")
        elif position == "SHORT":
            lines.append(f"{emoji} РЕКОМЕНДАЦИЯ: ОТКРЫТЬ SHORT")
            lines.append(f"   Уверенность: {confidence}")
        elif position == "SHORT (осторожно)":
            lines.append(f"{emoji} РЕКОМЕНДАЦИЯ: SHORT (с осторожностью)")
            lines.append(f"   Уверенность: {confidence}")
        else:
            lines.append(f"{emoji} РЕКОМЕНДАЦИЯ: ЖДАТЬ / ВНЕ РЫНКА")
            lines.append("   Нет чёткого сигнала")
        lines.append("="*50)
        
        # Причины ЗА
        if bullish_factors:
            lines.append("\n🟢 ФАКТОРЫ ЗА LONG:")
            for f in bullish_factors:
                lines.append(f"   • {f}")
        
        # Причины ПРОТИВ
        if bearish_factors:
            lines.append("\n🔴 ФАКТОРЫ ЗА SHORT:")
            for f in bearish_factors:
                lines.append(f"   • {f}")
        
        # Предупреждения
        if warnings:
            lines.append("\n⚠️ ВНИМАНИЕ:")
            for w in warnings:
                lines.append(f"   • {w}")
        
        # Конкретные действия
        lines.append("\n" + "─"*50)
        if position in ["LONG", "LONG (осторожно)"]:
            lines.append("📋 ЧТО ДЕЛАТЬ:")
            lines.append("   1. Открыть LONG позицию")
            if trade_levels:
                lines.append(f"   2. Вход: {trade_levels['entry_market']} (по рынку)")
                lines.append(f"      или {trade_levels['entry_limit']} (лимитный ордер на откате)")
                lines.append(f"   3. 🛑 Стоп-лосс: {trade_levels['stop_loss']} (-{trade_levels['risk_pct']}%)")
                lines.append(f"   4. 🎯 Тейк-профит: {trade_levels['take_profit']} (+{trade_levels['reward_pct']}%)")
                lines.append(f"")
                lines.append(f"   📊 Частичная фиксация:")
                lines.append(f"      TP1: {trade_levels['tp1']} (закрыть 50%)")
                lines.append(f"      TP2: {trade_levels['tp2']} (закрыть 30%)")
                lines.append(f"      TP3: {trade_levels['tp3']} (оставить 20% с трейлингом)")
                lines.append(f"")
                lines.append(f"   📈 Risk/Reward: 1:{trade_levels['rr_ratio']}")
            else:
                lines.append("   2. Стоп-лосс: ниже последнего локального минимума")
                lines.append("   3. Тейк-профит: 1.5-2x от риска")
            if confidence == "НИЗКАЯ":
                lines.append("   ⚠️ Использовать минимальный размер позиции!")
        elif position in ["SHORT", "SHORT (осторожно)"]:
            lines.append("📋 ЧТО ДЕЛАТЬ:")
            lines.append("   1. Открыть SHORT позицию")
            if trade_levels:
                lines.append(f"   2. Вход: {trade_levels['entry_market']} (по рынку)")
                lines.append(f"      или {trade_levels['entry_limit']} (лимитный ордер на откате)")
                lines.append(f"   3. 🛑 Стоп-лосс: {trade_levels['stop_loss']} (+{trade_levels['risk_pct']}%)")
                lines.append(f"   4. 🎯 Тейк-профит: {trade_levels['take_profit']} (-{trade_levels['reward_pct']}%)")
                lines.append(f"")
                lines.append(f"   📊 Частичная фиксация:")
                lines.append(f"      TP1: {trade_levels['tp1']} (закрыть 50%)")
                lines.append(f"      TP2: {trade_levels['tp2']} (закрыть 30%)")
                lines.append(f"      TP3: {trade_levels['tp3']} (оставить 20% с трейлингом)")
                lines.append(f"")
                lines.append(f"   📈 Risk/Reward: 1:{trade_levels['rr_ratio']}")
            else:
                lines.append("   2. Стоп-лосс: выше последнего локального максимума")
                lines.append("   3. Тейк-профит: 1.5-2x от риска")
            if confidence == "НИЗКАЯ":
                lines.append("   ⚠️ Использовать минимальный размер позиции!")
        else:
            lines.append("📋 ЧТО ДЕЛАТЬ:")
            lines.append("   1. Не открывать новые позиции")
            lines.append("   2. Дождаться чёткого сигнала")
            lines.append("   3. Следить за изменением тренда")
            if trade_levels and current_price:
                # Показываем важные уровни даже когда вне рынка
                lines.append(f"")
                lines.append(f"   📍 Ключевые уровни для наблюдения:")
                lines.append(f"      Сопротивление: {trade_levels['tp1']}")
                lines.append(f"      Поддержка: {trade_levels['stop_loss']}")
        
        return "\n".join(lines)
    
    def print_prediction(self, result: dict):
        """Компактный вывод — только суть"""
        
        prob = result['probability']
        trade_levels = result.get('trade_levels', {})
        analysis = result.get('analysis', {})
        
        # Определяем есть ли сигнал
        has_signal = False
        direction = trade_levels.get('direction', 'LONG')
        
        # Собираем ключевые факторы
        bullish = []
        bearish = []
        warnings = []
        
        # Тренд
        trend_dir = analysis.get('trend', {}).get('direction', '')
        if 'Восходящий' in trend_dir:
            bullish.append('Тренд ↑')
        elif 'Нисходящий' in trend_dir:
            bearish.append('Тренд ↓')
        
        # RSI
        rsi_val = analysis.get('momentum', {}).get('rsi_value', 50)
        if rsi_val > 70:
            bearish.append(f'RSI {rsi_val:.0f} (перекуплен)')
        elif rsi_val < 30:
            bullish.append(f'RSI {rsi_val:.0f} (перепродан)')
        
        # Buy pressure
        bp = analysis.get('volume', {}).get('buy_percent', 50)
        if bp > 60:
            bullish.append(f'Покупатели {bp:.0f}%')
        elif bp < 40:
            bearish.append(f'Продавцы {100-bp:.0f}%')
        
        # Объём
        vol = analysis.get('volume', {}).get('ratio', 1)
        if vol < 0.5:
            warnings.append('Низкий объём')
        elif vol > 2:
            bullish.append('Высокий объём') if prob > 0.5 else bearish.append('Высокий объём')
        
        # Multi-TF
        consensus = analysis.get('multi_tf', {}).get('consensus', '')
        if 'бычьи' in consensus:
            bullish.append('Все ТФ ↑')
        elif 'медвежьи' in consensus:
            bearish.append('Все ТФ ↓')
        elif 'расходятся' in consensus:
            warnings.append('ТФ расходятся')
        
        # Волатильность
        vol_risk = analysis.get('volatility', {}).get('risk', '')
        if vol_risk == 'высокий':
            warnings.append('Высокая волатильность')
        
        # Определяем сигнал
        bull_score = len(bullish)
        bear_score = len(bearish)
        
        # Получаем символ
        symbol = result.get('symbol', '')
        
        print("\n" + "═"*55)
        if symbol:
            print(f"  💎 {symbol}  |  {result['last_price']:.6g}  |  {result['last_time']}")
        else:
            print(f"  📊 {result['last_price']:.6g}  |  {result['last_time']}")
        print("═"*55)
        
        # LONG
        if prob > 0.65 and bull_score >= 2 and bull_score > bear_score:
            has_signal = True
            confidence = "🟢 ВЫСОКАЯ"
            print(f"\n  🟢 LONG  |  Вероятность: {prob*100:.0f}%  |  {confidence}")
        elif prob > 0.58 and bull_score > bear_score:
            has_signal = True
            confidence = "🟡 СРЕДНЯЯ"
            print(f"\n  🟢 LONG  |  Вероятность: {prob*100:.0f}%  |  {confidence}")
        elif prob > 0.52 and bull_score > bear_score + 1:
            has_signal = True
            confidence = "🟡 НИЗКАЯ"
            print(f"\n  🟢 LONG (осторожно)  |  Вероятность: {prob*100:.0f}%  |  {confidence}")
        # SHORT
        elif prob < 0.35 and bear_score >= 2 and bear_score > bull_score:
            has_signal = True
            confidence = "🔴 ВЫСОКАЯ"
            direction = 'SHORT'
            print(f"\n  🔴 SHORT  |  Вероятность падения: {(1-prob)*100:.0f}%  |  {confidence}")
        elif prob < 0.42 and bear_score > bull_score:
            has_signal = True
            confidence = "🟠 СРЕДНЯЯ"
            direction = 'SHORT'
            print(f"\n  🔴 SHORT  |  Вероятность падения: {(1-prob)*100:.0f}%  |  {confidence}")
        elif prob < 0.48 and bear_score > bull_score + 1:
            has_signal = True
            confidence = "🟠 НИЗКАЯ"
            direction = 'SHORT'
            print(f"\n  🔴 SHORT (осторожно)  |  Вероятность падения: {(1-prob)*100:.0f}%  |  {confidence}")
        
        # ═══════════════════════════════════════════════════════════
        # ЕСЛИ ЕСТЬ СИГНАЛ - показываем план
        # ═══════════════════════════════════════════════════════════
        if has_signal and trade_levels:
            print("─"*55)
            
            # Получаем данные о плече
            lev = trade_levels.get('leverage', 1)
            lev_cons = trade_levels.get('leverage_conservative', 1)
            lev_aggr = trade_levels.get('leverage_aggressive', lev)
            risk_pct = trade_levels.get('position_risk_pct', 1)
            
            if direction == 'LONG':
                tp1_pct = trade_levels['reward_pct'] * 0.5
                tp2_pct = trade_levels['reward_pct']
                tp3_pct = trade_levels['reward_pct'] * 1.5
                # Средняя прибыль при закрытии всех тейков (взвешенная)
                total_profit_pct = tp1_pct * 0.5 + tp2_pct * 0.3 + tp3_pct * 0.2
                print(f"""
  📥 ВХОД:     {trade_levels['entry_market']} (рынок)
               {trade_levels['entry_limit']} (лимит на откате)
  
  🛑 СТОП:     {trade_levels['stop_loss']}  (-{trade_levels['risk_pct']}%)
  
  🎯 ТЕЙКИ:    TP1: {trade_levels['tp1']}  (+{tp1_pct:.2f}%)  — закрыть 50%
               TP2: {trade_levels['tp2']}  (+{tp2_pct:.2f}%)  — закрыть 30%
               TP3: {trade_levels['tp3']}  (+{tp3_pct:.2f}%)  — 20% трейлинг
               ─────────────────────────────────
               💰 ИТОГО: +{total_profit_pct:.2f}% (если все TP)
  
  📈 R/R:      1:{trade_levels['rr_ratio']}""")
            else:
                tp1_pct = trade_levels['reward_pct'] * 0.5
                tp2_pct = trade_levels['reward_pct']
                tp3_pct = trade_levels['reward_pct'] * 1.5
                total_profit_pct = tp1_pct * 0.5 + tp2_pct * 0.3 + tp3_pct * 0.2
                print(f"""
  📥 ВХОД:     {trade_levels['entry_market']} (рынок)
               {trade_levels['entry_limit']} (лимит на откате)
  
  🛑 СТОП:     {trade_levels['stop_loss']}  (+{trade_levels['risk_pct']}%)
  
  🎯 ТЕЙКИ:    TP1: {trade_levels['tp1']}  (+{tp1_pct:.2f}%)  — закрыть 50%
               TP2: {trade_levels['tp2']}  (+{tp2_pct:.2f}%)  — закрыть 30%
               TP3: {trade_levels['tp3']}  (+{tp3_pct:.2f}%)  — 20% трейлинг
               ─────────────────────────────────
               💰 ИТОГО: +{total_profit_pct:.2f}% (если все TP)
  
  📈 R/R:      1:{trade_levels['rr_ratio']}""")
            
            # Блок с плечом и расчёт риска
            print("─"*55)
            stop_pct = trade_levels.get('risk_pct', 0.1)  # Движение цены до стопа
            
            print(f"  ⚡ ПЛЕЧО И РИСК:")
            print(f"")
            
            # Показываем агрессивное плечо если высокая уверенность
            if lev_aggr > lev:
                print(f"     Плечо:        {lev}x (консерв. {lev_cons}x / агрессив. {lev_aggr}x)")
            else:
                print(f"     Плечо:        {lev}x (консерв. {lev_cons}x)")
            
            print(f"     До стопа:     {stop_pct}% движения цены")
            print(f"     Потеря:       {stop_pct}% × {lev}x = {stop_pct * lev:.1f}% депозита")
            print(f"")
            print(f"     📌 Пример (депозит $1000):")
            print(f"        Позиция:   ${1000 * lev}")
            print(f"        При стопе: -${1000 * stop_pct * lev / 100:.0f} ({stop_pct * lev:.1f}%)")
            
            # Показываем пример для агрессивного плеча если доступно
            if lev_aggr > lev:
                print(f"")
                print(f"     🔥 Агрессивно ({lev_aggr}x):")
                print(f"        Позиция:   ${1000 * lev_aggr}")
                print(f"        При стопе: -${1000 * stop_pct * lev_aggr / 100:.0f} ({stop_pct * lev_aggr:.1f}%)")
            
            # Предупреждение о волатильности
            if trade_levels.get('vol_warning'):
                print(f"     ⚠️  {trade_levels['vol_warning']} — max {trade_levels.get('leverage_max', 10)}x")
            
            # Тип сделки (срок)
            print("─"*55)
            trade_type_ru = trade_levels.get('trade_type_ru', '⚡ СКАЛЬПИНГ')
            hold_time = trade_levels.get('hold_time', '15-60 мин')
            trade_advice = trade_levels.get('trade_advice', '')
            
            print(f"  🕐 ТИП СДЕЛКИ: {trade_type_ru}")
            print(f"     Держать: {hold_time}")
            print(f"     💡 {trade_advice}")
            
            # Долгосрочный потенциал (если сильные сигналы на старших ТФ)
            trend_strength = analysis.get('trend', {}).get('strength', '')
            tf_consensus = analysis.get('multi_tf', {}).get('consensus', '')
            
            has_long_potential = False
            long_potential_dir = None
            
            # Проверяем условия для долгосрочного прогноза
            if trend_strength == 'сильный' and 'бычьи' in tf_consensus:
                if prob > 0.65:
                    has_long_potential = True
                    long_potential_dir = 'рост'
            elif trend_strength == 'сильный' and 'медвежьи' in tf_consensus:
                if prob < 0.35:
                    has_long_potential = True
                    long_potential_dir = 'падение'
            
            if has_long_potential:
                print("")
                print("─"*55)
                if long_potential_dir == 'рост':
                    print(f"  📈 ДОЛГОСРОЧНЫЙ ПОТЕНЦИАЛ:")
                    print(f"     Все ТФ бычьи + сильный тренд")
                    print(f"     При пробое TP2 можно оставить часть позиции")
                    print(f"     на средне-срок (1-7 дней) с трейлинг-стопом")
                    print(f"     ⚠️  Это НЕ гарантия, а потенциал!")
                else:
                    print(f"  📉 ДОЛГОСРОЧНЫЙ ПОТЕНЦИАЛ (падение):")
                    print(f"     Все ТФ медвежьи + сильный тренд")
                    print(f"     SHORT можно держать дольше с трейлингом")
                    print(f"     ⚠️  Это НЕ гарантия, следи за разворотом!")
            
            # Краткие факторы
            print("─"*55)
            if bullish:
                print(f"  ✅ {' | '.join(bullish)}")
            if bearish:
                print(f"  ❌ {' | '.join(bearish)}")
            if warnings:
                print(f"  ⚠️  {' | '.join(warnings)}")
        
        # ═══════════════════════════════════════════════════════════
        # НЕТ СИГНАЛА
        # ═══════════════════════════════════════════════════════════
        else:
            print(f"\n  ⏸️  ЖДАТЬ  |  Вероятность: {prob*100:.0f}%")
            print("─"*55)
            print("  Нет чёткого сигнала для входа")
            print("")
            
            # Почему не входим
            reasons = []
            if 0.45 <= prob <= 0.55:
                reasons.append("Вероятность ~50% (неопределённость)")
            if bull_score == bear_score:
                reasons.append("Факторы уравновешены")
            if warnings:
                reasons.append(' | '.join(warnings))
            
            if reasons:
                print(f"  Причина: {'; '.join(reasons)}")
            
            # Ключевые уровни для наблюдения
            if trade_levels:
                print("")
                print(f"  👀 Следить за уровнями:")
                print(f"     Сопротивление: {trade_levels['tp1']}")
                print(f"     Поддержка:     {trade_levels['stop_loss']}")
        
        print("═"*55 + "\n")


def main():
    """Интерактивный режим"""
    print("="*60)
    print("  🤖 CRYPTO AI PREDICTOR")
    print("="*60)
    
    predictor = CryptoPredictor()
    
    print("""
Команды:
  analyze <путь к CSV>  - анализ файла с trades
  demo                  - демо на тестовых данных
  quit                  - выход
    """)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if cmd.lower() in ['quit', 'exit', 'q']:
                print("👋 Пока!")
                break
            
            elif cmd.lower() == 'demo':
                # Демо на последних данных
                from utils import load_trades
                trades = load_trades('../data/', pattern='BTCUSD*.csv', verbose=False)
                
                # Берём последние 2 часа
                trades = trades.tail(10000)
                
                result = predictor.predict_from_trades(trades)
                predictor.print_prediction(result)
            
            elif cmd.lower().startswith('analyze '):
                path = cmd[8:].strip()
                if os.path.exists(path):
                    result = predictor.predict_from_csv(path)
                    predictor.print_prediction(result)
                else:
                    print(f"❌ Файл не найден: {path}")
            
            else:
                print("❓ Неизвестная команда. Введите 'demo' или 'analyze <путь>'")
        
        except KeyboardInterrupt:
            print("\n👋 Пока!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()
