"""
Signal Generator - генерация готовых торговых сигналов

Формат сигнала:
    SYMBOL: ETHUSDT
    Direction: LONG
    Confidence: 0.69
    Entry: 2310–2325
    Stop Loss: 2278
    Take Profit: 2360 / 2420
    Leverage: 3x
    Risk: 0.8%
    Timeframe: Intraday

Risk Management Rules:
- max 1-2% риска на сделку
- max 3-5 одновременных позиций
- корреляция между монетами < threshold
- no trade в high-volatility regime
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAX_RISK_PER_TRADE, MAX_CONCURRENT_POSITIONS,
    LEVERAGE_MAP, HIGH_VOLATILITY_THRESHOLD
)
from meta.meta_engine import MetaDecision
from meta.ranking import RankedCoin


@dataclass
class TradingSignal:
    """Готовый торговый сигнал"""
    # Основное
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    confidence: float       # 0-1
    
    # Entry
    entry_price: float
    entry_range_low: float
    entry_range_high: float
    
    # Risk Management
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    
    # Position sizing
    leverage: int
    risk_percent: float
    position_size_pct: float  # % от капитала
    
    # Meta
    timeframe: str          # 'scalp', 'intraday', 'swing'
    risk_score: float
    expected_return: float
    
    # Timing
    generated_at: datetime = field(default_factory=datetime.now)
    valid_until: datetime = None
    
    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'entry_range': f"{self.entry_range_low:.2f}-{self.entry_range_high:.2f}",
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'leverage': self.leverage,
            'risk_percent': self.risk_percent,
            'position_size_pct': self.position_size_pct,
            'timeframe': self.timeframe,
            'risk_score': self.risk_score,
            'expected_return': self.expected_return,
            'generated_at': self.generated_at.isoformat(),
            'valid_until': self.valid_until.isoformat() if self.valid_until else None
        }
    
    def format(self) -> str:
        """Форматированный вывод сигнала"""
        return f"""
+==================================================+
|                 TRADING SIGNAL                    |
+==================================================+
|  SYMBOL:      {self.symbol:<20}             |
|  Direction:   {self.direction:<20}             |
|  Confidence:  {self.confidence:.2f}                              |
+--------------------------------------------------+
|  Entry:       {self.entry_range_low:.2f} - {self.entry_range_high:.2f}               |
|  Stop Loss:   {self.stop_loss:.2f}                            |
|  Take Profit: {self.take_profit_1:.2f} / {self.take_profit_2:.2f}              |
+--------------------------------------------------+
|  Leverage:    {self.leverage}x                                |
|  Risk:        {self.risk_percent:.1%}                              |
|  Position:    {self.position_size_pct:.1%} of capital                   |
|  Timeframe:   {self.timeframe:<20}             |
+--------------------------------------------------+
|  Risk Score:  {self.risk_score:.2f}                              |
|  Exp. Return: {self.expected_return:.2%}                           |
|  Generated:   {self.generated_at.strftime('%Y-%m-%d %H:%M')}                   |
+==================================================+
"""


class SignalGenerator:
    """
    Генератор торговых сигналов
    
    Преобразует решения Meta Engine в готовые торговые сигналы
    """
    
    def __init__(
        self,
        max_risk_per_trade: float = None,
        max_positions: int = None,
        high_vol_threshold: float = None
    ):
        """
        Args:
            max_risk_per_trade: максимальный риск на сделку (default 2%)
            max_positions: максимум позиций (default 5)
            high_vol_threshold: порог высокой волатильности
        """
        self.max_risk_per_trade = max_risk_per_trade or MAX_RISK_PER_TRADE
        self.max_positions = max_positions or MAX_CONCURRENT_POSITIONS
        self.high_vol_threshold = high_vol_threshold or HIGH_VOLATILITY_THRESHOLD
    
    def calculate_entry_range(
        self,
        current_price: float,
        direction: str,
        volatility: float
    ) -> tuple:
        """
        Рассчитать диапазон входа
        
        Даём ~0.5-1% от текущей цены для входа
        """
        entry_width = current_price * volatility * 0.5  # Половина ATR
        
        if direction == 'LONG':
            # Для long - ниже текущей цены
            entry_low = current_price - entry_width
            entry_high = current_price
        else:
            # Для short - выше текущей цены
            entry_low = current_price
            entry_high = current_price + entry_width
        
        return entry_low, entry_high
    
    def calculate_stop_loss(
        self,
        current_price: float,
        direction: str,
        volatility: float,
        risk_score: float
    ) -> float:
        """
        Рассчитать Stop Loss
        
        Базовый SL = 1.5-2 * ATR, корректируем на риск
        """
        # Множитель зависит от риска
        sl_multiplier = 1.5 + risk_score * 0.5  # 1.5 - 2.0 ATR
        sl_distance = current_price * volatility * sl_multiplier
        
        if direction == 'LONG':
            stop_loss = current_price - sl_distance
        else:
            stop_loss = current_price + sl_distance
        
        return stop_loss
    
    def calculate_take_profits(
        self,
        current_price: float,
        direction: str,
        stop_loss: float,
        risk_score: float
    ) -> tuple:
        """
        Рассчитать Take Profit уровни
        
        TP1 = 1.5 * SL distance (Risk:Reward 1:1.5)
        TP2 = 2.5 * SL distance (Risk:Reward 1:2.5)
        """
        sl_distance = abs(current_price - stop_loss)
        
        # R:R зависит от риска
        rr_1 = 1.5 if risk_score > 0.5 else 2.0
        rr_2 = 2.5 if risk_score > 0.5 else 3.0
        
        if direction == 'LONG':
            tp1 = current_price + sl_distance * rr_1
            tp2 = current_price + sl_distance * rr_2
        else:
            tp1 = current_price - sl_distance * rr_1
            tp2 = current_price - sl_distance * rr_2
        
        return tp1, tp2
    
    def calculate_leverage(self, risk_score: float) -> int:
        """
        Рассчитать рекомендуемое плечо
        """
        if risk_score < 0.3:
            return LEVERAGE_MAP['low_risk']
        elif risk_score < 0.5:
            return LEVERAGE_MAP['medium_risk']
        else:
            return LEVERAGE_MAP['high_risk']
    
    def calculate_position_size(
        self,
        risk_score: float,
        leverage: int
    ) -> tuple:
        """
        Рассчитать размер позиции
        
        Returns:
            (risk_percent, position_size_pct)
        """
        # Риск на сделку снижается при высоком risk_score
        risk_pct = self.max_risk_per_trade * (1 - risk_score * 0.3)
        
        # Position size = Risk / (SL distance * leverage)
        # Упрощённо: position_size зависит от риска и плеча
        position_size = risk_pct * leverage
        
        # Ограничиваем
        position_size = min(position_size, 0.2)  # Max 20% капитала
        
        return risk_pct, position_size
    
    def check_high_volatility(
        self,
        current_volatility: float,
        average_volatility: float
    ) -> bool:
        """
        Проверить высокую волатильность
        
        Returns:
            True если волатильность слишком высокая
        """
        if average_volatility == 0:
            return False
        
        vol_ratio = current_volatility / average_volatility
        return vol_ratio > self.high_vol_threshold
    
    def generate_signal(
        self,
        ranked_coin: RankedCoin,
        decision: MetaDecision,
        current_price: float,
        volatility: float,
        timeframe: str = None
    ) -> Optional[TradingSignal]:
        """
        Сгенерировать сигнал для одной монеты
        
        Args:
            ranked_coin: результат ранжирования
            decision: решение Meta Engine
            current_price: текущая цена
            volatility: текущая волатильность (ATR normalized)
            timeframe: таймфрейм сигнала
            
        Returns:
            TradingSignal или None если условия не выполнены
        """
        if not ranked_coin.selected:
            return None
        
        direction = ranked_coin.direction
        risk_score = ranked_coin.risk_score
        
        # Calculate all parameters
        entry_low, entry_high = self.calculate_entry_range(
            current_price, direction, volatility
        )
        
        stop_loss = self.calculate_stop_loss(
            current_price, direction, volatility, risk_score
        )
        
        tp1, tp2 = self.calculate_take_profits(
            current_price, direction, stop_loss, risk_score
        )
        
        leverage = self.calculate_leverage(risk_score)
        risk_pct, position_size = self.calculate_position_size(risk_score, leverage)
        
        # Create signal
        signal = TradingSignal(
            symbol=ranked_coin.symbol,
            direction=direction,
            confidence=ranked_coin.confidence,
            entry_price=current_price,
            entry_range_low=entry_low,
            entry_range_high=entry_high,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            leverage=leverage,
            risk_percent=risk_pct,
            position_size_pct=position_size,
            timeframe=timeframe or decision.timeframe_used,
            risk_score=risk_score,
            expected_return=ranked_coin.expected_return
        )
        
        return signal
    
    def generate_signals(
        self,
        ranked_coins: List[RankedCoin],
        decisions: Dict[str, MetaDecision],
        prices: Dict[str, float],
        volatilities: Dict[str, float],
        average_volatilities: Dict[str, float] = None
    ) -> List[TradingSignal]:
        """
        Сгенерировать сигналы для всех выбранных монет
        
        Args:
            ranked_coins: список отранжированных монет
            decisions: решения Meta Engine
            prices: текущие цены {symbol: price}
            volatilities: текущие волатильности {symbol: vol}
            average_volatilities: средние волатильности для проверки
            
        Returns:
            Список торговых сигналов
        """
        signals = []
        
        for coin in ranked_coins:
            if not coin.selected:
                continue
            
            symbol = coin.symbol
            
            # Проверяем наличие данных
            if symbol not in prices or symbol not in volatilities:
                continue
            
            current_price = prices[symbol]
            volatility = volatilities[symbol]
            
            # Проверка высокой волатильности
            if average_volatilities:
                avg_vol = average_volatilities.get(symbol, volatility)
                if self.check_high_volatility(volatility, avg_vol):
                    print(f"⚠️ Skipping {symbol}: High volatility regime")
                    continue
            
            # Генерируем сигнал
            decision = decisions.get(symbol)
            if decision is None:
                continue
            
            signal = self.generate_signal(
                coin, decision, current_price, volatility
            )
            
            if signal:
                signals.append(signal)
        
        # Ограничиваем количество позиций
        return signals[:self.max_positions]
    
    def format_all_signals(self, signals: List[TradingSignal]) -> str:
        """Форматированный вывод всех сигналов"""
        if not signals:
            return "No signals generated"
        
        output = [
            "="*50,
            f"GENERATED SIGNALS ({len(signals)})",
            "="*50
        ]
        
        for signal in signals:
            output.append(signal.format())
        
        return "\n".join(output)


def test_signal_generator():
    """Тест генератора сигналов"""
    from meta.meta_engine import MetaDecisionEngine
    from meta.ranking import CoinRanker
    
    # Setup
    engine = MetaDecisionEngine()
    ranker = CoinRanker()
    generator = SignalGenerator()
    
    # Test data
    test_data = {
        'BTCUSDT': {
            'scalp': {'P_up': 0.7, 'P_down': 0.15, 'expected_return': 0.02},
            'intraday': {'P_up': 0.65, 'P_down': 0.2, 'expected_return': 0.03},
            'swing': {'P_up': 0.6, 'P_down': 0.25, 'expected_return': 0.05},
            'risk_score': 0.3
        },
        'ETHUSDT': {
            'scalp': {'P_up': 0.75, 'P_down': 0.1, 'expected_return': 0.025},
            'intraday': {'P_up': 0.7, 'P_down': 0.15, 'expected_return': 0.035},
            'swing': {'P_up': 0.55, 'P_down': 0.3, 'expected_return': 0.04},
            'risk_score': 0.25
        }
    }
    
    prices = {'BTCUSDT': 42000, 'ETHUSDT': 2300}
    volatilities = {'BTCUSDT': 0.015, 'ETHUSDT': 0.02}
    
    # Generate decisions
    decisions = {}
    for symbol, preds in test_data.items():
        decisions[symbol] = engine.decide(
            symbol=symbol,
            scalp_pred=preds['scalp'],
            intraday_pred=preds['intraday'],
            swing_pred=preds['swing'],
            risk_score=preds['risk_score']
        )
    
    # Rank coins
    selected = ranker.select_top_coins(decisions)
    
    # Generate signals
    signals = generator.generate_signals(
        selected, decisions, prices, volatilities
    )
    
    # Output
    print(generator.format_all_signals(signals))


if __name__ == "__main__":
    test_signal_generator()
