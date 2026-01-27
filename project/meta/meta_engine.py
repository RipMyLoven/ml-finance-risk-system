"""
Meta Decision Engine - Rule-based логика для объединения сигналов

НЕ ML! Простая rule-based система.

Формула:
    score_long = 0.5 * scalp_P_up + 0.3 * intraday_P_up + 0.2 * swing_P_up

Условия входа:
- score > threshold (0.65)
- swing не против направления
- risk_score < max_risk

Выход:
- direction
- confidence
- timeframe_used
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import META_WEIGHTS, ENTRY_THRESHOLD, MAX_RISK_SCORE


@dataclass
class MetaDecision:
    """Результат Meta Decision Engine"""
    symbol: str
    direction: str          # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float       # 0-1
    score_long: float       # Score для long
    score_short: float      # Score для short
    timeframe_used: str     # 'scalp', 'intraday', 'swing', 'multi'
    expected_return: float  # Ожидаемый return
    risk_score: float       # Risk score из Risk AI
    should_trade: bool      # Итоговое решение
    reason: str             # Причина решения


class MetaDecisionEngine:
    """
    Rule-based Meta Decision Engine
    
    Объединяет сигналы от Scalp, Intraday и Swing моделей
    """
    
    def __init__(
        self,
        scalp_weight: float = None,
        intraday_weight: float = None,
        swing_weight: float = None,
        entry_threshold: float = None,
        max_risk_score: float = None
    ):
        """
        Args:
            scalp_weight: вес scalp модели (default 0.5)
            intraday_weight: вес intraday модели (default 0.3)
            swing_weight: вес swing модели (default 0.2)
            entry_threshold: порог для входа (default 0.65)
            max_risk_score: максимальный риск (default 0.7)
        """
        self.scalp_weight = scalp_weight or META_WEIGHTS['scalp']
        self.intraday_weight = intraday_weight or META_WEIGHTS['intraday']
        self.swing_weight = swing_weight or META_WEIGHTS['swing']
        
        self.entry_threshold = entry_threshold or ENTRY_THRESHOLD
        self.max_risk_score = max_risk_score or MAX_RISK_SCORE
        
        # Normalize weights
        total_weight = self.scalp_weight + self.intraday_weight + self.swing_weight
        self.scalp_weight /= total_weight
        self.intraday_weight /= total_weight
        self.swing_weight /= total_weight
    
    def calculate_scores(
        self,
        scalp_pred: Dict,
        intraday_pred: Dict,
        swing_pred: Dict
    ) -> Tuple[float, float]:
        """
        Рассчитать score для long и short
        
        Args:
            scalp_pred: {'P_up': float, 'P_down': float, 'expected_return': float}
            intraday_pred: то же
            swing_pred: то же
            
        Returns:
            (score_long, score_short)
        """
        # Score LONG
        score_long = (
            self.scalp_weight * scalp_pred.get('P_up', 0.33) +
            self.intraday_weight * intraday_pred.get('P_up', 0.33) +
            self.swing_weight * swing_pred.get('P_up', 0.33)
        )
        
        # Score SHORT
        score_short = (
            self.scalp_weight * scalp_pred.get('P_down', 0.33) +
            self.intraday_weight * intraday_pred.get('P_down', 0.33) +
            self.swing_weight * swing_pred.get('P_down', 0.33)
        )
        
        return score_long, score_short
    
    def check_swing_alignment(
        self,
        direction: str,
        swing_pred: Dict
    ) -> Tuple[bool, str]:
        """
        Проверить что swing не против направления
        
        Returns:
            (is_aligned, reason)
        """
        swing_up = swing_pred.get('P_up', 0.33)
        swing_down = swing_pred.get('P_down', 0.33)
        
        # Swing считается "против" если вероятность противоположного > 0.5
        if direction == 'LONG' and swing_down > 0.5:
            return False, f"Swing against LONG (P_down={swing_down:.2f})"
        
        if direction == 'SHORT' and swing_up > 0.5:
            return False, f"Swing against SHORT (P_up={swing_up:.2f})"
        
        return True, "Swing aligned"
    
    def determine_timeframe(
        self,
        scalp_pred: Dict,
        intraday_pred: Dict,
        swing_pred: Dict,
        direction: str
    ) -> str:
        """
        Определить основной таймфрейм сигнала
        
        Выбираем тот, где confidence максимальный
        """
        if direction == 'LONG':
            confidences = {
                'scalp': scalp_pred.get('P_up', 0),
                'intraday': intraday_pred.get('P_up', 0),
                'swing': swing_pred.get('P_up', 0)
            }
        else:
            confidences = {
                'scalp': scalp_pred.get('P_down', 0),
                'intraday': intraday_pred.get('P_down', 0),
                'swing': swing_pred.get('P_down', 0)
            }
        
        # Если все примерно равны - multi
        values = list(confidences.values())
        if max(values) - min(values) < 0.1:
            return 'multi'
        
        return max(confidences, key=confidences.get)
    
    def calculate_expected_return(
        self,
        scalp_pred: Dict,
        intraday_pred: Dict,
        swing_pred: Dict
    ) -> float:
        """
        Рассчитать ожидаемый return
        """
        exp_ret = (
            self.scalp_weight * scalp_pred.get('expected_return', 0) +
            self.intraday_weight * intraday_pred.get('expected_return', 0) +
            self.swing_weight * swing_pred.get('expected_return', 0)
        )
        return exp_ret
    
    def decide(
        self,
        symbol: str,
        scalp_pred: Dict,
        intraday_pred: Dict,
        swing_pred: Dict,
        risk_score: float
    ) -> MetaDecision:
        """
        Принять решение на основе всех входных данных
        
        Args:
            symbol: торговая пара
            scalp_pred: предсказания scalp модели
            intraday_pred: предсказания intraday модели
            swing_pred: предсказания swing модели
            risk_score: risk score из Risk AI
            
        Returns:
            MetaDecision
        """
        # Calculate scores
        score_long, score_short = self.calculate_scores(
            scalp_pred, intraday_pred, swing_pred
        )
        
        # Determine direction
        if score_long > score_short:
            direction = 'LONG'
            confidence = score_long
        elif score_short > score_long:
            direction = 'SHORT'
            confidence = score_short
        else:
            direction = 'NEUTRAL'
            confidence = 0.5
        
        # Expected return
        expected_return = self.calculate_expected_return(
            scalp_pred, intraday_pred, swing_pred
        )
        
        # Decision flags
        should_trade = True
        reasons = []
        
        # Check 1: Confidence threshold
        if confidence < self.entry_threshold:
            should_trade = False
            reasons.append(f"Low confidence ({confidence:.2f} < {self.entry_threshold})")
        
        # Check 2: Swing alignment
        if direction != 'NEUTRAL':
            aligned, align_reason = self.check_swing_alignment(direction, swing_pred)
            if not aligned:
                should_trade = False
                reasons.append(align_reason)
        
        # Check 3: Risk check
        if risk_score > self.max_risk_score:
            should_trade = False
            reasons.append(f"High risk ({risk_score:.2f} > {self.max_risk_score})")
        
        # Check 4: Neutral direction
        if direction == 'NEUTRAL':
            should_trade = False
            reasons.append("No clear direction")
        
        # Determine timeframe
        timeframe_used = self.determine_timeframe(
            scalp_pred, intraday_pred, swing_pred, direction
        ) if direction != 'NEUTRAL' else 'none'
        
        # Final reason
        if should_trade:
            reason = f"All conditions met. Direction: {direction}, Confidence: {confidence:.2f}"
        else:
            reason = "; ".join(reasons)
        
        return MetaDecision(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            score_long=score_long,
            score_short=score_short,
            timeframe_used=timeframe_used,
            expected_return=expected_return,
            risk_score=risk_score,
            should_trade=should_trade,
            reason=reason
        )
    
    def batch_decide(
        self,
        predictions: Dict[str, Dict]
    ) -> Dict[str, MetaDecision]:
        """
        Принять решения для множества символов
        
        Args:
            predictions: {
                'BTCUSDT': {
                    'scalp': {...},
                    'intraday': {...},
                    'swing': {...},
                    'risk_score': float
                },
                ...
            }
            
        Returns:
            Dict[symbol] = MetaDecision
        """
        decisions = {}
        
        for symbol, preds in predictions.items():
            decisions[symbol] = self.decide(
                symbol=symbol,
                scalp_pred=preds.get('scalp', {}),
                intraday_pred=preds.get('intraday', {}),
                swing_pred=preds.get('swing', {}),
                risk_score=preds.get('risk_score', 0.5)
            )
        
        return decisions


def test_meta_engine():
    """Тест Meta Engine"""
    engine = MetaDecisionEngine()
    
    # Test case 1: Strong LONG signal
    decision = engine.decide(
        symbol='BTCUSDT',
        scalp_pred={'P_up': 0.7, 'P_down': 0.15, 'P_flat': 0.15, 'expected_return': 0.02},
        intraday_pred={'P_up': 0.65, 'P_down': 0.2, 'P_flat': 0.15, 'expected_return': 0.03},
        swing_pred={'P_up': 0.6, 'P_down': 0.25, 'P_flat': 0.15, 'expected_return': 0.05},
        risk_score=0.3
    )
    
    print("Test 1: Strong LONG")
    print(f"  Direction: {decision.direction}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Should trade: {decision.should_trade}")
    print(f"  Reason: {decision.reason}")
    print()
    
    # Test case 2: Swing against
    decision = engine.decide(
        symbol='ETHUSDT',
        scalp_pred={'P_up': 0.7, 'P_down': 0.15, 'P_flat': 0.15, 'expected_return': 0.02},
        intraday_pred={'P_up': 0.65, 'P_down': 0.2, 'P_flat': 0.15, 'expected_return': 0.03},
        swing_pred={'P_up': 0.3, 'P_down': 0.55, 'P_flat': 0.15, 'expected_return': -0.02},
        risk_score=0.3
    )
    
    print("Test 2: Swing against LONG")
    print(f"  Direction: {decision.direction}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Should trade: {decision.should_trade}")
    print(f"  Reason: {decision.reason}")
    print()
    
    # Test case 3: High risk
    decision = engine.decide(
        symbol='SOLUSDT',
        scalp_pred={'P_up': 0.7, 'P_down': 0.15, 'P_flat': 0.15, 'expected_return': 0.02},
        intraday_pred={'P_up': 0.65, 'P_down': 0.2, 'P_flat': 0.15, 'expected_return': 0.03},
        swing_pred={'P_up': 0.6, 'P_down': 0.25, 'P_flat': 0.15, 'expected_return': 0.05},
        risk_score=0.8
    )
    
    print("Test 3: High risk")
    print(f"  Direction: {decision.direction}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Should trade: {decision.should_trade}")
    print(f"  Reason: {decision.reason}")


if __name__ == "__main__":
    test_meta_engine()
