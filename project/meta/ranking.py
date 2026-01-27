"""
Coin Ranking - ранжирование монет для выбора лучших сигналов

Формула:
    final_score = confidence * expected_return / risk_score

Отбираются TOP 5-10 монет
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOP_COINS_COUNT, MAX_CORRELATION, MAX_CONCURRENT_POSITIONS
from meta.meta_engine import MetaDecision


@dataclass
class RankedCoin:
    """Результат ранжирования монеты"""
    symbol: str
    direction: str
    final_score: float
    confidence: float
    expected_return: float
    risk_score: float
    rank: int
    selected: bool      # Выбрана ли для торговли
    reason: str


class CoinRanker:
    """
    Ранжирование монет по качеству сигнала
    """
    
    def __init__(
        self,
        top_n: int = None,
        max_correlation: float = None,
        max_positions: int = None,
        min_score: float = 0.1
    ):
        """
        Args:
            top_n: сколько монет выбирать (default 10)
            max_correlation: максимальная корреляция (default 0.7)
            max_positions: максимум позиций (default 5)
            min_score: минимальный score для рассмотрения
        """
        self.top_n = top_n or TOP_COINS_COUNT
        self.max_correlation = max_correlation or MAX_CORRELATION
        self.max_positions = max_positions or MAX_CONCURRENT_POSITIONS
        self.min_score = min_score
    
    def calculate_final_score(
        self,
        confidence: float,
        expected_return: float,
        risk_score: float
    ) -> float:
        """
        Рассчитать final_score
        
        Formula: confidence * expected_return / risk_score
        
        Чем выше confidence и expected_return и ниже risk - тем лучше
        """
        # Защита от деления на 0
        risk_score = max(risk_score, 0.1)
        
        # Expected return может быть отрицательным для short
        # Берём абсолютное значение для расчёта score
        abs_return = abs(expected_return)
        
        final_score = confidence * abs_return / risk_score
        
        return final_score
    
    def rank_coins(
        self,
        decisions: Dict[str, MetaDecision]
    ) -> List[RankedCoin]:
        """
        Ранжировать монеты по final_score
        
        Args:
            decisions: Dict[symbol] = MetaDecision
            
        Returns:
            List[RankedCoin] отсортированный по final_score
        """
        ranked = []
        
        for symbol, decision in decisions.items():
            # Пропускаем если не должны торговать
            if not decision.should_trade:
                continue
            
            final_score = self.calculate_final_score(
                decision.confidence,
                decision.expected_return,
                decision.risk_score
            )
            
            # Минимальный порог
            if final_score < self.min_score:
                continue
            
            ranked.append(RankedCoin(
                symbol=symbol,
                direction=decision.direction,
                final_score=final_score,
                confidence=decision.confidence,
                expected_return=decision.expected_return,
                risk_score=decision.risk_score,
                rank=0,
                selected=False,
                reason=""
            ))
        
        # Сортируем по score (убывание)
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Присваиваем ранги
        for i, coin in enumerate(ranked):
            coin.rank = i + 1
        
        return ranked
    
    def filter_by_correlation(
        self,
        ranked_coins: List[RankedCoin],
        correlation_matrix: Dict[Tuple[str, str], float] = None
    ) -> List[RankedCoin]:
        """
        Фильтрация по корреляции
        
        Не берём монеты, которые сильно коррелируют с уже выбранными
        
        Args:
            ranked_coins: отсортированный список
            correlation_matrix: Dict[(symbol1, symbol2)] = correlation
            
        Returns:
            Отфильтрованный список
        """
        if correlation_matrix is None:
            # Если матрицы нет - пропускаем фильтр
            return ranked_coins
        
        selected = []
        selected_symbols = set()
        
        for coin in ranked_coins:
            # Проверяем корреляцию с уже выбранными
            is_correlated = False
            
            for selected_symbol in selected_symbols:
                # Получаем корреляцию (проверяем оба направления ключа)
                corr = correlation_matrix.get(
                    (coin.symbol, selected_symbol),
                    correlation_matrix.get((selected_symbol, coin.symbol), 0)
                )
                
                if abs(corr) > self.max_correlation:
                    is_correlated = True
                    coin.reason = f"High correlation with {selected_symbol} ({corr:.2f})"
                    break
            
            if not is_correlated:
                selected.append(coin)
                selected_symbols.add(coin.symbol)
        
        return selected
    
    def select_top_coins(
        self,
        decisions: Dict[str, MetaDecision],
        correlation_matrix: Dict[Tuple[str, str], float] = None
    ) -> List[RankedCoin]:
        """
        Выбрать TOP монет для торговли
        
        Args:
            decisions: решения Meta Engine
            correlation_matrix: матрица корреляций
            
        Returns:
            Список выбранных монет (max top_n, max max_positions)
        """
        # Ранжируем
        ranked = self.rank_coins(decisions)
        
        if not ranked:
            return []
        
        # Фильтруем по корреляции
        filtered = self.filter_by_correlation(ranked, correlation_matrix)
        
        # Выбираем top N
        selected = filtered[:min(self.top_n, self.max_positions)]
        
        # Помечаем выбранные
        for coin in selected:
            coin.selected = True
            if not coin.reason:
                coin.reason = "Selected for trading"
        
        return selected
    
    def get_summary(self, selected_coins: List[RankedCoin]) -> str:
        """Получить текстовое резюме"""
        if not selected_coins:
            return "No coins selected for trading"
        
        lines = [
            "="*50,
            "COIN RANKING RESULTS",
            "="*50,
            f"Total selected: {len(selected_coins)}",
            "-"*50
        ]
        
        for coin in selected_coins:
            lines.append(
                f"#{coin.rank} {coin.symbol}: {coin.direction} | "
                f"Score: {coin.final_score:.4f} | "
                f"Conf: {coin.confidence:.2f} | "
                f"Risk: {coin.risk_score:.2f}"
            )
        
        lines.append("="*50)
        
        return "\n".join(lines)


def calculate_correlation_matrix(
    price_data: Dict[str, np.ndarray],
    window: int = 24
) -> Dict[Tuple[str, str], float]:
    """
    Рассчитать матрицу корреляций между монетами
    
    Args:
        price_data: Dict[symbol] = returns array
        window: окно для расчёта корреляции
        
    Returns:
        Dict[(symbol1, symbol2)] = correlation
    """
    symbols = list(price_data.keys())
    correlations = {}
    
    for i, sym1 in enumerate(symbols):
        for sym2 in symbols[i+1:]:
            returns1 = price_data[sym1]
            returns2 = price_data[sym2]
            
            # Выравниваем длины
            min_len = min(len(returns1), len(returns2))
            if min_len < window:
                correlations[(sym1, sym2)] = 0
                continue
            
            r1 = returns1[-min_len:]
            r2 = returns2[-min_len:]
            
            # Rolling correlation (берём последнее значение)
            if len(r1) >= window:
                corr = np.corrcoef(r1[-window:], r2[-window:])[0, 1]
            else:
                corr = np.corrcoef(r1, r2)[0, 1]
            
            correlations[(sym1, sym2)] = corr if not np.isnan(corr) else 0
    
    return correlations


def test_ranking():
    """Тест ранжирования"""
    from meta.meta_engine import MetaDecisionEngine
    
    # Создаём тестовые решения
    engine = MetaDecisionEngine()
    
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
        },
        'SOLUSDT': {
            'scalp': {'P_up': 0.6, 'P_down': 0.2, 'expected_return': 0.03},
            'intraday': {'P_up': 0.55, 'P_down': 0.25, 'expected_return': 0.04},
            'swing': {'P_up': 0.5, 'P_down': 0.35, 'expected_return': 0.06},
            'risk_score': 0.5
        }
    }
    
    decisions = {}
    for symbol, preds in test_data.items():
        decisions[symbol] = engine.decide(
            symbol=symbol,
            scalp_pred=preds['scalp'],
            intraday_pred=preds['intraday'],
            swing_pred=preds['swing'],
            risk_score=preds['risk_score']
        )
    
    # Ранжируем
    ranker = CoinRanker(top_n=5)
    selected = ranker.select_top_coins(decisions)
    
    print(ranker.get_summary(selected))


if __name__ == "__main__":
    test_ranking()
