"""
Optinus Backtest Runner

Обязательно:
- walk-forward validation
- out-of-sample тест
- комиссии + slippage
- noise test
- отдельные режимы рынка

Метрики:
- winrate
- profit factor
- max drawdown
- expectancy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMMISSION, SLIPPAGE, INITIAL_CAPITAL

# Try to import optinus
try:
    import optinus
    OPTINUS_AVAILABLE = True
except ImportError:
    OPTINUS_AVAILABLE = False
    print("Warning: optinus not installed. Using custom backtester.")


@dataclass
class Trade:
    """Информация о сделке"""
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime = None
    exit_price: float = None
    stop_loss: float = 0
    take_profit: float = 0
    size: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    commission: float = 0
    slippage: float = 0
    exit_reason: str = ""  # 'tp', 'sl', 'signal', 'timeout'


@dataclass 
class BacktestResult:
    """Результаты бэктеста"""
    # General
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    
    # Returns
    total_return: float
    total_return_pct: float
    annualized_return: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Trade metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    
    # Time metrics
    avg_trade_duration: float  # hours
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Per regime
    regime_metrics: Dict = field(default_factory=dict)
    
    # Trades list
    trades: List[Trade] = field(default_factory=list)
    
    # Equity curve
    equity_curve: pd.Series = None
    
    def summary(self) -> str:
        """Текстовое резюме результатов"""
        return f"""
╔══════════════════════════════════════════════════╗
║               BACKTEST RESULTS                    ║
╠══════════════════════════════════════════════════╣
║  Period:          {self.start_date.strftime('%Y-%m-%d')} - {self.end_date.strftime('%Y-%m-%d')}         ║
║  Initial Capital: ${self.initial_capital:,.2f}                    ║
║  Final Capital:   ${self.final_capital:,.2f}                    ║
╠══════════════════════════════════════════════════╣
║  RETURNS                                          ║
║  Total Return:    {self.total_return_pct:.2%}                         ║
║  Annualized:      {self.annualized_return:.2%}                         ║
╠══════════════════════════════════════════════════╣
║  RISK METRICS                                     ║
║  Max Drawdown:    {self.max_drawdown_pct:.2%}                         ║
║  Sharpe Ratio:    {self.sharpe_ratio:.2f}                            ║
║  Sortino Ratio:   {self.sortino_ratio:.2f}                            ║
║  Calmar Ratio:    {self.calmar_ratio:.2f}                            ║
╠══════════════════════════════════════════════════╣
║  TRADE METRICS                                    ║
║  Total Trades:    {self.total_trades}                              ║
║  Win Rate:        {self.win_rate:.2%}                         ║
║  Profit Factor:   {self.profit_factor:.2f}                            ║
║  Expectancy:      ${self.expectancy:.2f}                          ║
║  Avg Win:         ${self.avg_win:.2f}                          ║
║  Avg Loss:        ${self.avg_loss:.2f}                         ║
╠══════════════════════════════════════════════════╣
║  STREAKS                                          ║
║  Max Consec Wins:  {self.max_consecutive_wins}                             ║
║  Max Consec Losses:{self.max_consecutive_losses}                             ║
╚══════════════════════════════════════════════════╝
"""


class CustomBacktester:
    """
    Кастомный бэктестер (если optinus недоступен)
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission: float = None,
        slippage: float = None
    ):
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.commission = commission or COMMISSION
        self.slippage = slippage or SLIPPAGE
        
        self.capital = self.initial_capital
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = {}  # {symbol: Trade}
    
    def reset(self):
        """Сброс состояния"""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.positions = {}
    
    def apply_slippage(self, price: float, direction: str, is_entry: bool) -> float:
        """Применить slippage"""
        slip = price * self.slippage
        
        if direction == 'LONG':
            return price + slip if is_entry else price - slip
        else:
            return price - slip if is_entry else price + slip
    
    def open_position(
        self,
        symbol: str,
        direction: str,
        price: float,
        size: float,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime
    ):
        """Открыть позицию"""
        if symbol in self.positions:
            return  # Уже есть позиция
        
        entry_price = self.apply_slippage(price, direction, is_entry=True)
        commission = entry_price * size * self.commission
        
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_time=timestamp,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=size,
            commission=commission
        )
        
        self.positions[symbol] = trade
        self.capital -= commission
    
    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime,
        reason: str
    ):
        """Закрыть позицию"""
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        trade.exit_time = timestamp
        trade.exit_price = self.apply_slippage(price, trade.direction, is_entry=False)
        trade.exit_reason = reason
        
        # Calculate PnL
        if trade.direction == 'LONG':
            pnl = (trade.exit_price - trade.entry_price) * trade.size
        else:
            pnl = (trade.entry_price - trade.exit_price) * trade.size
        
        exit_commission = trade.exit_price * trade.size * self.commission
        trade.commission += exit_commission
        trade.pnl = pnl - trade.commission
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.size)
        
        self.capital += trade.pnl + (trade.entry_price * trade.size)  # Return capital + pnl
        self.trades.append(trade)
        del self.positions[symbol]
    
    def check_stops(self, symbol: str, high: float, low: float, timestamp: datetime):
        """Проверить стопы"""
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        if trade.direction == 'LONG':
            # Check SL
            if low <= trade.stop_loss:
                self.close_position(symbol, trade.stop_loss, timestamp, 'sl')
                return
            # Check TP
            if high >= trade.take_profit:
                self.close_position(symbol, trade.take_profit, timestamp, 'tp')
                return
        else:
            # Check SL
            if high >= trade.stop_loss:
                self.close_position(symbol, trade.stop_loss, timestamp, 'sl')
                return
            # Check TP
            if low <= trade.take_profit:
                self.close_position(symbol, trade.take_profit, timestamp, 'tp')
                return
    
    def update_equity(self, prices: Dict[str, float], timestamp: datetime):
        """Обновить equity curve"""
        equity = self.capital
        
        # Add unrealized PnL
        for symbol, trade in self.positions.items():
            if symbol in prices:
                current_price = prices[symbol]
                if trade.direction == 'LONG':
                    unrealized = (current_price - trade.entry_price) * trade.size
                else:
                    unrealized = (trade.entry_price - current_price) * trade.size
                equity += unrealized + (trade.entry_price * trade.size)
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity
        })
    
    def calculate_metrics(self) -> BacktestResult:
        """Рассчитать все метрики"""
        if not self.trades:
            return None
        
        # Equity curve
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df.set_index('timestamp', inplace=True)
        equity_series = eq_df['equity']
        
        # Returns
        returns = equity_series.pct_change().dropna()
        
        # Basic metrics
        total_return = self.capital - self.initial_capital
        total_return_pct = total_return / self.initial_capital
        
        # Dates
        start_date = self.trades[0].entry_time
        end_date = self.trades[-1].exit_time or datetime.now()
        days = (end_date - start_date).days or 1
        
        # Annualized return
        years = days / 365
        annualized_return = (1 + total_return_pct) ** (1/years) - 1 if years > 0 else 0
        
        # Drawdown
        rolling_max = equity_series.cummax()
        drawdown = equity_series - rolling_max
        max_drawdown = abs(drawdown.min())
        max_drawdown_pct = max_drawdown / rolling_max[drawdown.idxmin()] if len(drawdown) > 0 else 0
        
        # Sharpe ratio (assuming risk-free rate = 0)
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # Trade metrics
        total_trades = len(self.trades)
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        avg_trade = np.mean(pnls) if pnls else 0
        
        # Expectancy
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        # Average trade duration
        durations = []
        for t in self.trades:
            if t.exit_time:
                duration = (t.exit_time - t.entry_time).total_seconds() / 3600
                durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            avg_trade_duration=avg_duration,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            trades=self.trades,
            equity_curve=equity_series
        )


class OptinusBacktester:
    """
    Бэктестер на основе Optinus
    
    Features:
    - Walk-forward validation
    - Out-of-sample testing
    - Commission + slippage
    - Noise test
    - Market regime analysis
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission: float = None,
        slippage: float = None
    ):
        self.initial_capital = initial_capital or INITIAL_CAPITAL
        self.commission = commission or COMMISSION
        self.slippage = slippage or SLIPPAGE
        
        # Use custom backtester as fallback
        self.backtester = CustomBacktester(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=self.slippage
        )
    
    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        position_size: float = 0.1
    ) -> BacktestResult:
        """
        Запустить бэктест
        
        Args:
            signals: DataFrame с сигналами
                columns: timestamp, symbol, direction, entry, sl, tp, confidence
            prices: DataFrame с OHLCV
                columns: timestamp, symbol, open, high, low, close, volume
            position_size: размер позиции (% от капитала)
            
        Returns:
            BacktestResult
        """
        self.backtester.reset()
        
        # Sort by timestamp
        signals = signals.sort_values('timestamp')
        prices = prices.sort_values('timestamp')
        
        # Process each bar
        symbols = prices['symbol'].unique()
        timestamps = prices['timestamp'].unique()
        
        for ts in timestamps:
            ts_prices = prices[prices['timestamp'] == ts]
            ts_signals = signals[signals['timestamp'] == ts]
            
            # Update prices dict
            current_prices = {}
            for _, row in ts_prices.iterrows():
                current_prices[row['symbol']] = row['close']
                
                # Check stops for existing positions
                self.backtester.check_stops(
                    row['symbol'],
                    row['high'],
                    row['low'],
                    ts
                )
            
            # Process new signals
            for _, signal in ts_signals.iterrows():
                symbol = signal['symbol']
                direction = signal['direction']
                entry = signal['entry']
                sl = signal['sl']
                tp = signal['tp']
                
                # Position size in units
                size = (self.backtester.capital * position_size) / entry
                
                self.backtester.open_position(
                    symbol=symbol,
                    direction=direction,
                    price=entry,
                    size=size,
                    stop_loss=sl,
                    take_profit=tp,
                    timestamp=ts
                )
            
            # Update equity
            self.backtester.update_equity(current_prices, ts)
        
        # Close remaining positions
        for symbol in list(self.backtester.positions.keys()):
            if symbol in current_prices:
                self.backtester.close_position(
                    symbol,
                    current_prices[symbol],
                    timestamps[-1],
                    'timeout'
                )
        
        return self.backtester.calculate_metrics()
    
    def walk_forward_validation(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.7
    ) -> List[BacktestResult]:
        """
        Walk-forward валидация
        
        Делим данные на n_splits периодов, для каждого:
        - train на train_ratio% данных
        - test на оставшихся
        """
        results = []
        
        timestamps = sorted(prices['timestamp'].unique())
        n_bars = len(timestamps)
        split_size = n_bars // n_splits
        
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = min((i + 1) * split_size, n_bars)
            
            split_timestamps = timestamps[start_idx:end_idx]
            train_end = int(len(split_timestamps) * train_ratio)
            
            # Test period
            test_timestamps = split_timestamps[train_end:]
            
            # Filter data for test period
            test_prices = prices[prices['timestamp'].isin(test_timestamps)]
            test_signals = signals[signals['timestamp'].isin(test_timestamps)]
            
            # Run backtest
            result = self.run_backtest(test_signals, test_prices)
            if result:
                results.append(result)
        
        return results
    
    def out_of_sample_test(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[BacktestResult, BacktestResult]:
        """
        Out-of-sample тест
        
        Returns:
            (in_sample_result, out_of_sample_result)
        """
        timestamps = sorted(prices['timestamp'].unique())
        split_idx = int(len(timestamps) * train_ratio)
        
        train_ts = timestamps[:split_idx]
        test_ts = timestamps[split_idx:]
        
        # In-sample
        is_prices = prices[prices['timestamp'].isin(train_ts)]
        is_signals = signals[signals['timestamp'].isin(train_ts)]
        is_result = self.run_backtest(is_signals, is_prices)
        
        # Out-of-sample
        oos_prices = prices[prices['timestamp'].isin(test_ts)]
        oos_signals = signals[signals['timestamp'].isin(test_ts)]
        oos_result = self.run_backtest(oos_signals, oos_prices)
        
        return is_result, oos_result
    
    def noise_test(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        noise_levels: List[float] = [0.001, 0.002, 0.005],
        n_iterations: int = 10
    ) -> Dict[float, List[BacktestResult]]:
        """
        Noise test - проверка устойчивости к шуму
        
        Добавляем случайный шум к ценам и смотрим как меняются результаты
        """
        results = {}
        
        for noise in noise_levels:
            noise_results = []
            
            for _ in range(n_iterations):
                # Add noise to prices
                noisy_prices = prices.copy()
                for col in ['open', 'high', 'low', 'close']:
                    if col in noisy_prices.columns:
                        noise_factor = 1 + np.random.normal(0, noise, len(noisy_prices))
                        noisy_prices[col] = noisy_prices[col] * noise_factor
                
                result = self.run_backtest(signals, noisy_prices)
                if result:
                    noise_results.append(result)
            
            results[noise] = noise_results
        
        return results
    
    def regime_analysis(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        regime_column: str = 'regime'
    ) -> Dict[str, BacktestResult]:
        """
        Анализ по режимам рынка
        
        Если в prices есть колонка regime, разбиваем по режимам
        """
        results = {}
        
        if regime_column not in prices.columns:
            # Calculate regime from returns
            prices = prices.copy()
            prices['return'] = prices.groupby('symbol')['close'].pct_change(20)
            prices['regime'] = 'flat'
            prices.loc[prices['return'] > 0.05, 'regime'] = 'bull'
            prices.loc[prices['return'] < -0.05, 'regime'] = 'bear'
        
        for regime in prices[regime_column].unique():
            regime_prices = prices[prices[regime_column] == regime]
            regime_timestamps = regime_prices['timestamp'].unique()
            regime_signals = signals[signals['timestamp'].isin(regime_timestamps)]
            
            result = self.run_backtest(regime_signals, regime_prices)
            if result:
                results[regime] = result
        
        return results


def test_backtester():
    """Тест бэктестера"""
    import random
    
    # Generate test data
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    
    prices_data = []
    signals_data = []
    
    price = 42000
    for i, date in enumerate(dates):
        change = random.uniform(-0.02, 0.02)
        price = price * (1 + change)
        
        prices_data.append({
            'timestamp': date,
            'symbol': 'BTCUSDT',
            'open': price * 0.999,
            'high': price * 1.01,
            'low': price * 0.99,
            'close': price,
            'volume': random.uniform(1000, 10000)
        })
        
        # Random signals
        if random.random() < 0.1:
            direction = 'LONG' if random.random() > 0.5 else 'SHORT'
            sl_dist = price * 0.02
            tp_dist = price * 0.04
            
            signals_data.append({
                'timestamp': date,
                'symbol': 'BTCUSDT',
                'direction': direction,
                'entry': price,
                'sl': price - sl_dist if direction == 'LONG' else price + sl_dist,
                'tp': price + tp_dist if direction == 'LONG' else price - tp_dist,
                'confidence': random.uniform(0.6, 0.9)
            })
    
    prices_df = pd.DataFrame(prices_data)
    signals_df = pd.DataFrame(signals_data)
    
    # Run backtest
    backtester = OptinusBacktester()
    result = backtester.run_backtest(signals_df, prices_df)
    
    if result:
        print(result.summary())


if __name__ == "__main__":
    test_backtester()
