"""
⚡ Быстрый анализ - просто запусти и получи прогноз!

Использование:
    python quick_predict.py                    # Анализ последних BTC данных
    python quick_predict.py SOLUSD             # Анализ SOL
    python quick_predict.py path/to/file.csv   # Анализ конкретного файла
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict import CryptoPredictor
from utils import load_trades


def main():
    # Загружаем модель
    predictor = CryptoPredictor()
    
    # Определяем что анализировать
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if os.path.exists(arg):
            # Это путь к файлу
            trades = load_trades(arg, verbose=False)
        else:
            # Это название монеты
            pattern = f"{arg.upper()}*.csv"
            print(f"🔍 Ищу файлы: {pattern}")
            trades = load_trades('../data/', pattern=pattern, verbose=False)
    else:
        # По умолчанию - BTC
        print("🔍 Анализирую BTC...")
        trades = load_trades('../data/', pattern='BTCUSD*.csv', verbose=False)
    
    # Берём последние данные (последние 2 часа примерно)
    trades = trades.tail(50000)
    
    # Получаем прогноз
    result = predictor.predict_from_trades(trades)
    predictor.print_prediction(result)
    
    # Возвращаем код для автоматизации
    if result['probability'] > 0.6:
        return 1  # BUY
    elif result['probability'] < 0.4:
        return -1  # SELL
    else:
        return 0  # NEUTRAL


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code >= 0 else 255 + exit_code + 1)
