"""
Logging initialization for trading system.

Usage:
    from logging_init import setup_logging, get_logger
    
    setup_logging()
    logger = get_logger('ModelTraining')
    logger.info('Training started', model='scalp')
"""

import json
import logging
import logging.config
from pathlib import Path


def setup_logging(config_path: str = None):
    """Initialize logging from config file"""
    if config_path is None:
        config_path = Path(__file__).parent / 'logging_config.json'
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Ensure log directory exists
    Path('logs').mkdir(exist_ok=True)
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger"""
    return logging.getLogger(name)


class MetricsLogger:
    """Specialized logger for metrics"""
    
    def __init__(self, logger_name: str = 'Metrics'):
        self.logger = logging.getLogger(logger_name)
    
    def log_metric(self, name: str, value: float, **tags):
        """Log a metric with tags"""
        tag_str = ' '.join(f'{k}={v}' for k, v in tags.items())
        self.logger.info(f"METRIC {name}={value:.6f} {tag_str}")
    
    def log_batch_metrics(self, metrics: dict, **tags):
        """Log multiple metrics at once"""
        for name, value in metrics.items():
            self.log_metric(name, value, **tags)


class RiskLogger:
    """Specialized logger for risk decisions"""
    
    def __init__(self, logger_name: str = 'RiskManagement'):
        self.logger = logging.getLogger(logger_name)
    
    def log_decision(self, decision: str, reason: str, **context):
        """Log a risk decision"""
        ctx_str = ' '.join(f'{k}={v}' for k, v in context.items())
        self.logger.info(f"RISK_DECISION [{decision}] {reason} {ctx_str}")
    
    def log_signal(self, symbol: str, signal: str, confidence: float, **context):
        """Log a trade signal"""
        ctx_str = ' '.join(f'{k}={v}' for k, v in context.items())
        self.logger.info(f"TRADE_SIGNAL {symbol} {signal} confidence={confidence:.2%} {ctx_str}")


if __name__ == "__main__":
    setup_logging()
    
    logger = get_logger('ModelTraining')
    logger.info('Test log message')
    
    metrics = MetricsLogger()
    metrics.log_metric('accuracy', 0.85, model='scalp', split='test')
    
    risk = RiskLogger()
    risk.log_signal('BTCUSDT', 'BUY', 0.75, model='scalp')
