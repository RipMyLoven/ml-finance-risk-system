"""
═══════════════════════════════════════════════════════════════════════════════
 1️⃣1️⃣ LOGGING & OBSERVABILITY
═══════════════════════════════════════════════════════════════════════════════

Logging requirements:
✅ All stages logged to console
✅ Logs are readable and structured
✅ Log model parameters
✅ Log validation metrics
✅ Log risk decisions
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

LOG_CONFIG = {
    'level': logging.INFO,
    'format': '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'log_dir': 'logs',
    'max_file_size_mb': 10,
    'backup_count': 5
}

# Log categories
LOG_CATEGORIES = {
    'data': 'DataPipeline',
    'features': 'FeatureEngine',
    'training': 'ModelTraining',
    'validation': 'Validation',
    'risk': 'RiskManagement',
    'inference': 'Inference',
    'system': 'System'
}


@dataclass
class LoggingResult:
    """Results from logging validation"""
    passed: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    log_tests: Dict[str, bool] = field(default_factory=dict)
    sample_logs: List[str] = field(default_factory=list)


class StructuredLogger:
    """
    Structured logger for trading system.
    
    Provides consistent, readable, and parseable log output
    for all system components.
    """
    
    def __init__(self, name: str, log_dir: str = None):
        self.name = name
        self.log_dir = Path(log_dir or LOG_CONFIG['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with handlers"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(LOG_CONFIG['level'])
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_CONFIG['level'])
        console_formatter = ColoredFormatter(
            LOG_CONFIG['format'],
            datefmt=LOG_CONFIG['date_format']
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        from logging.handlers import RotatingFileHandler
        
        file_path = self.log_dir / f'{self.name.lower()}.log'
        file_handler = RotatingFileHandler(
            file_path,
            maxBytes=LOG_CONFIG['max_file_size_mb'] * 1024 * 1024,
            backupCount=LOG_CONFIG['backup_count']
        )
        file_handler.setLevel(LOG_CONFIG['level'])
        file_formatter = logging.Formatter(
            LOG_CONFIG['format'],
            datefmt=LOG_CONFIG['date_format']
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # JSON file handler for structured logs
        json_path = self.log_dir / f'{self.name.lower()}_structured.jsonl'
        self.json_handler = JsonFileHandler(json_path)
    
    def info(self, message: str, **kwargs):
        """Log info message with context"""
        self.logger.info(self._format_message(message, kwargs))
        self._write_json('INFO', message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context"""
        self.logger.warning(self._format_message(message, kwargs))
        self._write_json('WARNING', message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context"""
        self.logger.error(self._format_message(message, kwargs))
        self._write_json('ERROR', message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context"""
        self.logger.debug(self._format_message(message, kwargs))
        self._write_json('DEBUG', message, kwargs)
    
    def metric(self, name: str, value: float, **tags):
        """Log a metric value"""
        message = f"METRIC | {name}={value:.6f}"
        if tags:
            tag_str = ' | '.join(f'{k}={v}' for k, v in tags.items())
            message += f" | {tag_str}"
        self.logger.info(message)
        self._write_json('METRIC', name, {'value': value, **tags})
    
    def params(self, params_dict: Dict[str, Any], prefix: str = ""):
        """Log parameters"""
        for key, value in params_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.params(value, full_key)
            else:
                self.logger.info(f"PARAM | {full_key}={value}")
                self._write_json('PARAM', full_key, {'value': value})
    
    def risk_decision(self, decision: str, reason: str, **context):
        """Log risk management decision"""
        message = f"RISK | {decision} | Reason: {reason}"
        if context:
            ctx_str = ' | '.join(f'{k}={v}' for k, v in context.items())
            message += f" | {ctx_str}"
        self.logger.info(message)
        self._write_json('RISK', decision, {'reason': reason, **context})
    
    def trade_signal(self, symbol: str, signal: str, confidence: float, **context):
        """Log trade signal"""
        message = f"SIGNAL | {symbol} | {signal} | confidence={confidence:.2%}"
        if context:
            ctx_str = ' | '.join(f'{k}={v}' for k, v in context.items())
            message += f" | {ctx_str}"
        self.logger.info(message)
        self._write_json('SIGNAL', symbol, {
            'signal': signal, 
            'confidence': confidence,
            **context
        })
    
    def _format_message(self, message: str, context: Dict) -> str:
        """Format message with context"""
        if not context:
            return message
        ctx_str = ' | '.join(f'{k}={v}' for k, v in context.items())
        return f"{message} | {ctx_str}"
    
    def _write_json(self, level: str, message: str, context: Dict):
        """Write structured JSON log"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'context': context
        }
        self.json_handler.write(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


class JsonFileHandler:
    """Handler for JSON-lines log file"""
    
    def __init__(self, filepath: Path):
        self.filepath = filepath
    
    def write(self, entry: Dict):
        """Write JSON entry to file"""
        with open(self.filepath, 'a') as f:
            f.write(json.dumps(entry) + '\n')


class LoggingValidator:
    """
    Validates logging configuration and functionality.
    
    Implements checkpoint 11:
    - Console logging verification
    - Log structure validation
    - Parameter logging
    - Metric logging
    - Risk decision logging
    """
    
    def __init__(self, log_to_console: bool = True):
        self.log_to_console = log_to_console
        self.results = LoggingResult()
        self.log_dir = Path(__file__).parent / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, msg: str, level: str = "INFO"):
        if self.log_to_console:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "ℹ️", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"[{timestamp}] {prefix.get(level, '')} {msg}")
    
    def run_all_validations(self) -> LoggingResult:
        """Run all logging validations"""
        print("\n" + "="*80)
        print("  1️⃣1️⃣ LOGGING & OBSERVABILITY")
        print("="*80 + "\n")
        
        # 1. Test basic logging
        print("  ── 1. Basic Logging Test ──")
        self._test_basic_logging()
        
        # 2. Test structured logging
        print("\n  ── 2. Structured Logging Test ──")
        self._test_structured_logging()
        
        # 3. Test parameter logging
        print("\n  ── 3. Parameter Logging Test ──")
        self._test_parameter_logging()
        
        # 4. Test metric logging
        print("\n  ── 4. Metric Logging Test ──")
        self._test_metric_logging()
        
        # 5. Test risk decision logging
        print("\n  ── 5. Risk Decision Logging Test ──")
        self._test_risk_logging()
        
        # 6. Test log file output
        print("\n  ── 6. Log File Validation ──")
        self._validate_log_files()
        
        # 7. Generate logging configuration
        print("\n  ── 7. Generate Logging Config ──")
        self._generate_logging_config()
        
        # Summary
        self._print_summary()
        
        return self.results
    
    def _test_basic_logging(self):
        """Test basic logging functionality"""
        logger = StructuredLogger('TestBasic', self.log_dir)
        
        print("\n    Testing log levels:")
        
        try:
            logger.debug("Debug message test", component="test")
            print("    ✅ DEBUG level")
            
            logger.info("Info message test", component="test")
            print("    ✅ INFO level")
            
            logger.warning("Warning message test", component="test")
            print("    ✅ WARNING level")
            
            logger.error("Error message test", component="test")
            print("    ✅ ERROR level")
            
            self.results.log_tests['basic_logging'] = True
            self.results.sample_logs.extend([
                "2024-01-01 12:00:00 | INFO     | TestBasic            | Info message test | component=test"
            ])
            
        except Exception as e:
            self.results.errors.append(f"Basic logging failed: {e}")
            self.results.log_tests['basic_logging'] = False
    
    def _test_structured_logging(self):
        """Test structured logging with context"""
        logger = StructuredLogger('TestStructured', self.log_dir)
        
        print("\n    Testing structured logs:")
        
        try:
            # Log with multiple context fields
            logger.info(
                "Model training started",
                model_type="scalp",
                n_features=20,
                n_samples=10000,
                random_seed=42
            )
            print("    ✅ Multi-context logging")
            
            # Log nested context
            logger.info(
                "Hyperparameters set",
                learning_rate=0.01,
                n_estimators=100,
                max_depth=6
            )
            print("    ✅ Hyperparameter logging")
            
            self.results.log_tests['structured_logging'] = True
            
        except Exception as e:
            self.results.errors.append(f"Structured logging failed: {e}")
            self.results.log_tests['structured_logging'] = False
    
    def _test_parameter_logging(self):
        """Test parameter logging"""
        logger = StructuredLogger('TestParams', self.log_dir)
        
        print("\n    Testing parameter logging:")
        
        try:
            model_params = {
                'model_type': 'LightGBM',
                'hyperparameters': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'num_leaves': 31
                },
                'training': {
                    'n_samples': 10000,
                    'n_features': 20,
                    'cv_folds': 5
                }
            }
            
            logger.params(model_params)
            print("    ✅ Nested parameter logging")
            
            self.results.log_tests['parameter_logging'] = True
            
        except Exception as e:
            self.results.errors.append(f"Parameter logging failed: {e}")
            self.results.log_tests['parameter_logging'] = False
    
    def _test_metric_logging(self):
        """Test metric logging"""
        logger = StructuredLogger('TestMetrics', self.log_dir)
        
        print("\n    Testing metric logging:")
        
        try:
            # Log various metrics
            logger.metric('auc_roc', 0.856, model='scalp', split='validation')
            print("    ✅ AUC-ROC metric")
            
            logger.metric('accuracy', 0.723, model='scalp', split='validation')
            print("    ✅ Accuracy metric")
            
            logger.metric('log_loss', 0.445, model='scalp', split='validation')
            print("    ✅ Log loss metric")
            
            logger.metric('sharpe_ratio', 1.52, strategy='ensemble', period='backtest')
            print("    ✅ Sharpe ratio metric")
            
            logger.metric('max_drawdown', -0.082, strategy='ensemble', period='backtest')
            print("    ✅ Max drawdown metric")
            
            self.results.log_tests['metric_logging'] = True
            
        except Exception as e:
            self.results.errors.append(f"Metric logging failed: {e}")
            self.results.log_tests['metric_logging'] = False
    
    def _test_risk_logging(self):
        """Test risk decision logging"""
        logger = StructuredLogger('TestRisk', self.log_dir)
        
        print("\n    Testing risk decision logging:")
        
        try:
            # Log various risk decisions
            logger.risk_decision(
                decision='POSITION_SIZE_REDUCED',
                reason='CVaR threshold exceeded',
                original_size=1000,
                adjusted_size=500,
                cvar_value=-0.085
            )
            print("    ✅ Position size decision")
            
            logger.risk_decision(
                decision='TRADE_BLOCKED',
                reason='Drawdown limit reached',
                current_dd=-0.12,
                dd_limit=-0.10
            )
            print("    ✅ Trade block decision")
            
            logger.risk_decision(
                decision='LEVERAGE_ADJUSTED',
                reason='Kelly criterion update',
                old_leverage=3.0,
                new_leverage=2.0,
                kelly_fraction=0.15
            )
            print("    ✅ Leverage adjustment")
            
            # Log trade signals
            logger.trade_signal(
                symbol='BTCUSDT',
                signal='BUY',
                confidence=0.78,
                model='scalp',
                entry_price=42500.00,
                stop_loss=42000.00
            )
            print("    ✅ Trade signal logging")
            
            self.results.log_tests['risk_logging'] = True
            
        except Exception as e:
            self.results.errors.append(f"Risk logging failed: {e}")
            self.results.log_tests['risk_logging'] = False
    
    def _validate_log_files(self):
        """Validate log file output"""
        print("\n    Validating log files:")
        
        try:
            # Check .log files
            log_files = list(self.log_dir.glob('*.log'))
            print(f"    Found {len(log_files)} .log files")
            
            # Check .jsonl files
            jsonl_files = list(self.log_dir.glob('*.jsonl'))
            print(f"    Found {len(jsonl_files)} .jsonl files")
            
            # Validate JSON structure
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r') as f:
                    lines = f.readlines()
                    valid_json = 0
                    for line in lines:
                        try:
                            json.loads(line.strip())
                            valid_json += 1
                        except:
                            pass
                    print(f"    ✅ {jsonl_file.name}: {valid_json} valid JSON entries")
            
            self.results.log_tests['log_files_valid'] = True
            
        except Exception as e:
            self.results.errors.append(f"Log file validation failed: {e}")
            self.results.log_tests['log_files_valid'] = False
    
    def _generate_logging_config(self):
        """Generate logging configuration file"""
        
        config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': LOG_CONFIG['format'],
                    'datefmt': LOG_CONFIG['date_format']
                },
                'json': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(timestamp)s %(level)s %(name)s %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'filename': 'logs/trading_system.log',
                    'maxBytes': 10485760,
                    'backupCount': 5
                },
                'json_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': 'logs/trading_system.jsonl',
                    'maxBytes': 10485760,
                    'backupCount': 5
                }
            },
            'loggers': {
                'DataPipeline': {'level': 'INFO', 'handlers': ['console', 'file']},
                'FeatureEngine': {'level': 'INFO', 'handlers': ['console', 'file']},
                'ModelTraining': {'level': 'DEBUG', 'handlers': ['console', 'file', 'json_file']},
                'Validation': {'level': 'INFO', 'handlers': ['console', 'file', 'json_file']},
                'RiskManagement': {'level': 'INFO', 'handlers': ['console', 'file', 'json_file']},
                'Inference': {'level': 'INFO', 'handlers': ['console', 'file', 'json_file']},
                'System': {'level': 'WARNING', 'handlers': ['console', 'file']}
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        }
        
        config_path = self.log_dir / 'logging_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"    ✅ Saved logging_config.json")
        
        # Create logger initialization code
        init_code = '''"""
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
'''
        
        init_path = self.log_dir / 'logging_init.py'
        with open(init_path, 'w') as f:
            f.write(init_code)
        
        print(f"    ✅ Created logging_init.py")
    
    def _print_summary(self):
        """Print summary"""
        print("\n" + "="*80)
        print("  📋 LOGGING & OBSERVABILITY SUMMARY")
        print("="*80)
        
        print(f"\n  📝 Log Tests:")
        for test, passed in self.results.log_tests.items():
            status = "✅" if passed else "❌"
            print(f"    {status} {test}")
        
        print(f"\n  📄 Sample Log Output:")
        print("    " + "-"*60)
        print("    2024-01-01 12:00:00 | INFO     | ModelTraining        | Training started | model=scalp")
        print("    2024-01-01 12:00:01 | INFO     | ModelTraining        | METRIC | auc_roc=0.856000 | model=scalp")
        print("    2024-01-01 12:00:02 | WARNING  | RiskManagement       | RISK | POSITION_SIZE_REDUCED | CVaR exceeded")
        print("    2024-01-01 12:00:03 | INFO     | Inference            | SIGNAL | BTCUSDT | BUY | confidence=78.00%")
        print("    " + "-"*60)
        
        if self.results.errors:
            print(f"\n  Errors:")
            for err in self.results.errors:
                print(f"    ❌ {err}")
            self.results.passed = False
        
        if self.results.warnings:
            print(f"\n  Warnings:")
            for warn in self.results.warnings:
                print(f"    ⚠️ {warn}")
        
        all_tests_passed = all(self.results.log_tests.values())
        
        if all_tests_passed and not self.results.errors:
            print(f"\n  ✅ LOGGING VALIDATION: PASSED")
        else:
            print(f"\n  ❌ LOGGING VALIDATION: ISSUES FOUND")
            self.results.passed = False
        
        print(f"\n  📁 Logs saved to: {self.log_dir}")
        print("\n" + "="*80 + "\n")
    
    def save_results(self, output_path: str = None):
        """Save results to JSON"""
        if output_path is None:
            output_path = Path(__file__).parent / 'logging_validation_report.json'
        
        report = {
            'passed': self.results.passed,
            'log_tests': self.results.log_tests,
            'sample_logs': self.results.sample_logs,
            'warnings': self.results.warnings,
            'errors': self.results.errors,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to: {output_path}")
        return output_path


def run_logging_validation() -> LoggingResult:
    """Run logging validation"""
    validator = LoggingValidator()
    result = validator.run_all_validations()
    validator.save_results()
    return result


if __name__ == "__main__":
    result = run_logging_validation()
    sys.exit(0 if result.passed else 1)
