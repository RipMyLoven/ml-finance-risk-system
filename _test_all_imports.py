import sys
sys.path.insert(0, r'C:\Users\ripmy\Documents\GitHub\ml-finance-risk-system')
tests = [
    "tui.config",
    "tui.state",
    "tui.exchange.base",
    "tui.exchange.mexc",
    "tui.cache.memory",
    "tui.cache.disk",
    "tui.model_registry.registry",
    "tui.history",
]

for mod in tests:
    try:
        __import__(mod)
        print(f"  OK   {mod}")
    except Exception as e:
        print(f"  FAIL {mod}: {e}")

# Test feature builders separately (they need numpy/pandas/sklearn)
try:
    from tui.features.builders import ScalpFeatureBuilder, IntradayFeatureBuilder, SwingFeatureBuilder, RiskFeatureBuilder
    print(f"  OK   tui.features.builders")
except Exception as e:
    print(f"  FAIL tui.features.builders: {e}")

# Test orchestrator
try:
    from tui.orchestrator import Orchestrator
    print(f"  OK   tui.orchestrator")
except Exception as e:
    print(f"  FAIL tui.orchestrator: {e}")

# Test TUI app
try:
    from tui.app import TradingTUI
    print(f"  OK   tui.app")
except Exception as e:
    print(f"  FAIL tui.app: {e}")

# Test __main__
try:
    from tui.__main__ import main
    print(f"  OK   tui.__main__")
except Exception as e:
    print(f"  FAIL tui.__main__: {e}")
