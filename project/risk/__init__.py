"""
Advanced Risk Management System

CORE CONTROLLER for multi-model trading system with:
- CVaR (Conditional Value at Risk) estimation
- Kelly-based position sizing
- Drawdown-aware dynamic sizing
- Model authorization and veto power

This module provides institutional-grade risk management for:
- Scalp model
- Intraday model  
- Swing model
"""

from .cvar_engine import CVaREngine
from .kelly_sizing import KellySizer
from .drawdown_controller import DrawdownController
from .risk_model import AdvancedRiskModel
from .risk_features import build_advanced_risk_features

__all__ = [
    'CVaREngine',
    'KellySizer', 
    'DrawdownController',
    'AdvancedRiskModel',
    'build_advanced_risk_features'
]
