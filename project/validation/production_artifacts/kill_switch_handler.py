"""
Kill Switch Handler for Production Trading
"""

import os
import json
from pathlib import Path
from datetime import datetime


class KillSwitchHandler:
    """
    Emergency kill switch for trading system.
    
    Can be triggered:
    - Manually via activate()
    - Automatically via check_conditions()
    - Externally by creating KILL_SWITCH_ACTIVE file
    """
    
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_path = self.artifacts_dir / 'kill_switch_config.json'
        self.active_file = self.artifacts_dir / 'KILL_SWITCH_ACTIVE'
        
        self._load_config()
    
    def _load_config(self):
        """Load kill switch configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {'conditions': {}, 'auto_triggers': []}
    
    @property
    def is_active(self) -> bool:
        """Check if kill switch is active"""
        return self.active_file.exists()
    
    def activate(self, reason: str, auto: bool = False):
        """Activate kill switch"""
        activation = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'auto_triggered': auto
        }
        
        with open(self.active_file, 'w') as f:
            json.dump(activation, f, indent=2)
        
        # Log
        print(f"⚠️ KILL SWITCH ACTIVATED!")
        print(f"   Reason: {reason}")
        print(f"   Time: {activation['timestamp']}")
        
        return True
    
    def deactivate(self, confirm_code: str = None):
        """Deactivate kill switch (requires confirmation)"""
        if confirm_code != 'CONFIRM_DEACTIVATE':
            print("❌ Must provide confirmation code: CONFIRM_DEACTIVATE")
            return False
        
        if self.active_file.exists():
            self.active_file.unlink()
            print("✅ Kill switch deactivated")
            return True
        
        print("ℹ️ Kill switch was not active")
        return True
    
    def check_conditions(self, metrics: dict) -> bool:
        """
        Check if conditions warrant automatic activation.
        
        Args:
            metrics: Dict with current trading metrics
            
        Returns:
            True if kill switch was activated
        """
        conditions = self.config.get('conditions', {})
        
        # Check drawdown
        if metrics.get('current_drawdown', 0) > conditions.get('max_drawdown', 0.20):
            self.activate(f"Max drawdown exceeded: {metrics['current_drawdown']:.1%}", auto=True)
            return True
        
        # Check consecutive losses
        if metrics.get('consecutive_losses', 0) > conditions.get('max_consecutive_losses', 5):
            self.activate(f"Max consecutive losses: {metrics['consecutive_losses']}", auto=True)
            return True
        
        # Check daily loss
        if metrics.get('daily_loss_percent', 0) > conditions.get('max_daily_loss_percent', 5.0):
            self.activate(f"Max daily loss: {metrics['daily_loss_percent']:.1f}%", auto=True)
            return True
        
        return False


if __name__ == "__main__":
    import sys
    
    handler = KillSwitchHandler('./production_artifacts')
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'activate':
            reason = sys.argv[2] if len(sys.argv) > 2 else 'Manual activation'
            handler.activate(reason)
        elif sys.argv[1] == 'deactivate':
            handler.deactivate('CONFIRM_DEACTIVATE')
        elif sys.argv[1] == 'status':
            print(f"Kill switch active: {handler.is_active}")
