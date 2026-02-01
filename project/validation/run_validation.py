"""
═══════════════════════════════════════════════════════════════════════════════
 🚀 MAIN VALIDATION RUNNER
═══════════════════════════════════════════════════════════════════════════════

Complete 12-checkpoint validation pipeline for AI Crypto Trading System.

Usage:
    python run_validation.py              # Run all checkpoints
    python run_validation.py --checkpoint 1 2 3  # Run specific checkpoints
    python run_validation.py --skip 4 5   # Skip specific checkpoints
    python run_validation.py --quick      # Quick mode (skip heavy computations)
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import validators
from validation.data_sanity import run_data_sanity_check
from validation.feature_validation import run_feature_validation
from validation.labeling_validation import run_labeling_validation
from validation.baseline_models import run_baseline_training
from validation.optuna_optimization import run_optuna_optimization
from validation.risk_validation import run_risk_validation
from validation.ensemble_validation import run_ensemble_validation
from validation.overfitting_control import run_overfitting_check
from validation.onnx_validation import run_onnx_validation
from validation.production_readiness import run_production_readiness_check
from validation.logging_observability import run_logging_validation
from validation.final_review import run_final_review


# Checkpoint configuration
CHECKPOINTS = {
    1: {
        'name': 'Data & Sanity Checks',
        'runner': run_data_sanity_check,
        'description': 'File structure, NaN/outliers, temporal integrity, look-ahead bias, leakage',
        'estimated_time': '2-5 min',
        'critical': True
    },
    2: {
        'name': 'Feature Engineering Validation',
        'runner': run_feature_validation,
        'description': 'Feature split by model type, correlations, multicollinearity, stability',
        'estimated_time': '3-8 min',
        'critical': True
    },
    3: {
        'name': 'Labeling & Targets',
        'runner': run_labeling_validation,
        'description': 'Target definitions, horizon validation, leakage detection',
        'estimated_time': '1-3 min',
        'critical': True
    },
    4: {
        'name': 'Baseline Models',
        'runner': run_baseline_training,
        'description': 'Train baselines (random, constant), record reference metrics',
        'estimated_time': '5-10 min',
        'critical': False
    },
    5: {
        'name': 'Optuna Optimization',
        'runner': run_optuna_optimization,
        'description': 'Per-model hyperparameter studies, pruning, best params',
        'estimated_time': '15-30 min',
        'critical': True
    },
    6: {
        'name': 'Advanced Risk Model Validation',
        'runner': run_risk_validation,
        'description': 'CVaR validation, Fractional Kelly, Drawdown-aware scaling',
        'estimated_time': '5-10 min',
        'critical': True
    },
    7: {
        'name': 'Model Interaction & Ensemble',
        'runner': run_ensemble_validation,
        'description': 'Signal exchange, conflict detection, Risk veto',
        'estimated_time': '3-8 min',
        'critical': True
    },
    8: {
        'name': 'Metrics & Overfitting Control',
        'runner': run_overfitting_check,
        'description': 'Train/val/test metrics, error distributions, time stability',
        'estimated_time': '5-10 min',
        'critical': True
    },
    9: {
        'name': 'ONNX Export & Validation',
        'runner': run_onnx_validation,
        'description': 'Export to ONNX, verify I/O, compare outputs, latency',
        'estimated_time': '3-8 min',
        'critical': True
    },
    10: {
        'name': 'Production Readiness',
        'runner': run_production_readiness_check,
        'description': 'Seeds, determinism, artifacts, inference pipeline, kill-switch',
        'estimated_time': '2-5 min',
        'critical': True
    },
    11: {
        'name': 'Logging & Observability',
        'runner': run_logging_validation,
        'description': 'Console logging, structured logs, metrics/risk logging',
        'estimated_time': '1-3 min',
        'critical': False
    },
    12: {
        'name': 'Final Review & Sign-off',
        'runner': run_final_review,
        'description': 'Summary table, production assessment, GO/NO-GO verdict',
        'estimated_time': '1 min',
        'critical': True
    }
}


def print_header():
    """Print validation header"""
    header = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    █████╗ ██╗    ████████╗██████╗  █████╗ ██╗███╗   ██╗                       ║
║   ██╔══██╗██║    ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║                       ║
║   ███████║██║       ██║   ██████╔╝███████║██║██╔██╗ ██║                       ║
║   ██╔══██║██║       ██║   ██╔══██╗██╔══██║██║██║╚██╗██║                       ║
║   ██║  ██║██║       ██║   ██║  ██║██║  ██║██║██║ ╚████║                       ║
║   ╚═╝  ╚═╝╚═╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝                       ║
║                                                                               ║
║             🚀 CRYPTO TRADING SYSTEM - VALIDATION PIPELINE 🚀                 ║
║                                                                               ║
║   12-Checkpoint Production Readiness Validation                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(header)


def print_checkpoint_list():
    """Print list of all checkpoints"""
    print("\n  📋 CHECKPOINTS:")
    print("  " + "-"*75)
    
    for num, info in CHECKPOINTS.items():
        critical = "⚡" if info['critical'] else "  "
        print(f"  {critical} {num:2d}. {info['name']:<35} ({info['estimated_time']})")
    
    print("  " + "-"*75)
    print("  ⚡ = Critical checkpoint")
    print()


def run_checkpoint(checkpoint_num: int, quick_mode: bool = False) -> Tuple[bool, float]:
    """
    Run a single checkpoint.
    
    Args:
        checkpoint_num: Checkpoint number (1-12)
        quick_mode: Skip heavy computations
        
    Returns:
        Tuple of (passed, duration_seconds)
    """
    if checkpoint_num not in CHECKPOINTS:
        print(f"  ❌ Invalid checkpoint: {checkpoint_num}")
        return False, 0
    
    info = CHECKPOINTS[checkpoint_num]
    
    print(f"\n{'='*80}")
    print(f"  Starting Checkpoint {checkpoint_num}: {info['name']}")
    print(f"  {info['description']}")
    print(f"  Estimated time: {info['estimated_time']}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = info['runner']()
        passed = getattr(result, 'passed', True)
    except Exception as e:
        print(f"  ❌ Checkpoint {checkpoint_num} failed with error: {e}")
        import traceback
        traceback.print_exc()
        passed = False
    
    duration = time.time() - start_time
    
    print(f"\n  Checkpoint {checkpoint_num} completed in {duration:.1f}s")
    print(f"  Result: {'✅ PASSED' if passed else '❌ FAILED'}")
    
    return passed, duration


def run_all_checkpoints(
    checkpoint_list: List[int] = None,
    skip_list: List[int] = None,
    quick_mode: bool = False
) -> Dict:
    """
    Run validation checkpoints.
    
    Args:
        checkpoint_list: Specific checkpoints to run (default: all)
        skip_list: Checkpoints to skip
        quick_mode: Skip heavy computations
        
    Returns:
        Summary dict with results
    """
    # Determine which checkpoints to run
    if checkpoint_list:
        to_run = sorted([c for c in checkpoint_list if c in CHECKPOINTS])
    else:
        to_run = sorted(CHECKPOINTS.keys())
    
    if skip_list:
        to_run = [c for c in to_run if c not in skip_list]
    
    print(f"\n  Running {len(to_run)} checkpoints: {to_run}\n")
    
    results = {}
    total_start = time.time()
    
    for checkpoint_num in to_run:
        passed, duration = run_checkpoint(checkpoint_num, quick_mode)
        results[checkpoint_num] = {
            'name': CHECKPOINTS[checkpoint_num]['name'],
            'passed': passed,
            'duration': duration,
            'critical': CHECKPOINTS[checkpoint_num]['critical']
        }
        
        # Early exit on critical failure?
        if not passed and CHECKPOINTS[checkpoint_num]['critical']:
            print(f"\n  ⚠️ Critical checkpoint {checkpoint_num} failed!")
            # Continue anyway to gather full picture
    
    total_duration = time.time() - total_start
    
    # Print summary
    print_final_summary(results, total_duration)
    
    return results


def print_final_summary(results: Dict, total_duration: float):
    """Print final summary of all checkpoints"""
    print("\n" + "="*80)
    print("  📊 VALIDATION PIPELINE SUMMARY")
    print("="*80)
    
    print(f"\n  {'#':<4} {'Checkpoint':<35} {'Status':<10} {'Time':<10} {'Critical'}")
    print("  " + "-"*70)
    
    passed_count = 0
    critical_failures = 0
    
    for num, result in sorted(results.items()):
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        critical = "⚡" if result['critical'] else ""
        time_str = f"{result['duration']:.1f}s"
        
        print(f"  {num:<4} {result['name']:<35} {status:<10} {time_str:<10} {critical}")
        
        if result['passed']:
            passed_count += 1
        elif result['critical']:
            critical_failures += 1
    
    print("  " + "-"*70)
    print(f"  Total: {passed_count}/{len(results)} passed")
    print(f"  Total time: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    
    # Overall verdict
    print("\n" + "="*80)
    if critical_failures == 0 and passed_count == len(results):
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║              🎉  ALL CHECKPOINTS PASSED  🎉                  ║
    ║                                                               ║
    ║        System is ready for production deployment!             ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
""")
    elif critical_failures == 0:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║          🟡  NON-CRITICAL ISSUES FOUND  🟡                   ║
    ║                                                               ║
    ║     Review warnings before deployment. System may proceed.    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║            🔴  CRITICAL ISSUES FOUND  🔴                     ║
    ║                                                               ║
    ║     Address all critical failures before deployment!          ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
""")
    
    print("="*80 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AI Crypto Trading System - Validation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py                    Run all 12 checkpoints
  python run_validation.py --checkpoint 1 2   Run only checkpoints 1 and 2
  python run_validation.py --skip 4 5         Skip checkpoints 4 and 5
  python run_validation.py --quick            Quick mode (faster but less thorough)
  python run_validation.py --list             List all checkpoints
        """
    )
    
    parser.add_argument(
        '--checkpoint', '-c',
        type=int,
        nargs='+',
        help='Specific checkpoints to run (1-12)'
    )
    
    parser.add_argument(
        '--skip', '-s',
        type=int,
        nargs='+',
        help='Checkpoints to skip'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode - skip heavy computations'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all checkpoints and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    print_header()
    
    if args.list:
        print_checkpoint_list()
        return 0
    
    print_checkpoint_list()
    
    print(f"  📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  🖥️  Python: {sys.version.split()[0]}")
    print()
    
    results = run_all_checkpoints(
        checkpoint_list=args.checkpoint,
        skip_list=args.skip,
        quick_mode=args.quick
    )
    
    # Save results
    output_path = Path(__file__).parent / 'validation_results.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': {str(k): v for k, v in results.items()},
            'summary': {
                'total': len(results),
                'passed': sum(1 for r in results.values() if r['passed']),
                'critical_failures': sum(1 for r in results.values() if not r['passed'] and r['critical'])
            }
        }, f, indent=2)
    
    print(f"  📄 Results saved to: {output_path}")
    
    # Exit code based on results
    critical_failures = sum(1 for r in results.values() if not r['passed'] and r['critical'])
    return 0 if critical_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
