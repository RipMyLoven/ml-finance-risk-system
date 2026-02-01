"""
═══════════════════════════════════════════════════════════════════════════════
 1️⃣2️⃣ FINAL REVIEW & SIGN-OFF
═══════════════════════════════════════════════════════════════════════════════

Final review:
✅ Summary table for all checks
✅ Production readiness assessment
✅ Remaining risks
✅ Clear GO / NO-GO verdict
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class CheckpointResult:
    """Result for a single checkpoint"""
    name: str
    passed: bool
    warnings: int = 0
    errors: int = 0
    details: Dict = field(default_factory=dict)


@dataclass
class FinalReviewResult:
    """Final review result"""
    verdict: str = "PENDING"  # GO, NO-GO, CONDITIONAL
    checkpoints: Dict[str, CheckpointResult] = field(default_factory=dict)
    critical_issues: List[str] = field(default_factory=list)
    remaining_risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    production_ready: bool = False


class FinalReviewer:
    """
    Final Review and Sign-off
    
    Implements checkpoint 12:
    - Aggregates all checkpoint results
    - Produces summary table
    - Assesses production readiness
    - Issues GO/NO-GO verdict
    """
    
    # Checkpoint configuration
    CHECKPOINTS = {
        1: {'name': 'Data & Sanity', 'report': 'data_sanity_report.json', 'critical': True},
        2: {'name': 'Feature Validation', 'report': 'feature_validation_report.json', 'critical': True},
        3: {'name': 'Labeling & Targets', 'report': 'labeling_validation_report.json', 'critical': True},
        4: {'name': 'Baseline Models', 'report': 'baseline_report.json', 'critical': False},
        5: {'name': 'Optuna Optimization', 'report': 'optuna_optimization_report.json', 'critical': True},
        6: {'name': 'Risk Model Validation', 'report': 'risk_validation_report.json', 'critical': True},
        7: {'name': 'Ensemble Validation', 'report': 'ensemble_validation_report.json', 'critical': True},
        8: {'name': 'Overfitting Control', 'report': 'overfitting_control_report.json', 'critical': True},
        9: {'name': 'ONNX Validation', 'report': 'onnx_validation_report.json', 'critical': True},
        10: {'name': 'Production Readiness', 'report': 'production_readiness_report.json', 'critical': True},
        11: {'name': 'Logging & Observability', 'report': 'logging_validation_report.json', 'critical': False}
    }
    
    # Verdict criteria
    VERDICT_CRITERIA = {
        'GO': {
            'max_critical_failures': 0,
            'max_total_failures': 1,
            'max_warnings': 10
        },
        'CONDITIONAL': {
            'max_critical_failures': 0,
            'max_total_failures': 3,
            'max_warnings': 20
        }
    }
    
    def __init__(self, validation_dir: str = None):
        self.validation_dir = Path(validation_dir or Path(__file__).parent)
        self.results = FinalReviewResult()
    
    def run_final_review(self) -> FinalReviewResult:
        """Run final review and generate verdict"""
        print("\n" + "="*80)
        print("  1️⃣2️⃣ FINAL REVIEW & SIGN-OFF")
        print("="*80 + "\n")
        
        # 1. Load all checkpoint results
        print("  ── Loading Checkpoint Results ──\n")
        self._load_checkpoint_results()
        
        # 2. Generate summary table
        print("\n  ── Summary Table ──")
        self._print_summary_table()
        
        # 3. Assess production readiness
        print("\n  ── Production Readiness Assessment ──")
        self._assess_production_readiness()
        
        # 4. Identify remaining risks
        print("\n  ── Remaining Risks ──")
        self._identify_risks()
        
        # 5. Generate recommendations
        print("\n  ── Recommendations ──")
        self._generate_recommendations()
        
        # 6. Issue verdict
        print("\n  ── Final Verdict ──")
        self._issue_verdict()
        
        # 7. Generate final report
        self._generate_final_report()
        
        return self.results
    
    def _load_checkpoint_results(self):
        """Load results from all checkpoint reports"""
        for cp_num, cp_info in self.CHECKPOINTS.items():
            report_path = self.validation_dir / cp_info['report']
            
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    
                    self.results.checkpoints[cp_num] = CheckpointResult(
                        name=cp_info['name'],
                        passed=report.get('passed', False),
                        warnings=len(report.get('warnings', [])),
                        errors=len(report.get('errors', [])),
                        details=report
                    )
                    print(f"    ✅ Loaded checkpoint {cp_num}: {cp_info['name']}")
                except Exception as e:
                    self.results.checkpoints[cp_num] = CheckpointResult(
                        name=cp_info['name'],
                        passed=False,
                        errors=1,
                        details={'error': str(e)}
                    )
                    print(f"    ⚠️ Error loading checkpoint {cp_num}: {e}")
            else:
                # Create placeholder for missing checkpoint
                self.results.checkpoints[cp_num] = CheckpointResult(
                    name=cp_info['name'],
                    passed=False,
                    errors=1,
                    details={'error': 'Report not found'}
                )
                print(f"    ⏳ Missing checkpoint {cp_num}: {cp_info['name']}")
    
    def _print_summary_table(self):
        """Print checkpoint summary table"""
        print("\n" + "    " + "-"*75)
        print(f"    {'#':<4} {'Checkpoint':<25} {'Status':<10} {'Warnings':<10} {'Errors':<10} {'Critical'}")
        print("    " + "-"*75)
        
        total_warnings = 0
        total_errors = 0
        passed_count = 0
        critical_failures = 0
        
        for cp_num, cp_result in sorted(self.results.checkpoints.items()):
            status = "✅ PASS" if cp_result.passed else "❌ FAIL"
            critical = "⚡" if self.CHECKPOINTS[cp_num]['critical'] else ""
            
            print(f"    {cp_num:<4} {cp_result.name:<25} {status:<10} {cp_result.warnings:<10} {cp_result.errors:<10} {critical}")
            
            total_warnings += cp_result.warnings
            total_errors += cp_result.errors
            
            if cp_result.passed:
                passed_count += 1
            elif self.CHECKPOINTS[cp_num]['critical']:
                critical_failures += 1
        
        print("    " + "-"*75)
        print(f"    {'TOTAL':<4} {'':<25} {f'{passed_count}/{len(self.CHECKPOINTS)}':<10} {total_warnings:<10} {total_errors:<10}")
        print("    " + "-"*75)
        
        # Store totals
        self.results.checkpoints['_totals'] = CheckpointResult(
            name='TOTALS',
            passed=critical_failures == 0,
            warnings=total_warnings,
            errors=total_errors,
            details={
                'passed_count': passed_count,
                'total_checkpoints': len(self.CHECKPOINTS),
                'critical_failures': critical_failures
            }
        )
    
    def _assess_production_readiness(self):
        """Assess production readiness"""
        totals = self.results.checkpoints.get('_totals')
        
        if not totals:
            print("    ❌ Cannot assess - no checkpoint results")
            return
        
        criteria = [
            ('All critical checkpoints pass', totals.details['critical_failures'] == 0),
            ('Models validated and exported', self._check_models_ready()),
            ('Risk controls in place', self._check_risk_controls()),
            ('Kill-switch configured', self._check_kill_switch()),
            ('Logging operational', self._check_logging())
        ]
        
        print("\n    Production Readiness Criteria:")
        print("    " + "-"*50)
        
        ready_count = 0
        for criterion, passed in criteria:
            status = "✅" if passed else "❌"
            print(f"    {status} {criterion}")
            if passed:
                ready_count += 1
        
        print("    " + "-"*50)
        print(f"    Readiness: {ready_count}/{len(criteria)} criteria met")
        
        self.results.production_ready = ready_count == len(criteria)
    
    def _check_models_ready(self) -> bool:
        """Check if models are ready"""
        onnx_checkpoint = self.results.checkpoints.get(9)
        if onnx_checkpoint and onnx_checkpoint.passed:
            return True
        
        # Check for ONNX files
        onnx_dir = self.validation_dir / 'onnx_exports'
        if onnx_dir.exists():
            onnx_files = list(onnx_dir.glob('*.onnx'))
            return len(onnx_files) >= 4
        
        return False
    
    def _check_risk_controls(self) -> bool:
        """Check if risk controls are in place"""
        risk_checkpoint = self.results.checkpoints.get(6)
        return risk_checkpoint and risk_checkpoint.passed
    
    def _check_kill_switch(self) -> bool:
        """Check if kill switch is configured"""
        prod_checkpoint = self.results.checkpoints.get(10)
        if prod_checkpoint and prod_checkpoint.details:
            ks_status = prod_checkpoint.details.get('kill_switch_status', {})
            return ks_status.get('configured', False)
        return False
    
    def _check_logging(self) -> bool:
        """Check if logging is operational"""
        log_checkpoint = self.results.checkpoints.get(11)
        return log_checkpoint and log_checkpoint.passed
    
    def _identify_risks(self):
        """Identify remaining risks"""
        risks = []
        
        # Check for specific risks from each checkpoint
        for cp_num, cp_result in self.results.checkpoints.items():
            if cp_num == '_totals':
                continue
            
            if not cp_result.passed:
                risks.append(f"Checkpoint {cp_num} ({cp_result.name}) failed validation")
            
            if cp_result.warnings > 5:
                risks.append(f"Checkpoint {cp_num} has {cp_result.warnings} warnings")
        
        # Standard trading risks
        standard_risks = [
            "Market volatility may exceed historical patterns",
            "Model performance may degrade over time (concept drift)",
            "Exchange API changes could affect execution",
            "Network latency may impact scalp model timing",
            "Black swan events not captured in training data"
        ]
        
        risks.extend(standard_risks)
        
        self.results.remaining_risks = risks
        
        print("\n    Identified Risks:")
        for i, risk in enumerate(risks[:10], 1):
            print(f"    {i}. ⚠️ {risk}")
        
        if len(risks) > 10:
            print(f"    ... and {len(risks) - 10} more")
    
    def _generate_recommendations(self):
        """Generate recommendations"""
        recommendations = []
        
        # Based on checkpoint results
        for cp_num, cp_result in self.results.checkpoints.items():
            if cp_num == '_totals':
                continue
            
            if not cp_result.passed:
                if cp_num == 1:
                    recommendations.append("Fix data quality issues before production")
                elif cp_num == 2:
                    recommendations.append("Review and fix feature engineering issues")
                elif cp_num == 3:
                    recommendations.append("Verify target labeling is correct")
                elif cp_num == 5:
                    recommendations.append("Re-run Optuna optimization with fixed hyperparameters")
                elif cp_num == 6:
                    recommendations.append("Review and adjust risk model parameters")
                elif cp_num == 8:
                    recommendations.append("Address overfitting issues before deployment")
                elif cp_num == 9:
                    recommendations.append("Fix ONNX export issues")
        
        # Standard recommendations
        standard_recs = [
            "Start with paper trading before live deployment",
            "Set conservative initial position sizes",
            "Monitor model predictions for first 24-48 hours",
            "Have manual override ready for anomalous situations",
            "Review risk metrics daily during initial deployment",
            "Implement automated alerting for key metrics"
        ]
        
        recommendations.extend(standard_recs)
        
        self.results.recommendations = recommendations
        
        print("\n    Recommendations:")
        for i, rec in enumerate(recommendations[:8], 1):
            print(f"    {i}. 💡 {rec}")
    
    def _issue_verdict(self):
        """Issue final GO/NO-GO verdict"""
        totals = self.results.checkpoints.get('_totals')
        
        if not totals:
            self.results.verdict = "NO-GO"
            print("\n    ❌ VERDICT: NO-GO (Unable to assess)")
            return
        
        critical_failures = totals.details.get('critical_failures', 0)
        total_failures = totals.details['total_checkpoints'] - totals.details['passed_count']
        total_warnings = totals.warnings
        
        # Determine verdict
        go_criteria = self.VERDICT_CRITERIA['GO']
        cond_criteria = self.VERDICT_CRITERIA['CONDITIONAL']
        
        if (critical_failures <= go_criteria['max_critical_failures'] and
            total_failures <= go_criteria['max_total_failures'] and
            total_warnings <= go_criteria['max_warnings']):
            self.results.verdict = "GO"
            self.results.production_ready = True
        elif (critical_failures <= cond_criteria['max_critical_failures'] and
              total_failures <= cond_criteria['max_total_failures']):
            self.results.verdict = "CONDITIONAL GO"
        else:
            self.results.verdict = "NO-GO"
            self.results.production_ready = False
        
        # Print verdict with styling
        print("\n" + "    " + "="*60)
        
        if self.results.verdict == "GO":
            print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║         🟢  V E R D I C T :   G O   🟢                   ║
    ║                                                           ║
    ║   All critical checks passed. System is approved          ║
    ║   for production deployment.                              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
""")
        elif self.results.verdict == "CONDITIONAL GO":
            print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║    🟡  V E R D I C T :   C O N D I T I O N A L   🟡      ║
    ║                                                           ║
    ║   System may proceed with monitoring. Address             ║
    ║   remaining issues within 7 days.                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
""")
        else:
            print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║         🔴  V E R D I C T :   N O - G O   🔴             ║
    ║                                                           ║
    ║   Critical issues found. System is NOT approved           ║
    ║   for production. Address all critical failures.          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
""")
        
        # Summary stats
        print(f"    Summary:")
        print(f"    - Critical failures: {critical_failures}")
        print(f"    - Total failures: {total_failures}")
        print(f"    - Warnings: {total_warnings}")
        print(f"    - Production ready: {'Yes' if self.results.production_ready else 'No'}")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        report = {
            'verdict': self.results.verdict,
            'production_ready': self.results.production_ready,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'checkpoints_passed': sum(1 for cp in self.results.checkpoints.values() 
                                         if cp.passed and cp.name != 'TOTALS'),
                'checkpoints_total': len(self.CHECKPOINTS),
                'critical_failures': self.results.checkpoints.get('_totals', CheckpointResult('', False)).details.get('critical_failures', 0),
                'total_warnings': self.results.checkpoints.get('_totals', CheckpointResult('', False)).warnings,
                'total_errors': self.results.checkpoints.get('_totals', CheckpointResult('', False)).errors
            },
            'checkpoints': {
                str(k): {
                    'name': v.name,
                    'passed': v.passed,
                    'warnings': v.warnings,
                    'errors': v.errors
                }
                for k, v in self.results.checkpoints.items()
                if k != '_totals'
            },
            'critical_issues': self.results.critical_issues,
            'remaining_risks': self.results.remaining_risks,
            'recommendations': self.results.recommendations,
            'sign_off': {
                'reviewer': 'Automated Validation System',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'approved': self.results.verdict in ['GO', 'CONDITIONAL GO']
            }
        }
        
        report_path = self.validation_dir / 'final_review_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n    📄 Final report saved to: {report_path}")
        
        # Generate human-readable report
        self._generate_readable_report(report)
    
    def _generate_readable_report(self, report: Dict):
        """Generate human-readable markdown report"""
        md_content = f"""# Final Review Report

**Date:** {report['timestamp'][:10]}
**Verdict:** {report['verdict']}
**Production Ready:** {'Yes' if report['production_ready'] else 'No'}

---

## Summary

| Metric | Value |
|--------|-------|
| Checkpoints Passed | {report['summary']['checkpoints_passed']}/{report['summary']['checkpoints_total']} |
| Critical Failures | {report['summary']['critical_failures']} |
| Total Warnings | {report['summary']['total_warnings']} |
| Total Errors | {report['summary']['total_errors']} |

---

## Checkpoint Results

| # | Checkpoint | Status | Warnings | Errors |
|---|------------|--------|----------|--------|
"""
        
        for cp_num, cp_data in sorted(report['checkpoints'].items()):
            status = "✅ PASS" if cp_data['passed'] else "❌ FAIL"
            md_content += f"| {cp_num} | {cp_data['name']} | {status} | {cp_data['warnings']} | {cp_data['errors']} |\n"
        
        md_content += f"""
---

## Remaining Risks

"""
        for i, risk in enumerate(report['remaining_risks'][:10], 1):
            md_content += f"{i}. {risk}\n"
        
        md_content += f"""
---

## Recommendations

"""
        for i, rec in enumerate(report['recommendations'][:8], 1):
            md_content += f"{i}. {rec}\n"
        
        md_content += f"""
---

## Sign-Off

- **Reviewer:** {report['sign_off']['reviewer']}
- **Date:** {report['sign_off']['date']}
- **Approved:** {'Yes' if report['sign_off']['approved'] else 'No'}

---

*This report was automatically generated by the validation system.*
"""
        
        md_path = self.validation_dir / 'FINAL_REVIEW_REPORT.md'
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"    📄 Markdown report saved to: {md_path}")


def run_final_review() -> FinalReviewResult:
    """Run final review"""
    reviewer = FinalReviewer()
    return reviewer.run_final_review()


if __name__ == "__main__":
    result = run_final_review()
    
    # Exit code based on verdict
    if result.verdict == "GO":
        sys.exit(0)
    elif result.verdict == "CONDITIONAL GO":
        sys.exit(1)
    else:
        sys.exit(2)
