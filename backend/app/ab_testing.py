import numpy as np
from scipy import stats
from typing import List, Dict

class ABTester:
    """
    Compare different mixing algorithms statistically
    
    Research contribution: Rigorous evaluation methodology
    """
    
    @staticmethod
    def compare_algorithms(results_a: List[float], results_b: List[float], 
                          metric_name: str = 'score'):
        """
        Statistical comparison using t-test
        
        Args:
            results_a: List of scores from algorithm A
            results_b: List of scores from algorithm B
            metric_name: Name of metric being compared
        
        Returns:
            Dict with statistical results
        """
        # Descriptive statistics
        mean_a = np.mean(results_a)
        mean_b = np.mean(results_b)
        std_a = np.std(results_a)
        std_b = np.std(results_b)
        
        # T-test
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Determine significance
        significant = p_value < 0.05
        
        results = {
            'metric': metric_name,
            'algorithm_a': {
                'mean': mean_a,
                'std': std_a,
                'n': len(results_a)
            },
            'algorithm_b': {
                'mean': mean_b,
                'std': std_b,
                'n': len(results_b)
            },
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': significant,
            'winner': 'A' if mean_a > mean_b else 'B',
            'improvement_pct': abs((mean_a - mean_b) / mean_b * 100) if mean_b != 0 else 0
        }
        
        return results
    
    @staticmethod
    def print_comparison(results: Dict):
        """Print formatted comparison results"""
        print(f"\n{'='*60}")
        print(f"A/B Test Results: {results['metric']}")
        print(f"{'='*60}")
        print(f"Algorithm A: {results['algorithm_a']['mean']:.3f} ± {results['algorithm_a']['std']:.3f}")
        print(f"Algorithm B: {results['algorithm_b']['mean']:.3f} ± {results['algorithm_b']['std']:.3f}")
        print(f"\nStatistical Test:")
        print(f"  t-statistic: {results['t_statistic']:.3f}")
        print(f"  p-value: {results['p_value']:.4f}")
        print(f"  Cohen's d: {results['cohens_d']:.3f}")
        print(f"  Significant: {'✅ YES' if results['significant'] else '❌ NO'}")
        print(f"  Winner: Algorithm {results['winner']}")
        print(f"  Improvement: {results['improvement_pct']:.1f}%")
        print(f"{'='*60}\n")
