"""
Bootstrap confidence interval utilities for OpenCLIP evaluation.
"""

import numpy as np
import logging
from typing import Tuple, Optional

try:
    import wandb
except ImportError:
    wandb = None


def bootstrap_mean_ci(bits: np.ndarray, n_boot: int = 2000, ci: float = 0.95, 
                     rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for binary success rate (e.g., R@1).
    
    Args:
        bits: 1-D array of binary outcomes (1 = hit, 0 = miss)
        n_boot: Number of bootstrap samples
        ci: Confidence level (e.g., 0.95 for 95% CI)
        rng: Random number generator (for reproducibility)
    
    Returns:
        (mean, lower_bound, upper_bound) in percentage units
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Bootstrap resampling
    boots = rng.choice(bits, (n_boot, bits.size), replace=True).mean(axis=1)
    
    # Calculate percentiles
    alpha = 1 - ci
    lo_percentile = (alpha / 2) * 100
    hi_percentile = (1 - alpha / 2) * 100
    lo, hi = np.percentile(boots, [lo_percentile, hi_percentile])
    
    return bits.mean() * 100, lo * 100, hi * 100


def log_ci_metrics(metric_name: str, hits: np.ndarray, epoch: int, args, 
                  tb_writer=None, n_boot: int = 2000):
    """
    Compute and log confidence intervals for a metric.
    
    Args:
        metric_name: Name of the metric (e.g., "val_t2i_r1")
        hits: Binary array of hits/misses
        epoch: Current epoch
        args: Training arguments
        tb_writer: TensorBoard writer (optional)
        n_boot: Number of bootstrap samples
    """
    try:
        mean_score, ci_low, ci_high = bootstrap_mean_ci(hits, n_boot=n_boot)
        ci_width = ci_high - ci_low
        
        # Log to console
        logging.info(f"{metric_name}: {mean_score:.1f}% (95% CI [{ci_low:.1f}, {ci_high:.1f}])")
        
        # Log to TensorBoard
        if tb_writer is not None:
            tb_writer.add_scalar(f"{metric_name}/mean", mean_score, epoch)
            tb_writer.add_scalar(f"{metric_name}/ci_low", ci_low, epoch)
            tb_writer.add_scalar(f"{metric_name}/ci_high", ci_high, epoch)
            tb_writer.add_scalar(f"{metric_name}/ci_width", ci_width, epoch)
        
        # Log to Wandb
        if wandb is not None and wandb.run is not None:
            wandb.log({
                f"{metric_name}/mean": mean_score,
                f"{metric_name}/ci_low": ci_low,
                f"{metric_name}/ci_high": ci_high,
                f"{metric_name}/ci_width": ci_width,
                f"{metric_name}/n_samples": len(hits)
            }, step=epoch)
            
    except Exception as e:
        logging.warning(f"Failed to compute CI for {metric_name}: {e}")


# Alias for backwards compatibility
bootstrap_ci = bootstrap_mean_ci