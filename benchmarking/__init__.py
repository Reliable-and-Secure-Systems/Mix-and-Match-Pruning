"""Benchmarking framework for comparing pruning methods."""

try:
    from .benchmark_runner import BenchmarkRunner
except ModuleNotFoundError as e:
    # BenchmarkRunner needs the genetic_algorithm package, which is not part of this
    # repository. Without it, the pruning methods below still import.
    if not (e.name or '').startswith('genetic_algorithm'):
        raise
    BenchmarkRunner = None
from .reliability.reliability_test import ReliabilityTester

# Import all pruning methods
from .unstructured import (
    MagnitudePruning, RandomPruning, SNIPPruning, 
    WANDAPruning, PDPPruning
)
from .structured import (
    HRankPruning, TorqueBasedPruning, AdaSAPPruning
)

__all__ = [
    'BenchmarkRunner', 'ReliabilityTester',
    # Unstructured methods
    'MagnitudePruning', 'RandomPruning', 'SNIPPruning', 
    'WANDAPruning', 'PDPPruning',
    # Structured methods  
    'HRankPruning', 'TorqueBasedPruning', 'AdaSAPPruning'
]