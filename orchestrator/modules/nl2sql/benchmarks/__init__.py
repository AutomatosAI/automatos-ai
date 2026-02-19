"""
NL2SQL Benchmarking Module
===========================

Automated benchmark runner for measuring SQL generation accuracy
against verified Golden SQL examples.
"""

from .runner import NL2SQLBenchmarkRunner
from .comparator import SQLComparator

__all__ = ["NL2SQLBenchmarkRunner", "SQLComparator"]
