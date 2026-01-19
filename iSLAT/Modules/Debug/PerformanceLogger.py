# -*- coding: utf-8 -*-
"""
Performance logging utility for iSLAT startup profiling.

This module provides timing decorators and logging functions to identify
performance bottlenecks during startup and molecule loading.

Usage:
    from iSLAT.Modules.Debug.PerformanceLogger import perf_log, timed, get_performance_summary

    # Use as a decorator
    @timed("my_function")
    def my_function():
        ...

    # Use as a context manager
    with perf_log("loading molecules"):
        load_molecules()

    # Get summary
    get_performance_summary()
"""

import time
import functools
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from collections import defaultdict
import threading

# Global state for performance tracking
_perf_data: Dict[str, List[float]] = defaultdict(list)
_perf_lock = threading.Lock()
_enabled = False
_start_time: Optional[float] = None
_log_threshold_ms = 1.0  # Only log operations taking longer than this (ms)

def enable_performance_logging(enabled: bool = True) -> None:
    """Enable or disable performance logging globally."""
    global _enabled
    _enabled = enabled

def set_log_threshold(threshold_ms: float) -> None:
    """Set minimum threshold (in ms) for logging operations."""
    global _log_threshold_ms
    _log_threshold_ms = threshold_ms

def reset_performance_data() -> None:
    """Clear all collected performance data."""
    global _perf_data, _start_time
    with _perf_lock:
        _perf_data.clear()
        _start_time = time.perf_counter()

def mark_startup_begin() -> None:
    """Mark the beginning of startup for total time calculation."""
    global _start_time
    _start_time = time.perf_counter()

def _record_timing(operation: str, duration_s: float) -> None:
    """Record a timing measurement."""
    if not _enabled:
        return
    with _perf_lock:
        _perf_data[operation].append(duration_s)

def _format_time(seconds: float) -> str:
    """Format time in appropriate units."""
    if seconds >= 1.0:
        return f"{seconds:.3f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.2f}ms"
    else:
        return f"{seconds * 1000000:.1f}µs"

@contextmanager
def perf_log(operation: str, verbose: bool = True):
    """
    Context manager for timing a code block.
    
    Parameters
    ----------
    operation : str
        Name/description of the operation being timed
    verbose : bool
        If True, prints timing immediately; if False, only records
        
    Example
    -------
    with perf_log("loading HITRAN data"):
        load_hitran_data()
    """
    if not _enabled:
        yield
        return
        
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        _record_timing(operation, duration)
        duration_ms = duration * 1000
        if verbose and duration_ms >= _log_threshold_ms:
            print(f"[PERF] {operation}: {_format_time(duration)}")

def timed(operation: str = None, verbose: bool = True):
    """
    Decorator for timing a function.
    
    Parameters
    ----------
    operation : str, optional
        Name for the operation. If None, uses function name.
    verbose : bool
        If True, prints timing on each call
        
    Example
    -------
    @timed("calculate_intensity")
    def calculate_intensity(self, ...):
        ...
    """
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not _enabled:
                return func(*args, **kwargs)
                
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                _record_timing(op_name, duration)
                duration_ms = duration * 1000
                if verbose and duration_ms >= _log_threshold_ms:
                    print(f"[PERF] {op_name}: {_format_time(duration)}")
        
        return wrapper
    return decorator

def get_performance_summary(sort_by: str = "total", top_n: int = 20, print_output: bool = True) -> str | None:
    """
    Get a formatted summary of all recorded performance data.
    
    Parameters
    ----------
    sort_by : str
        Sort criterion: "total", "avg", "count", "max"
    top_n : int
        Number of top operations to show
        
    Returns
    -------
    str
        Formatted performance summary
    """
    if not _perf_data:
        return #"No performance data collected."
    
    with _perf_lock:
        summary_data = []
        for operation, times in _perf_data.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            max_time = max(times) if times else 0
            min_time = min(times) if times else 0
            summary_data.append({
                'operation': operation,
                'total': total,
                'count': count,
                'avg': avg,
                'max': max_time,
                'min': min_time
            })
    
    # Sort by specified criterion
    sort_key = {'total': 'total', 'avg': 'avg', 'count': 'count', 'max': 'max'}.get(sort_by, 'total')
    summary_data.sort(key=lambda x: x[sort_key], reverse=True)
    
    # Build output
    lines = [
        "",
        "=" * 80,
        "PERFORMANCE SUMMARY (sorted by {})".format(sort_by),
        "=" * 80,
        f"{'Operation':<45} {'Total':>10} {'Count':>6} {'Avg':>10} {'Max':>10}",
        "-" * 80
    ]
    
    for item in summary_data[:top_n]:
        lines.append(
            f"{item['operation'][:44]:<45} "
            f"{_format_time(item['total']):>10} "
            f"{item['count']:>6} "
            f"{_format_time(item['avg']):>10} "
            f"{_format_time(item['max']):>10}"
        )
    
    # Add total startup time if available
    if _start_time is not None:
        total_elapsed = time.perf_counter() - _start_time
        lines.append("-" * 80)
        lines.append(f"Total elapsed time since startup: {_format_time(total_elapsed)}")
    
    # Add grand total of tracked operations
    grand_total = sum(sum(times) for times in _perf_data.values())
    lines.append(f"Total tracked time: {_format_time(grand_total)}")
    lines.append("=" * 80)
    
    if print_output:
        print("\n".join(lines))
    else:
        return "\n".join(lines)

def log_timing(operation: str, duration_s: float, verbose: bool = True) -> None:
    """
    Manually log a timing measurement.
    
    Parameters
    ----------
    operation : str
        Name of the operation
    duration_s : float  
        Duration in seconds
    verbose : bool
        If True, prints the timing
    """
    _record_timing(operation, duration_s)
    if verbose and _enabled:
        duration_ms = duration_s * 1000
        if duration_ms >= _log_threshold_ms:
            print(f"[PERF] {operation}: {_format_time(duration_s)}")

class PerformanceSection:
    """
    Class for tracking performance of a section with sub-operations.
    
    Example
    -------
    section = PerformanceSection("molecule_loading")
    section.start()
    
    section.mark("parsing_file")
    # ... parse file ...
    
    section.mark("converting_data")
    # ... convert data ...
    
    section.end()
    section.get_breakdown(print_output=True)
    """
    
    def __init__(self, name: str):
        self.name = name
        self._start_time: Optional[float] = None
        self._marks: List[tuple] = []  # (name, timestamp)
        self._end_time: Optional[float] = None
    
    def start(self) -> 'PerformanceSection':
        self._start_time = time.perf_counter()
        self._marks = [("start", self._start_time)]
        return self
    
    def mark(self, label: str) -> None:
        """Mark a point in the section."""
        if self._start_time is None:
            self.start()
        self._marks.append((label, time.perf_counter()))
    
    def end(self) -> float:
        """End the section and return total duration."""
        self._end_time = time.perf_counter()
        self._marks.append(("end", self._end_time))
        total = self._end_time - self._start_time if self._start_time else 0
        _record_timing(self.name, total)
        return total
    
    def get_breakdown(self, print_output: bool = False) -> str | None:
        """Get a breakdown of time spent between marks."""
        if not _enabled:
            #return f"{self.name}: Performance logging disabled"
            return

        if len(self._marks) < 2:
            if print_output:
                print(f"{self.name}: No timing data")
            else:
                return f"{self.name}: No timing data"
        
        lines = [f"\n[PERF BREAKDOWN] {self.name}:"]
        for i in range(1, len(self._marks)):
            prev_name, prev_time = self._marks[i - 1]
            curr_name, curr_time = self._marks[i]
            duration = curr_time - prev_time
            lines.append(f"  {prev_name} -> {curr_name}: {_format_time(duration)}")
        
        if self._start_time and self._end_time:
            total = self._end_time - self._start_time
            lines.append(f"  TOTAL: {_format_time(total)}")
                
        if print_output:
            print("\n".join(lines))
        else:
            return "\n".join(lines)