# -*- coding: utf-8 -*-
"""
Performance logging utility for iSLAT startup profiling.

This module provides timing decorators and logging functions to identify
performance bottlenecks during startup and molecule loading. It also
automatically tracks memory usage (via tracemalloc) for every timed
operation, showing memory deltas and peak usage.

Usage:
    from iSLAT.Modules.Debug.PerformanceLogger import perf_log, timed, get_performance_summary

    # Use as a decorator
    @timed("my_function")
    def my_function():
        ...

    # Use as a context manager
    with perf_log("loading molecules"):
        load_molecules()

    # Get summary (includes memory columns automatically)
    get_performance_summary()

    # Disable memory tracking if not needed
    enable_memory_tracking(False)
"""

import time
import functools
import os
import tracemalloc
from typing import Dict, List, Optional, Callable, Any, Tuple
from contextlib import contextmanager
from collections import defaultdict
import threading

try:
    import resource  # Unix/macOS only
    _has_resource = True
except ImportError:
    _has_resource = False

# Global state for performance tracking
_perf_data: Dict[str, List[float]] = defaultdict(list)
_mem_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # (mem_before_MB, mem_after_MB)
_perf_lock = threading.Lock()
_enabled = False
_track_memory = False
_tracemalloc_started = False
_start_time: Optional[float] = None
_start_memory: Optional[float] = None  # RSS at startup in MB
_log_threshold_ms = 1.0  # Only log operations taking longer than this (ms)

def enable_performance_logging(enabled: bool = True) -> None:
    """Enable or disable performance logging globally."""
    global _enabled
    _enabled = enabled

def set_log_threshold(threshold_ms: float) -> None:
    """Set minimum threshold (in ms) for logging operations."""
    global _log_threshold_ms
    _log_threshold_ms = threshold_ms

def enable_memory_tracking(enabled: bool = True) -> None:
    """Enable or disable memory tracking alongside performance logging."""
    global _track_memory
    _track_memory = enabled

def _ensure_tracemalloc() -> None:
    """Start tracemalloc if not already running."""
    global _tracemalloc_started
    if _track_memory and not _tracemalloc_started:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        _tracemalloc_started = True

def _get_process_rss_mb() -> float:
    """Get current process RSS (Resident Set Size) in MB."""
    if _has_resource:
        # resource.getrusage returns maxrss in bytes on macOS, KB on Linux
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = rusage.ru_maxrss
        if os.uname().sysname == 'Darwin':
            return maxrss / (1024 * 1024)  # bytes -> MB
        else:
            return maxrss / 1024  # KB -> MB
    return 0.0

def _get_tracemalloc_mb() -> float:
    """Get current tracemalloc-tracked memory in MB."""
    if _track_memory and tracemalloc.is_tracing():
        current, _ = tracemalloc.get_traced_memory()
        return current / (1024 * 1024)
    return 0.0

def _format_memory(mb: float) -> str:
    """Format memory size in appropriate units."""
    if mb >= 1024:
        return f"{mb / 1024:.2f}GB"
    elif mb >= 1.0:
        return f"{mb:.2f}MB"
    elif mb >= 0.001:
        return f"{mb * 1024:.1f}KB"
    else:
        return f"{mb * 1024 * 1024:.0f}B"

def reset_performance_data() -> None:
    """Clear all collected performance data."""
    global _perf_data, _mem_data, _start_time, _start_memory
    with _perf_lock:
        _perf_data.clear()
        _mem_data.clear()
        _start_time = time.perf_counter()
        _start_memory = _get_tracemalloc_mb()

def mark_startup_begin() -> None:
    """Mark the beginning of startup for total time calculation."""
    global _start_time, _start_memory
    _ensure_tracemalloc()
    _start_time = time.perf_counter()
    _start_memory = _get_tracemalloc_mb()

def _record_timing(operation: str, duration_s: float, mem_before_mb: float = 0.0, mem_after_mb: float = 0.0) -> None:
    """Record a timing and memory measurement."""
    if not _enabled:
        return
    with _perf_lock:
        _perf_data[operation].append(duration_s)
        if _track_memory:
            _mem_data[operation].append((mem_before_mb, mem_after_mb))

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

    _ensure_tracemalloc()
    mem_before = _get_tracemalloc_mb()
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        mem_after = _get_tracemalloc_mb()
        _record_timing(operation, duration, mem_before, mem_after)
        duration_ms = duration * 1000
        if verbose and duration_ms >= _log_threshold_ms:
            mem_delta = mem_after - mem_before
            mem_str = f" | Δmem: {'+' if mem_delta >= 0 else ''}{_format_memory(mem_delta)} (curr: {_format_memory(mem_after)})" if _track_memory else ""
            print(f"[PERF] {operation}: {_format_time(duration)}{mem_str}")

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

            _ensure_tracemalloc()
            mem_before = _get_tracemalloc_mb()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start
                mem_after = _get_tracemalloc_mb()
                _record_timing(op_name, duration, mem_before, mem_after)
                duration_ms = duration * 1000
                if verbose and duration_ms >= _log_threshold_ms:
                    mem_delta = mem_after - mem_before
                    mem_str = f" | Δmem: {'+' if mem_delta >= 0 else ''}{_format_memory(mem_delta)} (curr: {_format_memory(mem_after)})" if _track_memory else ""
                    print(f"[PERF] {op_name}: {_format_time(duration)}{mem_str}")
        
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
    
    show_mem = _track_memory and _mem_data
    width = 100 if show_mem else 80

    with _perf_lock:
        summary_data = []
        for operation, times in _perf_data.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            max_time = max(times) if times else 0
            min_time = min(times) if times else 0

            # Memory stats
            mem_entries = _mem_data.get(operation, [])
            if mem_entries:
                mem_deltas = [after - before for before, after in mem_entries]
                mem_total_delta = sum(mem_deltas)
                mem_peak = max(after for _, after in mem_entries)
            else:
                mem_total_delta = 0.0
                mem_peak = 0.0

            summary_data.append({
                'operation': operation,
                'total': total,
                'count': count,
                'avg': avg,
                'max': max_time,
                'min': min_time,
                'mem_delta': mem_total_delta,
                'mem_peak': mem_peak,
            })
    
    # Sort by specified criterion
    sort_key = {'total': 'total', 'avg': 'avg', 'count': 'count', 'max': 'max', 'memory': 'mem_delta'}.get(sort_by, 'total')
    summary_data.sort(key=lambda x: x[sort_key], reverse=True)
    
    # Build output
    if show_mem:
        header = f"{'Operation':<40} {'Total':>9} {'Count':>5} {'Avg':>9} {'Max':>9} {'ΔMem':>9} {'MemPeak':>9}"
    else:
        header = f"{'Operation':<45} {'Total':>10} {'Count':>6} {'Avg':>10} {'Max':>10}"

    lines = [
        "",
        "=" * width,
        "PERFORMANCE SUMMARY (sorted by {})".format(sort_by),
        "=" * width,
        header,
        "-" * width
    ]
    
    for item in summary_data[:top_n]:
        if show_mem:
            mem_delta_str = f"{'+' if item['mem_delta'] >= 0 else ''}{_format_memory(item['mem_delta'])}"
            lines.append(
                f"{item['operation'][:39]:<40} "
                f"{_format_time(item['total']):>9} "
                f"{item['count']:>5} "
                f"{_format_time(item['avg']):>9} "
                f"{_format_time(item['max']):>9} "
                f"{mem_delta_str:>9} "
                f"{_format_memory(item['mem_peak']):>9}"
            )
        else:
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
        lines.append("-" * width)
        lines.append(f"Total elapsed time since startup: {_format_time(total_elapsed)}")
    
    # Add grand total of tracked operations
    grand_total = sum(sum(times) for times in _perf_data.values())
    lines.append(f"Total tracked time: {_format_time(grand_total)}")

    # Memory summary
    if show_mem:
        current_mem = _get_tracemalloc_mb()
        rss = _get_process_rss_mb()
        lines.append(f"Current traced memory: {_format_memory(current_mem)}")
        if rss > 0:
            lines.append(f"Process peak RSS: {_format_memory(rss)}")
        if _start_memory is not None:
            lines.append(f"Memory growth since startup: {'+' if current_mem - _start_memory >= 0 else ''}{_format_memory(current_mem - _start_memory)}")

    lines.append("=" * width)
    
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
    mem_current = _get_tracemalloc_mb() if _track_memory else 0.0
    _record_timing(operation, duration_s, mem_current, mem_current)
    if verbose and _enabled:
        duration_ms = duration_s * 1000
        if duration_ms >= _log_threshold_ms:
            mem_str = f" | mem: {_format_memory(mem_current)}" if _track_memory else ""
            print(f"[PERF] {operation}: {_format_time(duration_s)}{mem_str}")

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
        self._start_mem: float = 0.0
        self._marks: List[tuple] = []  # (name, timestamp, mem_mb)
        self._end_time: Optional[float] = None
    
    def start(self) -> 'PerformanceSection':
        _ensure_tracemalloc()
        self._start_time = time.perf_counter()
        self._start_mem = _get_tracemalloc_mb()
        self._marks = [("start", self._start_time, self._start_mem)]
        return self
    
    def mark(self, label: str) -> None:
        """Mark a point in the section."""
        if self._start_time is None:
            self.start()
        self._marks.append((label, time.perf_counter(), _get_tracemalloc_mb()))
    
    def end(self) -> float:
        """End the section and return total duration."""
        self._end_time = time.perf_counter()
        end_mem = _get_tracemalloc_mb()
        self._marks.append(("end", self._end_time, end_mem))
        total = self._end_time - self._start_time if self._start_time else 0
        _record_timing(self.name, total, self._start_mem, end_mem)
        return total
    
    def get_breakdown(self, print_output: bool = False) -> str | None:
        """Get a breakdown of time and memory spent between marks."""
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
            prev_name, prev_time, prev_mem = self._marks[i - 1]
            curr_name, curr_time, curr_mem = self._marks[i]
            duration = curr_time - prev_time
            mem_delta = curr_mem - prev_mem
            if _track_memory:
                mem_str = f" | Δmem: {'+' if mem_delta >= 0 else ''}{_format_memory(mem_delta)}"
            else:
                mem_str = ""
            lines.append(f"  {prev_name} -> {curr_name}: {_format_time(duration)}{mem_str}")
        
        if self._start_time and self._end_time:
            total = self._end_time - self._start_time
            if _track_memory and len(self._marks) >= 2:
                total_mem_delta = self._marks[-1][2] - self._marks[0][2]
                lines.append(f"  TOTAL: {_format_time(total)} | Δmem: {'+' if total_mem_delta >= 0 else ''}{_format_memory(total_mem_delta)}")
            else:
                lines.append(f"  TOTAL: {_format_time(total)}")
                
        if print_output:
            print("\n".join(lines))
        else:
            return "\n".join(lines)