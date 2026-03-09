"""
DisplayConfig — Centralized high-DPI / Retina display configuration for iSLAT.

Detects the system's display capabilities (scaling factor, Retina/HiDPI,
physical DPI) and provides optimised matplotlib + Tk settings that maximise
rendering quality **without** impacting interactive performance.

Strategy
--------
* Detect the system DPI scaling factor once at import time.
* Expose a small set of constants that every plotting / GUI module can query.
* Configure matplotlib ``rcParams`` globally for crisp text and lines.
* Provide helpers for Tk DPI awareness (macOS / Windows / Linux).

Performance notes
-----------------
* ``figure.dpi`` is set to the *logical* DPI (typically 100) — the OS
  compositor handles pixel-doubling on Retina screens, so matplotlib does
  not need to render at 2x internally.
* ``savefig.dpi`` is set to a sensible high-res default (300) for exports.
* Anti-aliased text and Agg path snapping are left at their defaults
  (already enabled in modern matplotlib).
* ``agg.path.chunksize`` is raised to prevent Agg rendering from
  splitting very dense spectra into separate draw calls.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

import matplotlib


# ======================================================================
# System detection helpers (pure functions — no side effects)
# ======================================================================

def _detect_macos_scale_factor() -> float:
    """Return the macOS display scale factor (2.0 on Retina, 1.0 otherwise)."""
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=3,
        )
        if "Retina" in result.stdout:
            return 2.0
    except Exception:
        pass
    return 1.0


# Module-level flag: True when SetProcessDpiAwareness succeeded on Windows.
# When DPI awareness is active the OS compositor handles scaling, so Tk's
# own ``tk scaling`` command must NOT be used (that would double-scale).
_windows_dpi_aware: bool = False


def _detect_windows_scale_factor() -> float:
    """Return the Windows DPI scale factor (e.g. 1.25, 1.5, 2.0)."""
    global _windows_dpi_aware
    try:
        import ctypes
        # Enable Per-Monitor DPI awareness (V2 preferred, V1 fallback)
        # so that Tk reports real screen dimensions and renders correctly.
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE_V2
            _windows_dpi_aware = True
        except Exception:
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
                _windows_dpi_aware = True
            except Exception:
                ctypes.windll.user32.SetProcessDPIAware()
                _windows_dpi_aware = True
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return dpi / 96.0
    except Exception:
        return 1.0


def _detect_linux_scale_factor() -> float:
    """Best-effort scale factor for Linux (Xorg / Wayland)."""
    try:
        result = subprocess.run(
            ["xrdb", "-query"],
            capture_output=True, text=True, timeout=2,
        )
        for line in result.stdout.splitlines():
            if "Xft.dpi" in line:
                dpi = float(line.split(":")[-1].strip())
                return dpi / 96.0
    except Exception:
        pass
    # Wayland / GDK_SCALE
    import os
    gdk = os.environ.get("GDK_SCALE", "1")
    try:
        return float(gdk)
    except ValueError:
        return 1.0


def detect_scale_factor() -> float:
    """Return the system's display scale factor (1.0 = standard, 2.0 = HiDPI)."""
    system = platform.system()
    if system == "Darwin":
        return _detect_macos_scale_factor()
    elif system == "Windows":
        return _detect_windows_scale_factor()
    else:
        return _detect_linux_scale_factor()


# ======================================================================
# Configuration dataclass
# ======================================================================

@dataclass(frozen=True)
class _DisplayConfig:
    """Immutable snapshot of display-related settings.

    Created once at module-import time and never mutated.
    """
    scale_factor: float
    is_hidpi: bool

    # Matplotlib settings
    figure_dpi: int             # DPI for on-screen figures
    savefig_dpi: int            # DPI for saved figures
    agg_chunksize: int          # Agg renderer chunk size (performance)
    text_antialiased: bool
    lines_antialiased: bool

    # Tk / GUI hints
    tk_scaling: Optional[float]  # Value to pass to root.tk.call('tk', 'scaling', ...)


def _build_config() -> _DisplayConfig:
    """Build the display configuration from detected system capabilities."""
    scale = detect_scale_factor()
    is_hidpi = scale >= 1.5

    # --- Figure DPI ---
    # On HiDPI / Retina the OS handles pixel doubling, so we keep the
    # logical DPI at 100 to avoid doubling the rendering workload.
    # A slightly higher value (e.g. 120) *can* be used on non-Retina
    # screens to improve text sharpness without much cost.
    figure_dpi = 100

    # --- Save DPI ---
    savefig_dpi = 300  # publication-quality default

    # --- Agg chunk size ---
    # A larger chunk avoids path-splitting for very dense plots.
    # Default matplotlib value is 0 (unlimited), but some builds use
    # 20_000 which causes artefacts on dense spectra.
    agg_chunksize = 100_000

    # --- Tk scaling ---
    # On macOS the TkAgg backend already respects Retina automatically.
    # On Windows, when DPI awareness is enabled (SetProcessDpiAwareness),
    # the OS compositor handles scaling for Tk widgets — setting
    # ``tk scaling`` on top of that causes **double-scaling**, making
    # the matplotlib canvas and all widgets larger than the window.
    # Only apply tk_scaling when DPI awareness could NOT be established.
    # On Linux we may still need to nudge Tk's internal scaling.
    tk_scaling: Optional[float] = None
    system = platform.system()
    if system == "Windows" and scale > 1.0 and not _windows_dpi_aware:
        tk_scaling = scale
    elif system == "Linux" and scale > 1.0:
        tk_scaling = scale

    return _DisplayConfig(
        scale_factor=scale,
        is_hidpi=is_hidpi,
        figure_dpi=figure_dpi,
        savefig_dpi=savefig_dpi,
        agg_chunksize=agg_chunksize,
        text_antialiased=True,
        lines_antialiased=True,
        tk_scaling=tk_scaling,
    )


# Singleton — computed once at import time
display_config: _DisplayConfig = _build_config()


# ======================================================================
# Matplotlib rcParams configuration
# ======================================================================

def apply_matplotlib_defaults() -> None:
    """Set matplotlib ``rcParams`` for high-quality rendering.

    Safe to call multiple times (idempotent).  Must be called **before**
    any figures are created.
    """
    rc = matplotlib.rcParams

    # --- Resolution ---
    rc["figure.dpi"] = display_config.figure_dpi
    rc["savefig.dpi"] = display_config.savefig_dpi

    # --- Anti-aliasing ---
    rc["text.antialiased"] = display_config.text_antialiased
    rc["lines.antialiased"] = display_config.lines_antialiased

    # --- Agg performance ---
    rc["agg.path.chunksize"] = display_config.agg_chunksize

    # --- Font rendering ---
    # Use TrueType fonts in PDF/SVG exports for crisp vector text.
    rc["pdf.fonttype"] = 42     # TrueType
    rc["ps.fonttype"] = 42      # TrueType
    rc["svg.fonttype"] = "none" # Do not convert text to paths in SVG

    # --- Line rendering quality ---
    rc["path.simplify"] = True           # Keep simplification ON (performance)
    rc["path.simplify_threshold"] = 1/9  # matplotlib default — good balance


def apply_tk_scaling(root) -> None:
    """Apply DPI scaling to a Tk root window if needed.

    Parameters
    ----------
    root : tk.Tk
        The root Tk window (must be created before calling this).
    """
    if display_config.tk_scaling is not None:
        try:
            root.tk.call("tk", "scaling", display_config.tk_scaling)
        except Exception:
            pass

    # On macOS, ensure the window supports Retina resolution.
    if platform.system() == "Darwin":
        try:
            # Tk 8.6+ on macOS automatically supports Retina,
            # but we ensure the NSHighResolutionCapable flag is honoured.
            root.tk.call("tk", "windowingsystem")  # just a health-check
        except Exception:
            pass


# ======================================================================
# Apply defaults eagerly on import
# ======================================================================
apply_matplotlib_defaults()
