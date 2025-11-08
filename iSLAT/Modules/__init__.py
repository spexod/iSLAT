"""
Modules for iSLAT (interactive Spectral Line Analysis Tool).

This module contains core components for spectral analysis, data processing,
file handling, GUI widgets, and debugging utilities.
"""

import sys
import warnings

# Package metadata
from iSLAT import __version__
__author__ = "iSLAT Development Team"
__description__ = "interactive Spectral Line Analysis Tool (iSLAT) Components"

# Check Python version
if sys.version_info < (3, 8):
    raise RuntimeError("iSLAT COMPONENTS requires Python 3.8 or higher")

# Check for required dependencies
try:
    import numpy as np
except ImportError:
    raise ImportError("NumPy is required for iSLAT COMPONENTS module")

try:
    import pandas as pd
except ImportError:
    raise ImportError("Pandas is required for iSLAT COMPONENTS module")

try:
    import scipy
except ImportError:
    warnings.warn("SciPy not found. Some fitting features may be limited.")

try:
    import matplotlib
except ImportError:
    warnings.warn("Matplotlib not found. Plotting features will be limited.")

try:
    import astroquery
except ImportError:
    warnings.warn("Astroquery not found. HITRAN data download will be limited.")

# Core imports
from .Hitran_data import download_hitran_data, get_Hitran_data

# Define public API
__all__ = [
    'download_hitran_data',
    'get_Hitran_data'
]

# Configuration constants
DEFAULT_SAVE_PATH = "DATAFILES/SAVES"
DEFAULT_CONFIG_PATH = "DATAFILES/CONFIG"
DEFAULT_THEME_PATH = "DATAFILES/CONFIG/GUIThemes"