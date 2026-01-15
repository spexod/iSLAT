"""
Constants module for iSLAT (interactive Spectral Line Analysis Tool).

This module contains all physical constants, default parameters, and molecular data
used throughout the iSLAT application. Constants are merged from the original
Constants.py and IRconstants.py files.
"""

import numpy as np
from collections import namedtuple

'''__all__ = [
    # Physical constants
    "BOLTZMANN_CONSTANT", "SPEED_OF_LIGHT_CGS", "SPEED_OF_LIGHT_KMS", 
    "SPEED_OF_LIGHT_MICRONS", "PLANCK_CONSTANT", "FGAUSS_PREFACTOR",
    "ASTRONOMICAL_UNIT_CM", "ASTRONOMICAL_UNIT_M", "PARSEC_CM", "PARSEC_CM_ALT",
    
    # Backward compatibility
    "CONSTANTS",
    
    # Molecular data
    "MOLECULES_DATA", "DEFAULT_MOLECULES_DATA",
    
    # Default parameters
    "WAVELENGTH_RANGE", "MIN_WAVELENGTH", "MAX_WAVELENGTH", "DEFAULT_DISTANCE",
    "DEFAULT_STELLAR_RV", "DEFAULT_FWHM", "PIXELS_PER_FWHM", "DEFAULT_BROADENING",
    "INTRINSIC_LINE_WIDTH", "MODEL_LINE_WIDTH", "MODEL_PIXEL_RESOLUTION",
    
    # Spectroscopic analysis
    "DEFAULT_SPAN_MOLECULE", "SPECTRAL_SEPARATION", "FWHM_TOLERANCE", 
    "CENTROID_TOLERANCE"
]'''

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

BOLTZMANN_CONSTANT = 1.3806504e-16
"""Boltzmann constant in erg/K."""

BOLTZMANN_CONSTANT_JOULE = 1.380649e-23
"""Boltzmann constant in J/K."""

SPEED_OF_LIGHT_CGS = 2.99792458e10
"""Speed of light in cm/s."""

SPEED_OF_LIGHT_KMS = 2.99792458e5
"""Speed of light in km/s."""

SPEED_OF_LIGHT_MICRONS = 2.99792458e14
"""Speed of light in μm/s."""

PLANCK_CONSTANT = 6.62606896e-27
"""Planck constant in erg·s."""

FGAUSS_PREFACTOR: float = np.sqrt(np.pi) / (2.0 * np.sqrt(np.log(2.0)))
#FGAUSS_PREFACTOR = np.sqrt(np.log(2.0)/np.pi)
"""Prefactor for opacity and intensity calculations."""

ASTRONOMICAL_UNIT_CM = 1.496e13
"""Astronomical unit in cm."""

ASTRONOMICAL_UNIT_M = 1.496e11
"""Astronomical unit in m."""

PARSEC_CM = 3.086e18
"""Parsec in cm."""

PARSEC_CM_ALT = 3.08567758128e18
"""Alternative parsec conversion from parsec to cm."""

AVAGADRO_NUMBER = 6.02214076e23
"""Avogadro's number in 1/mol."""

PI = np.pi
"""Mathematical constant pi."""

# =============================================================================
# NAMEDTUPLE FOR BACKWARD COMPATIBILITY
# =============================================================================

_Constants = namedtuple('Constants', ['k', 'c', 'h', 'fgauss', 'aucm', 'pccm'])

CONSTANTS = _Constants(
    BOLTZMANN_CONSTANT,
    SPEED_OF_LIGHT_CGS,
    PLANCK_CONSTANT,
    FGAUSS_PREFACTOR,
    ASTRONOMICAL_UNIT_CM,
    PARSEC_CM
)
"""Namedtuple containing physical constants for backward compatibility."""

# =============================================================================
# MOLECULAR DATA DEFINITIONS
# =============================================================================

MOLECULES_DATA = [
    {"name": "H2O", "file": "HITRANdata/data_Hitran_2024_H2O.par", "label": "H$_2$O"},
    {"name": "OH",  "file": "HITRANdata/data_Hitran_2024_OH.par",  "label": "OH"},
    {"name": "HCN", "file": "HITRANdata/data_Hitran_2024_HCN.par", "label": "HCN"},
    {"name": "C2H2", "file": "HITRANdata/data_Hitran_2024_C2H2.par", "label": "C$_2$H$_2$"},
    {"name": "CO2", "file": "HITRANdata/data_Hitran_2024_CO2.par", "label": "CO$_2$"},
    {"name": "CO",  "file": "HITRANdata/data_Hitran_2024_CO.par",  "label": "CO"}
]
"""List of dictionaries containing molecular data with names, file paths, and labels."""

DEFAULT_MOLECULES_DATA = [
    ("H2O", "HITRANdata/data_Hitran_2024_H2O.par", "H$_2$O"),
    ("OH", "HITRANdata/data_Hitran_2024_OH.par", "OH"),
    ("HCN", "HITRANdata/data_Hitran_2024_HCN.par", "HCN"),
    ("C2H2", "HITRANdata/data_Hitran_2024_C2H2.par", "C$_2$H$_2$"),
    ("CO2", "HITRANdata/data_Hitran_2024_CO2.par", "CO$_2$"),
    ("CO", "HITRANdata/data_Hitran_2024_CO.par", "CO")
]
"""Default molecular data as tuples for backward compatibility."""

# =============================================================================
# DEFAULT MODEL PARAMETERS
# =============================================================================

WAVELENGTH_RANGE = (4.5, 28.0)
"""Default wavelength range in microns (min, max)."""

MIN_WAVELENGTH, MAX_WAVELENGTH = WAVELENGTH_RANGE
"""Individual wavelength bounds extracted from WAVELENGTH_RANGE."""

DEFAULT_DISTANCE = 160.0
"""Default distance in parsecs."""

DEFAULT_STELLAR_RV = 0.0
"""Default stellar radial velocity in km/s."""

DEFAULT_FWHM = 130.0
"""Default FWHM of observed lines or instrument in km/s."""

PIXELS_PER_FWHM = 10
"""Number of pixels per FWHM element."""

DEFAULT_BROADENING = 1.0
"""Default broadening parameter."""

INTRINSIC_LINE_WIDTH = 1.0
"""Default intrinsic line width."""

MODEL_LINE_WIDTH = SPEED_OF_LIGHT_KMS / DEFAULT_FWHM
"""Calculated model line width based on speed of light and FWHM."""

MODEL_PIXEL_RESOLUTION = (np.mean([MIN_WAVELENGTH, MAX_WAVELENGTH]) / SPEED_OF_LIGHT_KMS * DEFAULT_FWHM) / PIXELS_PER_FWHM
"""Calculated model pixel resolution."""

FWHM_TOLERANCE = 5
"""Default tolerance in FWHM for the de-blender in km/s."""