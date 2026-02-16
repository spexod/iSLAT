"""
DataProcessing - Spectral data processing, fitting, and analysis.

Data is passed directly through method parameters, but most methods also accept an optional reference to the main iSLAT instance for accessing shared data and configuration when needed. 
This allows for flexible usage of the processing services both within the main application flow and in more standalone contexts (e.g., batch processing scripts).
"""

from .chi2spectrum import Chi2Spectrum
from .Slabfit import SlabModel, SlabFit
from .FittingEngine import FittingEngine
from .LineAnalyzer import LineAnalyzer
from .BatchFittingService import BatchFittingService
from .DeblendingService import DeblendingService
from .LineSaveService import LineSaveService
from .LineListMaker import LineListMaker

__all__ = [
    "Chi2Spectrum",
    "SlabModel",
    "SlabFit",
    "FittingEngine",
    "LineAnalyzer",
    "BatchFittingService",
    "DeblendingService",
    "LineSaveService",
    "LineListMaker",
]