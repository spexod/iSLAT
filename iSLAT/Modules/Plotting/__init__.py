"""
iSLAT Plotting Module
=====================

Provides standalone, class-based plot objects that work **both** inside the
iSLAT GUI and as regular matplotlib figures in scripts / Jupyter notebooks.

Quick-start (notebook)::

    from iSLAT.Modules.Plotting import (
        LineInspectionPlot,
        PopulationDiagramPlot,
        FullSpectrumPlot,
        MainPlotGrid,
        FitLinesPlotGrid,
    )

Class hierarchy::

    BasePlot (ABC)
    ├── LineInspectionPlot     — zoomed wavelength region
    ├── PopulationDiagramPlot  — Boltzmann / rotation diagram
    ├── FullSpectrumPlot       — multi-panel full spectrum overview
    ├── MainPlotGrid           — 3-panel composite (spectrum + inspection + pop-diagram)
    └── FitLinesPlotGrid       — grid of individual line-fit results
"""

from .BasePlot import BasePlot, DEFAULT_THEME
from .LineInspectionPlot import LineInspectionPlot
from .PopulationDiagramPlot import PopulationDiagramPlot
from .FullSpectrumPlot import FullSpectrumPlot
from .MainPlotGrid import MainPlotGrid
from .FitLinesPlotGrid import FitLinesPlotGrid

__all__ = [
    "BasePlot",
    "DEFAULT_THEME",
    "LineInspectionPlot",
    "PopulationDiagramPlot",
    "FullSpectrumPlot",
    "MainPlotGrid",
    "FitLinesPlotGrid",
]