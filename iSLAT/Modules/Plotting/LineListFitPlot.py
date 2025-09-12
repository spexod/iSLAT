import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import iSLAT.Constants as c

from .PlotRenderer import PlotRenderer
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine

from typing import Optional, List, Dict, Any, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList

class LineListFitPlot():
    """Class to display a grid of plots that each show the fit results of a single line"""

    def __init__(self):
        self.canvas = None
        pass

    def load_fit_results(self):
        pass

    def start(self):
        """Render the grid of plots showing line fit results"""
        pass