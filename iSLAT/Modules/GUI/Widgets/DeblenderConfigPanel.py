from typing import Optional, List, Dict, Any, Tuple, Union, TYPE_CHECKING
from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import iSLAT.Constants as c

import traceback
import tkinter as tk
from tkinter import ttk
#from iSLAT.Modules.DataTypes.Molecule import Molecule
#from ..GUIFunctions import create_wrapper_frame, create_scrollable_frame, ColorButton
from ..Tooltips import CreateToolTip

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

class DeblenderConfigPanel(ttk.Frame):
    def __init__(self, parent: tk.Tk, molecule_dict: MoleculeDict):
        super().__init__(parent)
        self.molecule_dict = molecule_dict