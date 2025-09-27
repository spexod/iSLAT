import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iSLAT.Modules.Plotting.FitLinesPlotGrid import FitLinesPlotGrid

from typing import Dict, List, Optional, Tuple, Callable, Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from lmfit.model import ModelResult

class PlotGridWindow(tk.Toplevel):
    def __init__(self, parent, 
                 plot_grid_list: List[FitLinesPlotGrid] = None,
                 **kwargs):
        super().__init__(parent)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.title("Fit Lines Plot Grid Window")
        #self.geometry("800x600")
        
        #self.plot_grid = FitLinesPlotGrid(fit_data=fit_data, rows=rows, cols=cols, figsize=figsize, **kwargs)
        #self.plot_grid.plot()
        
        self.plot_grid_list = plot_grid_list if plot_grid_list is not None else []
        
        self.generate_plots()

        '''self.toolbar_frame = ttk.Frame(self)
        self.toolbar_frame.pack(fill=tk.X)
        
        toolbar = ttk.Frame(self.toolbar_frame)
        toolbar.pack(side=tk.LEFT)'''
        
        '''btn_save = ttk.Button(toolbar, text="Save Figure", command=self.save_figure)
        btn_save.pack(side=tk.LEFT, padx=2, pady=2)'''
        
    def generate_plots(self):
        for idx, plot_grid in enumerate(self.plot_grid_list):
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=f"{plot_grid.spectrum_name}")
            
            canvas = FigureCanvasTkAgg(plot_grid.fig, master=frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            toolbar_frame = ttk.Frame(frame)
            toolbar_frame.pack(fill=tk.X)
            
            toolbar = ttk.Frame(toolbar_frame)
            toolbar.pack(side=tk.LEFT)
            
            btn_save = ttk.Button(toolbar, text="Save Figure", command=lambda pg=plot_grid: self.save_figure(pg))
            btn_save.pack(side=tk.LEFT, padx=2, pady=2)