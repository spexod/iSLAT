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

# Minimum size for each subplot cell (in pixels) to ensure visibility
SUBPLOT_MIN_SIZE = 200  # Reduced for better fit
SUBPLOT_ASPECT_RATIO = 1.2  # Width:Height ratio (slightly wider than tall for axis labels)

class PlotGridWindow(tk.Toplevel):
    def __init__(self, parent, 
                 plot_grid_list: List[FitLinesPlotGrid] = None,
                 **kwargs):
        super().__init__(parent)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.title("Fit Lines Plot Grid Window")
        self.geometry("1200x800")  # Set a reasonable default window size
        
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

    def _create_scrollable_frame(self, parent):
        """Create a vertically scrollable frame using canvas."""
        # Create a canvas and scrollbar
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        
        # Create the scrollable frame inside the canvas
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure the scrollable frame to update scroll region when resized
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create a window inside the canvas for the scrollable frame
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Make the scrollable frame expand to canvas width
        def configure_canvas_width(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind("<Configure>", configure_canvas_width)
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mouse wheel scrolling with faster scroll speed
        def _on_mousewheel(event):
            # Increase scroll speed by using larger multiplier
            canvas.yview_scroll(int(-1 * (event.delta / 40)), "units")
        
        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        scrollable_frame.bind("<Enter>", _bind_mousewheel)
        scrollable_frame.bind("<Leave>", _unbind_mousewheel)
        
        return scrollable_frame, canvas
        
    def generate_plots(self):
        for idx, plot_grid in enumerate(self.plot_grid_list):
            # Create main frame for this tab
            tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=f"{plot_grid.spectrum_name}")
            
            # Create toolbar at the top (outside scrollable area)
            toolbar_frame = ttk.Frame(tab_frame)
            toolbar_frame.pack(fill=tk.X, side=tk.TOP)
            
            toolbar = ttk.Frame(toolbar_frame)
            toolbar.pack(side=tk.LEFT)
            
            btn_save = ttk.Button(toolbar, text="Save Figure", command=lambda pg=plot_grid: self.save_figure(pg))
            btn_save.pack(side=tk.LEFT, padx=2, pady=2)
            
            # Create scrollable container for the plot
            scroll_container = ttk.Frame(tab_frame)
            scroll_container.pack(fill=tk.BOTH, expand=True)
            
            scrollable_frame, scroll_canvas = self._create_scrollable_frame(scroll_container)
            
            # Calculate required figure size based on subplot count
            # Use reasonable sizes that fit well on screen
            subplot_width_inches = SUBPLOT_MIN_SIZE / plot_grid.fig.dpi
            subplot_height_inches = subplot_width_inches / SUBPLOT_ASPECT_RATIO
            
            fig_width_inches = subplot_width_inches * plot_grid.cols
            fig_height_inches = subplot_height_inches * plot_grid.rows
            
            # Update the figure size
            plot_grid.fig.set_size_inches(fig_width_inches, fig_height_inches)
            
            # Configure subplots with tight spacing to reduce whitespace
            plot_grid.fig.subplots_adjust(
                left=0.05, right=0.98,
                top=0.95, bottom=0.05,
                wspace=0.3, hspace=0.4
            )
            
            # Ensure axes use auto aspect (no forced square - let data determine)
            for ax in plot_grid.axs.flat:
                ax.set_aspect('auto')
                # Reduce tick label size for compactness
                ax.tick_params(axis='both', labelsize=7)
                ax.title.set_fontsize(8)
            
            # Create matplotlib canvas with rasterization for better scroll performance
            fig_canvas = FigureCanvasTkAgg(plot_grid.fig, master=scrollable_frame)
            
            # Rasterize the figure for faster rendering during scrolling
            for ax in plot_grid.axs.flat:
                ax.set_rasterization_zorder(0)
            
            fig_canvas.draw()
            
            # Get the required size for the figure widget
            fig_width_px = int(plot_grid.fig.get_figwidth() * plot_grid.fig.dpi)
            fig_height_px = int(plot_grid.fig.get_figheight() * plot_grid.fig.dpi)
            
            # Configure the canvas widget with explicit size
            fig_widget = fig_canvas.get_tk_widget()
            fig_widget.configure(width=fig_width_px, height=fig_height_px)
            fig_widget.pack(fill=tk.NONE, expand=False)
    
    def save_figure(self, plot_grid: FitLinesPlotGrid):
        """Save the figure to a file."""
        from tkinter import filedialog
        filetypes = [
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            initialfile=f"{plot_grid.spectrum_name}_fit_grid"
        )
        if filepath:
            plot_grid.fig.savefig(filepath, dpi=150, bbox_inches='tight')