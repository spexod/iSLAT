import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from iSLAT.Modules.Plotting.FitLinesPlotGrid import FitLinesPlotGrid

from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from lmfit.model import ModelResult

# Size settings - each subplot will be approximately this size
SUBPLOT_WIDTH_INCHES = 1.8
SUBPLOT_HEIGHT_INCHES = 1.5

class PlotGridWindow(tk.Toplevel):
    def __init__(self, parent, 
                 plot_grid_list: List[FitLinesPlotGrid] = None,
                 **kwargs):
        super().__init__(parent)
        
        self.title("Fit Lines Plot Grid Window")
        self.geometry("1200x800")
        
        self.plot_grid_list = plot_grid_list if plot_grid_list is not None else []
        
        # Create notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Generate all tabs
        self._generate_tabs()

    def _generate_tabs(self):
        """Generate all tabs with their plots."""
        for idx, plot_grid in enumerate(self.plot_grid_list):
            self._create_tab(plot_grid)

    def _create_tab(self, plot_grid: FitLinesPlotGrid):
        """Create a single tab with scrollable plot grid."""
        # Main tab frame
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=f"{plot_grid.spectrum_name}")
        
        # Toolbar at top
        toolbar_frame = ttk.Frame(tab_frame)
        toolbar_frame.pack(fill=tk.X, side=tk.TOP, pady=2)
        
        btn_save = ttk.Button(toolbar_frame, text="Save Figure", 
                              command=lambda pg=plot_grid: self.save_figure(pg))
        btn_save.pack(side=tk.LEFT, padx=5)
        
        # Create scrollable container
        canvas_frame = ttk.Frame(tab_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar and canvas
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")
        
        scroll_canvas = tk.Canvas(canvas_frame, highlightthickness=0,
                                   yscrollcommand=scrollbar.set)
        scroll_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=scroll_canvas.yview)
        
        # Frame inside canvas to hold the matplotlib figure
        inner_frame = ttk.Frame(scroll_canvas)
        canvas_window = scroll_canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        # Configure figure size based on grid dimensions
        fig_width = SUBPLOT_WIDTH_INCHES * plot_grid.cols
        fig_height = SUBPLOT_HEIGHT_INCHES * plot_grid.rows
        
        plot_grid.fig.set_size_inches(fig_width, fig_height)
        plot_grid.fig.set_dpi(100)
        
        # Compact subplot spacing
        plot_grid.fig.subplots_adjust(
            left=0.05, right=0.98,
            top=0.96, bottom=0.04,
            wspace=0.3, hspace=0.4
        )
        
        # Reduce font sizes for compact display
        for ax in plot_grid.axs.flat:
            ax.tick_params(axis='both', labelsize=7)
            ax.title.set_fontsize(8)
            if ax.yaxis.label:
                ax.yaxis.label.set_fontsize(7)
        
        # Create matplotlib canvas widget
        fig_canvas = FigureCanvasTkAgg(plot_grid.fig, master=inner_frame)
        fig_canvas.draw()
        fig_widget = fig_canvas.get_tk_widget()
        fig_widget.pack(fill=tk.BOTH, expand=True)
        
        # Update scroll region when inner frame changes size
        def update_scroll_region(event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))
        
        inner_frame.bind("<Configure>", update_scroll_region)
        
        # Make inner frame expand horizontally with canvas
        def configure_inner_frame(event):
            scroll_canvas.itemconfig(canvas_window, width=event.width)
        
        scroll_canvas.bind("<Configure>", configure_inner_frame)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 60)), "units")
        
        def bind_mousewheel(event):
            scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        def unbind_mousewheel(event):
            scroll_canvas.unbind_all("<MouseWheel>")
        
        scroll_canvas.bind("<Enter>", bind_mousewheel)
        scroll_canvas.bind("<Leave>", unbind_mousewheel)

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