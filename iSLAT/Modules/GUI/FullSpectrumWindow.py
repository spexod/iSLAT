import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from iSLAT.Modules.FileHandling.OutputFullSpectrum import FullSpectrumPlot
from tkinter import filedialog

from typing import Dict, List, Optional, Tuple, Callable, Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from iSLAT.iSLATClass import iSLAT

class FullSpectrumWindow(tk.Toplevel):
    """
    GUI window for displaying and interacting with full spectrum plots.
    Similar structure to PlotGridWindow.
    """
    
    def __init__(self, parent, islat_ref: "iSLAT", **kwargs):
        """
        Initialize the FullSpectrumWindow.
        
        Parameters:
        -----------
        parent : tk.Widget
            Parent widget
        islat_ref : iSLAT
            Reference to the main iSLAT instance
        **kwargs : dict
            Additional parameters passed to FullSpectrumPlot
        """
        super().__init__(parent)
        
        self.islat_ref = islat_ref
        self.kwargs = kwargs
        
        # Window configuration
        self.title("Full Spectrum Output")
        #self.geometry("1200x800")
        
        # Create the spectrum plot
        self.spectrum_plot = FullSpectrumPlot(islat_ref, **kwargs)
        
        # Create main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create toolbar frame
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Create canvas frame
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Generate plot and create canvas
        self.spectrum_plot.generate_plot()
        self.canvas = FigureCanvasTkAgg(self.spectrum_plot.fig, master=self.canvas_frame)
        #self.canvas.draw()
        
        # Pack canvas
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Create navigation toolbar
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.nav_toolbar.update()
        
        # Create control buttons
        self._create_toolbar_buttons()
        
        # Handle window closing
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _create_toolbar_buttons(self):
        """Create toolbar buttons for various actions."""
        # Save button
        self.save_btn = ttk.Button(
            self.toolbar_frame,
            text="Save as PDF",
            command=self.save_figure
        )
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Save PNG button
        self.save_png_btn = ttk.Button(
            self.toolbar_frame,
            text="Save as PNG",
            command=self.save_figure_png
        )
        self.save_png_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Refresh button
        self.refresh_btn = ttk.Button(
            self.toolbar_frame,
            text="Refresh Plot",
            command=self.refresh_plot
        )
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Separator
        ttk.Separator(self.toolbar_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Settings frame
        self.settings_frame = ttk.LabelFrame(self.toolbar_frame, text="Plot Settings")
        self.settings_frame.pack(side=tk.LEFT, padx=(5, 0))
        
        # Wavelength step control
        ttk.Label(self.settings_frame, text="Step:").grid(row=0, column=0, padx=2, sticky='w')
        self.step_var = tk.DoubleVar(value=self.spectrum_plot.step)
        self.step_spinbox = ttk.Spinbox(
            self.settings_frame,
            from_=0.5, to=5.0, increment=0.1,
            textvariable=self.step_var,
            width=8,
            command=self.update_step
        )
        self.step_spinbox.grid(row=0, column=1, padx=2)
        
        # Y-axis factor control
        ttk.Label(self.settings_frame, text="Y margin:").grid(row=0, column=2, padx=2, sticky='w')
        self.ymax_var = tk.DoubleVar(value=self.spectrum_plot.ymax_factor)
        self.ymax_spinbox = ttk.Spinbox(
            self.settings_frame,
            from_=0.0, to=1.0, increment=0.05,
            textvariable=self.ymax_var,
            width=8,
            command=self.update_ymax_factor
        )
        self.ymax_spinbox.grid(row=0, column=3, padx=2)
        
        # Apply button
        self.apply_btn = ttk.Button(
            self.settings_frame,
            text="Apply",
            command=self.apply_settings
        )
        self.apply_btn.grid(row=0, column=4, padx=(5, 2))
    
    def save_figure(self):
        """Save the figure as PDF."""
        self.spectrum_plot.save_figure()
    
    def save_figure_png(self):
        """Save the figure as PNG."""
        default_name = self.spectrum_plot.spectrum_path.stem + "_full_output.png"
        save_path = filedialog.asksaveasfilename(
            title="Save Spectrum Output as PNG",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if save_path:
            self.spectrum_plot.fig.savefig(save_path, bbox_inches='tight', dpi=300)
            self.islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")
    
    def refresh_plot(self):
        """Refresh the plot with current data."""
        try:
            # Clear the current plot
            #self.spectrum_plot.close()
            
            # Create new plot with current settings
            figsize = self.spectrum_plot.fig.get_size_inches()
            self.spectrum_plot = FullSpectrumPlot(self.islat_ref, figsize=figsize, **self.kwargs)
            self.spectrum_plot.generate_plot()
            
            # Update canvas
            self.canvas.figure = self.spectrum_plot.fig
            #self.canvas.draw_idle()
            self.canvas.draw_idle()
            
            # Update navigation toolbar
            self.nav_toolbar.update()
            
            self.islat_ref.GUI.data_field.insert_text("Full spectrum plot refreshed")
            
        except Exception as e:
            self.islat_ref.GUI.data_field.insert_text(f"Error refreshing plot: {str(e)}")
    
    def update_step(self):
        """Update the wavelength step from spinbox."""
        self.spectrum_plot.step = self.step_var.get()
        self.spectrum_plot.xlim1 = np.arange(
            self.spectrum_plot.xlim_start,
            self.spectrum_plot.xlim_end,
            self.spectrum_plot.step
        )

    def update_ymax_factor(self):
        """Update the Y-axis margin factor from spinbox.""" 
        self.spectrum_plot.ymax_factor = self.ymax_var.get()
        self.spectrum_plot.ylim1 = np.array([0, self.spectrum_plot.ymax_factor])
    
    def apply_settings(self):
        """Apply the current settings and regenerate the plot."""
        try:
            # Update plot parameters
            self.spectrum_plot.step = self.step_var.get()
            self.spectrum_plot.ymax_factor = self.ymax_var.get()
            self.spectrum_plot.xlim1 = np.arange(
                self.spectrum_plot.xlim_start,
                self.spectrum_plot.xlim_end,
                self.spectrum_plot.step
            )
            
            # Regenerate plot
            self.refresh_plot()
            
        except Exception as e:
            self.islat_ref.GUI.data_field.insert_text(f"Error applying settings: {str(e)}")
    
    def on_closing(self):
        """Handle window closing event."""
        try:
            self.spectrum_plot.close()
        except:
            pass
        self.destroy()