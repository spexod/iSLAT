import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path 
from tkinter import filedialog

from iSLAT.Modules.FileHandling import *
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
from iSLAT.Constants import SPEED_OF_LIGHT_KMS

from typing import Optional, Union, Literal, TYPE_CHECKING, Any, List, Dict, Tuple
if TYPE_CHECKING:
    from iSLAT.iSLATClass import iSLAT
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

class FullSpectrumPlot:
    """
    Class for generating full spectrum plots with multiple panels and line annotations.
    Similar structure to FitLinesPlotGrid.
    """
    
    def __init__(self, islat_ref: "iSLAT", **kwargs):
        """
        Initialize the FullSpectrumPlot class.
        
        Parameters:
        -----------
        islat_ref : iSLAT
            Reference to the main iSLAT instance
        **kwargs : dict
            Additional parameters for plot customization
        """
        self.islat_ref = islat_ref
        
        # Plot configuration
        self.step = kwargs.get('step', 2.3)
        self.xlim_start = kwargs.get('xlim_start', 4.9)
        self.xlim_end = kwargs.get('xlim_end', 26)
        self.xlim1 = np.arange(self.xlim_start, self.xlim_end, self.step)
        self.offset_label = kwargs.get('offset_label', 0.003)
        self.ymax_factor = kwargs.get('ymax_factor', 0.2)
        self.figsize = kwargs.get('figsize', (12, 16))
        
        # Data attributes
        self.spectrum_path: Optional[Path] = None
        self.saved_lines_path: Optional[Path] = None
        self.spectrum_data: Optional[pd.DataFrame] = None
        self.line_data: Optional[pd.DataFrame] = None
        self.wave: Optional[np.ndarray] = None
        self.flux: Optional[np.ndarray] = None
        self.saved_lines: List[np.ndarray] = []
        
        # Plot attributes
        self.fig: Optional["Figure"] = None
        self.subplots: Dict[int, "Axes"] = {}
        self.mol_labels: List[str] = []
        self.mol_colors: List[str] = []
        
        # Initialize data
        self._load_data()
        self._prepare_molecule_info()
    
    def _load_data(self):
        """Load spectrum and line data from files."""
        try:
            self.spectrum_path = Path(self.islat_ref.loaded_spectrum_file)
            self.saved_lines_path = Path(self.islat_ref.input_line_list)
            
            self.line_data = pd.read_csv(self.saved_lines_path, sep=',')
            self.spectrum_data = pd.read_csv(self.spectrum_path, sep=',')

            self.saved_lines = []
            
            # Apply radial velocity correction
            rv = self.islat_ref.molecules_dict.global_stellar_rv
            self.wave = self.spectrum_data['wave'].values
            self.wave = self.wave - (self.wave / SPEED_OF_LIGHT_KMS * rv)
            self.flux = self.spectrum_data['flux'].values
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _prepare_molecule_info(self):
        """Prepare molecule labels and colors for legend."""
        mol_dict = self.islat_ref.molecules_dict
        self.mol_labels = []
        self.mol_colors = []
        
        for key, mol in mol_dict.items():
            if mol._is_visible == "True":
                print(f"Adding {key} to plot legend")
                self.mol_labels.append(mol.displaylabel)
                self.mol_colors.append(mol.color)
    
    def _plot_line_list(self, ax: "Axes", xr: List[float], ymin: float, ymax: float):
        """
        Plot line annotations for a given wavelength range.
        
        Parameters:
        -----------
        ax : Axes
            Matplotlib axes to plot on
        xr : List[float]
            Wavelength range [xmin, xmax]
        ymin, ymax : float
            Y-axis limits for line annotations
        """
        # Get wavelength column (flexible column naming)
        svd_lamb = np.array(self.line_data['wave'] if 'wave' in self.line_data.columns else self.line_data['lam'])
        svd_species = self.line_data['species']
        
        # Handle missing line ID column
        if 'line' not in self.line_data.columns:
            self.line_data['line'] = [''] * len(self.line_data)
        svd_lineID = np.array(self.line_data['line'])
        
        # Plot lines in the current wavelength range
        for i in range(len(svd_lamb)):
            if xr[0] < svd_lamb[i] < xr[1]:
                ax.vlines(svd_lamb[i], ymin, ymax, linestyles='dotted', 
                         color='grey', linewidth=0.7)
                
                # Position label
                label_y = ymax
                label_x = svd_lamb[i] + self.offset_label
                
                ax.text(label_x, label_y, f"{svd_species[i]} {svd_lineID[i]}", 
                       fontsize=6, rotation=90, va='top', ha='left', color='grey')
    
    def generate_plot(self):
        """Generate the full spectrum plot with multiple panels."""
        self.fig = plt.figure(figsize=self.figsize)
        
        plot_renderer = self.islat_ref.GUI.get_plot_renderer()
        
        for n, xlim in enumerate(self.xlim1):
            # Create subplot for current wavelength range
            xr = [self.xlim1[n], self.xlim1[n] + self.step]
            self.subplots[n] = plt.subplot(len(self.xlim1), 1, n + 1)
            
            # Calculate y-axis limits
            flux_mask = (self.wave > xr[0] - 0.02) & (self.wave < xr[1])
            maxv = np.nanmax(self.flux[flux_mask])
            ymax = maxv + maxv * self.ymax_factor
            ymin = -0.005
            
            # Set axis properties
            plt.xlim(xr)
            plt.xticks(np.arange(xr[0], xr[1], 0.25))
            plt.ylim([ymin, ymax])
            plt.ylabel("Flux dens. (Jy)")
            
            # Plot line annotations
            self._plot_line_list(self.subplots[n], xr, ymin, ymax)
            
            # Get summed flux for molecules
            summed_wavelengths, summed_flux = self.islat_ref.molecules_dict.get_summed_flux(
                self.islat_ref.wave_data_original, visible_only=True
            )
            
            # Render the spectrum and molecules
            plot_renderer.render_main_spectrum_output(
                subplot=self.subplots[n],
                wave_data=self.wave,
                flux_data=self.flux,
                molecules=self.islat_ref.molecules_dict,
                summed_wavelengths=summed_wavelengths,
                summed_flux=summed_flux
            )
            if self.islat_ref.GUI.top_bar.line_toggle:
                plot_renderer.plot_saved_lines(self.line_data, self.saved_lines, fig = self.subplots[n])
            
            plt.draw()
            
            # Add legend to first panel
            if n == 0:
                plt.legend()
                plt.legend(
                    self.mol_labels,
                    labelcolor=self.mol_colors,
                    loc='upper center',
                    ncols=9,
                    handletextpad=0.2,
                    bbox_to_anchor=(0.5, 1.4),
                    handlelength=0,
                    fontsize=10,
                    prop={'weight': 'bold'},
                )
            
            # Add x-axis label to last panel
            if n == len(self.xlim1) - 1:
                plt.xlabel("Wavelength (μm)")
    
    def show(self):
        """Display the plot."""
        if self.fig is None:
            self.generate_plot()
        plt.show(block=False)
    
    def save_figure(self, save_path: Optional[str] = None):
        """
        Save the figure to a file.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, opens a file dialog.
        """
        if self.fig is None:
            self.generate_plot()
        
        if save_path is None:
            default_name = self.spectrum_path.stem + "_full_output.pdf"
            save_path = filedialog.asksaveasfilename(
                title="Save Spectrum Output",
                defaultextension=".pdf",
                initialfile=default_name,
                initialdir=absolute_data_files_path,
                filetypes=[("PDF files", "*.pdf")]
            )
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
            self.islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")
            return save_path
        return None
    
    def close(self):
        """Close the figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

# Backward compatibility function
def output_full_spectrum(islat_ref: "iSLAT"):
    """
    Backward-compatible function that uses the new FullSpectrumPlot class.
    Maintains the same interface for existing code.
    """
    spectrum_plotter = FullSpectrumPlot(islat_ref)
    spectrum_plotter.generate_plot()
    save_path = spectrum_plotter.save_figure()
    spectrum_plotter.close()
    return save_path

def plot_line_list(xr, ymin, ymax, svd_lns, svd_lamb=None, svd_species=None, svd_lineID=None, offslabl=0.003):
    """
    Backward-compatible function for plotting line annotations.
    """
    if svd_lamb is None:
        svd_lamb = np.array(svd_lns['wave'] if 'wave' in svd_lns.columns else svd_lns['lam'])
    if svd_species is None:
        svd_species = svd_lns['species']
    if svd_lineID is None:
        if 'line' not in svd_lns.columns:
            svd_lns['line'] = [''] * len(svd_lns)
        svd_lineID = np.array(svd_lns['line'])
    
    for i in range(len(svd_lamb)):
        if xr[0] < svd_lamb[i] < xr[1]:
            plt.vlines(svd_lamb[i], ymin, ymax, linestyles='dotted', color='grey', linewidth=0.7)
            label_y = ymax
            label_x = svd_lamb[i] + offslabl
            plt.text(label_x, label_y, svd_species[i] + ' ' + svd_lineID[i] + ' ', 
                    fontsize=6, rotation=90, va='top', ha='left', color='grey')