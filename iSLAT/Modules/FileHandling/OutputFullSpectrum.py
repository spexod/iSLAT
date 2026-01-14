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
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines

from typing import Optional, Union, Literal, TYPE_CHECKING, Any, List, Dict, Tuple
if TYPE_CHECKING:
    from iSLAT.iSLATClass import iSLAT
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend

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
        #self.figsize = kwargs.get('figsize', (12, 16))
        
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
        if "figsize" in kwargs:
            self.figsize = kwargs["figsize"]
        
        self.plot_renderer = self.islat_ref.GUI.get_plot_renderer()

        # Initialize data
        self._load_data()
        self._prepare_molecule_info()
    
    def _load_data(self):
        """Load spectrum and line data from files."""
        try:
            self.spectrum_path = Path(self.islat_ref.loaded_spectrum_file)
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
        try:
            self.saved_lines_path = Path(self.islat_ref.input_line_list)
            self.line_data = pd.read_csv(self.saved_lines_path, sep=',')
        except Exception as e:
            #print(f"Error loading line data: {e}")
            self.line_data = None
    
    def _prepare_molecule_info(self):
        """Prepare molecule labels and colors for legend."""
        mol_dict = self.islat_ref.molecules_dict
        #self.mol_labels = []
        #self.mol_colors = []
        
        self.visible_molecules = mol_dict.get_visible_molecules(return_objects=True)

        '''for mol in visible_molecules:
            #print(f"Adding {key} to plot legend")
            self.mol_labels.append(mol.displaylabel)
            self.mol_colors.append(mol.color)'''
        
        self.mol_labels = [mol.displaylabel for mol in self.visible_molecules]
        self.mol_colors = [mol.color for mol in self.visible_molecules]

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
        if hasattr(self, 'line_data') and self.line_data is not None:
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
        else:
            pass
            #print("No line data available for annotations.")
    
    def generate_plot(self):
        """Generate the full spectrum plot with multiple panels."""
        # Create figure if it doesn't exist
        if not hasattr(self, 'fig') or self.fig is None:
            if hasattr(self, 'figsize'):
                self.fig = plt.figure(figsize=self.figsize, layout='constrained')
            else:
                self.fig = plt.figure(layout='constrained')
        
        # Get summed flux for molecules
        summed_wavelengths, summed_flux = self.islat_ref.molecules_dict.get_summed_flux(
            self.islat_ref.wave_data_original, visible_only=True
        )

        for n, xlim in enumerate(self.xlim1):
            # Create subplot for current wavelength range
            xr = [self.xlim1[n], self.xlim1[n] + self.step]
            
            # Reuse existing subplot if available, otherwise create new one
            # This allows update-in-place optimization to work
            if n not in self.subplots or self.subplots[n] not in self.fig.axes:
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
            #plt.ylabel("Flux dens. (Jy)")
            
            #self.plot_renderer.clear_model_lines(ax=self.subplots[n], lines=self.plot_renderer.model_lines, do_clear_self=False)

            # Plot line annotations
            self._plot_line_list(self.subplots[n], xr, ymin, ymax)

            # Temporarily set render_out to use thinner lines for output
            original_render_out = self.plot_renderer.render_out
            self.plot_renderer.render_out = True
            
            # Determine if this is an update (subplots already exist) or initial render
            # For updates, don't clear axes to enable update-in-place optimization
            is_update = (n in self.subplots and self.subplots[n] in self.fig.axes)
            
            # Render the spectrum and molecules without automatic legend. We'll add a custom legend at the end
            self.plot_renderer.render_main_spectrum_plot(
                wave_data=self.wave,
                flux_data=self.flux,
                molecules=self.islat_ref.molecules_dict,
                summed_wavelengths=summed_wavelengths,
                summed_flux=summed_flux,
                axes=self.subplots[n],
                update_legend=False,  # Disable automatic legend - we'll add custom one
                clear_axes=not is_update  # Don't clear on updates for faster rendering
            )
            if self.islat_ref.GUI.top_bar.line_toggle:
                self.plot_renderer.plot_saved_lines(self.line_data, self.saved_lines, fig = self.subplots[n])

            if self.islat_ref.GUI.top_bar.atomic_toggle:
                atomic_lines = load_atomic_lines()

                atomic_lines = atomic_lines[
                    (atomic_lines['wave'] >= xr[0]) &
                    (atomic_lines['wave'] <= xr[1])
                ]

                wavelengths = atomic_lines['wave'].values
                species = atomic_lines['species'].values
                line_ids = atomic_lines['line'].values

                self.plot_renderer.render_atomic_lines(atomic_lines, self.subplots[n], 
                wavelengths, species, line_ids, using_subplot=True)

            
            # Restore original setting
            self.plot_renderer.render_out = original_render_out

        if hasattr(self, 'legend_subplot') and self.legend_subplot is not None:
            #self.legend.update()
            pass
            '''handles, labels = self.legend_subplot.get_legend_handles_labels()
            if handles:
                self.legend_subplot.legend()'''
        else:
            # Add legend to first panel
            self.legend_subplot = self.subplots[0]
            '''self.legend_subplot.legend(
                self.mol_labels,
                labelcolor=self.mol_colors,
                loc='upper center',
                ncols=9,
                handletextpad=0.2,
                bbox_to_anchor=(0.5, 1.4),
                handlelength=0,
                fontsize=10,
                prop={'weight': 'bold'},
            )'''
        
        handles, labels = self.legend_subplot.get_legend_handles_labels()
        if handles:
            self.legend_subplot.legend(
                self.mol_labels,
                labelcolor=self.mol_colors,
                loc='upper center',
                ncols=12,
                handletextpad=0.2,
                bbox_to_anchor=(0.5, 1.4),
                handlelength=0,
                fontsize=10,
                prop={'weight': 'bold'},
            )

        # Add x-axis label to last panel
        self.subplots[len(self.xlim1) - 1].set_xlabel("Wavelength (μm)")
        # Add y-axis label to the side of the pannels
        self.fig.supylabel("Flux Density (Jy)", fontsize=10)
        self.fig.canvas.draw_idle()
    
    def reload_data(self):
        """Refresh the plot data with any updates from the molecules dictionary."""
        self._load_data()
        self._prepare_molecule_info()
        self.generate_plot()

    def show(self):
        """Display the plot."""
        if self.fig is None:
            self.generate_plot()
        plt.show(block=False)
    
    def save_figure(self, save_path: Optional[str] = None, rasterized: bool = False):
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
        if rasterized:
            for ax in self.subplots.values():
                ax.set_rasterized(True)
            dpi=300

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=dpi if rasterized else None)
            self.islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")
            return save_path
        return None
    
    def close(self):
        """Close the figure to free memory."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
    
    def get_legend(self) -> Optional["Legend"]:
        """Return the legend object if it exists."""
        return self.legend_subplot.legend_ if hasattr(self, 'legend_subplot') and self.legend_subplot is not None else None

# Backward compatibility function
def output_full_spectrum(islat_ref: "iSLAT", rasterized: bool = False):
    """
    Backward-compatible function that uses the new FullSpectrumPlot class.
    Maintains the same interface for existing code.
    """
    spectrum_plotter = FullSpectrumPlot(islat_ref)
    spectrum_plotter.figsize = (12, 16)
    spectrum_plotter.generate_plot()
    save_path = spectrum_plotter.save_figure(rasterized=rasterized)
    spectrum_plotter.figsize = None
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