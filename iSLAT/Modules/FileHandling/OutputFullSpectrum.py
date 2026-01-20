from os.path import dirname
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
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
        self.span_selectors: Dict[int, SpanSelector] = {}  # Span selectors for each subplot
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
                    line = ax.vlines(svd_lamb[i], ymin, ymax, linestyles='dotted', 
                            color='grey', linewidth=0.7)
                    line._islat_saved_line = True  # Mark for easy removal
                    
                    # Position label
                    label_y = ymax
                    label_x = svd_lamb[i] + self.offset_label
                    
                    text = ax.text(label_x, label_y, f"{svd_species[i]} {svd_lineID[i]}", 
                        fontsize=6, rotation=90, va='top', ha='left', color='grey')
                    text._islat_saved_line = True  # Mark for easy removal
        else:
            pass
            #print("No line data available for annotations.")
    
    def generate_plot(self, force_clear: bool = False):
        """Generate the full spectrum plot with multiple panels.
        
        Parameters:
        -----------
        force_clear : bool
            If True, forces clearing of axes even on updates
        """
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
                # Add span selector for this subplot
                self._setup_span_selector(n)
            
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

            # Temporarily set render_out to use thinner lines for output
            original_render_out = self.plot_renderer.render_out
            self.plot_renderer.render_out = True
            
            # Determine if this is an update (subplots already exist) or initial render
            # For updates, don't clear axes to enable update-in-place optimization
            # Unless force_clear is True (e.g., when toggling lines off)
            is_update = (n in self.subplots and self.subplots[n] in self.fig.axes)
            should_clear = (not is_update) or force_clear
            
            # Render the spectrum and molecules without automatic legend. We'll add a custom legend at the end
            self.plot_renderer.render_main_spectrum_plot(
                wave_data=self.wave,
                flux_data=self.flux,
                molecules=self.islat_ref.molecules_dict,
                summed_wavelengths=summed_wavelengths,
                summed_flux=summed_flux,
                axes=self.subplots[n],
                update_legend=False,  # Disable automatic legend - we'll add custom one
                clear_axes=should_clear  # Clear axes when needed
            )
            
            # Plot line annotations AFTER main plot (so they're not cleared)
            if self.islat_ref.GUI.top_bar.line_toggle:
                self._plot_line_list(self.subplots[n], xr, ymin, ymax)

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
    
    def reload_data(self, force_clear: bool = False):
        """Refresh the plot data with any updates from the molecules dictionary.
        
        Parameters:
        -----------
        force_clear : bool
            If True, forces clearing of axes even on updates (needed for toggle off)
        """
        self._load_data()
        self._prepare_molecule_info()
        self.generate_plot(force_clear=force_clear)
    
    def toggle_saved_lines(self, show: bool):
        """
        Optimized toggle for saved lines - only adds/removes line artists.
        Avoids full replot for better performance.
        
        Parameters:
        -----------
        show : bool
            If True, show saved lines. If False, hide them.
        """
        if not hasattr(self, 'subplots') or not self.subplots:
            return
        
        if show:
            # Add saved lines to each subplot
            for n, xlim in enumerate(self.xlim1):
                if n not in self.subplots:
                    continue
                ax = self.subplots[n]
                xr = [self.xlim1[n], self.xlim1[n] + self.step]
                
                # Calculate y limits
                flux_mask = (self.wave > xr[0] - 0.02) & (self.wave < xr[1])
                if np.any(flux_mask):
                    ymax = np.nanmax(self.flux[flux_mask]) * (1 + self.ymax_factor)
                    ymin = -0.01
                else:
                    ymin, ymax = -0.01, 0.15
                
                self._plot_line_list(ax, xr, ymin, ymax)
        else:
            # Remove saved lines from each subplot
            self._remove_saved_line_artists()
        
        # Redraw canvas
        if hasattr(self, 'fig') and self.fig is not None:
            self.fig.canvas.draw_idle()
    
    def _remove_saved_line_artists(self):
        """Remove all saved line artists from subplots."""
        for n, ax in self.subplots.items():
            # Remove marked line collections
            for collection in ax.collections[:]:
                if hasattr(collection, '_islat_saved_line'):
                    collection.remove()
            # Remove marked text annotations
            for text in ax.texts[:]:
                if hasattr(text, '_islat_saved_line'):
                    text.remove()
    
    def toggle_atomic_lines(self, show: bool):
        """
        Optimized toggle for atomic lines - only adds/removes line artists.
        Avoids full replot for better performance.
        
        Parameters:
        -----------
        show : bool
            If True, show atomic lines. If False, hide them.
        """
        if not hasattr(self, 'subplots') or not self.subplots:
            return
        
        if show:
            # Add atomic lines to each subplot
            atomic_lines_data = load_atomic_lines()
            
            for n, xlim in enumerate(self.xlim1):
                if n not in self.subplots:
                    continue
                ax = self.subplots[n]
                xr = [self.xlim1[n], self.xlim1[n] + self.step]
                
                # Filter atomic lines for this range
                filtered_atomic = atomic_lines_data[
                    (atomic_lines_data['wave'] >= xr[0]) &
                    (atomic_lines_data['wave'] <= xr[1])
                ]
                
                if len(filtered_atomic) > 0:
                    wavelengths = filtered_atomic['wave'].values
                    species = filtered_atomic['species'].values
                    line_ids = filtered_atomic['line'].values
                    
                    self.plot_renderer.render_atomic_lines(
                        filtered_atomic, ax, wavelengths, species, line_ids, using_subplot=True
                    )
        else:
            # Remove atomic lines from each subplot
            self._remove_atomic_line_artists()
        
        # Redraw canvas
        if hasattr(self, 'fig') and self.fig is not None:
            self.fig.canvas.draw_idle()
    
    def _remove_atomic_line_artists(self):
        """Remove all atomic line artists from subplots."""
        for n, ax in self.subplots.items():
            # Remove marked lines
            for line in ax.lines[:]:
                if hasattr(line, '_islat_atomic_line'):
                    line.remove()
            # Remove marked text annotations
            for text in ax.texts[:]:
                if hasattr(text, '_islat_atomic_line'):
                    text.remove()

    def _setup_span_selector(self, subplot_index: int):
        """
        Set up a span selector for a specific subplot.
        
        Parameters:
        -----------
        subplot_index : int
            Index of the subplot to add span selector to
        """
        if subplot_index not in self.subplots:
            return
        
        ax = self.subplots[subplot_index]
        
        # Create span selector for this subplot
        span = SpanSelector(
            ax,
            lambda xmin, xmax, idx=subplot_index: self._on_span_select(xmin, xmax, idx),
            direction='horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='lime'),
            interactive=True,
            drag_from_anywhere=True
        )
        self.span_selectors[subplot_index] = span
    
    def _on_span_select(self, xmin: float, xmax: float, subplot_index: int):
        """
        Handle span selection on a subplot.
        Switches back to regular plot mode and triggers line inspection.
        
        Parameters:
        -----------
        xmin : float
            Minimum wavelength of selection
        xmax : float
            Maximum wavelength of selection
        subplot_index : int
            Index of the subplot where selection was made
        """
        # Ignore tiny selections (likely accidental clicks)
        if abs(xmax - xmin) < 0.001:
            return
        
        # Store the selection for the main plot to use
        self._pending_selection = (xmin, xmax)
        
        # Get reference to main plot (it's GUI.plot, not GUI.main_plot)
        main_plot = self.islat_ref.GUI.plot
        
        # Switch back to regular mode if in embedded full spectrum mode
        if hasattr(main_plot, 'is_full_spectrum') and main_plot.is_full_spectrum:
            # Toggle off full spectrum mode
            main_plot.toggle_full_spectrum()
            
            # After switching back, trigger line inspection with the selected range
            # Need to use after_idle to ensure the canvas is ready
            if hasattr(self.islat_ref, 'root'):
                self.islat_ref.root.after(100, lambda: self._apply_selection_to_main_plot(xmin, xmax))
            else:
                self._apply_selection_to_main_plot(xmin, xmax)
        else:
            # We're in a separate FullSpectrumWindow, just apply selection to main plot
            self._apply_selection_to_main_plot(xmin, xmax)
    
    def _apply_selection_to_main_plot(self, xmin: float, xmax: float):
        """
        Apply the selection to the main plot after switching modes.
        
        Parameters:
        -----------
        xmin : float
            Minimum wavelength of selection
        xmax : float
            Maximum wavelength of selection
        """
        main_plot = self.islat_ref.GUI.plot
        
        # Invalidate the population diagram cache to force a full redraw
        # This is necessary because switching from full spectrum mode may leave
        # the plot renderer's cached state out of sync
        if hasattr(main_plot, 'plot_renderer'):
            main_plot.plot_renderer._pop_diagram_molecule = None
            main_plot.plot_renderer._pop_diagram_cache_key = None
            # Also clear the active scatter collection reference
            main_plot.plot_renderer._active_scatter_collection = None
            main_plot.plot_renderer._active_scatter_count = 0
        
        # Set the current selection
        main_plot.current_selection = (xmin, xmax)
        
        # Update the main plot's view to center on the selection
        # Calculate a reasonable view range around the selection
        selection_center = (xmin + xmax) / 2
        selection_width = xmax - xmin
        view_padding = max(selection_width * 2, 0.5)  # At least 0.5 um padding
        
        view_xmin = selection_center - view_padding
        view_xmax = selection_center + view_padding
        
        # Set the x-axis limits on the main spectrum plot
        main_plot.ax1.set_xlim(view_xmin, view_xmax)
        
        # Update the span selector's visual extents to show the selection
        if hasattr(main_plot, 'interaction_handler') and hasattr(main_plot.interaction_handler, 'span_selector'):
            span = main_plot.interaction_handler.span_selector
            if span is not None:
                try:
                    # Set the span visible and update its extents
                    span.set_visible(True)
                    span.extents = (xmin, xmax)
                    # Force update the span's visual
                    span.update()
                except Exception as e:
                    print(f"[DEBUG] Could not set span extents: {e}")
        
        # Trigger the line inspection plot
        main_plot.onselect(xmin, xmax)
        
        # Redraw the canvas
        main_plot.canvas.draw_idle()

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
    file_name = str(Path(save_path).with_suffix("")) + "_parameters.csv"
    print(f"file name is: {file_name}\n")
    ifh.write_molecules_to_csv(islat_ref.molecules_dict, file_path=dirname(save_path), file_name=file_name)
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