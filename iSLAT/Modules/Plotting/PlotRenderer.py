from typing import Optional, List, Dict, Any, Tuple, Union, TYPE_CHECKING
#from matplotlib import lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
#from matplotlib.collections import PolyCollection
import iSLAT.Constants as c

from matplotlib.axes import Axes

# Import debug configuration
try:
    from iSLAT.Modules.Debug import debug_config, DebugLevel
except ImportError:
    # Fallback if debug module is not available
    class DebugLevel:
        NONE = 0
        ERROR = 1
        WARNING = 2
        INFO = 3
        VERBOSE = 4
        TRACE = 5
    
    class FallbackDebugConfig:
        def __init__(self):
            self.level = DebugLevel.WARNING
        
        def should_log(self, component, level):
            return level <= self.level
        
        def log(self, component, level, message, **kwargs):
            if self.should_log(component, level):
                print(f"[{component.upper()}] {message}")
        
        def verbose(self, component, message, **kwargs):
            self.log(component, DebugLevel.VERBOSE, message, **kwargs)
        
        def info(self, component, message, **kwargs):
            self.log(component, DebugLevel.INFO, message, **kwargs)
        
        def warning(self, component, message, **kwargs):
            self.log(component, DebugLevel.WARNING, message, **kwargs)
        
        def error(self, component, message, **kwargs):
            self.log(component, DebugLevel.ERROR, message, **kwargs)
    
    debug_config = FallbackDebugConfig()

# Import actual data types for proper type hinting
if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    #from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
    #from iSLAT.Modules.DataTypes.Intensity import Intensity
    #from iSLAT.Modules.DataTypes.Spectrum import Spectrum

class PlotRenderer:
    """
    Handles all plot rendering and visual updates for the iSLAT spectroscopy tool.
    
    This class provides comprehensive rendering of:
    - Observed spectrum data with error bars
    - Individual molecule model spectra leveraging molecule caching
    - Summed model spectra
    - Population diagrams using molecule cached data
    - Line inspection plots
    - Saved line markers
    
    Features:
    - Efficient molecule visibility filtering
    - Direct use of molecule cached calculations for optimal performance
    - Memory-conscious plotting with line limit management
    - Batch operations for better performance with many molecules
    - Zero cache conflicts by relying entirely on molecule caching
    """
    
    def __init__(self, plot_manager: Any) -> None:
        self.plot_manager = plot_manager
        self.islat = plot_manager.islat
        self.theme: Dict[str, Any] = plot_manager.theme
        
        self.fig: plt.Figure = plot_manager.fig
        self.ax1: Axes = plot_manager.ax1
        self.ax2: Axes = plot_manager.ax2
        self.ax3: Axes = plot_manager.ax3
        self.canvas = plot_manager.canvas
        
        self.model_lines: List[Line2D] = []
        self.active_lines: List[Line2D] = []
        
        self.render_out = False
        
        # Simplified stats - only for performance monitoring, no data caching
        self._plot_stats = {
            'renders_count': 0,
            'molecules_rendered': 0
        }
    
    # Helper methods for common operations
    def _get_molecule_display_name(self, molecule: 'Molecule') -> str:
        """Get display name for a molecule"""
        return getattr(molecule, 'displaylabel', getattr(molecule, 'name', 'unknown'))
    
    def _get_molecule_identifier(self, molecule: 'Molecule') -> Optional[str]:
        """Get unique identifier for a molecule"""
        return getattr(molecule, 'name', getattr(molecule, 'displaylabel', None)) if molecule else None
    
    def _get_theme_value(self, key: str, default: Any = None) -> Any:
        """Get theme value with fallback"""
        return self.theme.get(key, default)
    
    def _get_molecule_color(self, molecule: 'Molecule') -> str:
        """Get color for molecule from theme or molecule properties"""
        mol_name = self._get_molecule_display_name(molecule)
        
        # Check molecule's own color first
        if hasattr(molecule, 'color') and molecule.color:
            return molecule.color
            
        # Then check theme colors
        molecule_colors = self._get_theme_value('molecule_colors', {})
        if mol_name in molecule_colors:
            molecule.color = molecule_colors[mol_name]
            return molecule_colors[mol_name]
            
        # Default fallback
        molecule.color = self._get_theme_value('default_molecule_color', 'blue')
        return self._get_theme_value('default_molecule_color', 'blue')
    
    def set_summed_spectrum_visibility(self, visible: bool) -> None:
        """Toggle visibility of the summed spectrum (gray fill)."""
        for collection in self.ax1.collections[:]:
            if hasattr(collection, '_islat_summed'):
                collection.set_visible(visible)
    
    def clear_all_plots(self) -> None:
        """Clear all plots and reset stats"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.model_lines.clear()
        self.active_lines.clear()
        
        self._plot_stats = {
            'renders_count': 0,
            'molecules_rendered': 0
        }
        
    def clear_model_lines(self, ax: "Axes" = None, lines: List["Line2D"] = None, do_clear_self: bool = True) -> None:
        """Clear only the model spectrum lines from the main plot"""
        if ax is None:
            ax = self.ax1
        if lines is None:
            lines = self.model_lines
        
        for line in lines:
            if line in ax.lines:
                line.remove()
        
        if do_clear_self:
            self.model_lines.clear()

    def remove_molecule_lines(self, molecule_name: str, ax: "Axes" = None, lines: List["Line2D"] = None, update_legend: bool = True) -> None:
        """
        Remove lines associated with a specific molecule
        
        Parameters
        ----------
        molecule_name : str
            Name of molecule whose lines should be removed
        ax : Axes, optional
            Axes to remove from (default: ax1)
        lines : List[Line2D], optional
            List of lines to search in (default: self.model_lines)
        update_legend : bool, optional
            Whether to update legend after removal (default: True)
        """
        print(f"removing lines from {molecule_name}")

        if ax is None:
            ax = self.ax1
        if lines is None:
            lines = self.model_lines

        lines_to_remove = []
        for line in lines:
            # Check if line belongs to this molecule (by stored metadata first)
            if hasattr(line, '_molecule_name') and line._molecule_name == molecule_name:
                print(f"found lines matching with {line._molecule_name} (metadata)")
                lines_to_remove.append(line)

        # Remove lines from plot and list
        for line in lines_to_remove:
            if line in ax.lines:
                line.remove()
            if line in lines:
                lines.remove(line)
        
        # Only update legend if requested
        if update_legend and lines_to_remove:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ncols = 2 if len(handles) > 8 else 1
                ax.legend(ncols = ncols)
    
    def set_molecule_visibility(self, molecule_name: str, visible: bool, ax: "Axes" = None, lines: List["Line2D"] = None) -> bool:
        """
        Toggle molecule line visibility without removing/recreating (fastest method).
        
        This is much faster than remove/recreate as it just changes visibility state.
        
        Parameters
        ----------
        molecule_name : str
            Name of molecule to show/hide
        visible : bool
            True to show, False to hide
        ax : Axes, optional
            Axes containing the lines (default: ax1)
        lines : List[Line2D], optional
            List of lines to search in (default: self.model_lines)
            
        Returns
        -------
        bool
            True if lines were found and updated, False otherwise
        """
        if ax is None:
            ax = self.ax1
        if lines is None:
            lines = self.model_lines
        
        found_lines = False
        for line in lines:
            if hasattr(line, '_molecule_name') and line._molecule_name == molecule_name:
                line.set_visible(visible)
                found_lines = True
        
        return found_lines
        
    def render_main_spectrum_plot(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                                 molecules: Union[List['Molecule'], 'MoleculeDict'], 
                                 summed_flux: Optional[np.ndarray] = None, 
                                 summed_wavelengths: Optional[np.ndarray] = None,
                                 error_data: Optional[np.ndarray] = None,
                                 axes: Optional["Axes"] = None,
                                 update_legend: bool = True,
                                 clear_axes: bool = True) -> None:
        """Render the main spectrum plot with observed data, model spectra, and sum
        
        Parameters
        ----------
        wave_data : np.ndarray
            Wavelength grid for model calculations (may extend beyond observed data)
        flux_data : np.ndarray
            Observed flux data
        molecules : Union[List['Molecule'], 'MoleculeDict']
            Molecules to render
        summed_flux : Optional[np.ndarray]
            Summed model flux (same size as wave_data)
        error_data : Optional[np.ndarray]
            Error data for observed spectrum
        axes : Optional["Axes"]
            Axes to render on; if None, uses self.ax1
        update_legend : bool
            Whether to create/update legend (default: True)
            Set to False when you want to add custom legend later
        clear_axes : bool
            Whether to clear axes before rendering (default: True)
            Set to False to update existing plot in place (faster for parameter updates)
        """
        # Define the axes to use
        if axes is None:
            axes = self.ax1

        # Store current view limits
        current_xlim = axes.get_xlim() if hasattr(axes, 'get_xlim') else None
        current_ylim = axes.get_ylim() if hasattr(axes, 'get_ylim') else None
        
        # Clear the plot only if requested
        # When clear_axes=False, existing molecule lines will be updated in place
        if clear_axes:
            axes.clear()
        
        # Early return if no data
        if wave_data is None or len(wave_data) == 0:
            axes.set_title("No spectrum data loaded", color=self._get_theme_value("foreground", "black"))
            return
        
        # Use observed wavelength data if provided
        obs_wave_for_plotting = wave_data
        
        # Plot observed spectrum using appropriate wavelength grid
        self._plot_observed_spectrum(obs_wave_for_plotting, flux_data, error_data, subplot=axes)

        # Plot individual molecule spectra
        if molecules:
            self.render_visible_molecules(wave_data, molecules, subplot=axes, update_legend=update_legend)
            
        # Plot summed spectrum
        if summed_flux is not None and len(summed_flux) > 0:
            self._plot_summed_spectrum(summed_wavelengths, summed_flux, subplot=axes)
        
        # Configure plot appearance
        self._configure_main_plot_appearance()
        
        # Restore view limits if they existed
        if current_xlim and current_ylim:
            if current_xlim != (0.0, 1.0) and current_ylim != (0.0, 1.0):
                axes.set_xlim(current_xlim)
                axes.set_ylim(current_ylim)
        
    def _plot_observed_spectrum(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                               error_data: Optional[np.ndarray] = None, subplot: Optional[Axes] = None) -> None:
        """Plot the observed spectrum data"""
        #print("plotting spectrum")
        
        plot = subplot if subplot else self.ax1 # Use subplot for output if given.
        #alpha = 0.8 if self.render_out else 1
        
        # Remove existing observed spectrum lines to avoid duplicates
        for line in plot.lines[:]:  # Use slice to avoid modification during iteration
            if hasattr(line, '_islat_observed'):
                line.remove()
        for collection in plot.collections[:]:
            if hasattr(collection, '_islat_observed'):
                collection.remove()
        
        if flux_data is not None and len(flux_data) > 0:
            if error_data is not None and len(error_data) == len(flux_data):
                # Plot with error bars
                errorbar_container = plot.errorbar(
                    wave_data, 
                    flux_data,
                    yerr=error_data,
                    fmt='-', 
                    color="black", # self._get_theme_value("foreground", "black")
                    linewidth=1,
                    label='Data',
                    zorder=self._get_theme_value("zorder_observed", 2),
                    elinewidth=0.5,
                    capsize=0,
                )
                # Mark all parts of errorbar for identification
                # errorbar_container is a tuple: (data_line, caplines, barlinecols)
                # Mark the data line
                errorbar_container[0]._islat_observed = True
                # Mark caplines (list of lines for caps)
                for line in errorbar_container[1]:
                    line._islat_observed = True
                # Mark the error bar line collection
                for collection in errorbar_container[2]:
                    collection._islat_observed = True
            else:
                # Plot without error bars
                line, = plot.plot(
                    wave_data, 
                    flux_data,
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=1,
                    label='Data',
                    zorder=self._get_theme_value("zorder_observed", 2)
                )
                line._islat_observed = True
    
    def _plot_summed_spectrum(self, wave_data: np.ndarray, summed_flux: np.ndarray, subplot: Optional[Axes] = None) -> None:
        """Plot the summed model spectrum"""
        if subplot is None:
            subplot = self.ax1
        
        # Remove existing summed spectrum to avoid duplicates
        for collection in subplot.collections[:]:
            if hasattr(collection, '_islat_summed'):
                collection.remove()
        
        if len(summed_flux) > 0 and np.any(summed_flux > 0):
            fill = subplot.fill_between(
                wave_data,
                0,
                summed_flux,
                color=self._get_theme_value("summed_spectra_color", "lightgray"),
                alpha=1.0,
                label='Sum',
                zorder=self._get_theme_value("zorder_summed", 1)
            )
            # Mark for identification
            fill._islat_summed = True
    
    def _configure_main_plot_appearance(self) -> None:
        """Configure the appearance of the main plot"""
        self.ax1.set_xlabel('Wavelength (μm)', color=self._get_theme_value("foreground", "black"))
        self.ax1.set_ylabel('Flux density (Jy)', color=self._get_theme_value("foreground", "black"))
        self.ax1.set_title("Full Spectrum with Line Inspection", color=self._get_theme_value("foreground", "black"))
        
        # Only show legend if there are labeled items
        handles, labels = self.ax1.get_legend_handles_labels()
        if handles:
            ncols = 2 if len(handles) > 8 else 1
            self.ax1.legend(ncols = ncols)
        
    def render_line_inspection_plot(self, line_wave: Optional[np.ndarray], 
                                   line_flux: Optional[np.ndarray], 
                                   line_label: Optional[str] = None) -> None:
        """Render the line inspection subplot"""
        self.ax2.clear()
        
        if line_wave is not None and line_flux is not None:
            # Plot data in selected range
            self.ax2.plot(line_wave, line_flux, 
                         color=self._get_theme_value("foreground", "black"), 
                         linewidth=1, 
                         label="Data")

            self.ax2.set_xlabel("Wavelength (μm)", color=self._get_theme_value("foreground", "black"))
            self.ax2.set_ylabel("Flux (Jy) denisty", color=self._get_theme_value("foreground", "black"))
            self.ax2.set_title("Line inspection plot", color=self._get_theme_value("foreground", "black"))
            
            # Show legend if there are labeled items
            handles, labels = self.ax2.get_legend_handles_labels()
            if handles:
                self.ax2.legend()
    
    def render_complete_line_inspection_plot(self, wave_data: np.ndarray, flux_data: np.ndarray,
                                           xmin: float, xmax: float, active_molecule: Optional['Molecule'] = None,
                                           fit_result: Optional[Any] = None) -> None:
        """
        Render complete line inspection plot with observed data and active molecule model.
        Uses PlotRenderer logic exclusively with molecule caching.
        
        Parameters
        ----------
        wave_data : np.ndarray
            Full wavelength array
        flux_data : np.ndarray
            Full flux array
        xmin : float
            Minimum wavelength for selected range
        xmax : float
            Maximum wavelength for selected range
        active_molecule : Molecule, optional
            Active molecule to plot model for
        fit_result : Any, optional
            Fit results to plot if available
        """
        # Always clear the line inspection plot to start fresh
        self.ax2.clear()
        
        if xmin is None or xmax is None or (xmax - xmin) < 0.0001:
            return
        
        # Plot observed data in selected range
        data_mask = (wave_data >= xmin) & (wave_data <= xmax)
        observed_wave = wave_data[data_mask]
        observed_flux = flux_data[data_mask]
        
        if len(observed_wave) > 0 and len(observed_flux) > 0:
            self.ax2.plot(observed_wave, observed_flux, 
                         color=self._get_theme_value("foreground", "black"), 
                         linewidth=1, label="Data")
        
        # Calculate max_y for plot scaling
        max_y = np.nanmax(observed_flux) if len(observed_flux) > 0 else 0.15
        
        # Plot the active molecule model using PlotRenderer's molecule spectrum method
        if active_molecule is not None:
            try:
                # Use PlotRenderer's get_molecule_spectrum_data which leverages molecule caching
                plot_lam, model_flux = self.get_molecule_spectrum_data(active_molecule, wave_data)
                
                
                if plot_lam is not None and model_flux is not None and len(model_flux) > 0:
                    # Filter the molecule data to the selected wavelength range
                    model_mask = (plot_lam >= xmin) & (plot_lam <= xmax)
                    if np.any(model_mask):
                        model_wave_range = plot_lam[model_mask]
                        model_flux_range = model_flux[model_mask]
                        
                        if len(model_wave_range) > 0 and len(model_flux_range) > 0:
                            label = self._get_molecule_display_name(active_molecule)
                            color = self._get_molecule_color(active_molecule)
                            self.ax2.plot(model_wave_range, model_flux_range, 
                                         color=color, linestyle="--", 
                                         linewidth=2, label=label)
                            if len(observed_flux) <= 0:
                                max_y = np.nanmax(model_flux_range)
            except Exception as e:
                mol_name = self._get_molecule_display_name(active_molecule)
                debug_config.warning("plot_renderer", f"Could not get model data for molecule {mol_name}: {e}")

        # Plot fit results if available
        if fit_result is not None:
            self._render_fit_results_in_line_inspection(fit_result, xmin, xmax, max_y)
        
        # Set plot properties
        self.ax2.set_xlim(xmin, xmax)
        self.ax2.set_ylim(0, max_y * 1.1)
        self.ax2.set_xlabel("Wavelength (μm)", color=self._get_theme_value("foreground", "black"))
        self.ax2.set_ylabel("Flux density (Jy)", color=self._get_theme_value("foreground", "black"))
        self.ax2.set_title("Line inspection plot", color=self._get_theme_value("foreground", "black"))
        
        # Show legend if there are labeled items
        handles, labels = self.ax2.get_legend_handles_labels()
        if handles:
            self.ax2.legend()
    
    def _render_fit_results_in_line_inspection(self, fit_result: Any, xmin: float, xmax: float, max_y: float) -> None:
        """Helper method to render fit results in the line inspection plot."""
        
        # Clear old fit results if setting is enabled (but preserve data and molecule plots)
        if self._should_clear_old_fits():
            self._clear_old_fit_results_in_range(xmin, xmax)
        
        try:
            gauss_fit, fitted_wave, fitted_flux = fit_result
            if gauss_fit is not None and fitted_wave is not None and fitted_flux is not None:
                # Filter fit data to range
                fit_mask = (fitted_wave >= xmin) & (fitted_wave <= xmax)
                if np.any(fit_mask):
                    fit_line = self.ax2.plot(fitted_wave[fit_mask], fitted_flux[fit_mask], 
                                color='red', linewidth=1, label='Total Fit', linestyle='--')[0]
                    # Mark as fit result for future removal
                    fit_line._islat_fit_result = True
                    
                    # Check if this is a multi-component fit by looking at the fit result structure
                    if hasattr(gauss_fit, 'params') and gauss_fit.params:
                        # Count components by looking for numbered prefixes (g1_, g2_, etc.)
                        component_prefixes = set()
                        for param_name in gauss_fit.params:
                            if '_' in param_name:
                                prefix = param_name.split('_')[0] + '_'
                                if prefix.startswith('g') and prefix[1:-1].isdigit():
                                    component_prefixes.add(prefix)
                        
                        # If multi-component, plot individual components
                        if len(component_prefixes) > 1:
                            try:
                                components = gauss_fit.eval_components(x=fitted_wave[fit_mask])
                                for i, prefix in enumerate(sorted(component_prefixes)):
                                    if prefix in components:
                                        component_flux = components[prefix]
                                        comp_line = self.ax2.plot(fitted_wave[fit_mask], component_flux, 
                                                    linestyle='--', linewidth=1, 
                                                    label=f"Component {i+1}")[0]
                                        # Mark as fit result for future removal
                                        comp_line._islat_fit_result = True
                            except Exception as e:
                                debug_config.warning("plot_renderer", f"Could not plot fit components: {e}")
                        else:
                            # Single component fit, fill uncertainty area
                            dely = gauss_fit.eval_uncertainty(sigma = self.islat.user_settings.get('fit_line_uncertainty', 1.0))
                            fill_collection = self.ax2.fill_between(fitted_wave, fitted_flux - dely, fitted_flux + dely,
                                                color='gray', alpha=0.3, label=r'3-$\sigma$ uncertainty band')
                            # Mark as fit result for future removal
                            fill_collection._islat_fit_result = True

        except Exception as e:
            debug_config.warning("plot_renderer", f"Could not render fit results: {e}")
        
        handles, labels = self.ax2.get_legend_handles_labels()
        if handles:
            self.ax2.legend()
        # Don't call canvas.draw_idle() here - let caller batch it
        self.canvas.draw_idle()
    
    def _should_clear_old_fits(self) -> bool:
        """Check if old fit results should be cleared when making new selections."""
        try:
            if hasattr(self.islat, 'user_settings'):
                return self.islat.user_settings.get('clear_old_fits', True)
            return True  # Default to True if setting not found
        except Exception as e:
            debug_config.warning("plot_renderer", f"Could not check clear_old_fits setting: {e}")
            return True
    
    def _clear_old_fit_results_in_range(self, xmin: float, xmax: float) -> None:
        """Clear old fit results that overlap with the new selection range."""
        # Remove existing fit lines from the line inspection plot
        lines_to_remove = []
        for line in self.ax2.lines:
            if hasattr(line, '_islat_fit_result') or (hasattr(line, 'get_label') and 
                                                     line.get_label() and 
                                                     ('Fit' in line.get_label() or 'Component' in line.get_label())):
                # Check if the fit line overlaps with the current selection
                if hasattr(line, 'get_xdata'):
                    line_xdata = line.get_xdata()
                    if len(line_xdata) > 0:
                        line_xmin = np.min(line_xdata)
                        line_xmax = np.max(line_xdata)
                        # Check for overlap
                        if (line_xmin <= xmax and line_xmax >= xmin):
                            lines_to_remove.append(line)
        
        # Remove existing fit collections (fill_between objects) from the line inspection plot
        collections_to_remove = []
        for collection in self.ax2.collections:
            if hasattr(collection, '_islat_fit_result') or (hasattr(collection, 'get_label') and 
                                                           collection.get_label() and 
                                                           ('uncertainty' in collection.get_label().lower() or 'sigma' in collection.get_label().lower())):
                # For collections, we need to check their path bounds
                try:
                    paths = collection.get_paths()
                    if paths:
                        # Get bounds from the first path
                        bounds = paths[0].get_extents()
                        coll_xmin = bounds.xmin
                        coll_xmax = bounds.xmax
                        # Check for overlap
                        if (coll_xmin <= xmax and coll_xmax >= xmin):
                            collections_to_remove.append(collection)
                except:
                    # If we can't determine bounds, remove it to be safe
                    collections_to_remove.append(collection)
        
        # Remove the overlapping fit lines
        for line in lines_to_remove:
            line.remove()
            debug_config.trace("plot_renderer", f"Removed old fit result line: {line.get_label()}")
            
        # Remove the overlapping fit collections
        for collection in collections_to_remove:
            collection.remove()
            debug_config.trace("plot_renderer", f"Removed old fit result collection: {collection.get_label()}")
        
        # Don't call canvas.draw_idle() here - let caller batch it
    
    def render_population_diagram(self, molecule: 'Molecule', wave_range: Optional[Tuple[float, float]] = None, force_redraw: bool = False) -> None:
        """
        Render population diagram using molecule's cached intensity data.
        
        This method relies on the molecule's internal caching system rather than
        maintaining its own cache to avoid conflicts with cached parameter restoration.
        
        Parameters
        ----------
        molecule : Molecule
            The molecule to render
        wave_range : tuple, optional
            Wavelength range (not currently used)
        force_redraw : bool
            If True, forces redraw even if cached. If False, skips if same molecule.
        """
        # Check if we can skip redrawing (same molecule, no parameter changes)
        # Use molecule's _compute_intensity_hash() for cache validation
        current_hash = None
        if molecule is not None and hasattr(molecule, '_compute_intensity_hash'):
            current_hash = (molecule.name, molecule._compute_intensity_hash())
        
        if not force_redraw and hasattr(self, '_pop_diagram_molecule'):
            if (self._pop_diagram_molecule is molecule and 
                current_hash is not None and
                hasattr(self, '_pop_diagram_cache_key') and
                self._pop_diagram_cache_key == current_hash):
                # Same molecule with same parameters - skip full redraw
                return
        
        self.ax3.clear()
        
        if molecule is None:
            self.ax3.set_title("No molecule selected")
            self._pop_diagram_molecule = None
            self._pop_diagram_cache_key = None
            return
        
        # Cache the molecule reference and cache key
        self._pop_diagram_molecule = molecule
        self._pop_diagram_cache_key = current_hash
            
        try:
            # Use molecule's cached intensity data directly
            int_pars = self.get_intensity_data(molecule)
            if int_pars is None:
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f"{mol_label} - No intensity data", color=self._get_theme_value("foreground", "black"))
                return

            wavelength = int_pars['lam']
            intens_mod = int_pars['intens']
            Astein_mod = int_pars['a_stein']
            gu = int_pars['g_up']
            eu = int_pars['e_up']

            radius = getattr(molecule, 'radius', None)
            distance = getattr(molecule, 'distance', None)

            area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
            dist = distance * c.PARSEC_CM
            beam_s = area / dist ** 2
            F = intens_mod * beam_s
            frequency = c.SPEED_OF_LIGHT_MICRONS / wavelength
            rd_yax = np.log(4 * np.pi * F / (Astein_mod * c.PLANCK_CONSTANT * frequency * gu))
            threshold = np.nanmax(F) / 100

            # Set limits with bounds checking
            valid_rd = rd_yax[F > threshold]
            valid_eu = eu[F > threshold]
            
            if len(valid_rd) > 0 and len(valid_eu) > 0:
                self.ax3.set_ylim(np.nanmin(valid_rd), np.nanmax(rd_yax) + 0.5)
                self.ax3.set_xlim(np.nanmin(eu) - 50, np.nanmax(valid_eu))

                # Populating the population diagram graph with the lines
                self.ax3.scatter(eu, rd_yax, s=0.5, color=self._get_theme_value("scatter_main_color", '#838B8B'))

                # Set labels
                self.ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))', color=self._get_theme_value("foreground", "black"), labelpad = -1)
                self.ax3.set_xlabel(r'$E_{u}$ (K)', color=self._get_theme_value("foreground", "black"))
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f'{mol_label} Population diagram', fontsize='medium', color=self._get_theme_value("foreground", "black"))
            else:
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f"{mol_label} - No valid data for population diagram", color=self._get_theme_value("foreground", "black"))

        except Exception as e:
            debug_config.error("plot_renderer", f"Error rendering population diagram: {e}")
            mol_label = self._get_molecule_display_name(molecule)
            self.ax3.set_title(f"{mol_label} - Error in calculation", color=self._get_theme_value("foreground", "black"))

    def plot_saved_lines(self, loaded_lines: pd.DataFrame, saved_lines, fig = None) -> None:
        """Plot saved lines on given plot (defaults main plot)"""
        if not fig: 
            fig = self.ax1

        for index, line in loaded_lines.iterrows():
            # Plot vertical lines at saved positions
            if 'lam' in line:
                saved_lines.append(fig.axvline(
                    line['lam'], 
                    color=self._get_theme_value("saved_line_color", self._get_theme_value("saved_line_color_one", "red")),
                    alpha=0.7, 
                    linestyle=':', 
                    # label=f"Saved: {line.get('label', 'Line')}"
                ))
            
            if 'xmin' in line and 'xmax' in line:
                # Plot wavelength range
                saved_lines.append(fig.axvline(
                    line['xmin'],
                    color=self._get_theme_value("saved_line_color_two", "orange"),
                    alpha=0.7,
                ))
                saved_lines.append(fig.axvline(
                    line['xmax'],
                    color=self._get_theme_value("saved_line_color_two", "orange"),
                    alpha=0.7,
                ))
        # make sure that a refresh of the plot is triggered
        self.canvas.draw_idle()

    def remove_saved_lines(self, saved_lines) -> None:
        for line in saved_lines:
            try:
                line.remove()
            except ValueError:
                pass
        
        self.canvas.draw_idle()
    
    def highlight_line_selection(self, xmin: float, xmax: float) -> None:
        """Highlight a selected wavelength range"""
        # Remove previous highlights
        for patch in self.ax1.patches:
            if hasattr(patch, '_islat_highlight'):
                patch.remove()
        
        # Add new highlight
        highlight = self.ax1.axvspan(xmin, xmax, alpha=0.3, color=self._get_theme_value("highlighted_line_color", "yellow"))
        highlight._islat_highlight = True
    
    def plot_vertical_lines(self, wavelengths: List[float], heights: Optional[List[float]] = None, 
                           colors: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> None:
        """Plot vertical lines at specified wavelengths"""
        if heights is None:
            # Get current y-limits for line height
            ylim = self.ax2.get_ylim()
            height = ylim[1] - ylim[0]
            heights = [height] * len(wavelengths)
        
        if colors is None:
            colors = ['green'] * len(wavelengths)
        
        if labels is None:
            labels = [None] * len(wavelengths)
        
        for i, (wave, height, color, label) in enumerate(zip(wavelengths, heights, colors, labels)):
            # Plot vertical line from bottom to specified height
            self.ax2.axvline(wave, color=color, alpha=0.7, linewidth=1, 
                           linestyle='-', picker=True, label=label)
            
            # Add scatter point at top of line for picking
            self.ax2.scatter([wave], [height], color=color, s=20, 
                           alpha=0.8, picker=True, zorder=5)
    
    def plot_single_lines(self, wavelengths: List[float], heights: Optional[List[float]] = None, 
                           colors: Optional[List[str]] = None, labels: Optional[List[str]] = None) -> None:
        """Plot vertical lines at specified wavelengths"""
        if heights is None:
            # Get current y-limits for line height
            ylim = self.ax1.get_ylim()
            height = ylim[1] - ylim[0]
            heights = [height] * len(wavelengths)
        
        if colors is None:
            colors = ['blue'] * len(wavelengths)
        
        if labels is None:
            labels = [None] * len(wavelengths)
        
        for i, (wave, height, color, label) in enumerate(zip(wavelengths, heights, colors, labels)):
            # Plot vertical line from bottom to specified height
            self.ax1.axvline(wave, color=color, alpha=0.7, linewidth=1, 
                           linestyle='-', picker=True, label=label)
    
    def render_visible_molecules(self, wave_data: np.ndarray, molecules: 'MoleculeDict', 
                                subplot: Optional[Axes] = None, update_legend: bool = True) -> None:
        """
        Render molecules using their built-in caching for optimal performance.
        
        Each molecule's caching system is leveraged to avoid redundant calculations
        based on parameter hashes and cache validity.
        
        Optimized to batch render all molecules and update legend only once at the end.
        Also removes lines for molecules that are no longer visible.
        
        Parameters
        ----------
        wave_data : np.ndarray
            Wavelength data for plotting
        molecules : MoleculeDict
            Dictionary of molecules to render
        subplot : Optional[Axes]
            Axes to render on (default: ax1)
        update_legend : bool
            Whether to create/update legend (default: True)
        """
        if not molecules:
            return
        
        # Get visible molecules
        visible_molecules = molecules.get_visible_molecules(return_objects=True)
        visible_molecule_names = set(getattr(mol, 'name', None) for mol in visible_molecules)
        
        plot = subplot if subplot else self.ax1
        
        # Remove lines for molecules that are no longer visible
        lines_to_remove = []
        for line in plot.lines[:]:
            if hasattr(line, '_molecule_name'):
                if line._molecule_name not in visible_molecule_names:
                    lines_to_remove.append(line)
        
        for line in lines_to_remove:
            line.remove()
            # Also remove from self.model_lines if it's there
            if line in self.model_lines:
                self.model_lines.remove(line)
        
        if not visible_molecules:
            # If no molecules are visible, just update legend and return
            if update_legend:
                handles, labels = plot.get_legend_handles_labels()
                if handles:
                    ncols = 2 if len(handles) > 8 else 1
                    plot.legend(ncols=ncols)
            return
        
        # Batch render all visible molecules without updating legend each time
        rendered_count = 0
        for mol in visible_molecules:
            #mol_name = getattr(mol, 'name', 'unknown')
            try:
                # Render without updating legend (update_legend=False)
                success = self.render_individual_molecule_spectrum(mol, wave_data, subplot=subplot, update_legend=False)
                if success:
                    rendered_count += 1
                else:
                    debug_config.warning("plot_renderer", f"Could not render molecule {getattr(mol, 'name', 'unknown')}")
            except Exception as e:
                print(f"Error rendering molecule {getattr(mol, 'name', 'unknown')}: {e}")
                continue
        
        # Update legend only once after all molecules are rendered (if requested)
        if update_legend:
            handles, labels = plot.get_legend_handles_labels()
            if handles:
                ncols = 2 if len(handles) > 8 else 1
                plot.legend(ncols=ncols)
    
    def handle_molecule_visibility_change(self, molecule_name: str, is_visible: bool, 
                                        molecules_dict: 'MoleculeDict', 
                                        wave_data: np.ndarray,
                                        active_molecule: Optional['Molecule'] = None,
                                        current_selection: Optional[Tuple[float, float]] = None,
                                        is_full_spectrum: bool = False) -> None:
        """
        Handle molecule visibility changes with comprehensive PlotRenderer logic.
        Leverages MoleculeDict's advanced caching and visibility management.
        
        Parameters
        ----------
        molecule_name : str
            Name of the molecule whose visibility changed
        is_visible : bool
            New visibility state
        molecules_dict : Union[MoleculeDict, Dict]
            Dictionary containing all molecules
        wave_data : np.ndarray
            Wavelength data for plotting
        active_molecule : Molecule, optional
            Currently active molecule for line inspection
        current_selection : Tuple[float, float], optional
            Current wavelength selection range (xmin, xmax)
        """
        if molecule_name not in molecules_dict:
            debug_config.warning("plot_renderer", f"Molecule {molecule_name} not found in molecules_dict")
            return
        
        #molecule = molecules_dict[molecule_name]
        #print(f"changing plotting of: {molecule}")
        
        if is_full_spectrum:
            # For full spectrum output, simply re-render everything
            self.plot_manager.load_full_spectrum()
        # Handle visibility change using PlotRenderer methods
        if is_visible:
            pass
            # Add molecule spectrum using PlotRenderer
            #success = self.render_individual_molecule_spectrum(molecule, wave_data)
            #if not success:
            #    debug_config.warning("plot_renderer", f"Failed to render molecule {molecule_name}")
        else:
            # Remove molecule spectrum using PlotRenderer
            self.remove_molecule_lines(molecule_name)
        
        '''# Optional: Clear MoleculeDict's flux caches if needed after visibility change
        if hasattr(molecules_dict, '_clear_flux_caches'):
            try:
                # This ensures that the next summed flux calculation uses fresh visibility state
                molecules_dict._clear_flux_caches()
                debug_config.trace("plot_renderer", f"Cleared MoleculeDict flux caches after visibility change for {molecule_name}")
            except Exception as e:
                debug_config.trace("plot_renderer", f"Could not clear MoleculeDict flux caches: {e}")'''
        
        # Update summed spectrum using MoleculeDict's optimized caching system
        self._update_summed_spectrum_with_molecules(molecules_dict, wave_data)

        # Handle active molecule line inspection update if needed
        if (active_molecule and 
            hasattr(active_molecule, 'name') and 
            active_molecule.name == molecule_name and 
            current_selection):
            
            # Delegate line inspection update to plot manager
            # This is the only callback needed to MainPlot
            if hasattr(self.plot_manager, 'plot_spectrum_around_line'):
                xmin, xmax = current_selection
                self.plot_manager.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
    
    def update_summed_spectrum_only(self, wave_data: np.ndarray, summed_flux: np.ndarray) -> None:
        """Update only the summed spectrum without affecting individual molecule plots"""
        # Remove existing summed spectrum
        for collection in self.ax1.collections:
            if hasattr(collection, '_islat_summed'):
                collection.remove()
        
        # Add new summed spectrum
        if len(summed_flux) > 0 and np.any(summed_flux > 0):
            fill = self.ax1.fill_between(
                wave_data,
                0,
                summed_flux,
                color=self._get_theme_value("summed_spectra_color", "lightgray"),
                alpha=1.0,
                label='Sum',
                zorder=self._get_theme_value("zorder_summed", 1)
            )
            # Mark as summed spectrum for future removal
            fill._islat_summed = True

    def _update_summed_spectrum_with_molecules(self, molecules_dict: Union['MoleculeDict', Dict], 
                                             wave_data: np.ndarray) -> None:
        """
        Update summed spectrum using MoleculeDict's advanced caching system.
        This method leverages the built-in summed flux calculations with optimal caching.
        """
        if not molecules_dict or len(molecules_dict) == 0:
            # Clear summed spectrum if no molecules
            self.update_summed_spectrum_only(wave_data, np.zeros_like(wave_data))
            return
        try:
            debug_config.trace("plot_renderer", "Using MoleculeDict.get_summed_flux() with caching")
            summed_wavelengths, summed_flux = molecules_dict.get_summed_flux(wave_data, visible_only=True)
            wave_data = summed_wavelengths  # Use the combined wavelength grid
        except Exception as e:
            debug_config.warning("plot_renderer", f"Error in summed flux calculation: {e}")
        
        # Update summed spectrum using PlotRenderer
        self.update_summed_spectrum_only(wave_data, summed_flux)
    
    def render_atomic_lines(self, atomic_lines, axis: Axes, wavelengths, species, line_ids, using_subplot = False):
        if using_subplot:
            pass

        for i in range(len(wavelengths)):
            line = axis.axvline(wavelengths[i], linestyle='--', color='tomato', alpha=0.7)
            line._islat_atomic_line = True  # Mark for easy removal
            
            # Adjust the y-coordinate to place labels within the plot borders
            ylim = axis.get_ylim()
            label_y = ylim[1]
            
            # Adjust the x-coordinate to place labels just to the right of the line
            xlim = axis.get_xlim()
            label_x = wavelengths[i] + 0.006 * (xlim[1] - xlim[0])

            # Add text label for the line
            label_text = f"{species[i]} {line_ids[i]}"
            label = axis.text(label_x, label_y, label_text, fontsize=8, rotation=90, 
                                                va='top', ha='left', color='tomato')
            label._islat_atomic_line = True  # Mark for easy removal
            if using_subplot is False:
                atomic_lines.append((line, label))

    def remove_atomic_lines(self, lines):
        for (line, text) in lines:
            try:
                line.remove()
                text.remove()
            except ValueError:
                pass
        lines.clear()

    def clear_active_lines(self, active_lines_list: List[Any]) -> None:
        """
        Properly clear active lines by removing matplotlib artists first.
        
        Parameters
        ----------
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
        """
        for line_data in active_lines_list:
            if len(line_data) >= 2:
                line_artist = line_data[0] # Line artist (vlines)
                text_artist = line_data[1] 
                scatter_artist = line_data[2] if len(line_data) > 2 else None # Scatter artist
                
                # Remove line artist if it exists
                if line_artist is not None and getattr(line_artist, 'axes', None) is not None:
                    try:
                        line_artist.remove()
                    except (ValueError, AttributeError):
                        pass
                
                # Remove text artist if it exists
                if text_artist is not None and getattr(text_artist, 'axes', None) is not None:
                    try:
                        text_artist.remove()
                    except (ValueError, AttributeError):
                        pass

                # Remove scatter artist if it exists
                if scatter_artist is not None and getattr(scatter_artist, 'axes', None) is not None:
                    try:
                        scatter_artist.remove()
                    except (ValueError, AttributeError):
                        pass
        
        # Clear the list after removing all artists
        active_lines_list.clear()
    
    def render_active_lines_in_population_diagram(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], active_lines_list: List[Any]) -> None:
        """
        Render active lines as scatter points in the population diagram.
        
        Uses vectorized operations for better performance.
        Lines are filtered based on intensity threshold from user settings.
        Only lines with intensity above threshold_percent of the strongest line are rendered.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
        active_lines_list : List[Any]
            List to store active line data for interaction
        """
        if not line_data:
            return
        
        # Get threshold and filter lines
        threshold_percent = self.get_line_intensity_threshold()
        filtered_line_data = self.filter_lines_by_threshold(line_data, threshold_percent)
        
        if not filtered_line_data:
            debug_config.trace("plot_renderer", f"No lines above threshold ({threshold_percent*100:.1f}%)")
            return
            
        debug_config.trace("plot_renderer", f"Rendering {len(filtered_line_data)}/{len(line_data)} lines in pop diagram")
        
        # Get max intensity from original data for consistent percentage calculation
        original_intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(original_intensities) if original_intensities else 1.0
        
        # Get molecule properties once (not in loop)
        molecule = getattr(self.islat, 'active_molecule', None)
        if molecule is None:
            return
        radius = getattr(molecule, 'radius', 1.0)
        distance = getattr(molecule, 'distance', getattr(self.islat, 'global_dist', 140.0))
        
        # Pre-calculate constants
        area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
        dist = distance * c.PARSEC_CM
        beam_s = area / dist ** 2
        
        # Cache theme value once
        active_color = self._get_theme_value("active_scatter_line_color", 'green')
        
        # Collect data for vectorized scatter (faster than individual scatter calls)
        e_ups = []
        rd_yaxs = []
        value_data_list = []
        
        for line, intensity, tau_val in filtered_line_data:
            if all(x is not None for x in [intensity, line.a_stein, line.g_up, line.lam]):
                # Calculate rd_yax
                F = intensity * beam_s
                freq = c.SPEED_OF_LIGHT_MICRONS / line.lam
                rd_yax = np.log(4 * np.pi * F / (line.a_stein * c.PLANCK_CONSTANT * freq * line.g_up))
                
                e_ups.append(line.e_up)
                rd_yaxs.append(rd_yax)
                
                # Store line information
                value_data_list.append({
                    'lam': line.lam,
                    'e': line.e_up,
                    'a': line.a_stein,
                    'g': line.g_up,
                    'e_low': line.e_low if line.e_low else 'N/A',
                    'g_low': line.g_low if line.g_low else 'N/A',
                    'rd_yax': rd_yax,
                    'inten': intensity,
                    'up_lev': line.lev_up if line.lev_up else 'N/A',
                    'low_lev': line.lev_low if line.lev_low else 'N/A',
                    'tau': tau_val if tau_val is not None else 'N/A',
                    'intensity_percent': (intensity / max_intensity) * 100
                })
        
        if not e_ups:
            return
            
        # Create single vectorized scatter call (much faster than N individual calls)
        sc = self.ax3.scatter(e_ups, rd_yaxs, s=30, 
                             color=active_color, 
                             edgecolors='black', picker=True)
        
        # Store the scatter collection for later recoloring
        self._active_scatter_collection = sc
        self._active_scatter_count = len(e_ups)
        
        # Store reference to scatter and value data in active_lines_list
        # Each entry gets the same scatter ref but different point_index for recoloring
        for idx, value_data in enumerate(value_data_list):
            value_data['_scatter_point_index'] = idx  # Store index within scatter collection
            if idx < len(active_lines_list):
                # Update existing entry with scatter artist reference
                active_lines_list[idx][2] = sc  # All points share same scatter collection
                active_lines_list[idx][3].update(value_data)
            else:
                # Create new entry: [line_artist, text_obj, scatter_artist, value_data]
                active_lines_list.append([None, None, sc, value_data])
    
    def render_active_lines_in_line_inspection(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], active_lines_list: List[Any], 
                                              max_y: float) -> None:
        """
        Render active lines as vertical lines in the line inspection plot.
        
        Lines are filtered based on intensity threshold from user settings.
        Only lines with intensity above threshold_percent of the strongest line are rendered.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
        active_lines_list : List[Any]
            List to store active line data for interaction
        max_y : float
            Maximum y value for scaling line heights
        """
        if not line_data:
            return
        
        # Get threshold and filter lines
        threshold_percent = self.get_line_intensity_threshold()
        filtered_line_data = self.filter_lines_by_threshold(line_data, threshold_percent)
        
        if not filtered_line_data:
            debug_config.trace("plot_renderer", f"No lines above threshold ({threshold_percent*100:.1f}%)")
            return
            
        debug_config.trace("plot_renderer", f"Rendering {len(filtered_line_data)}/{len(line_data)} lines in line inspection")
        
        # Get max intensity from original data for consistent scaling
        original_intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(original_intensities) if original_intensities else 1.0
        
        # Cache theme value once (not in loop)
        active_color = self._get_theme_value("active_scatter_line_color", "green")
        
        # Plot vertical lines for each filtered molecular line and create/update active_lines entries
        for idx, (line, intensity, tau_val) in enumerate(filtered_line_data):
            # Calculate line height
            lineheight = 0
            if max_intensity > 0:
                lineheight = (intensity / max_intensity) * max_y
            
            if lineheight > 0:
                # Create vertical line
                vline = self.ax2.vlines(line.lam, 0, lineheight,
                                       color=active_color, 
                                       linestyle='dashed', linewidth=1, picker=True)
                
                # Add text label
                text = self.ax2.text(line.lam, lineheight,
                                   f"{line.e_up:.0f},{line.a_stein:.3f}", 
                                   fontsize='x-small', 
                                   color=active_color, 
                                   rotation=45)
                
                # Create value data for this line
                value_data = {
                    'lam': line.lam,
                    'e': line.e_up,
                    'a': line.a_stein,
                    'g': line.g_up,
                    'e_low': line.e_low if line.e_low else 'N/A',
                    'g_low': line.g_low if line.g_low else 'N/A',
                    'inten': intensity,
                    'up_lev': line.lev_up if line.lev_up else 'N/A',
                    'low_lev': line.lev_low if line.lev_low else 'N/A',
                    'tau': tau_val if tau_val is not None else 'N/A',
                    #'text_obj': text,
                    'lineheight': lineheight,
                    'intensity_percent': (intensity / max_intensity) * 100  # Store percentage for debugging
                }
                
                # Add new entry to active_lines or update existing one
                if idx < len(active_lines_list):
                    # Update existing entry
                    active_lines_list[idx][0] = vline  # Set line artist
                    active_lines_list[idx][1] = text  # Set text artist
                    active_lines_list[idx][3].update(value_data)  # Update value data
                else:
                    # Create new entry: [line_artist, text_obj, scatter_artist, value_data]
                    active_lines_list.append([vline, text, None, value_data])
    
    def highlight_strongest_line(self, active_lines_list: List[Any]) -> Any:
        """
        Find and highlight the strongest line in the active lines.
        
        Parameters
        ----------
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
            
        Returns
        -------
        Any
            The strongest line triplet or None
        """
        if not active_lines_list:
            return None
        
        # Get the active scatter collection and count
        scatter_collection = getattr(self, '_active_scatter_collection', None)
        scatter_count = getattr(self, '_active_scatter_count', 0)
        active_color = self._get_theme_value("active_scatter_line_color", 'green')
        
        # Reset all line inspection lines to green first
        for line, text_obj, scatter, value in active_lines_list:
            if line is not None:
                line.set_color(active_color)
            if text_obj is not None:
                text_obj.set_color(active_color)

        # Find the line with the highest intensity and its scatter index
        highest_intensity = -float('inf')
        strongest_triplet = None
        strongest_scatter_idx = None
        
        for line, text_obj, scatter, value in active_lines_list:
            intensity = value.get('inten', 0) if value else 0
            if intensity > highest_intensity:
                highest_intensity = intensity
                strongest_triplet = [line, text_obj, scatter, value]
                strongest_scatter_idx = value.get('_scatter_point_index', None) if value else None
        
        # Reset scatter collection to all green, then highlight strongest in orange
        if scatter_collection is not None and scatter_count > 0:
            # Create color array - all green initially
            import matplotlib.colors as mcolors
            colors = [mcolors.to_rgba(active_color)] * scatter_count
            
            # Set the strongest point to orange
            if strongest_scatter_idx is not None and strongest_scatter_idx < scatter_count:
                colors[strongest_scatter_idx] = mcolors.to_rgba('orange')
            
            scatter_collection.set_facecolors(colors)
            scatter_collection.set_zorder(1)  # Reset base z-order
        
        # Highlight the strongest line inspection elements in orange
        if strongest_triplet is not None:
            line, text_obj, scatter, value = strongest_triplet
            if line is not None:
                line.set_color('orange')
            if text_obj is not None:
                text_obj.set_color('orange')
        
        return strongest_triplet
    
    def handle_line_pick_event(self, event: Any, active_lines_list: List[Any]) -> Any:
        """
        Handle line pick events and highlight the selected line.
        
        Parameters
        ----------
        event : Any
            The matplotlib pick event
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
            
        Returns
        -------
        Any
            The value data of the picked line or None
        """
        picked_value = None
        picked_scatter_idx = None
        picked_artist = event.artist
        
        # Get the active scatter collection and count
        scatter_collection = getattr(self, '_active_scatter_collection', None)
        scatter_count = getattr(self, '_active_scatter_count', 0)
        active_color = self._get_theme_value("active_scatter_line_color", 'green')
        
        # Check if the picked artist is the scatter collection
        # If so, use event.ind to determine which specific point was clicked
        scatter_point_clicked = None
        if picked_artist is scatter_collection and hasattr(event, 'ind') and len(event.ind) > 0:
            scatter_point_clicked = event.ind[0]  # Get first clicked point index
        
        # Find which entry in active_lines was picked and reset line inspection colors
        for line, text_obj, scatter, value in active_lines_list:
            # Check if this specific line was picked
            is_line_picked = (picked_artist is line)
            # Check if this specific scatter point was picked (by matching index)
            point_idx = value.get('_scatter_point_index', None) if value else None
            is_scatter_picked = (scatter_point_clicked is not None and point_idx == scatter_point_clicked)
            is_picked = is_line_picked or is_scatter_picked
            
            # Reset line inspection elements to green first
            if line is not None:
                line.set_color(active_color)
            if text_obj is not None:
                text_obj.set_color(active_color)

            # If this was the picked item, highlight in orange
            if is_picked:
                picked_value = value
                picked_scatter_idx = point_idx
                if line is not None:
                    line.set_color('orange')
                if text_obj is not None:
                    text_obj.set_color('orange')

        # Update scatter collection colors - all green except picked point
        if scatter_collection is not None and scatter_count > 0:
            import matplotlib.colors as mcolors
            colors = [mcolors.to_rgba(active_color)] * scatter_count
            
            # Set the picked point to orange
            if picked_scatter_idx is not None and picked_scatter_idx < scatter_count:
                colors[picked_scatter_idx] = mcolors.to_rgba('orange')
            
            scatter_collection.set_facecolors(colors)

        return picked_value
    
    def get_molecule_spectrum_data(self, molecule: 'Molecule', wave_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get spectrum data directly from molecule's caching system.
        
        Uses get_flux() method which returns consistent wavelength grids for all molecules
        when they use the same global wavelength range, eliminating grid size mismatches.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule with internal flux caching
        wave_data : np.ndarray
            Wavelength array (not used for interpolation, just for caching key)
            
        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (wavelength, flux) arrays or (None, None) if unavailable
        """
        if molecule is None or wave_data is None:
            return None, None
        
        try:
            # Use get_flux with return_wavelengths=True to get consistent grids
            # All molecules with the same global wavelength range will return identical grids
            result_wavelengths, result_flux = molecule.get_flux(
                wavelength_array=wave_data, 
                return_wavelengths=True, 
                interpolate_to_input=False  # Use native grid for consistency
            )
            
            if result_wavelengths is not None and result_flux is not None and len(result_flux) > 0:
                debug_config.verbose("plot_renderer", 
                                   f"Retrieved flux data for {self._get_molecule_display_name(molecule)}",
                                   data_points=len(result_flux))
                return result_wavelengths, result_flux
            
            debug_config.warning("plot_renderer", f"No flux data available for {self._get_molecule_display_name(molecule)}")
            return None, None
                
        except Exception as e:
            debug_config.error("plot_renderer", f"Could not get model data for molecule {self._get_molecule_display_name(molecule)}: {e}")
            return None, None
    
    def get_molecule_line_data(self, molecule: 'Molecule', xmin: float, xmax: float) -> List[Tuple['MoleculeLine', float, Optional[float]]]:
        """
        Get molecule lines in wavelength range.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object
        xmin, xmax : float
            Wavelength range
            
        Returns
        -------
        List[Tuple['MoleculeLine', float, Optional[float]]]
            List of (MoleculeLine, intensity, tau) tuples
        """
        try:
            # Method 1: Use intensity API
            if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                intensity_obj = molecule.intensity
                if hasattr(intensity_obj, 'get_lines_in_range_with_intensity'):
                    return intensity_obj.get_lines_in_range_with_intensity(xmin, xmax)
            
            # Method 2: Use MoleculeLineList directly
            if hasattr(molecule, 'lines') and molecule.lines is not None:
                lines = molecule.lines
                if hasattr(lines, 'get_lines_in_range'):
                    lines_in_range = lines.get_lines_in_range(xmin, xmax)
                    # Try to get corresponding intensities
                    if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                        intensity_obj = molecule.intensity
                        if hasattr(intensity_obj, 'intensity') and intensity_obj.intensity is not None:
                            intensities = intensity_obj.intensity
                            tau_values = getattr(intensity_obj, 'tau', None)
                            
                            result = []
                            for i, line in enumerate(lines_in_range):
                                intensity = intensities[i] if i < len(intensities) else 0.0
                                tau = tau_values[i] if tau_values is not None and i < len(tau_values) else None
                                result.append((line, intensity, tau))
                            return result
                    else:
                        # Return lines with zero intensity
                        return [(line, 0.0, None) for line in lines_in_range]
            return []
            
        except Exception as e:
            print(f"Error getting molecule lines: {e}")
            return []
    
    def render_individual_molecule_spectrum(self, molecule: 'Molecule', wave_data: np.ndarray, 
                                         plot_name: Optional[str] = None, subplot: Optional[Axes] = None,
                                         update_legend: bool = True) -> bool:
        """
        Render a single molecule spectrum using the molecule's cached data.
        
        Works with any axes - searches for existing lines on the target axes and updates
        them in place, or creates new ones. Does not depend on self.model_lines.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object with internal caching
        wave_data : np.ndarray
            Wavelength array
        plot_name : Optional[str]
            Custom name for plotting
        subplot : Optional[Axes]
            Subplot to render on (default: ax1)
        update_legend : bool
            Whether to update the legend after rendering (default: True)
            Set to False when rendering multiple molecules to update legend once at the end
            
        Returns
        -------
        bool
            True if successfully plotted, False otherwise
        """
        try:
            plot = subplot if subplot else self.ax1
            # Increment render stats
            self._plot_stats['renders_count'] += 1
            
            molecule_name = plot_name or self._get_molecule_display_name(molecule)
            mol_identifier = getattr(molecule, 'name', molecule_name)
            
            # Search for existing line with this molecule name directly on the target axes
            existing_line = None
            for line in plot.lines:
                if (hasattr(line, '_molecule_name') and 
                    line._molecule_name == mol_identifier):
                    existing_line = line
                    break
            
            # Get spectrum data directly from molecule's caching system
            plot_lam, plot_flux = self.get_molecule_spectrum_data(molecule, wave_data)
            
            if plot_lam is None or plot_flux is None:
                print(f"No spectrum data available for {molecule_name}")
                return False
            
            # Check if we actually have meaningful flux data
            if len(plot_flux) == 0 or np.all(plot_flux == 0):
                print(f"Spectrum data is empty or all zeros for {molecule_name}")
                return False
            
            # Get molecule properties
            color = self._get_molecule_color(molecule)
            label = getattr(molecule, 'displaylabel', molecule_name)
            
            lw = 1 if self.render_out else 2
            alpha = 1 if self.render_out else 0.8
            
            # Update existing line if found, otherwise create new
            if existing_line is not None:
                # Update existing line data - much faster than remove/recreate
                existing_line.set_data(plot_lam, plot_flux)
                existing_line.set_color(color)
                existing_line.set_alpha(alpha)
                existing_line.set_linewidth(lw)
                existing_line.set_label(label)
                line = existing_line
            else:
                # Create new line only if it doesn't exist on this axes
                line, = plot.plot(
                    plot_lam,
                    plot_flux,
                    linestyle='--',
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    label=label,
                    zorder=self._get_theme_value("zorder_model", 3)
                )
                
                # Store molecule name in line metadata for selective removal
                line._molecule_name = mol_identifier
                
                # Only track in self.model_lines if this is the main plot (ax1)
                if plot == self.ax1:
                    self.model_lines.append(line)
    
            # Only update legend if requested (batch operations can skip this)
            if update_legend:
                handles, labels = plot.get_legend_handles_labels()
                if handles:
                    ncols = 2 if len(handles) > 8 else 1
                    plot.legend(ncols = ncols)
            
            self._plot_stats['molecules_rendered'] += 1
            
            return True
            
        except Exception as e:
            print(f"Error plotting molecule {self._get_molecule_display_name(molecule)}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_intensity_data(self, molecule: 'Molecule') -> Optional[pd.DataFrame]:
        """
        Get intensity table directly from molecule's caching system.
        
        Fixed to properly access molecule's intensity object using standard methods.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object with internal caching
            
        Returns
        -------
        Optional[pd.DataFrame]
            Intensity table with columns: lam, intens, a_stein, g_up, e_up, etc.
        """
        try:
            if not hasattr(molecule, 'intensity') or molecule.intensity is None:
                return None
                
            molecule_name = self._get_molecule_display_name(molecule)
            intensity_obj = molecule.intensity
            
            # Check if intensity data is already computed (cache hit indicator)
            has_computed_data = (hasattr(intensity_obj, 'intensity') and 
                               intensity_obj.intensity is not None and 
                               len(intensity_obj.intensity) > 0)
            
            # Ensure intensity is calculated (this uses molecule's internal caching)
            if hasattr(molecule, 'calculate_intensity'):
                molecule.calculate_intensity()
            
            # Check if data was already computed (cache hit) or newly computed
            cache_status = "from cache" if has_computed_data else "newly computed"
            
            # Get the intensity table from the intensity object
            if hasattr(intensity_obj, 'get_table'):
                table = intensity_obj.get_table
                if table is not None:
                    # Reset index for consistent access
                    if hasattr(table, 'index'):
                        table.index = range(len(table.index))
                    debug_config.verbose("plot_renderer", 
                                       f"Retrieved intensity table for {molecule_name} ({cache_status})",
                                       cache_hit=has_computed_data,
                                       table_rows=len(table))
                    return table
            
            return None
            
        except Exception as e:
            debug_config.error("plot_renderer", f"Error getting intensity table for {self._get_molecule_display_name(molecule)}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_line_intensity_threshold(self) -> float:
        """
        Get the line intensity threshold from user settings.
        
        Returns
        -------
        float
            Threshold as a percentage (0.0 to 1.0) of the strongest line intensity.
            Default is 0.3 (30%) if not specified in settings.
        """
        return getattr(self.islat, 'user_settings', {}).get('line_threshold', 0.3)
    
    def set_line_intensity_threshold(self, threshold_percent: float) -> None:
        """
        Set the line intensity threshold dynamically.
        
        This updates the user settings and can be used for runtime adjustments.
        
        Parameters
        ----------
        threshold_percent : float
            Threshold as a percentage (0.0 to 1.0) of the strongest line intensity.
        """
        if not hasattr(self.islat, 'user_settings'):
            print("Warning: user_settings not available, cannot set threshold")
            return
            
        # Validate threshold value
        threshold_percent = max(0.0, min(1.0, threshold_percent))
        
        # Update user settings
        self.islat.user_settings['line_threshold'] = threshold_percent
        
        print(f"Line intensity threshold updated to {threshold_percent*100:.1f}%")
    
    def filter_lines_by_threshold(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], 
                                 threshold_percent: Optional[float] = None) -> List[Tuple['MoleculeLine', float, Optional[float]]]:
        """
        Filter line data based on intensity threshold.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
        threshold_percent : Optional[float]
            Threshold percentage (0.0 to 1.0). If None, uses user settings.
            
        Returns
        -------
        List[Tuple['MoleculeLine', float, Optional[float]]]
            Filtered list of lines above threshold
        """
        if not line_data:
            return []
            
        if threshold_percent is None:
            threshold_percent = self.get_line_intensity_threshold()
        
        # Extract intensities for threshold calculation
        intensities = [intensity for _, intensity, _ in line_data]
        if not intensities:
            return []
            
        max_intensity = max(intensities)
        if max_intensity <= 0:
            return []
            
        # Calculate threshold intensity
        threshold_intensity = max_intensity * threshold_percent
        
        # Filter lines based on threshold
        filtered_lines = [(line, intensity, tau_val) for line, intensity, tau_val in line_data 
                         if intensity >= threshold_intensity]
        
        return filtered_lines

    def plot_fitted_saved_lines(self, fit_data, ax: Optional[plt.Axes] = None) -> None:
        """
        Plot the fitted saved lines on the provided axes using flux integral calculation.

        Parameters
        ----------
        fit_data : Any
            The data to plot.
        ax : Optional[plt.Axes]
            The axes to plot on. If None, uses the current axes.
        """
        if ax is None:
            ax = plt.gca()

        #print(f"fit data: {fit_data}")

        # Unpack the fit_data tuple
        gauss_fits, fitted_waves, fitted_fluxes = fit_data
        
        # Iterate through each fit
        for i, (gauss_fit, fitted_wave, fitted_flux) in enumerate(zip(gauss_fits, fitted_waves, fitted_fluxes)):
            #line = gauss_fit
            #print(f"Line: {line}")
            #print(f"Fitted wave: {fitted_wave}")
            #print(f"Fitted flux: {fitted_flux}")
            
            lam_min = np.min(fitted_wave)
            lam_max = np.max(fitted_wave)
            
            # plot the fit result
            ax.plot(fitted_wave, fitted_flux, color='lime', linewidth=2, zorder=10, linestyle='--')[0]#, label=f'Gauss Fit {i}')[0]
            dely = gauss_fit.eval_uncertainty(sigma = self.islat.user_settings.get('fit_line_uncertainty', 3.0))
            ax.fill_between(fitted_wave, fitted_flux - dely, fitted_flux + dely,
                                    color='lime', alpha=0.3)#, label=r'3-$\sigma$ uncertainty band')
            
            # plot the xmin and xmax for each line
            ax.vlines([lam_min, lam_max], -2, 10, colors='lime', alpha=0.5)

            #ax.legend()
            self.canvas.draw_idle()