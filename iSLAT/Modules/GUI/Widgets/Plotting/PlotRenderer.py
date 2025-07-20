from typing import Optional, List, Dict, Any, Tuple, Union, TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import iSLAT.Constants as c

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
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    from iSLAT.Modules.DataTypes.Spectrum import Spectrum

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
        
        self.fig = plot_manager.fig
        self.ax1 = plot_manager.ax1
        self.ax2 = plot_manager.ax2
        self.ax3 = plot_manager.ax3
        self.canvas = plot_manager.canvas
        
        self.model_lines: List[Line2D] = []
        self.active_lines: List[Line2D] = []
        
        # Simplified stats - only for performance monitoring, no data caching
        self._plot_stats = {
            'renders_count': 0,
            'molecules_rendered': 0
        }
        
        # Remove molecule change callbacks since we don't cache data anymore
    
    # Helper methods for common operations
    def _convert_visibility_to_bool(self, is_visible_raw: Any) -> bool:
        """Convert various visibility representations to boolean"""
        if isinstance(is_visible_raw, str):
            return is_visible_raw.lower() in ('true', '1', 'yes', 'on')
        return bool(is_visible_raw)
    
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
            return molecule_colors[mol_name]
            
        # Default fallback
        return self._get_theme_value('default_molecule_color', 'blue')
    
    def _safe_get_molecule_attribute(self, molecule: 'Molecule', attr_name: str, default_value: Any = None) -> Any:
        """Safely get molecule attribute with error handling"""
        try:
            return getattr(molecule, attr_name, default_value)
        except Exception as e:
            debug_config.error("plot_renderer", f"Error accessing {attr_name} for molecule {self._get_molecule_display_name(molecule)}: {e}")
            return default_value
    
    def cleanup_callbacks(self) -> None:
        """No callbacks to cleanup since we rely entirely on molecule caching"""
        pass
    
    def clear_all_plots(self) -> None:
        """Clear all plots and reset stats"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.model_lines.clear()
        self.active_lines.clear()
        
        # Reset only performance stats, no cache data
        self._plot_stats = {
            'renders_count': 0,
            'molecules_rendered': 0
        }
        
    def clear_model_lines(self) -> None:
        """Clear only the model spectrum lines from the main plot"""
        for line in self.model_lines:
            if line in self.ax1.lines:
                line.remove()
        self.model_lines.clear()

    def remove_molecule_lines(self, molecule_name: str) -> None:
        """Remove lines associated with a specific molecule"""
        lines_to_remove = []
        for line in self.model_lines:
            # Check if line belongs to this molecule (by stored metadata first)
            if hasattr(line, '_molecule_name') and line._molecule_name == molecule_name:
                lines_to_remove.append(line)
            # Fallback: check by label if metadata not available
            elif hasattr(line, 'get_label') and line.get_label():
                label = line.get_label()
                # Check both the molecule name and display label
                if molecule_name in label or any(mol_part in label for mol_part in molecule_name.split('_')):
                    lines_to_remove.append(line)
        
        # Remove lines from plot and list
        for line in lines_to_remove:
            if line in self.ax1.lines:
                line.remove()
            if line in self.model_lines:
                self.model_lines.remove(line)

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
        
    def render_main_spectrum_plot(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                                 molecules: Union[List['Molecule'], 'MoleculeDict'], 
                                 summed_flux: Optional[np.ndarray] = None, 
                                 error_data: Optional[np.ndarray] = None) -> None:
        """Render the main spectrum plot with observed data, model spectra, and sum"""
        # Store current view limits
        current_xlim = self.ax1.get_xlim() if hasattr(self.ax1, 'get_xlim') else None
        current_ylim = self.ax1.get_ylim() if hasattr(self.ax1, 'get_ylim') else None
        
        # Clear the plot
        self.ax1.clear()
        
        # Early return if no data
        if wave_data is None or len(wave_data) == 0:
            self.ax1.set_title("No spectrum data loaded")
            return
        
        # Plot observed spectrum
        self._plot_observed_spectrum(wave_data, flux_data, error_data)
        
        # Plot individual molecule spectra
        if molecules:
            self.render_visible_molecules(wave_data, molecules)
            
        # Plot summed spectrum
        if summed_flux is not None and len(summed_flux) > 0:
            self._plot_summed_spectrum(wave_data, summed_flux)
        
        # Configure plot appearance
        self._configure_main_plot_appearance()
        
        # Restore view limits if they existed
        if current_xlim and current_ylim:
            if current_xlim != (0.0, 1.0) and current_ylim != (0.0, 1.0):
                self.ax1.set_xlim(current_xlim)
                self.ax1.set_ylim(current_ylim)
        
    def _plot_observed_spectrum(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                               error_data: Optional[np.ndarray] = None) -> None:
        """Plot the observed spectrum data"""
        if flux_data is not None and len(flux_data) > 0:
            if error_data is not None and len(error_data) == len(flux_data):
                # Plot with error bars
                self.ax1.errorbar(
                    wave_data, 
                    flux_data,
                    yerr=error_data,
                    fmt='-', 
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=self._get_theme_value("zorder_observed", 3),
                    elinewidth=0.5,
                    capsize=0
                )
            else:
                # Plot without error bars
                self.ax1.plot(
                    wave_data, 
                    flux_data,
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=1,
                    label='Observed',
                    zorder=self._get_theme_value("zorder_observed", 3)
                )
    
    def _plot_summed_spectrum(self, wave_data: np.ndarray, summed_flux: np.ndarray) -> None:
        """Plot the summed model spectrum"""
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
    
    def _configure_main_plot_appearance(self) -> None:
        """Configure the appearance of the main plot"""
        self.ax1.set_xlabel('Wavelength (μm)')
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax1.set_title("Full Spectrum with Line Inspection")
        
        # Only show legend if there are labeled items
        handles, labels = self.ax1.get_legend_handles_labels()
        if handles:
            self.ax1.legend()
        
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
                         label="Observed")
            
            self.ax2.set_xlabel("Wavelength (μm)")
            self.ax2.set_ylabel("Flux (Jy)")
            self.ax2.set_title("Line inspection plot")
            
            # Show legend if there are labeled items
            handles, labels = self.ax2.get_legend_handles_labels()
            if handles:
                self.ax2.legend()
    
    def _get_molecule_parameters_hash(self, molecule: 'Molecule') -> Optional[int]:
        if molecule is None:
            return None
        
        try:
            if hasattr(molecule, 'get_parameter_hash'):
                return molecule.get_parameter_hash('full')
            elif hasattr(molecule, '_compute_full_parameter_hash'):
                return molecule._compute_full_parameter_hash()
            else:
                return hash((molecule.name, molecule.temp, molecule.radius, molecule.n_mol, 
                           molecule.distance, molecule.fwhm, molecule.broad))
        except Exception:
            return None
    
    def render_population_diagram(self, molecule: 'Molecule', wave_range: Optional[Tuple[float, float]] = None) -> None:
        """
        Render population diagram using molecule's cached intensity data.
        
        This method relies on the molecule's internal caching system rather than
        maintaining its own cache to avoid conflicts with cached parameter restoration.
        """
        self.ax3.clear()
        
        if molecule is None:
            self.ax3.set_title("No molecule selected")
            return
            
        try:
            # Use molecule's cached intensity data directly
            int_pars = self.get_intensity_data(molecule)
            if int_pars is None:
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f"{mol_label} - No intensity data")
                return

            wavelength = int_pars['lam']
            intens_mod = int_pars['intens']
            Astein_mod = int_pars['a_stein']
            gu = int_pars['g_up']
            eu = int_pars['e_up']

            radius = getattr(molecule, 'radius', 1.0)
            distance = getattr(molecule, 'distance', getattr(self.islat, 'global_dist', 140.0))
            
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
                self.ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
                self.ax3.set_xlabel(r'$E_{u}$ (K)')
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f'{mol_label} Population diagram', fontsize='medium')
            else:
                mol_label = self._get_molecule_display_name(molecule)
                self.ax3.set_title(f"{mol_label} - No valid data for population diagram")
            
        except Exception as e:
            debug_config.error("plot_renderer", f"Error rendering population diagram: {e}")
            mol_label = self._get_molecule_display_name(molecule)
            self.ax3.set_title(f"{mol_label} - Error in calculation")
    
    def plot_saved_lines(self, saved_lines: pd.DataFrame) -> None:
        """Plot saved lines on the main spectrum"""
        if saved_lines.empty:
            return

        for index, line in saved_lines.iterrows():
            #print("Line:", line)
            #print("Index:", index)
            # Plot vertical lines at saved positions
            if 'lam' in line:
                self.ax1.axvline(
                    line['lam'], 
                    color=self._get_theme_value("saved_line_color", self._get_theme_value("saved_line_color_one", "red")),
                    alpha=0.7, 
                    linestyle=':', 
                    label=f"Saved: {line.get('label', 'Line')}"
                )
            elif 'xmin' in line and 'xmax' in line:
                # Plot wavelength range
                self.ax1.axvspan(
                    line['xmin'], 
                    line['xmax'], 
                    alpha=0.2, 
                    color=self._get_theme_value("saved_line_color_two", "coral"),
                    label=f"Saved Range: {line.get('label', 'Range')}"
                )
        # make sure that a refresh of the plot is triggered
        self.update_plot_display()
    
    def highlight_line_selection(self, xmin: float, xmax: float) -> None:
        """Highlight a selected wavelength range"""
        # Remove previous highlights
        for patch in self.ax1.patches:
            if hasattr(patch, '_islat_highlight'):
                patch.remove()
        
        # Add new highlight
        highlight = self.ax1.axvspan(xmin, xmax, alpha=0.3, color=self._get_theme_value("highlighted_line_color", "yellow"))
        highlight._islat_highlight = True
    
    def update_plot_display(self) -> None:
        """Update the plot display"""
        self.canvas.draw_idle()
    
    def force_plot_refresh(self) -> None:
        """Force a complete plot refresh"""
        self.canvas.draw()
    
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
    
    def get_visible_molecules(self, molecules: Union['MoleculeDict', List['Molecule']]) -> List['Molecule']:
        """Get visible molecules using the most efficient method available"""
        
        debug_config.trace("plot_renderer", f"molecules type = {type(molecules)}")
        debug_config.trace("plot_renderer", f"hasattr(molecules, 'get_visible_molecules_fast') = {hasattr(molecules, 'get_visible_molecules_fast')}")
        debug_config.trace("plot_renderer", f"hasattr(molecules, 'values') = {hasattr(molecules, 'values')}")
        debug_config.trace("plot_renderer", f"hasattr(molecules, '__iter__') = {hasattr(molecules, '__iter__')}")
        
        if hasattr(molecules, 'get_visible_molecules_fast'):
            # MoleculeDict with fast access
            visible_names = molecules.get_visible_molecules_fast()
            visible_molecules = [molecules[name] for name in visible_names if name in molecules]
            debug_config.trace("plot_renderer", f"get_visible_molecules_fast(): {len(visible_molecules)}/{len(molecules)} molecules visible: {visible_names}")
            return visible_molecules
        elif hasattr(molecules, 'values'):
            # Regular dict-like object - check each molecule's is_visible attribute
            debug_config.trace("plot_renderer", "Using values() path for dict-like object")
            visible_molecules = []
            for mol in molecules.values():
                is_visible_raw = getattr(mol, 'is_visible', False)
                mol_name = getattr(mol, 'name', 'unknown')
                
                # Use consolidated conversion method
                is_visible = self._convert_visibility_to_bool(is_visible_raw)
                    
                debug_config.trace("plot_renderer", f"Checking {mol_name}: is_visible_raw = {is_visible_raw} -> is_visible = {is_visible}")
                if is_visible:
                    visible_molecules.append(mol)
            debug_config.trace("plot_renderer", f"values() path result: {len(visible_molecules)}/{len(molecules)} molecules visible")
            return visible_molecules
        elif hasattr(molecules, '__iter__'):
            # List-like object
            debug_config.trace("plot_renderer", "Using __iter__ path for list-like object")
            visible_molecules = []
            total_count = 0
            for mol in molecules:
                total_count += 1
                is_visible_raw = getattr(mol, 'is_visible', False)
                mol_name = getattr(mol, 'name', 'unknown')
                mol_id = id(mol)
                
                # Use consolidated conversion method
                is_visible = self._convert_visibility_to_bool(is_visible_raw)
                
                debug_config.trace("plot_renderer", f"Checking {mol_name} (id:{mol_id}): is_visible_raw = {is_visible_raw} (type: {type(is_visible_raw)}) -> is_visible = {is_visible}")
                
                if is_visible:
                    visible_molecules.append(mol)
                    debug_config.trace("plot_renderer", f"ADDED {mol_name} to visible_molecules list (list length now: {len(visible_molecules)})")
                else:
                    debug_config.trace("plot_renderer", f"SKIPPED {mol_name} (not visible)")
                
            debug_config.trace("plot_renderer", f"Final visible_molecules list length: {len(visible_molecules)}")
            debug_config.trace("plot_renderer", f"__iter__ path result: {len(visible_molecules)}/{total_count} molecules visible")
            return visible_molecules
        else:
            # Single molecule
            is_visible_raw = getattr(molecules, 'is_visible', False)
            
            # Use consolidated conversion method
            is_visible = self._convert_visibility_to_bool(is_visible_raw)
                
            result = [molecules] if is_visible else []
            debug_config.trace("plot_renderer", f"Single molecule visibility: is_visible_raw = {is_visible_raw} -> is_visible = {is_visible} -> {'visible' if is_visible else 'hidden'}")
            return result
    
    def render_visible_molecules(self, wave_data: np.ndarray, molecules: Union['MoleculeDict', List['Molecule']]) -> None:
        """
        Render molecules using their built-in caching for optimal performance.
        
        Each molecule's caching system is leveraged to avoid redundant calculations
        based on parameter hashes and cache validity.
        """
        if not molecules:
            return
        
        # Get visible molecules
        visible_molecules = self.get_visible_molecules(molecules)
        if not visible_molecules:
            return
        
        # Use each molecule's internal caching for rendering
        for mol in visible_molecules:
            mol_name = getattr(mol, 'name', 'unknown')
            try:
                # The render_individual_molecule_spectrum method will use the molecule's
                # built-in caching via get_flux() which respects parameter hash validation
                success = self.render_individual_molecule_spectrum(mol, wave_data)
                if not success:
                    debug_config.warning("plot_renderer", f"Could not render molecule {mol_name}")
            except Exception as e:
                print(f"Error rendering molecule {mol_name}: {e}")
                continue
    
    def optimize_plot_memory_usage(self) -> None:
        """Optimize memory usage for plotting operations"""
        # Limit the number of cached model lines
        if len(self.model_lines) > 50:
            # Remove oldest lines from plot
            for line in self.model_lines[:25]:
                if line in self.ax1.lines:
                    line.remove()
            self.model_lines = self.model_lines[25:]
        
        # Clear inactive lines
        self.active_lines = [line for line in self.active_lines if line in self.ax2.lines]
    
    def batch_update_molecule_colors(self, molecule_color_map: Dict[str, str]) -> None:
        """Update molecule colors in batch for better performance"""
        for line in self.model_lines:
            label = line.get_label()
            if label in molecule_color_map:
                line.set_color(molecule_color_map[label])
        
        # Update canvas once at the end
        self.canvas.draw_idle()
    
    def get_plot_performance_stats(self) -> Dict[str, Any]:
        """Get simplified performance statistics"""
        return {
            'model_lines_count': len(self.model_lines),
            'active_lines_count': len(self.active_lines),
            'total_renders': self._plot_stats.get('renders_count', 0),
            'molecules_rendered': self._plot_stats.get('molecules_rendered', 0),
            'caching_strategy': 'molecule_only',  # Document our strategy
            'line_intensity_threshold_percent': self.get_line_intensity_threshold() * 100
        }
    
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
                line_artist = line_data[0]  # Line artist (vlines)
                scatter_artist = line_data[1]  # Scatter artist
                
                # Remove line artist if it exists
                if line_artist is not None:
                    try:
                        line_artist.remove()
                    except (ValueError, AttributeError):
                        pass
                
                # Remove scatter artist if it exists
                if scatter_artist is not None:
                    try:
                        scatter_artist.remove()
                    except (ValueError, AttributeError):
                        pass
        
        # Clear the list after removing all artists
        active_lines_list.clear()
    
    def render_active_lines_in_population_diagram(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]], active_lines_list: List[Any]) -> None:
        """
        Render active lines as scatter points in the population diagram.
        
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
            print(f"No lines above threshold ({threshold_percent*100:.1f}% of strongest line) for population diagram")
            return
            
        print(f"Rendering {len(filtered_line_data)}/{len(line_data)} lines in population diagram above threshold ({threshold_percent*100:.1f}%)")
        
        # Get max intensity from original data for consistent percentage calculation
        original_intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(original_intensities) if original_intensities else 1.0
        
        # Calculate rd_yax values for each filtered line and add scatter points
        for idx, (line, intensity, tau_val) in enumerate(filtered_line_data):
            if all(x is not None for x in [intensity, line.a_stein, line.g_up, line.lam]):
                # Get molecule properties safely
                molecule = getattr(self.islat, 'active_molecule', None)
                if molecule is None:
                    continue
                    
                radius = getattr(molecule, 'radius', 1.0)
                distance = getattr(molecule, 'distance', getattr(self.islat, 'global_dist', 140.0))
                
                # Calculate rd_yax
                area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
                dist = distance * c.PARSEC_CM
                beam_s = area / dist ** 2
                F = intensity * beam_s
                freq = c.SPEED_OF_LIGHT_MICRONS / line.lam
                rd_yax = np.log(4 * np.pi * F / (line.a_stein * c.PLANCK_CONSTANT * freq * line.g_up))
                
                # Create scatter point
                sc = self.ax3.scatter(line.e_up, rd_yax, s=30, 
                                     color=self._get_theme_value("scatter_main_color", 'green'), 
                                     edgecolors='black', picker=True)
                
                # Store line information
                value_data = {
                    'lam': line.lam,
                    'e': line.e_up,
                    'a': line.a_stein,
                    'g': line.g_up,
                    'rd_yax': rd_yax,
                    'inten': intensity,
                    'up_lev': line.lev_up if line.lev_up else 'N/A',
                    'low_lev': line.lev_low if line.lev_low else 'N/A',
                    'tau': tau_val if tau_val is not None else 'N/A',
                    'intensity_percent': (intensity / max_intensity) * 100  # Store percentage for debugging
                }
                
                # Update existing entry or create new one
                if idx < len(active_lines_list):
                    # Update existing entry with scatter artist
                    active_lines_list[idx][1] = sc  # Set scatter artist
                    active_lines_list[idx][2].update(value_data)  # Update value data
                else:
                    # Create new entry: [line_artist, scatter_artist, value_data]
                    active_lines_list.append([None, sc, value_data])
    
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
            print(f"No lines above threshold ({threshold_percent*100:.1f}% of strongest line)")
            return
            
        print(f"Rendering {len(filtered_line_data)}/{len(line_data)} lines above threshold ({threshold_percent*100:.1f}%)")
        
        # Get max intensity from original data for consistent scaling
        original_intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(original_intensities) if original_intensities else 1.0
        
        # Plot vertical lines for each filtered molecular line and create/update active_lines entries
        for idx, (line, intensity, tau_val) in enumerate(filtered_line_data):
            # Calculate line height
            lineheight = 0
            if max_intensity > 0:
                lineheight = (intensity / max_intensity) * max_y
            
            if lineheight > 0:
                # Create vertical line
                vline = self.ax2.vlines(line.lam, 0, lineheight,
                                       color=self._get_theme_value("active_scatter_line_color", "green"), 
                                       linestyle='dashed', linewidth=1, picker=True)
                
                # Add text label
                text = self.ax2.text(line.lam, lineheight,
                                   f"{line.e_up:.0f},{line.a_stein:.3f}", 
                                   fontsize='x-small', 
                                   color=self._get_theme_value("active_scatter_line_color", "green"), 
                                   rotation=45)
                
                # Create value data for this line
                value_data = {
                    'lam': line.lam,
                    'e': line.e_up,
                    'a': line.a_stein,
                    'g': line.g_up,
                    'inten': intensity,
                    'up_lev': line.lev_up if line.lev_up else 'N/A',
                    'low_lev': line.lev_low if line.lev_low else 'N/A',
                    'tau': tau_val if tau_val is not None else 'N/A',
                    'text_obj': text,
                    'lineheight': lineheight,
                    'intensity_percent': (intensity / max_intensity) * 100  # Store percentage for debugging
                }
                
                # Add new entry to active_lines or update existing one
                if idx < len(active_lines_list):
                    # Update existing entry
                    active_lines_list[idx][0] = vline  # Set line artist
                    active_lines_list[idx][2].update(value_data)  # Update value data
                else:
                    # Create new entry: [line_artist, scatter_artist, value_data]
                    active_lines_list.append([vline, None, value_data])
    
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
            
        # Reset all lines to green first
        for line, scatter, value in active_lines_list:
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
                scatter.set_zorder(1)  # Reset z-order
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('green')
        
        # Find the line with the highest intensity
        highest_intensity = -float('inf')
        strongest_triplet = None
        
        for line, scatter, value in active_lines_list:
            intensity = value.get('inten', 0) if value else 0
            if intensity > highest_intensity:
                highest_intensity = intensity
                strongest_triplet = [line, scatter, value]
        
        # Highlight the strongest line in orange
        if strongest_triplet is not None:
            line, scatter, value = strongest_triplet
            if line is not None:
                line.set_color('orange')
            if scatter is not None:
                scatter.set_facecolor('orange')
                scatter.set_zorder(10)  # Bring to front
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('orange')
        
        return strongest_triplet
    
    def handle_line_pick_event(self, picked_artist: Any, active_lines_list: List[Any]) -> Any:
        """
        Handle line pick events and highlight the selected line.
        
        Parameters
        ----------
        picked_artist : Any
            The matplotlib artist that was picked
        active_lines_list : List[Any]
            List of [line_artist, scatter_artist, value_data] tuples
            
        Returns
        -------
        Any
            The value data of the picked line or None
        """
        picked_value = None
        
        # Find which entry in active_lines was picked and reset colors
        for line, scatter, value in active_lines_list:
            is_picked = (picked_artist is line or picked_artist is scatter)
            
            # Reset all to green first
            if line is not None:
                line.set_color('green')
            if scatter is not None:
                scatter.set_facecolor('green')
            if 'text_obj' in value and value['text_obj'] is not None:
                value['text_obj'].set_color('green')
            
            # If this was the picked item, highlight in orange
            if is_picked:
                picked_value = value
                if line is not None:
                    line.set_color('orange')
                if scatter is not None:
                    scatter.set_facecolor('orange')
                if 'text_obj' in value and value['text_obj'] is not None:
                    value['text_obj'].set_color('orange')
        
        return picked_value
    
    def get_molecule_spectrum_data(self, molecule: 'Molecule', wave_data: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get spectrum data directly from molecule's caching system.
        
        Fixed to use prepare_plot_data() which has proper parameter-hash-aware caching,
        instead of get_flux() which clears cache on parameter changes.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule with internal flux caching
        wave_data : np.ndarray
            Wavelength array
            
        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            (wavelength, flux) arrays or (None, None) if unavailable
        """
        if molecule is None or wave_data is None:
            return None, None
        
        try:
            # Check wave_data_cache size before and after to detect cache hits
            initial_wave_cache_size = len(molecule._wave_data_cache) if hasattr(molecule, '_wave_data_cache') and molecule._wave_data_cache else 0
            
            # Use prepare_plot_data which has parameter-hash-aware caching
            # This method checks parameter hash and reuses cached data when parameters return to previous values
            result = molecule.prepare_plot_data(wave_data)
            
            # Check if cache was used by seeing if a new entry was added
            final_wave_cache_size = len(molecule._wave_data_cache) if hasattr(molecule, '_wave_data_cache') and molecule._wave_data_cache else 0
            used_cache = final_wave_cache_size == initial_wave_cache_size  # No new entry = cache hit
            
            if result is not None and len(result) == 2:
                plot_lam, plot_flux = result
                if plot_lam is not None and plot_flux is not None and len(plot_flux) > 0:
                    cache_status = "from cache" if used_cache else "newly computed"
                    debug_config.verbose("plot_renderer", 
                                       f"Retrieved flux data for {self._get_molecule_display_name(molecule)} ({cache_status})",
                                       cache_hit=used_cache,
                                       data_points=len(plot_flux))
                    return plot_lam, plot_flux
            
            debug_config.warning("plot_renderer", f"No flux data available for {self._get_molecule_display_name(molecule)}")
            return None, None
                
        except Exception as e:
            print(f"Error getting flux data for {self._get_molecule_display_name(molecule)}: {e}")
            import traceback
            traceback.print_exc()
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
                                         plot_name: Optional[str] = None) -> bool:
        """
        Render a single molecule spectrum using ONLY the molecule's cached data.
        
        No additional caching layers - complete reliance on molecule's caching system.
        Enhanced with cache debugging to diagnose 850°→950°→850° issues.
        
        Parameters
        ----------
        molecule : Molecule
            Molecule object with internal caching
        wave_data : np.ndarray
            Wavelength array
        plot_name : Optional[str]
            Custom name for plotting
            
        Returns
        -------
        bool
            True if successfully plotted, False otherwise
        """
        try:
            # Increment render stats
            self._plot_stats['renders_count'] += 1
            
            molecule_name = plot_name or self._get_molecule_display_name(molecule)
            
            # Debug cache status before attempting to get spectrum data
            cache_debug = self.debug_molecule_cache_status(molecule)
            wave_cache_size = cache_debug.get('wave_data_cache_size', 0)
            debug_config.trace("plot_renderer", 
                             f"Rendering {molecule_name} - Cache debug: flux_cache_size={cache_debug.get('flux_cache_size', 0)}, "
                             f"wave_data_cache_size={wave_cache_size}, "
                             f"spectrum_cache_size={cache_debug.get('spectrum_cache_size', 0)}, "
                             f"intensity_cache_size={cache_debug.get('intensity_cache_size', 0)}")
            if wave_cache_size > 0:
                debug_config.trace("plot_renderer", f"  Wave data cache contains {wave_cache_size} entries with parameter hashes")
            
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
            
            # Plot the spectrum
            line, = self.ax1.plot(
                plot_lam,
                plot_flux,
                linestyle='--',
                color=color,
                alpha=0.7,
                linewidth=self._get_theme_value("model_plot_line_width", 1.5),
                label=label,
                zorder=self._get_theme_value("zorder_model", 2)
            )
            
            # Store molecule name in line metadata for selective removal
            line._molecule_name = getattr(molecule, 'name', molecule_name)
            
            self.model_lines.append(line)
            self._plot_stats['molecules_rendered'] += 1
            
            print(f"Successfully rendered spectrum for {molecule_name} with {len(plot_flux)} data points")
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
        
    def get_plot_performance_stats(self) -> Dict[str, Any]:
        """Get simplified performance statistics"""
        return {
            'model_lines_count': len(self.model_lines),
            'active_lines_count': len(self.active_lines),
            'total_renders': self._plot_stats.get('renders_count', 0),
            'molecules_rendered': self._plot_stats.get('molecules_rendered', 0),
            'caching_strategy': 'molecule_only'  # Document our strategy
        }
    
    def debug_molecule_cache_status(self, molecule: 'Molecule') -> Dict[str, Any]:
        """
        Debug the cache status of a molecule to understand why plots aren't updating.
        
        This helps diagnose issues like 850°→950°→850° not updating correctly.
        """
        if molecule is None:
            return {'error': 'No molecule provided'}
        
        try:
            molecule_name = self._get_molecule_display_name(molecule)
            debug_info = {
                'molecule_name': molecule_name,
                'molecule_id': id(molecule),
                'has_intensity': hasattr(molecule, 'intensity') and molecule.intensity is not None,
                'has_spectrum': hasattr(molecule, 'spectrum') and molecule.spectrum is not None,
                'has_lines': hasattr(molecule, 'lines') and molecule.lines is not None,
            }
            
            # Check parameter hash methods
            debug_info['parameter_hash_methods'] = []
            if hasattr(molecule, 'get_parameter_hash'):
                debug_info['parameter_hash_methods'].append('get_parameter_hash')
                try:
                    debug_info['current_spectrum_hash'] = molecule.get_parameter_hash('spectrum')
                    debug_info['current_intensity_hash'] = molecule.get_parameter_hash('intensity')
                    debug_info['current_full_hash'] = molecule.get_parameter_hash('full')
                except Exception as e:
                    debug_info['hash_error'] = str(e)
            
            if hasattr(molecule, '_compute_spectrum_parameter_hash'):
                debug_info['parameter_hash_methods'].append('_compute_spectrum_parameter_hash')
            if hasattr(molecule, '_compute_intensity_parameter_hash'):
                debug_info['parameter_hash_methods'].append('_compute_intensity_parameter_hash')
            
            # Check cache attributes
            debug_info['cache_attributes'] = []
            if hasattr(molecule, '_flux_cache'):
                debug_info['cache_attributes'].append('_flux_cache')
                debug_info['flux_cache_size'] = len(molecule._flux_cache) if molecule._flux_cache else 0
                if molecule._flux_cache:
                    debug_info['flux_cache_keys'] = list(molecule._flux_cache.keys())
            
            if hasattr(molecule, '_wave_data_cache'):
                debug_info['cache_attributes'].append('_wave_data_cache')
                debug_info['wave_data_cache_size'] = len(molecule._wave_data_cache) if molecule._wave_data_cache else 0
                if molecule._wave_data_cache:
                    # Show composite keys and their parameter hashes
                    cache_info = []
                    for cache_key in molecule._wave_data_cache.keys():
                        if isinstance(cache_key, tuple) and len(cache_key) == 2:
                            wave_hash, param_hash = cache_key
                            cache_info.append(f"wave:{wave_hash}, params:{param_hash}")
                        else:
                            cache_info.append(str(cache_key))
                    debug_info['wave_data_cache_keys'] = cache_info
            
            if hasattr(molecule, '_intensity_cache'):
                debug_info['cache_attributes'].append('_intensity_cache')
                debug_info['intensity_cache_size'] = len(molecule._intensity_cache) if molecule._intensity_cache else 0
                if molecule._intensity_cache:
                    debug_info['intensity_cache_keys'] = list(molecule._intensity_cache.keys())
            
            if hasattr(molecule, '_spectrum_cache'):
                debug_info['cache_attributes'].append('_spectrum_cache')
                debug_info['spectrum_cache_size'] = len(molecule._spectrum_cache) if molecule._spectrum_cache else 0
                if molecule._spectrum_cache:
                    debug_info['spectrum_cache_keys'] = list(molecule._spectrum_cache.keys())
            
            # Check current parameter values
            debug_info['current_parameters'] = {}
            for param in ['temp', 'radius', 'n_mol', 'distance', 'fwhm', 'broad']:
                if hasattr(molecule, param):
                    debug_info['current_parameters'][param] = getattr(molecule, param)
            
            # Check if molecule reports its cache as valid
            debug_info['cache_validity'] = {}
            if hasattr(molecule, 'is_cache_valid'):
                try:
                    debug_info['cache_validity']['spectrum'] = molecule.is_cache_valid('spectrum')
                    debug_info['cache_validity']['intensity'] = molecule.is_cache_valid('intensity')
                    debug_info['cache_validity']['full'] = molecule.is_cache_valid('full')
                except Exception as e:
                    debug_info['cache_validity']['error'] = str(e)
            
            return debug_info
            
        except Exception as e:
            return {'error': f"Error debugging molecule cache: {e}"}
    
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
        
        # Optionally trigger a plot refresh if plots are currently visible
        # This could be enhanced to automatically refresh active plots
    
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
    
    def get_threshold_debug_info(self, line_data: List[Tuple['MoleculeLine', float, Optional[float]]]) -> Dict[str, Any]:
        """
        Get debugging information about threshold filtering.
        
        Parameters
        ----------
        line_data : List[Tuple]
            List of (MoleculeLine, intensity, tau) tuples
            
        Returns
        -------
        Dict[str, Any]
            Debug information about threshold filtering
        """
        if not line_data:
            return {'total_lines': 0, 'filtered_lines': 0, 'threshold_percent': 0}
            
        threshold_percent = self.get_line_intensity_threshold()
        filtered_lines = self.filter_lines_by_threshold(line_data, threshold_percent)
        
        intensities = [intensity for _, intensity, _ in line_data]
        max_intensity = max(intensities) if intensities else 0
        min_intensity = min(intensities) if intensities else 0
        threshold_intensity = max_intensity * threshold_percent if max_intensity > 0 else 0
        
        return {
            'total_lines': len(line_data),
            'filtered_lines': len(filtered_lines),
            'threshold_percent': threshold_percent * 100,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'threshold_intensity': threshold_intensity,
            'lines_below_threshold': len(line_data) - len(filtered_lines)
        }