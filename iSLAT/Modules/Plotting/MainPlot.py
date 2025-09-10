import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

import iSLAT.Constants as c

from .PlotRenderer import PlotRenderer
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine

# Import debug configuration with fallback
try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:
    # Fallback to a simple debug class for compatibility
    class FallbackDebugConfig:
        def verbose(self, component, message, **kwargs):
            pass
        def info(self, component, message, **kwargs):
            pass
        def warning(self, component, message, **kwargs):
            print(f"[{component.upper()}] WARNING: {message}")
        def error(self, component, message, **kwargs):
            print(f"[{component.upper()}] ERROR: {message}")
        def trace(self, component, message, **kwargs):
            pass
    debug_config = FallbackDebugConfig()
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer

class iSLATPlot:
    """
    Main plotting class for iSLAT spectroscopy tool.
    
    This class coordinates between specialized modules to provide comprehensive
    plotting functionality including spectrum visualization, line analysis,
    population diagrams, and interactive features.
    
    Architecture:
    - PlotRenderer: Handles matplotlib rendering and visual updates
    - InteractionHandler: Processes mouse/keyboard interactions
    - FittingEngine: Handles line fitting operations
    - LineAnalyzer: Provides line detection and analysis capabilities
    
    Data Sources:
    - Uses MoleculeLine objects for line data access
    - Leverages Spectrum and Intensity classes for computation
    - Integrates with MoleculeDict for parameter management
    """
    def __init__(self, parent_frame, wave_data, flux_data, theme, islat_class_ref):
        self.theme = theme
        self.islat = islat_class_ref

        self.active_lines = []  # List of (line, text, scatter, values) tuples for active molecular lines

        self.fig = plt.Figure(figsize=(10, 7))
        # Adjust subplot parameters to minimize margins and maximize plot area
        self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.15, hspace=0.25)
        gs = GridSpec(2, 2, height_ratios=[2, 3], figure=self.fig)
        self.ax1 = self.full_spectrum = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.line_inspection = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.population_diagram = self.fig.add_subplot(gs[1, 1])

        self.ax1.set_title("Full Spectrum with Line Inspection")
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax2.set_title("Line inspection plot")
        self.ax3.set_title(f"{self.islat.active_molecule.displaylabel} Population diagram")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        
        # Apply theme to matplotlib figure and toolbar
        self._apply_plot_theming()

        # Initialize the modular classes
        self.plot_renderer = PlotRenderer(self)
        self.interaction_handler = InteractionHandler(self)
        self.fitting_engine = FittingEngine(self.islat)
        self.line_analyzer = LineAnalyzer(self.islat)

        # Set up interaction handler callbacks
        self.interaction_handler.set_span_select_callback(self.onselect)
        self.interaction_handler.set_click_callback(self.on_click)
        
        # self.toolbar.pack(side="top", fill="x", padx=0, pady=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)
            
        self.canvas.draw()

        self.selected_wave = None
        self.selected_flux = None
        self.fit_result = None

        # Initial data and model computation using new data structures
        self.summed_flux = np.array([])
        
        # Register callbacks for parameter and molecule changes
        self._register_update_callbacks()
        
        # Set initial zoom range to display_range if available
        self._set_initial_zoom_range()

    def create_toolbar(self, frame):
        self.toolbar = NavigationToolbar2Tk(self.canvas, window = frame)
        return self.toolbar
    
    def toggle_legend(self):
        if self.ax1.legend_ is None:
            self.ax1.legend()
        else:
            self.ax1.legend_.remove()
        self.canvas.draw_idle()

    def _apply_plot_theming(self):
        """Apply theme colors to matplotlib figure and toolbar"""
        try:
            # Set figure background color
            self.fig.patch.set_facecolor(self.theme.get("background", "#181A1B"))
            
            # Set axes background colors and text colors
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_facecolor(self.theme.get("graph_fill_color", "#282C34"))
                ax.tick_params(colors=self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")), which='both')
                ax.xaxis.label.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                ax.yaxis.label.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                ax.title.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))
                
                # Set spine colors
                for spine in ax.spines.values():
                    spine.set_color(self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")))   

                if self.theme.get(f'ax{ax.get_gid()}_grid', False):
                    ax.grid(True, color=self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")), alpha=0.3, linestyle='-', linewidth=0.5)
                    
            # Apply theme to canvas
            if hasattr(self.canvas.get_tk_widget(), 'configure'):
                try:
                    self.canvas.get_tk_widget().configure(bg=self.theme.get("background", "#181A1B"))
                except:
                    pass
                    
        except Exception as e:
            debug_config.error("main_plot", f"Could not apply plot theming: {e}")
    
    def _get_molecule_display_name(self, molecule):
        """Get display name for a molecule"""
        return getattr(molecule, 'displaylabel', getattr(molecule, 'name', 'unknown'))

    def _register_update_callbacks(self):
        """Register callbacks to handle parameter and molecule changes"""
        Molecule.add_molecule_parameter_change_callback(self.on_molecule_parameter_changed)
        
        # Register for global parameter changes if molecules_dict exists
        if hasattr(self.islat, 'molecules_dict'):
            self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_changed)

        # Register for active molecule changes
        self.islat.add_active_molecule_change_callback(self._on_active_molecule_changed)
    
    def _on_active_molecule_changed(self, old_molecule, new_molecule):
        """Handle active molecule changes"""
        self.on_active_molecule_changed()
    
    def _on_global_parameter_changed(self, parameter_name, old_value, new_value):
        """Handle global parameter changes that affect all molecules"""
        # Refresh plots when global parameters change - molecules handle their own caching
        self.update_all_plots()

    def match_display_range(self, match_y=False):
        # Sync plot xlim to islat.display_range if set, else update islat.display_range from plot
        if hasattr(self.islat, 'display_range'):
            # If display_range is set elsewhere, update plot xlim
            if self.islat.display_range:
                wmin, wmax = self.islat.display_range
                current_xlim = self.ax1.get_xlim()
                
                # Only update plot xlim if it's actually different (prevent infinite loops)
                if (abs(current_xlim[0] - wmin) > 1e-10 or 
                    abs(current_xlim[1] - wmax) > 1e-10):
                    self.ax1.set_xlim(wmin, wmax)
            else:
                # If not set, initialize from current plot xlim
                self.islat.display_range = tuple(self.ax1.get_xlim())
        else:
            # If islat has no display_range attribute, do nothing
            return

        # Adjust y-limits
        wmin, wmax = self.ax1.get_xlim()
        mask = (self.islat.wave_data >= wmin) & (self.islat.wave_data <= wmax)
        range_flux_cnts = self.islat.flux_data[mask]
        if range_flux_cnts.size == 0:
            fig_height = np.nanmax(self.islat.flux_data)
            fig_bottom_height = 0
        else:
            fig_height = np.nanmax(range_flux_cnts)
            fig_bottom_height = np.nanmin(range_flux_cnts)
        
        if match_y:
            self.ax1.set_ylim(ymin=fig_bottom_height, ymax=fig_height + (fig_height / 8))

        self.canvas.draw_idle()

    def _set_initial_zoom_range(self):
        """Set the initial zoom range based on display_range or data range"""
        # Use display_range if available
        if hasattr(self.islat, 'display_range') and self.islat.display_range:
            xmin, xmax = self.islat.display_range
            self.ax1.set_xlim(xmin, xmax)
        elif hasattr(self.islat, 'wave_data') and self.islat.wave_data is not None:
            # Fallback to full data range
            self.ax1.set_xlim(self.islat.wave_data.min(), self.islat.wave_data.max())
        
        # Update display to match and set optimal y-limits
        self.match_display_range(match_y=True)
        self.canvas.draw_idle()

    def make_span_selector(self):
        """Creates a SpanSelector for the main plot to select a region for line inspection."""
        self.span = self.interaction_handler.create_span_selector(self.ax1, self.theme["selection_color"])

    def update_all_plots(self):
        """
        Updates all plots in the GUI.
        This method leverages the molecular data model for updates and avoids redundant rendering.
        """    
        self.update_model_plot()
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        self.plot_spectrum_around_line()

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.
        Uses molecules' built-in caching and hashing for optimal performance.
        """
        if not hasattr(self.islat, 'molecules_dict') or len(self.islat.molecules_dict) == 0:
            self.plot_renderer.clear_model_lines()
            self.canvas.draw_idle()
            return
        
        wave_data = self._get_model_wavelength_grid()
        
        try:
            if hasattr(self.islat.molecules_dict, 'get_summed_flux'):
                debug_config.trace("main_plot", "Using MoleculeDict.get_summed_flux() for model plot")
                summed_wavelengths, summed_flux = self.islat.molecules_dict.get_summed_flux(wave_data, visible_only=True)
                wave_data = summed_wavelengths  # Use the combined wavelength grid
        except Exception as e:
                #mol_name = self._get_molecule_display_name(molecule)
                debug_config.warning("main_plot", f"Could not get flux form molecule dict: {e}")
    
        wave_data = self.islat.wave_data_original
        wave_data = wave_data - (wave_data / c.SPEED_OF_LIGHT_KMS * self.islat.molecules_dict.global_stellar_rv)
        self.islat.wave_data = wave_data # Update islat wave_data to match adjusted grid

        self.plot_renderer.render_main_spectrum_plot(
            wave_data=wave_data,
            flux_data=self.islat.flux_data,
            molecules=self.islat.molecules_dict,
            summed_wavelengths=summed_wavelengths,
            summed_flux=summed_flux,
            error_data=getattr(self.islat, 'err_data', None),
            #observed_wave_data=self.islat.wave_data  # Pass observed data separately
        )
        
        # Recreate span selector and redraw
        self.make_span_selector()
        self.canvas.draw_idle()

    def onselect(self, xmin, xmax):
        self.current_selection = (xmin, xmax)
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.selected_wave = self.islat.wave_data[mask]
        self.selected_flux = self.islat.flux_data[mask]

        # if xmax - xmin < 0.025:
        #     self.ax2.clear()
        #     self.current_selection = None
        #     self.canvas.draw_idle()
        #     return

        self.plot_spectrum_around_line(
            xmin=xmin,
            xmax=xmax
        )

    def plot_spectrum_around_line(self, xmin=None, xmax=None, highlight_strongest=True):
        """
        Plot spectrum around selected lines using molecule's built-in caching.
        
        This method leverages the active molecule's intensity and line caching
        to avoid redundant calculations.
        """
        debug_config.verbose("line_inspection", f"plot_spectrum_around_line called", 
                           xmin=xmin, xmax=xmax, highlight_strongest=highlight_strongest)

        if xmin is None or xmax is None:
            # If no selection but we need to update population diagram due to molecule/parameter changes
            debug_config.verbose("line_inspection", "No selection, updating population diagram only")
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.plot_renderer.ax2.clear()
            self.current_selection = None
            self.canvas.draw_idle()
            return

        debug_config.trace("line_inspection", f"Processing selection: {xmin:.3f} - {xmax:.3f}")
        
        # Get lines in range using the molecule's built-in line caching
        try:
            # Use PlotRenderer which leverages molecule's internal caching
            line_data = self.plot_renderer.get_molecule_line_data(self.islat.active_molecule, xmin, xmax)
            if not line_data:
                self.islat.GUI.data_field.insert_text(f"No transitions found for {self.islat.active_molecule.name} in the selected range")
                # Clear active lines and update population diagram even if no lines in range
                debug_config.verbose("line_inspection", "No lines in range, clearing active lines")
                self.clear_active_lines()
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.canvas.draw_idle()
                return
        except Exception as e:
            debug_config.warning("main_plot", f"Could not get line data: {e}")
            # Clear active lines and update population diagram even if no lines in range
            debug_config.verbose("line_inspection", "Error getting line data, clearing active lines")
            self.clear_active_lines()
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()
            return

        # Clear previous active_lines before plotting
        debug_config.trace("line_inspection", f"Found {len(line_data)} lines, plotting line inspection and population diagram")
        self.clear_active_lines()
        self.plot_line_inspection(xmin, xmax, line_data, highlight_strongest=highlight_strongest)
        self.plot_population_diagram(line_data)
        self.canvas.mpl_connect('pick_event', self.on_pick_line)
        debug_config.verbose("line_inspection", "plot_spectrum_around_line completed")
    
    def on_pick_line(self, event):
        """Handle line pick events by delegating to PlotRenderer."""
        picked_value = self.plot_renderer.handle_line_pick_event(event.artist, self.active_lines)
        if picked_value:
            self.selected_line = picked_value
            self._display_line_info(picked_value)
        self.canvas.draw_idle()

    def highlight_strongest_line(self):
        """
        Highlight the strongest line by delegating to PlotRenderer.
        """
        strongest = self.plot_renderer.highlight_strongest_line(self.active_lines)
        if strongest is not None:
            # Display strongest line information in data field
            line, text, scatter, value = strongest
            self.selected_line = value
            if value:
                self._display_line_info(value)
        
        self.canvas.draw_idle()

    def _display_line_info(self, value, clear_data_field=True):
        """
        Helper method to display line information in the data field.
        """
        # Calculate flux integral in selected range
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            # Calculate flux integral
            err_data = getattr(self.islat, 'err_data', None)
            line_flux, line_err = self.flux_integral(
                lam=self.islat.wave_data, 
                flux=self.islat.flux_data, 
                lam_min=xmin, 
                lam_max=xmax,
                err=err_data
            )
            molecule_wave, molecule_flux = self.islat.active_molecule.get_flux(return_wavelengths=True)
            molecule_flux_in_range, _ = self.flux_integral(
                lam=molecule_wave, 
                flux=molecule_flux, 
                lam_min=xmin, 
                lam_max=xmax,
                err=None
            )
        else:
            line_flux = [0.0]
        
        # Extract line information
        lam = value.get('lam', None)
        e_up = value.get('e', None)
        a_stein = value.get('a', None)
        g_up = value.get('g', None)
        inten = value.get('inten', None)
        up_lev = value.get('up_lev', 'N/A')
        low_lev = value.get('low_lev', 'N/A')
        tau_val = value.get('tau', 'N/A')
        
        # Format values to match original output
        wavelength_str = f"{lam:.6f}" if lam is not None else 'N/A'
        einstein_str = f"{a_stein:.3e}" if a_stein is not None else 'N/A'
        energy_str = f"{e_up:.0f}" if e_up is not None else 'N/A'
        tau_str = f"{tau_val:.3f}" if isinstance(tau_val, (float, int)) else str(tau_val)
        flux_str = f"{line_flux[0]:.3e}" if isinstance(line_flux, (list, tuple)) and len(line_flux) > 0 else f"{line_flux:.3e}"

        # Display line information in the original format
        info_str = (
            "\n--- Line Information ---\n"
            "Selected line:\n"
            f"Upper level = {up_lev}\n"
            f"Lower level = {low_lev}\n"
            f"Wavelength (μm) = {wavelength_str}\n"
            f"Einstein-A coeff. (1/s) = {einstein_str}\n"
            f"Upper level energy (K) = {energy_str}\n"
            f"Opacity = {tau_str}\n"
            f"Data flux in sel. range (erg/s/cm2) = {flux_str}\n"
            f"Model flux in sel. range (erg/s/cm2) = {molecule_flux_in_range:.3e}\n"
        )
        
        # Add the information without clearing the data field, with error protection
        if (hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field') and
            self.islat.GUI.data_field is not None):
            try:
                # Check if the widget still exists before accessing it
                if hasattr(self.islat.GUI.data_field, 'text') and self.islat.GUI.data_field.text.winfo_exists():
                    self.islat.GUI.data_field.insert_text(info_str, clear_after=clear_data_field)
            except Exception as e:
                # Silently ignore GUI access errors during initialization
                print(f"Warning: Could not update data field: {e}")
                pass

    def clear_selection(self):
        self.current_selection = None
        self.ax2.clear()
        self.canvas.draw_idle()
        return

    def plot_line_inspection(self, xmin=None, xmax=None, line_data=None, highlight_strongest=True):
        if xmin is None or xmax is None:
            self.plot_renderer.ax2.clear()
            self.current_selection = None
            self.canvas.draw_idle()
            return
        
        # Get line data using the molecular line API
        if line_data is None:
            try:
                line_data = self.plot_renderer.get_molecule_line_data(self.islat.active_molecule, xmin, xmax)
                if not line_data:
                    self.ax2.clear()
                    self.current_selection = None
                    self.canvas.draw_idle()
                    return
            except Exception as e:
                debug_config.warning("main_plot", f"Could not get line data: {e}")
                self.ax2.clear()
                self.current_selection = None
                self.canvas.draw_idle()
                return

        # First update the basic line inspection plot
        self.update_line_inspection_plot(xmin=xmin, xmax=xmax)
        
        # Get the max y value for scaling line heights
        data_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        data_region_y = self.islat.flux_data[data_mask]
        max_y = np.nanmax(data_region_y) if len(data_region_y) > 0 else (self.plot_renderer.ax2.get_ylim()[1] / 1.1) #returns ymin, ymax
            
        # Clear and add vertical lines using PlotRenderer
        self.clear_active_lines()
        self.plot_renderer.render_active_lines_in_line_inspection(line_data, self.active_lines, max_y)

        self.canvas.draw_idle()

        # Highlight the strongest line
        if highlight_strongest:
            self.highlight_strongest_line()

    def plot_population_diagram(self, line_data):
        """
        Plot population diagram for the currently active lines in the selected region.
        Uses the MoleculeLine-based data structure.
        
        Parameters
        ----------
        line_data : list
            List of (MoleculeLine, intensity, tau) tuples
        """
        # First update the base population diagram with current molecule parameters
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        
        # Add active line scatter points
        if line_data:
            self.plot_renderer.render_active_lines_in_population_diagram(line_data, self.active_lines)
        
        self.canvas.draw_idle()
        self.highlight_strongest_line()

    def update_line_inspection_plot(self, xmin=None, xmax=None):
        """
        Update the line inspection plot showing data and active molecule model in the selected range.
        Uses only PlotRenderer logic with molecule's built-in caching for optimal performance.
        """
        # Delegate all rendering logic to PlotRenderer
        fit_result = getattr(self, 'fit_result', None)
        if hasattr(self, 'old_fit_result'):
            if fit_result == self.old_fit_result:
                self.old_fit_result = fit_result
                fit_result = None  # Avoid re-rendering if fit result hasn't changed
        else:
            self.old_fit_result = fit_result
        self.plot_renderer.render_complete_line_inspection_plot(
            wave_data=self.islat.wave_data,
            flux_data=self.islat.flux_data,
            xmin=xmin,
            xmax=xmax,
            active_molecule=self.islat.active_molecule,
            fit_result=fit_result
        )

        self.canvas.draw_idle()

    def toggle_legend(self):
        ax1_leg = self.ax1.get_legend()
        ax2_leg = self.ax2.get_legend()
        if ax1_leg is not None:
            vis = not ax1_leg.get_visible()
            ax1_leg.set_visible(vis)
        if ax2_leg is not None:
            vis = not ax2_leg.get_visible()
            ax2_leg.set_visible(vis)
        self.canvas.draw_idle()

    def flux_integral(self, lam, flux, err, lam_min, lam_max):
        """
        Calculate flux integral in the selected wavelength range.
        
        Parameters
        ----------
        lam : array
            Wavelength array in microns
        flux : array  
            Flux array in Jy
        err : array
            Error array in Jy
        lam_min : float
            Minimum wavelength in microns
        lam_max : float
            Maximum wavelength in microns
            
        Returns
        -------
        tuple
            (line_flux_meas, line_err_meas) in erg/s/cm^2
        """
        return self.line_analyzer.flux_integral(lam, flux, err, lam_min, lam_max)

    def clear_active_lines(self) -> None:
        """
        Clear active lines by delegating to PlotRenderer.
        """
        self.plot_renderer.clear_active_lines(self.active_lines)

    def clear_model_lines(self):
        """
        Clear model spectrum lines from the main plot.
        Delegates to PlotRenderer for efficient line management.
        """
        self.plot_renderer.clear_model_lines()
    
    def clear_all_plots(self):
        """
        Clear all plots and reset visual state.
        Delegates to PlotRenderer for comprehensive plot clearing.
        """
        self.plot_renderer.clear_all_plots()
        self.canvas.draw_idle()

    def highlight_line_selection(self, xmin, xmax):
        """
        Highlight a selected wavelength range.
        Delegates to PlotRenderer for visual highlighting.
        """
        self.plot_renderer.highlight_line_selection(xmin, xmax)
        self.canvas.draw_idle()
    
    def plot_vertical_lines(self, wavelengths, heights=None, colors=None, labels=None):
        """
        Plot vertical lines at specified wavelengths.
        Delegates to PlotRenderer for efficient line plotting.
        """
        self.plot_renderer.plot_vertical_lines(wavelengths, heights, colors, labels)
        self.canvas.draw_idle()

    def on_click(self, event):
        """Handle mouse click events on the plot."""
        self.interaction_handler.handle_click_event(event)
    
    def on_active_molecule_changed(self):
        """
        Called when the active molecule changes.
        Updates plot titles and refreshes displays with current selection if available.
        """
        debug_config.info("active_molecule", "on_active_molecule_changed() called")
        
        # Update the population diagram title
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            self.ax3.set_title(f'{self.islat.active_molecule.displaylabel} Population diagram')
            debug_config.verbose("active_molecule", f"Set title for molecule: {self.islat.active_molecule.displaylabel}")
        
        # Clear active lines since they belong to the previous molecule
        debug_config.verbose("active_molecule", "Clearing active lines")
        self.clear_active_lines()
        
        # If we have a current selection, refresh the line inspection and population diagram
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            debug_config.verbose("active_molecule", f"Refreshing line inspection for selection: {xmin:.3f} - {xmax:.3f}")
            self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
        else:
            # Just update the population diagram without active lines
            debug_config.verbose("active_molecule", "Updating population diagram only")
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()
        
        debug_config.info("active_molecule", "on_active_molecule_changed() completed")

    def on_molecule_parameter_changed(self, molecule_name, parameter_name, old_value, new_value):
        """
        Called when any molecule parameter changes.
        """
        debug_config.info("main_plot", f"Parameter change: {molecule_name}.{parameter_name}: {old_value} → {new_value}")
        
        # Debug cache status before and after parameter change
        if (hasattr(self.islat, 'molecules_dict') and 
            molecule_name in self.islat.molecules_dict):
            
            molecule = self.islat.molecules_dict[molecule_name]
        
        # Check if this molecule is visible - if so, we need to update the main spectrum plot
        if (hasattr(self.islat, 'molecules_dict') and 
            molecule_name in self.islat.molecules_dict):
            
            molecule = self.islat.molecules_dict[molecule_name]
            
            # Only update plots if the molecule is visible
            if molecule.is_visible:
                self.update_model_plot()
        
        # Check if the changed molecule is the active one for additional updates
        if (hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule and 
            hasattr(self.islat.active_molecule, 'name') and
            self.islat.active_molecule.name == molecule_name):
            
            # If we have a current selection, refresh the line inspection and population diagram
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
                self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
            else:
                # Just update the population diagram without active lines
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.canvas.draw_idle()

    def on_molecule_deleted(self, molecule_name):
        """
        Handle molecule deletion by clearing relevant plot elements and updating displays.
        
        Parameters
        ----------
        molecule_name : str
            Name of the deleted molecule
        """
        # Clear model lines first
        self.clear_model_lines()
        
        # Clear active lines if they belong to the deleted molecule
        if (hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule and 
            hasattr(self.islat.active_molecule, 'name') and
            self.islat.active_molecule.name == molecule_name):
            self.clear_active_lines()
        
        # Update all plots to reflect the change
        self.update_all_plots()
    
    def on_molecule_visibility_changed(self, molecule_name, is_visible):
        """
        Handle molecule visibility changes by delegating to PlotRenderer.
        Minimal logic here - PlotRenderer handles all the complexity and caching.
        
        Parameters
        ----------
        molecule_name : str
            Name of the molecule whose visibility changed
        is_visible : bool
            New visibility state
        """
        if not hasattr(self.islat, 'molecules_dict'):
            return
        
        # Get current selection for potential line inspection update
        current_selection = getattr(self, 'current_selection', None)
        active_molecule = getattr(self.islat, 'active_molecule', None)
        
        # Delegate everything to PlotRenderer's comprehensive method
        self.plot_renderer.handle_molecule_visibility_change(
            molecule_name=molecule_name,
            is_visible=is_visible,
            molecules_dict=self.islat.molecules_dict,
            wave_data=self.islat.wave_data,
            active_molecule=active_molecule,
            current_selection=current_selection
        )
        
        # Only canvas update needed in MainPlot
        self.canvas.draw_idle()
    
    def compute_fit_line(self, xmin=None, xmax=None, deblend=False, update_plot=True):
        """
        Compute fit line using FittingEngine with data access.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range. Uses current_selection if not provided.
        deblend : bool
            Whether to perform multi-component deblending
            
        Returns
        -------
        tuple or None
            (fit_result, fitted_wave, fitted_flux) or None if fitting fails
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                return None
        
        # Use vectorized mask for efficient data selection
        fit_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        x_fit = self.islat.wave_data[fit_mask]
        y_fit = self.islat.flux_data[fit_mask]
        
        if len(x_fit) < 5:
            return None
        
        try:
            fit_result, fitted_wave, fitted_flux = self.fitting_engine.fit_gaussian_line(
                x_fit, y_fit, xmin=xmin, xmax=xmax, deblend=deblend
            )
            self.fit_result = fit_result, fitted_wave, fitted_flux
            if update_plot:
                self.plot_renderer._render_fit_results_in_line_inspection(fit_result=self.fit_result, xmin=xmin, xmax=xmax, max_y=fitted_flux.max())
            return self.fit_result
        except Exception as e:
            debug_config.error("main_plot", f"Error in fitting: {str(e)}")
            return None
    
    def save_fig(self, filename, dpi=10):
        """Save the current figure to a file."""
        try:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            debug_config.info("main_plot", f"Figure saved to {filename}")
        except Exception as e:
            debug_config.error("main_plot", f"Error saving figure: {e}")

    def find_single_lines(self, xmin=None, xmax=None):
        """
        Find single lines using LineAnalyzer with data processing.
        
        Parameters
        ----------
        xmin, xmax : float, optional
            Wavelength range. Uses current_selection if not provided.
            
        Returns
        -------
        list
            List of detected lines with properties
        """
        if xmin is None or xmax is None:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
            else:
                return []
        
        # Efficient data selection using vectorized operations
        range_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        range_wave = self.islat.wave_data[range_mask]
        range_flux = self.islat.flux_data[range_mask]
        
        if len(range_wave) < 10:
            return []
        
        try:
            detected_lines = self.line_analyzer.find_single_lines(
                range_wave, range_flux
            )
            
            # Create optimized line data structure
            self.single_lines_list = []
            ylim = self.ax1.get_ylim()
            
            for line in detected_lines:
                line_info = {
                    "wavelength": line['wavelength'], 
                    "ylim": ylim,
                    #"strength": line['line_strength'],
                    #"type": line['type']
                }
                self.single_lines_list.append(line_info)
            
            return self.single_lines_list
        except Exception as e:
            debug_config.error("main_plot", f"Error in line detection: {str(e)}")
            return []
    
    def plot_single_lines(self):
        """Plot single lines on main plot with rendering."""
        self.update_model_plot()
        if not hasattr(self, 'single_lines_list') or not self.single_lines_list:
            return
            
        # Extract wavelengths for batch plotting
        wavelengths = [line['wavelength'] for line in self.single_lines_list]
        # Delegate to PlotRenderer for plotting
        self.plot_renderer.plot_single_lines(wavelengths)
    
    def plot_saved_lines(self, saved_lines):
        """Plot saved lines using PlotRenderer with delegation."""
        self.plot_renderer.plot_saved_lines(saved_lines)

    def apply_theme(self, theme=None):
        """Apply theme to the plot and update colors"""
        if theme:
            self.theme = theme
        self._apply_plot_theming()
        # Refresh the plots to apply colors
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        # Also update any existing plots to use colors
        if hasattr(self, 'update_all_plots'):
                self.update_all_plots()

    def _get_model_wavelength_grid(self):
        """
        Generate wavelength grid for model calculations.
        
        If a global wavelength range is set and extends beyond the observed data,
        create an extended grid. Otherwise, use the observed data wavelength grid.
        
        Returns
        -------
        np.ndarray
            Wavelength grid for model calculations
        """
        # Start with observed data wavelength grid
        obs_wave_data = self.islat.wave_data
        
        # Check if molecules_dict has a global wavelength range set
        if (hasattr(self.islat.molecules_dict, '_global_wavelength_range') and 
            self.islat.molecules_dict._global_wavelength_range is not None):
            
            global_min, global_max = self.islat.molecules_dict._global_wavelength_range
            obs_min, obs_max = obs_wave_data.min(), obs_wave_data.max()
            
            # Check if global range extends beyond observed data
            extends_beyond = global_min < obs_min or global_max > obs_max
            
            if extends_beyond:
                # Create extended wavelength grid
                # Use the same resolution as the observed data
                obs_resolution = np.median(np.diff(obs_wave_data))
                
                # Create extended grid from global range
                n_points = int((global_max - global_min) / obs_resolution) + 1
                extended_wave_data = np.linspace(global_min, global_max, n_points)
                
                debug_config.info("main_plot", 
                    f"Using extended wavelength grid: {global_min:.3f} - {global_max:.3f} µm "
                    f"(vs observed: {obs_min:.3f} - {obs_max:.3f} µm)")
                
                return extended_wave_data
        
        # Use observed data wavelength grid (default behavior)
        return obs_wave_data