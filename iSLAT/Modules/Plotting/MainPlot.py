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

        self.active_lines = []  # List of (line, scatter) tuples for active molecular lines

        self.fig = plt.Figure(figsize=(10, 7))
        # Adjust subplot parameters to minimize margins and maximize plot area
        self.fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.95, wspace=0.15, hspace=0.25)
        gs = GridSpec(2, 2, height_ratios=[2, 3], figure=self.fig)
        self.ax1 = self.full_spectrum = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.line_inspection = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.population_diagram = self.fig.add_subplot(gs[1, 1])

        self.ax1.set_title("Full Spectrum with Line Inspection")
        self.ax2.set_title("Line inspection plot")
        self.ax3.set_title(f"{self.islat.active_molecule.displaylabel} Population diagram")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, window = parent_frame)
        
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
        
        self.toolbar.pack(side="top", fill="x", padx=0, pady=0)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)
        
        # Apply tight layout to maximize space usage
        try:
            self.fig.tight_layout(pad=0.5)
        except:
            pass  # Fallback if tight_layout fails
            
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
                    #print(f"Applying grid to ax{ax.get_gid()}")
                    ax.grid(True, color=self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")), alpha=0.3, linestyle='-', linewidth=0.5)
                # Set grid colors if grid is enabled
                #ax.grid(True, color=self.theme.get("axis_text_label_color", self.theme.get("foreground", "#F0F0F0")), alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Apply theme to toolbar if possible
            if hasattr(self.toolbar, 'configure'):
                try:
                    self.toolbar.configure(bg=self.theme.get("toolbar", "#23272A"))
                except:
                    pass
            
            # Try to style toolbar buttons
            if hasattr(self.toolbar, 'winfo_children'):
                for child in self.toolbar.winfo_children():
                    try:
                        if hasattr(child, 'configure'):
                            child.configure(
                                bg=self.theme.get("toolbar", "#23272A"),
                                fg=self.theme.get("foreground", "#F0F0F0"),
                                activebackground=self.theme.get("selection_color", "#00FF99"),
                                activeforeground=self.theme.get("foreground", "#F0F0F0")
                            )
                    except:
                        pass
                    
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

    def match_display_range(self):
        # Sync plot xlim to islat.display_range if set, else update islat.display_range from plot
        if hasattr(self.islat, 'display_range'):
            # If display_range is set elsewhere, update plot xlim
            if self.islat.display_range:
                wmin, wmax = self.islat.display_range
                self.ax1.set_xlim(wmin, wmax)
            else:
                # If not set, initialize from current plot xlim
                self.islat.display_range = tuple(self.ax1.get_xlim())
        else:
            # If islat has no display_range attribute, do nothing
            return

        # Connect callback to update islat.display_range when user changes xlim
        def on_xlim_changed(ax):
            # Only update if changed by user (not programmatically)
            new_xlim = list(ax.get_xlim())
            if self.islat.display_range != new_xlim:
                self.islat.display_range = new_xlim

        # Avoid multiple connections
        if not hasattr(self, '_xlim_callback_connected'):
            self.ax1.callbacks.connect('xlim_changed', on_xlim_changed)
            self._xlim_callback_connected = True

        # Adjust y-limits as before
        wmin, wmax = self.ax1.get_xlim()
        mask = (self.islat.wave_data >= wmin) & (self.islat.wave_data <= wmax)
        range_flux_cnts = self.islat.flux_data[mask]
        if range_flux_cnts.size == 0:
            fig_height = np.nanmax(self.islat.flux_data)
            fig_bottom_height = 0
        else:
            fig_height = np.nanmax(range_flux_cnts)
            fig_bottom_height = np.nanmin(range_flux_cnts)
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
        self.match_display_range()
        self.canvas.draw_idle()

    def make_span_selector(self):
        """Creates a SpanSelector for the main plot to select a region for line inspection."""
        self.span = self.interaction_handler.create_span_selector(self.ax1, self.theme["selection_color"])

    def update_all_plots(self):
        """
        Updates all plots in the GUI.
        This method leverages the molecular data model for updates and avoids redundant rendering.
        """
        # Check if we should defer rendering
        if getattr(self.islat, '_defer_spectrum_rendering', False):
            return
            
        # Check if a batch update is in progress to avoid redundant calls
        if getattr(self.islat, '_batch_update_in_progress', False):
            return
            
        self.update_model_plot()
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        self.plot_spectrum_around_line()

    def update_model_plot(self):
        """
        Updates the main spectrum plot with observed data, model spectra, and summed flux.
        Uses molecules' built-in caching and hashing for optimal performance.
        """
        if not hasattr(self.islat, 'molecules_dict') or len(self.islat.molecules_dict) == 0:
            # No molecules to plot, just clear and return
            self.plot_renderer.clear_model_lines()
            self.canvas.draw_idle()
            return
            
        '''# Get visible molecules - use their own visibility property
        visible_molecules = [mol for mol in self.islat.molecules_dict.values() 
                           if self._convert_visibility_to_bool(mol.is_visible)]'''
        
        # Calculate summed flux using MoleculeDict's advanced caching system
        try:
            if hasattr(self.islat.molecules_dict, 'get_summed_flux_optimized'):
                # Use MoleculeDict's optimized summed flux calculation with advanced caching
                debug_config.trace("main_plot", "Using MoleculeDict.get_summed_flux_optimized() for model plot")
                summed_flux = self.islat.molecules_dict.get_summed_flux_optimized(self.islat.wave_data, visible_only=True)
            elif hasattr(self.islat.molecules_dict, 'get_summed_flux'):
                # Use MoleculeDict's standard summed flux calculation with caching
                debug_config.trace("main_plot", "Using MoleculeDict.get_summed_flux() for model plot")
                summed_flux = self.islat.molecules_dict.get_summed_flux(self.islat.wave_data, visible_only=True)
        except Exception as e:
                #mol_name = self._get_molecule_display_name(molecule)
                debug_config.warning("main_plot", f"Could not get flux form molecule dict: {e}")
    
        # Delegate rendering to PlotRenderer for clean separation of concerns
        self.plot_renderer.render_main_spectrum_plot(
            self.islat.wave_data,
            self.islat.flux_data,
            molecules=self.islat.molecules_dict,
            summed_flux=summed_flux,
            error_data=getattr(self.islat, 'err_data', None)
        )
        
        # Recreate span selector and redraw
        self.make_span_selector()
        self.canvas.draw_idle()

    def onselect(self, xmin, xmax):
        self.current_selection = (xmin, xmax)
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.selected_wave = self.islat.wave_data[mask]
        self.selected_flux = self.islat.flux_data[mask]
        self.islat.selected_wave = self.selected_wave
        self.islat.selected_flux = self.selected_flux
        self.last_xmin = xmin
        self.last_xmax = xmax

        if len(self.selected_wave) < 5:
            self.ax2.clear()
            self.canvas.draw_idle()
            return

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
        
        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None

        if xmin is None or xmax is None:
            # If no selection but we need to update population diagram due to molecule/parameter changes
            debug_config.verbose("line_inspection", "No selection, updating population diagram only")
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.canvas.draw_idle()
            return

        debug_config.trace("line_inspection", f"Processing selection: {xmin:.3f} - {xmax:.3f}")
        
        # Get lines in range using the molecule's built-in line caching
        try:
            # Use PlotRenderer which leverages molecule's internal caching
            line_data = self.plot_renderer.get_molecule_line_data(self.islat.active_molecule, xmin, xmax)
            if not line_data:
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
            self._display_line_info(picked_value)
        self.canvas.draw_idle()

    def flux_integral_basic(self, wave_data, flux_data, err_data, xmin, xmax):
        """
        Calculate the flux integral in a given wavelength range.
        
        Parameters:
        -----------
        wave_data : array
            Wavelength array
        flux_data : array
            Flux array
        err_data : array or None
            Error array (optional)
        xmin, xmax : float
            Wavelength range
            
        Returns:
        --------
        line_flux : float
            Integrated flux
        line_err : float
            Error on integrated flux
        """
        mask = (wave_data >= xmin) & (wave_data <= xmax)
        if not np.any(mask):
            return 0.0, 0.0
            
        wave_region = wave_data[mask]
        flux_region = flux_data[mask]
        
        if len(wave_region) < 2:
            return 0.0, 0.0
            
        # Integrate using trapezoidal rule
        line_flux = np.trapz(flux_region, wave_region)
        
        # Calculate error if available
        if err_data is not None:
            err_region = err_data[mask]
            # Simple error propagation for integration
            line_err = np.sqrt(np.sum(err_region**2)) * (wave_region[-1] - wave_region[0]) / len(wave_region)
        else:
            line_err = 0.0
            
        return line_flux, line_err

    def highlight_strongest_line(self):
        """
        Highlight the strongest line by delegating to PlotRenderer.
        """
        strongest = self.plot_renderer.highlight_strongest_line(self.active_lines)
        if strongest is not None:
            # Display strongest line information in data field
            line, scatter, value = strongest
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
            line_flux, line_err = self.flux_integral_basic(
                self.islat.wave_data, 
                self.islat.flux_data, 
                err_data, 
                xmin, 
                xmax
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
            f"Flux in sel. range (erg/s/cm2) = {flux_str}\n"
        )
        
        # Add the information without clearing the data field, with error protection
        if (hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field') and
            self.islat.GUI.data_field is not None):
            try:
                # Check if the widget still exists before accessing it
                if hasattr(self.islat.GUI.data_field, 'text') and self.islat.GUI.data_field.text.winfo_exists():
                    self.islat.GUI.data_field.insert_text(info_str, clear_first=clear_data_field)
            except Exception as e:
                # Silently ignore GUI access errors during initialization
                print(f"Warning: Could not update data field: {e}")
                pass

    def plot_line_inspection(self, xmin=None, xmax=None, line_data=None, highlight_strongest=True):
        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None
        
        if xmin is None or xmax is None:
            self.canvas.draw_idle()
            return
        
        # Get line data using the molecular line API
        if line_data is None:
            try:
                line_data = self.plot_renderer.get_molecule_line_data(self.islat.active_molecule, xmin, xmax)
                if not line_data:
                    self.ax2.clear()
                    self.canvas.draw_idle()
                    return
            except Exception as e:
                debug_config.warning("main_plot", f"Could not get line data: {e}")
                self.ax2.clear()
                self.canvas.draw_idle()
                return

        # First update the basic line inspection plot
        self.update_line_inspection_plot(xmin=xmin, xmax=xmax)
        
        # Get the max y value for scaling line heights
        data_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        data_region_y = self.islat.flux_data[data_mask]
        max_y = np.nanmax(data_region_y) if len(data_region_y) > 0 else 1.0

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
        if xmin is None:
            xmin = self.last_xmin if hasattr(self, 'last_xmin') else None
        if xmax is None:
            xmax = self.last_xmax if hasattr(self, 'last_xmax') else None

        if xmin is None or xmax is None or (xmax - xmin) < 0.0001:
            self.plot_renderer.ax2.clear()
            self.canvas.draw_idle()
            return

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
        # Use vectorized operations for efficiency
        wavelength_mask = (lam >= lam_min) & (lam <= lam_max)
        
        if not np.any(wavelength_mask):
            return 0.0, 0.0
            
        lam_range = lam[wavelength_mask]
        flux_range = flux[wavelength_mask]
        
        if len(lam_range) < 2:
            return 0.0, 0.0
        
        # Convert to frequency space for proper integration
        freq_range = c.SPEED_OF_LIGHT_KMS / lam_range
        
        # Integrate in frequency space (reverse order for proper frequency ordering)
        line_flux_meas = np.trapz(flux_range[::-1], x=freq_range[::-1])
        line_flux_meas = -line_flux_meas * 1e-23  # Convert Jy*Hz to erg/s/cm^2
        
        # Calculate error propagation if error data provided
        if err is not None:
            err_range = err[wavelength_mask]
            line_err_meas = np.trapz(err_range[::-1], x=freq_range[::-1])
            line_err_meas = -line_err_meas * 1e-23
        else:
            line_err_meas = 0.0
            
        return line_flux_meas, line_err_meas

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
    
    def optimize_plot_memory(self):
        """
        Optimize memory usage for plotting operations.
        Delegates to PlotRenderer for memory management.
        """
        self.plot_renderer.optimize_plot_memory_usage()
    
    def get_plot_performance_stats(self):
        """
        Get performance statistics for debugging.
        Returns dict with plot performance metrics.
        """
        return self.plot_renderer.get_plot_performance_stats()

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
    
    def update_plot_display(self):
        """
        Update the plot display.
        Delegates to PlotRenderer for display updates.
        """
        self.plot_renderer.update_plot_display()
    
    def force_plot_refresh(self):
        """
        Force a complete plot refresh.
        Delegates to PlotRenderer for comprehensive refresh.
        """
        self.plot_renderer.force_plot_refresh()

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
        Since we rely entirely on molecule caching, we just need to trigger plot updates.
        """
        debug_config.info("main_plot", f"Parameter change: {molecule_name}.{parameter_name}: {old_value} → {new_value}")
        
        # Debug cache status before and after parameter change
        if (hasattr(self.islat, 'molecules_dict') and 
            molecule_name in self.islat.molecules_dict):
            
            molecule = self.islat.molecules_dict[molecule_name]
            
            # Debug molecule cache status
            if hasattr(self.plot_renderer, 'debug_molecule_cache_status'):
                cache_debug = self.plot_renderer.debug_molecule_cache_status(molecule)
                debug_config.trace("main_plot", f"Cache debug for {molecule_name}: {cache_debug}")
        
        # Check if this molecule is visible - if so, we need to update the main spectrum plot
        if (hasattr(self.islat, 'molecules_dict') and 
            molecule_name in self.islat.molecules_dict):
            
            molecule = self.islat.molecules_dict[molecule_name]
            
            # Only update plots if the molecule is visible
            #if self._convert_visibility_to_bool(molecule.is_visible):
            if molecule.is_visible:
                # The molecule's cache has already been invalidated by its parameter setter
                # We just need to trigger a plot update
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

    def _add_molecule_to_plot(self, molecule):
        """Add a single molecule's spectrum using PlotRenderer - minimal logic"""
        if not molecule.is_visible:
            return
        
        # Delegate to PlotRenderer with minimal logic
        return self.plot_renderer.render_individual_molecule_spectrum(
            molecule, self.islat.wave_data
        )

    def _update_summed_spectrum(self):
        """Update summed spectrum by delegating to PlotRenderer - minimal logic"""
        if not hasattr(self.islat, 'molecules_dict'):
            return
        
        # Delegate to PlotRenderer's optimized method that uses advanced caching
        self.plot_renderer._update_summed_spectrum_with_molecules(
            self.islat.molecules_dict, 
            self.islat.wave_data
        )
    
    def batch_update_molecule_colors(self, molecule_color_map):
        """
        Update multiple molecule colors.
        
        Parameters
        ----------
        molecule_color_map : dict
            Dictionary mapping molecule names to colors
        """
        self.plot_renderer.batch_update_molecule_colors(molecule_color_map)
    
    # Convenience methods that delegate to specialized modules
    
    def compute_fit_line(self, xmin=None, xmax=None, deblend=False):
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
            return self.fit_result
        except Exception as e:
            debug_config.error("main_plot", f"Error in fitting: {str(e)}")
            return None
    
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
            detected_lines = self.line_analyzer.detect_lines_automatic(
                range_wave, range_flux, detection_type='both'
            )
            
            # Create optimized line data structure
            self.single_lines_list = []
            ylim = self.ax1.get_ylim()
            
            for line in detected_lines:
                line_info = {
                    "wavelength": line['wavelength'], 
                    "ylim": ylim,
                    "strength": line['line_strength'],
                    "type": line['type']
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
        self.plot_vertical_lines(wavelengths)
    
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
        self._refresh_existing_plots()
    
    def _refresh_existing_plots(self):
        """Refresh existing plots to use theme colors"""
        try:
            # This method can be called to refresh plots after theme changes
            if hasattr(self, 'update_all_plots'):
                self.update_all_plots()
        except:
            pass
    
    '''def on_cached_data_loaded(self, molecules_dict=None):
        """
        Called when cached molecular data is loaded from file.
        
        This method ensures that plots are properly updated when cache history
        is restored, utilizing the existing cached calculations rather than 
        recalculating everything.
        
        Parameters
        ----------
        molecules_dict : dict, optional
            Dictionary of loaded molecules. If None, uses self.islat.molecules_dict
        """
        if molecules_dict is None and hasattr(self.islat, 'molecules_dict'):
            molecules_dict = self.islat.molecules_dict
        
        if not molecules_dict:
            return'''
        
    def on_cached_data_loaded(self, molecules_dict=None):
        """
        Called when cached molecular data is loaded from file.
        
        Since we rely entirely on molecule caching, we just need to refresh
        the plots to display the loaded cached data.
        """
        if molecules_dict is None and hasattr(self.islat, 'molecules_dict'):
            molecules_dict = self.islat.molecules_dict
        
        if not molecules_dict:
            return
        
        debug_config.verbose("main_plot", f"Loading cached data for {len(molecules_dict)} molecules - using molecule cached calculations")
        
        # No cache clearing needed - just refresh plots to show molecule cached data
        self.update_all_plots()
        
        # Update population diagram if we have an active molecule
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            
        # If we have a current selection, refresh the line inspection
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
        
        self.canvas.draw_idle()
        debug_config.verbose("main_plot", "Plot refresh completed - displaying cached molecular data")
        
        # If we have an active molecule, ensure its population diagram is updated
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            
        # If we have a current selection, refresh the line inspection
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
        
        self.canvas.draw_idle()
        debug_config.verbose("main_plot", "Plot refresh completed - should now display cached molecular data")
    
    def validate_molecule_caches(self):
        """
        Validate all molecule caches and trigger updates if needed.
        
        This method checks each molecule's cache validity and updates
        plots only for molecules whose caches have been invalidated.
        """
        if not hasattr(self.islat, 'molecules_dict'):
            return
        
        molecules_needing_update = []
        
        for molecule_name, molecule in self.islat.molecules_dict.items():
            try:
                # Use molecule's built-in cache validation
                if hasattr(molecule, 'is_cache_valid'):
                    if not molecule.is_cache_valid():
                        molecules_needing_update.append(molecule_name)
                elif hasattr(molecule, '_dirty_flags'):
                    # Check if any important flags are dirty
                    if any(molecule._dirty_flags.get(flag, False) 
                          for flag in ['intensity', 'spectrum', 'flux']):
                        molecules_needing_update.append(molecule_name)
                        
            except Exception as e:
                debug_config.warning("main_plot", f"Error checking cache for molecule {molecule_name}: {e}")
                continue
        
        # Update plots only if there are molecules needing updates
        if molecules_needing_update:
            debug_config.info("main_plot", f"Updating plots for molecules with invalid caches: {molecules_needing_update}")
            self.update_all_plots()
        
        return molecules_needing_update
    
    def on_bulk_molecule_parameters_changed(self, parameter_changes):
        """
        Handle bulk parameter changes efficiently.
        
        This method is optimized for cases where multiple molecule parameters
        change at once (e.g., loading a saved state, applying global changes).
        It leverages molecule caching to update only what's necessary.
        
        Parameters
        ----------
        parameter_changes : dict
            Dictionary with structure: {molecule_name: {param_name: new_value, ...}, ...}
        """
        if not parameter_changes:
            return
        
        affected_visible_molecules = []
        active_molecule_affected = False
        
        # Check which molecules actually need visual updates
        for molecule_name, param_dict in parameter_changes.items():
            if (hasattr(self.islat, 'molecules_dict') and 
                molecule_name in self.islat.molecules_dict):
                
                molecule = self.islat.molecules_dict[molecule_name]
                
                # Check if molecule is visible (affects main plot)
                #if self._convert_visibility_to_bool(molecule.is_visible):
                if molecule.is_visible:
                    affected_visible_molecules.append(molecule_name)
                
                # Check if this is the active molecule (affects line inspection and population diagram)
                if (hasattr(self.islat, 'active_molecule') and 
                    self.islat.active_molecule and 
                    hasattr(self.islat.active_molecule, 'name') and
                    self.islat.active_molecule.name == molecule_name):
                    active_molecule_affected = True
        
        # Update main plot only if visible molecules were affected
        if affected_visible_molecules:
            self.update_model_plot()
        
        # Update line inspection and population diagram if active molecule was affected
        if active_molecule_affected:
            if hasattr(self, 'current_selection') and self.current_selection:
                xmin, xmax = self.current_selection
                self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
            else:
                # Just update the population diagram
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.canvas.draw_idle()
    
    def get_molecule_cache_status(self):
        """
        Get cache status for all molecules for debugging purposes.
        
        Returns
        -------
        dict
            Dictionary with cache status information for each molecule
        """
        cache_status = {}
        
        if not hasattr(self.islat, 'molecules_dict'):
            return cache_status
        
        for molecule_name, molecule in self.islat.molecules_dict.items():
            try:
                status = {
                    'name': molecule_name,
                    'visible': molecule.is_visible,
                    'cache_valid': None,
                    'cache_stats': None,
                    'parameter_hash': None
                }
                
                # Get cache validity
                if hasattr(molecule, 'is_cache_valid'):
                    status['cache_valid'] = molecule.is_cache_valid()
                
                # Get cache statistics
                if hasattr(molecule, 'get_cache_stats'):
                    status['cache_stats'] = molecule.get_cache_stats()
                
                # Get current parameter hash
                if hasattr(molecule, 'get_parameter_hash'):
                    status['parameter_hash'] = molecule.get_parameter_hash()
                
                cache_status[molecule_name] = status
                
            except Exception as e:
                cache_status[molecule_name] = {
                    'name': molecule_name,
                    'error': str(e)
                }
        
        return cache_status
    
    def debug_molecule_cache_usage(self):
        """
        Debug method to check if molecules are using their cached data properly.
        
        Returns information about cache state and usage for troubleshooting.
        """
        if not hasattr(self.islat, 'molecules_dict'):
            return "No molecules_dict found"
        
        debug_info = []
        
        for molecule_name, molecule in self.islat.molecules_dict.items():
            try:
                info = {
                    'name': molecule_name,
                    'has_cached_intensity': False,
                    'has_cached_spectrum': False,
                    'has_plot_data': False,
                    'cache_stats': {},
                    'visible': molecule.is_visible
                }
                
                # Check for cached intensity
                if hasattr(molecule, '_intensity_cache') and molecule._intensity_cache.get('data'):
                    info['has_cached_intensity'] = True
                
                # Check for cached spectrum
                if hasattr(molecule, '_spectrum_cache') and molecule._spectrum_cache.get('data'):
                    info['has_cached_spectrum'] = True
                elif hasattr(molecule, 'spectrum') and molecule.spectrum:
                    info['has_cached_spectrum'] = True
                
                # Check for plot data
                if hasattr(molecule, 'plot_lam') and hasattr(molecule, 'plot_flux'):
                    if molecule.plot_lam is not None and molecule.plot_flux is not None:
                        info['has_plot_data'] = True
                
                # Get cache statistics
                if hasattr(molecule, 'get_cache_stats'):
                    info['cache_stats'] = molecule.get_cache_stats()
                
                debug_info.append(info)
                
            except Exception as e:
                debug_info.append({
                    'name': molecule_name,
                    'error': str(e)
                })
        
        return debug_info