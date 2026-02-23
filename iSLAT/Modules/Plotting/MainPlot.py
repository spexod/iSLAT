import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

from typing import Optional, List, Tuple, Any, TYPE_CHECKING

import iSLAT.Constants as c

from .PlotRenderer import PlotRenderer
from .BasePlot import BasePlot
from .PlotView import PlotView
from .ThreePanelView import ThreePanelView
from .FullSpectrumView import FullSpectrumView
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.GUI.InteractionHandler import InteractionHandler
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

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
        
        # Flag to defer data-dependent operations until window is visible
        # This prevents blocking during startup while lazy loading triggers
        self._data_initialized = False

        self.active_lines = []  # List of (line, text, scatter, values) tuples for active molecular lines
        self.atomic_lines = []
        self.saved_lines = []

        # Single source of truth for every overlay toggle.
        # Views read this dict on activate() to reconcile their visual state.
        self.toggle_state: dict = {
            "atomic_lines": False,
            "saved_lines":  False,
            "summed":       True,
            "legend":       True,
            "current_selection": None,   # (xmin, xmax) or None
        }

        #self.fig = plt.Figure(figsize=(15, 8.5))
        self.fig = plt.Figure(constrained_layout=True)
        # Adjust subplot parameters to minimize margins and maximize plot area
        #self.fig.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.96, hspace=0.15, wspace=0.15)
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.5], figure=self.fig)
        self.ax1 = self.full_spectrum = self.fig.add_subplot(gs[0, :])
        self.ax2 = self.line_inspection = self.fig.add_subplot(gs[1, 0])
        self.ax3 = self.population_diagram = self.fig.add_subplot(gs[1, 1])

        self.ax1.set_title("Full Spectrum with Line Inspection")
        self.ax1.set_ylabel('Flux density (Jy)')
        self.ax2.set_title("Line inspection plot")
        # Use placeholder title - will be updated when data is initialized
        self.ax3.set_title("Population diagram")

        self.parent_frame = parent_frame

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        
        # Apply theme to matplotlib figure and toolbar
        #self._apply_plot_theming()

        # Initialize the modular classes
        self.plot_renderer = PlotRenderer(self)
        self.interaction_handler = InteractionHandler(self)
        self.fitting_engine = FittingEngine()
        self.line_analyzer = LineAnalyzer()

        # --- View strategy pattern ---
        # The active_view is the current rendering strategy.
        # ThreePanelView delegates to the existing axes + PlotRenderer.
        # FullSpectrumView provides the self-contained multi-panel full spectrum layout.
        self._three_panel_view: PlotView = ThreePanelView(self)
        self._full_spectrum_view: PlotView = FullSpectrumView(self)
        self.active_view: PlotView = self._three_panel_view
        self.is_full_spectrum: bool = False

        # Molecules whose parameters changed while they were hidden.
        # Consumed by on_molecule_visibility_changed to force a re-render
        # instead of simply toggling the (now stale) artists.
        self._stale_molecules: set = set()

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
        
        # DEFERRED: Don't set initial zoom range here - wait for initialize_data()
        # This prevents triggering molecule calculations before window is visible
        # self._set_initial_zoom_range()
    
    # ------------------------------------------------------------------
    # Backward-compatible properties — read / write the toggle_state dict
    # ------------------------------------------------------------------
    @property
    def atomic_toggle(self) -> bool:
        return self.toggle_state["atomic_lines"]

    @atomic_toggle.setter
    def atomic_toggle(self, value: bool) -> None:
        self.toggle_state["atomic_lines"] = value

    @property
    def line_toggle(self) -> bool:
        return self.toggle_state["saved_lines"]

    @line_toggle.setter
    def line_toggle(self, value: bool) -> None:
        self.toggle_state["saved_lines"] = value

    @property
    def summed_toggle(self) -> bool:
        return self.toggle_state["summed"]

    @summed_toggle.setter
    def summed_toggle(self, value: bool) -> None:
        self.toggle_state["summed"] = value

    @property
    def legend_toggle(self) -> bool:
        return self.toggle_state["legend"]

    @legend_toggle.setter
    def legend_toggle(self, value: bool) -> None:
        self.toggle_state["legend"] = value

    def initialize_data(self):
        """
        Initialize data-dependent plot elements.
        
        Call this AFTER the window is visible to avoid blocking during startup.
        This triggers lazy molecule loading and intensity calculations.
        """
        if self._data_initialized:
            return
        
        # debug print
        #print("Initializing plot data...")
        
        # Update population diagram title with actual molecule name
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            self.ax3.set_title(f"{self.islat.active_molecule.displaylabel} Population diagram")
            # Render the population diagram on startup
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        
        # Set initial zoom range (may trigger flux calculations)
        self._set_initial_zoom_range()
        
        self._data_initialized = True

    def create_toolbar(self, frame):
        self.toolbar = NavigationToolbar2Tk(self.canvas, window = frame)
        return self.toolbar

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
        """Get display name for a molecule (delegates to :class:`BasePlot`)."""
        return BasePlot.get_molecule_display_name(molecule)

    # ------------------------------------------------------------------
    # Backward-compatibility properties for external code that still
    # references full_spectrum_plot / full_spectrum_plot_canvas directly.
    # Now the FullSpectrumView *is* the plot (no inner _fsp wrapper).
    # ------------------------------------------------------------------
    @property
    def full_spectrum_plot(self):
        """Return the FullSpectrumView (replaces the old FullSpectrumPlot)."""
        return self._full_spectrum_view

    @property
    def full_spectrum_plot_canvas(self):
        """Return the full-spectrum canvas if it exists."""
        return self._full_spectrum_view._canvas

    @property
    def line_inspection_plot(self):
        """Access the reusable :class:`LineInspectionPlot` delegate."""
        return self.plot_renderer._line_inspection_plot

    @property
    def population_diagram_plot(self):
        """Access the reusable :class:`PopulationDiagramPlot` delegate."""
        return self.plot_renderer._population_diagram_plot

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
        # For match_spectral_sampling, update plots but preserve line inspection
        if parameter_name == 'match_spectral_sampling':
            # Delegate to active view for model update
            self.active_view.update_model_plot()
            # If there's an active line inspection selection in 3-panel mode, refresh it
            if not self.is_full_spectrum:
                if hasattr(self, 'current_selection') and self.current_selection:
                    xmin, xmax = self.current_selection
                    self.plot_spectrum_around_line(xmin, xmax, highlight_strongest=True)
            return
        
        # For other global parameters, refresh all plots
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

    # ================================
    # Loading Indicator for Async Display
    # ================================
    def show_loading_indicator(self, message="Loading..."):
        """
        Show a loading indicator on the plot while calculations are in progress.
        
        Parameters
        ----------
        message : str
            Message to display on the loading indicator
        """
        self._loading_text = self.ax1.text(
            0.5, 0.5, message,
            transform=self.ax1.transAxes,
            ha='center', va='center',
            fontsize=16, fontweight='bold',
            color='gray', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
        )
        self.canvas.draw_idle()
        # Process pending events to show immediately
        self.canvas.get_tk_widget().update_idletasks()
    
    def hide_loading_indicator(self):
        """Remove the loading indicator from the plot."""
        if hasattr(self, '_loading_text') and self._loading_text is not None:
            try:
                self._loading_text.remove()
            except ValueError:
                pass  # Already removed
            self._loading_text = None
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

        Delegates to the active view so the correct panel layout is refreshed.
        The inactive view is marked stale so it re-renders on the next activate().
        """
        # Always update the active view
        self.active_view.update_model_plot()

        # Mark the *other* view as needing a refresh next time it's activated,
        # instead of doing an expensive silent update now.
        if self.is_full_spectrum:
            self._three_panel_view._needs_refresh = True
        else:
            self._full_spectrum_view._needs_refresh = True

    def onselect(self, xmin, xmax):
        self.current_selection = (xmin, xmax)
        self.toggle_state["current_selection"] = (xmin, xmax)
        # Clear previous fit result — it belongs to the old selection
        self.fit_result = None
        mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        self.selected_wave = self.islat.wave_data[mask]
        self.selected_flux = self.islat.flux_data[mask]

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
            self.clear_active_lines()
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.plot_renderer.ax2.clear()
            self.current_selection = None
            self.canvas.draw_idle()
            return

        debug_config.trace("line_inspection", f"Processing selection: {xmin:.3f} - {xmax:.3f}")
        
        # Get lines in range using the molecule's built-in line caching
        try:
            line_data = self.get_molecule_line_data(self.islat.active_molecule, xmin, xmax)
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
        # Highlight strongest line AFTER both line inspection and population diagram are rendered
        # so that both the vertical line and scatter point get the orange color
        if highlight_strongest:
            self.highlight_strongest_line()
        # Only connect pick event once (check if already connected)
        if not hasattr(self, '_pick_event_connected'):
            self.canvas.mpl_connect('pick_event', self.on_pick_line)
            self._pick_event_connected = True
        # Single canvas update at the end
        self.canvas.draw_idle()
        debug_config.verbose("line_inspection", "plot_spectrum_around_line completed")
    
    def on_pick_line(self, event):
        """Handle line pick events — interaction logic owned by the controller."""
        picked_value = self._handle_line_pick_event(event, self.active_lines)
        if picked_value:
            self.selected_line = picked_value
            self._display_line_info(picked_value)
        self.canvas.draw_idle()

    def highlight_strongest_line(self):
        """
        Highlight the strongest line — interaction logic owned by the controller.
        Note: Does NOT call canvas.draw_idle() — caller is responsible for batching.
        """
        strongest = self._highlight_strongest_line(self.active_lines)
        if strongest is not None:
            # Display strongest line information in data field
            line, text, scatter, value = strongest
            self.selected_line = value
            if value:
                self._display_line_info(value)
        # Don't call canvas.draw_idle() here - let caller batch it

    # ------------------------------------------------------------------
    # Data-access & interaction helpers (moved from PlotRenderer)
    # ------------------------------------------------------------------

    def get_molecule_line_data(
        self, molecule: 'Molecule', xmin: float, xmax: float,
    ) -> List[Tuple['MoleculeLine', float, Optional[float]]]:
        """
        Get molecule lines in a wavelength range.

        This is a data-access operation and belongs in the controller
        rather than the renderer.

        Parameters
        ----------
        molecule : Molecule
            Molecule object
        xmin, xmax : float
            Wavelength range

        Returns
        -------
        List[Tuple[MoleculeLine, float, Optional[float]]]
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
                        return [(line, 0.0, None) for line in lines_in_range]
            return []

        except Exception as e:
            print(f"Error getting molecule lines: {e}")
            return []

    def _handle_line_pick_event(
        self, event: Any, active_lines_list: List[Any],
    ) -> Any:
        """
        Handle line pick events and highlight the selected line.

        Parameters
        ----------
        event : Any
            The matplotlib pick event
        active_lines_list : List[Any]
            List of [line_artist, text_obj, scatter_artist, value_data] tuples

        Returns
        -------
        Any
            The value data of the picked line or None
        """
        picked_value = None
        picked_scatter_idx = None
        picked_artist = event.artist

        # Get the active scatter collection and count from the renderer
        scatter_collection = getattr(self.plot_renderer, '_active_scatter_collection', None)
        scatter_count = getattr(self.plot_renderer, '_active_scatter_count', 0)
        active_color = self.plot_renderer._get_theme_value("active_scatter_line_color", 'green')

        # Check if the picked artist is the scatter collection
        scatter_point_clicked = None
        if picked_artist is scatter_collection and hasattr(event, 'ind') and len(event.ind) > 0:
            scatter_point_clicked = event.ind[0]

        # Find which entry in active_lines was picked and reset line inspection colors
        for line, text_obj, scatter, value in active_lines_list:
            is_line_picked = (picked_artist is line)
            point_idx = value.get('_scatter_point_index', None) if value else None
            is_scatter_picked = (scatter_point_clicked is not None and point_idx == scatter_point_clicked)
            is_picked = is_line_picked or is_scatter_picked

            if line is not None:
                line.set_color(active_color)
            if text_obj is not None:
                text_obj.set_color(active_color)

            if is_picked:
                picked_value = value
                picked_scatter_idx = point_idx
                if line is not None:
                    line.set_color('orange')
                if text_obj is not None:
                    text_obj.set_color('orange')

        # Update scatter collection colors
        if scatter_collection is not None and scatter_count > 0:
            import matplotlib.colors as mcolors
            colors = [mcolors.to_rgba(active_color)] * scatter_count
            if picked_scatter_idx is not None and picked_scatter_idx < scatter_count:
                colors[picked_scatter_idx] = mcolors.to_rgba('orange')
            scatter_collection.set_facecolors(colors)

        return picked_value

    def _highlight_strongest_line(
        self, active_lines_list: List[Any],
    ) -> Any:
        """
        Find and highlight the strongest line in the active lines.

        Parameters
        ----------
        active_lines_list : List[Any]
            List of [line_artist, text_obj, scatter_artist, value_data] tuples

        Returns
        -------
        Any
            The strongest line quadruplet or None
        """
        if not active_lines_list:
            return None

        scatter_collection = getattr(self.plot_renderer, '_active_scatter_collection', None)
        scatter_count = getattr(self.plot_renderer, '_active_scatter_count', 0)
        active_color = self.plot_renderer._get_theme_value("active_scatter_line_color", 'green')

        # Reset all line inspection lines to green first
        for line, text_obj, scatter, value in active_lines_list:
            if line is not None:
                line.set_color(active_color)
            if text_obj is not None:
                text_obj.set_color(active_color)

        # Find the line with the highest intensity
        highest_intensity = -float('inf')
        strongest_triplet = None
        strongest_scatter_idx = None

        for line, text_obj, scatter, value in active_lines_list:
            intensity = value.get('intensity', 0) if value else 0
            if intensity > highest_intensity:
                highest_intensity = intensity
                strongest_triplet = [line, text_obj, scatter, value]
                strongest_scatter_idx = value.get('_scatter_point_index', None) if value else None

        # Reset scatter collection to all green, then highlight strongest in orange
        if scatter_collection is not None and scatter_count > 0:
            import matplotlib.colors as mcolors
            colors = [mcolors.to_rgba(active_color)] * scatter_count
            if strongest_scatter_idx is not None and strongest_scatter_idx < scatter_count:
                colors[strongest_scatter_idx] = mcolors.to_rgba('orange')
            scatter_collection.set_facecolors(colors)
            scatter_collection.set_zorder(1)

        # Highlight the strongest line inspection elements in orange
        if strongest_triplet is not None:
            line, text_obj, scatter, value = strongest_triplet
            if line is not None:
                line.set_color('orange')
            if text_obj is not None:
                text_obj.set_color('orange')

        return strongest_triplet

    def _display_line_info(self, value, clear_data_field=True):
        """
        Helper method to display line information in the data field.

        Delegates formatting to :meth:`LineInspectionPlot.get_line_info` and
        enriches the result with observed / model flux integrals when a
        selection range is active.
        """
        from .LineInspectionPlot import LineInspectionPlot

        # Calculate flux integrals in the selected range ----------------
        data_flux = None
        model_flux = None
        if hasattr(self, 'current_selection') and self.current_selection:
            xmin, xmax = self.current_selection
            err_data = getattr(self.islat, 'err_data', None)
            line_flux, _ = self.flux_integral(
                lam=self.islat.wave_data,
                flux=self.islat.flux_data,
                lam_min=xmin, lam_max=xmax,
                err=err_data,
            )
            data_flux = line_flux[0] if isinstance(line_flux, (list, tuple)) else line_flux
            molecule_wave, molecule_flux_arr = self.islat.active_molecule.get_flux(return_wavelengths=True)
            model_flux, _ = self.flux_integral(
                lam=molecule_wave,
                flux=molecule_flux_arr,
                lam_min=xmin, lam_max=xmax,
                err=None,
            )

        # If the value dict already comes from get_line_info, update flux
        # fields and regenerate the formatted text.  Otherwise fall back
        # to the legacy key names.
        if 'formatted_text' in value:
            # Re-generate info with actual flux values (the original was
            # created at render time without them).
            class _Line2:
                pass
            _l2 = _Line2()
            _l2.lam = value.get('lam')
            _l2.e_up = value.get('e_up')
            _l2.e_low = value.get('e_low')
            _l2.a_stein = value.get('a_stein')
            _l2.g_up = value.get('g_up')
            _l2.g_low = value.get('g_low')
            _l2.lev_up = value.get('up_lev')
            _l2.lev_low = value.get('low_lev')
            info = LineInspectionPlot.get_line_info(
                _l2,
                intensity=value.get('intensity', 0),
                tau=value.get('tau'),
                data_flux_in_range=data_flux,
                model_flux_in_range=model_flux,
            )
        else:
            # Legacy value_data dict (keys: lam/e/a/g/inten/…)
            # Build a minimal namespace so get_line_info can work.
            class _Line:
                pass
            _l = _Line()
            _l.lam = value.get('lam')
            _l.e_up = value.get('e_up', value.get('e'))
            _l.e_low = value.get('e_low')
            _l.a_stein = value.get('a_stein', value.get('a'))
            _l.g_up = value.get('g_up', value.get('g'))
            _l.g_low = value.get('g_low')
            _l.lev_up = value.get('up_lev')
            _l.lev_low = value.get('low_lev')
            info = LineInspectionPlot.get_line_info(
                _l,
                intensity=value.get('intensity', value.get('inten', 0)),
                tau=value.get('tau'),
                data_flux_in_range=data_flux,
                model_flux_in_range=model_flux,
            )

        info_str = LineInspectionPlot.format_line_info(info)

        # Push to the GUI data-field (with error protection) -----------
        if (hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field') and
            self.islat.GUI.data_field is not None):
            try:
                if hasattr(self.islat.GUI.data_field, 'text') and self.islat.GUI.data_field.text.winfo_exists():
                    self.islat.GUI.data_field.insert_text(info_str, clear_after=clear_data_field)
            except Exception as e:
                print(f"Warning: Could not update data field: {e}")
                pass

    def clear_selection(self):
        self.current_selection = None
        self.toggle_state["current_selection"] = None
        self.clear_active_lines()
        self.ax2.clear()
        # Refresh population diagram without active line dots
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        self.canvas.draw_idle()
        return

    def plot_line_inspection(self, xmin=None, xmax=None, line_data=None, highlight_strongest=True):
        if xmin is None or xmax is None:
            self.clear_active_lines()
            self.plot_renderer.ax2.clear()
            # Refresh population diagram without active line dots
            if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
                self.plot_renderer.render_population_diagram(self.islat.active_molecule)
            self.current_selection = None
            self.canvas.draw_idle()
            return
        
        # Get line data using the molecular line API
        if line_data is None:
            try:
                line_data = self.get_molecule_line_data(self.islat.active_molecule, xmin, xmax)
                if not line_data:
                    self.clear_active_lines()
                    self.ax2.clear()
                    # Refresh population diagram without active line dots
                    if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
                        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                    self.current_selection = None
                    self.canvas.draw_idle()
                    return
            except Exception as e:
                debug_config.warning("main_plot", f"Could not get line data: {e}")
                self.clear_active_lines()
                self.ax2.clear()
                # Refresh population diagram without active line dots
                if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
                    self.plot_renderer.render_population_diagram(self.islat.active_molecule)
                self.current_selection = None
                self.canvas.draw_idle()
                return

        # First update the basic line inspection plot
        self.update_line_inspection_plot(xmin=xmin, xmax=xmax)
        
        # Get the max y value for scaling line heights
        data_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
        data_region_y = self.islat.flux_data[data_mask]
        max_y = np.nanmax(data_region_y) if len(data_region_y) > 0 else (self.plot_renderer.ax2.get_ylim()[1] / 1.1) #returns ymin, ymax
            
        # Add vertical lines using PlotRenderer (clear_active_lines already called by caller)
        self.plot_renderer.render_active_lines_in_line_inspection(line_data, self.active_lines, max_y)

        # Don't call canvas.draw_idle() here - let caller batch it
        # Don't highlight here - do it after population diagram scatter points are created

    def plot_population_diagram(self, line_data):
        """
        Plot population diagram for the currently active lines in the selected region.
        Uses the MoleculeLine-based data structure.
        
        Parameters
        ----------
        line_data : list
            List of (MoleculeLine, intensity, tau) tuples
        """
        # Only redraw base population diagram if molecule changed (uses internal caching)
        # This avoids expensive full redraw on every selection change
        self.plot_renderer.render_population_diagram(self.islat.active_molecule)
        
        # Add active line scatter points
        if line_data:
            self.plot_renderer.render_active_lines_in_population_diagram(line_data, self.active_lines)
        
        # Don't call canvas.draw_idle() here - let caller batch it
        # Don't call highlight_strongest_line() here - already called in plot_line_inspection

    def update_line_inspection_plot(self, xmin=None, xmax=None):
        """
        Update the line inspection plot showing data and active molecule model in the selected range.
        Uses only PlotRenderer logic with molecule's built-in caching for optimal performance.

        The stored ``fit_result`` is always forwarded so that fit overlays
        survive ax2.clear() calls.  ``fit_result`` is reset to *None* in
        :meth:`onselect` whenever the user drags a new span selection.
        """
        fit_result = getattr(self, 'fit_result', None)
        self.plot_renderer.render_complete_line_inspection_plot(
            wave_data=self.islat.wave_data,
            flux_data=self.islat.flux_data,
            xmin=xmin,
            xmax=xmax,
            active_molecule=self.islat.active_molecule,
            fit_result=fit_result
        )
        # Don't call canvas.draw_idle() here - let caller batch it

    def toggle_atomic_lines(self, show: Optional[bool] = None) -> None:
        """
        Toggle atomic line annotations on the active view.

        Parameters
        ----------
        show : bool or None
            If *None* the toggle is flipped; otherwise the explicit state
            is forwarded.
        """
        if show is None:
            self.atomic_toggle = not self.atomic_toggle
        else:
            self.atomic_toggle = show
        self.active_view.toggle_atomic_lines(self.atomic_toggle)

    def toggle_saved_lines(self, show: Optional[bool] = None, loaded_lines=None) -> None:
        """
        Toggle saved-line annotations on the active view.

        Parameters
        ----------
        show : bool or None
            If *None* the toggle is flipped; otherwise the explicit state
            is forwarded.
        loaded_lines : DataFrame or None
            Pre-loaded line data.  If *None* the view will load from disk.
        """
        if show is None:
            self.line_toggle = not self.line_toggle
        else:
            self.line_toggle = show
        self.active_view.toggle_saved_lines(self.line_toggle, loaded_lines=loaded_lines)

    def toggle_legend(self):
        self.legend_toggle = not self.legend_toggle
        self.active_view.toggle_legend(self.legend_toggle)

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
    
    def remove_atomic_lines(self):
        """Remove atomic lines — delegates to the active view."""
        self.atomic_toggle = False
        self.active_view.toggle_atomic_lines(False)

    def plot_atomic_lines(self, data_field=None, atomic_lines=None):
        """Plot atomic lines — delegates to the active view."""
        self.atomic_toggle = True
        self.active_view.toggle_atomic_lines(True)

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

        # Visibility changes are handled by on_molecule_visibility_changed
        # (called explicitly from ControlPanel) — skip here to avoid
        # double-rendering.
        if parameter_name == 'is_visible':
            return
        
        # Check if this molecule is visible - if so, we need to update plots
        if (hasattr(self.islat, 'molecules_dict') and 
            molecule_name in self.islat.molecules_dict):
            
            molecule = self.islat.molecules_dict[molecule_name]
            
            if molecule.is_visible:
                # Delegate to update_model_plot so the inactive view is
                # also marked stale and refreshes on next activate().
                self.update_model_plot()
            else:
                # Molecule is hidden — record it as stale so that when it
                # is next made visible we re-render from fresh data rather
                # than just toggling the old (now outdated) artists.
                self._stale_molecules.add(molecule_name)
                debug_config.trace("main_plot", f"{molecule_name} parameter changed while hidden — marked stale")
        
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
        Handle molecule visibility changes by delegating to the active view.

        The active view handles the rendering update (lightweight artist
        toggling in full-spectrum mode, or PlotRenderer-based update in
        three-panel mode).  The inactive view is marked stale so it
        re-renders with the correct visibility on next activate().
        
        Parameters
        ----------
        molecule_name : str
            Name of the molecule whose visibility changed
        is_visible : bool
            New visibility state
        """
        if not hasattr(self.islat, 'molecules_dict'):
            return
        
        current_selection = getattr(self, 'current_selection', None)
        active_molecule = getattr(self.islat, 'active_molecule', None)

        # If the molecule had its parameters changed while hidden, we need
        # to force a full re-render (not just an artist toggle) so the
        # displayed spectrum reflects the updated parameters.
        force_rerender = is_visible and molecule_name in self._stale_molecules
        if force_rerender:
            self._stale_molecules.discard(molecule_name)
            debug_config.trace("main_plot", f"{molecule_name} was stale — forcing re-render on visibility toggle")

        self.active_view.on_molecule_visibility_changed(
            molecule_name=molecule_name,
            is_visible=is_visible,
            molecules_dict=self.islat.molecules_dict,
            wave_data=self.islat.wave_data,
            active_molecule=active_molecule,
            current_selection=current_selection,
            force_rerender=force_rerender,
        )

        # Mark the *other* view as needing a full refresh next time it is
        # activated, so molecule visibility stays consistent across views.
        if self.is_full_spectrum:
            self._three_panel_view._needs_refresh = True
        else:
            self._full_spectrum_view._needs_refresh = True
    
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
            # Build extra kwargs for deblend mode
            fit_kwargs = dict(xmin=xmin, xmax=xmax, deblend=deblend)
            if deblend:
                fit_kwargs.update(
                    wave_data_full=self.islat.wave_data,
                    err_data_full=self.islat.err_data,
                    user_settings=getattr(self.islat, 'user_settings', {}),
                    active_molecule_fwhm=getattr(self.islat.active_molecule, 'fwhm', None) if getattr(self.islat, 'active_molecule', None) else None,
                    lines_with_intensity=(
                        self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
                        if getattr(self.islat, 'active_molecule', None) and hasattr(self.islat.active_molecule, 'intensity')
                        else None
                    ),
                    line_threshold=(
                        self.islat.user_settings.get('line_threshold', 0.03)
                        if getattr(self.islat, 'user_settings', None) else 0.03
                    )
                )
            
            fit_result, fitted_wave, fitted_flux = self.fitting_engine.fit_gaussian_line(
                x_fit, y_fit, **fit_kwargs
            )
            self.fit_result = (fit_result, fitted_wave, fitted_flux)
            if update_plot and self.current_selection is not None:
                # Overlay the fit directly on the existing line inspection plot
                # without clearing ax2 (which would destroy the active lines).
                max_y = fitted_flux.max() if len(fitted_flux) > 0 else 0.15
                self.plot_renderer._render_fit_results_in_line_inspection(
                    fit_result=self.fit_result, xmin=xmin, xmax=xmax, max_y=max_y,
                )
                self.canvas.draw_idle()
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
    
    def plot_saved_lines(self, loaded_lines=None, data_field=None):
        """Plot saved lines — delegates to the active view."""
        self.line_toggle = True
        self.active_view.toggle_saved_lines(True, loaded_lines=loaded_lines)

    def remove_saved_lines(self):
        """Remove saved lines — delegates to the active view."""
        self.line_toggle = False
        self.active_view.toggle_saved_lines(False)

    def apply_theme(self, theme=None):
        """Apply theme to the plot and update colors"""
        if theme:
            self.theme = theme
        self._apply_plot_theming()
        # Refresh the plots to apply colors
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        # Only update plots if data has been initialized (avoid blocking during startup)
        if hasattr(self, '_data_initialized') and self._data_initialized:
            if hasattr(self, 'update_all_plots'):
                self.update_all_plots()
    
    def load_full_spectrum(self):
        """Activate the full-spectrum view (called by toggle_full_spectrum)."""
        self._three_panel_view.deactivate()
        self._full_spectrum_view.activate(self.parent_frame)
        self.active_view = self._full_spectrum_view

    def toggle_summed_spectrum(self):
        """Toggle visibility of the summed spectral flux."""
        self.summed_toggle = not self.summed_toggle
        self.active_view.toggle_summed_spectrum(self.summed_toggle)

    def toggle_full_spectrum(self):
        """Toggle between the regular three-panel view and the full spectrum view."""
        self.is_full_spectrum = not self.is_full_spectrum
        debug_config.info("main_plot", f"toggle_full_spectrum: is_full_spectrum = {self.is_full_spectrum}")

        if self.is_full_spectrum:
            self.load_full_spectrum()
        else:
            # Switch back to three-panel view
            self._full_spectrum_view.deactivate()
            self.active_view = self._three_panel_view
            self._three_panel_view.activate(self.parent_frame)