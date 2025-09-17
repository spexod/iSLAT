import numpy as np
import time
from matplotlib.widgets import SpanSelector
from typing import Optional, Tuple, Callable, Dict, Any

class InteractionHandler:
    """Mouse/keyboard events - handles all user interactions with plots"""
    
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.islat = plot_manager.islat
        
        # Plot references
        self.fig = plot_manager.fig
        self.ax1 = plot_manager.ax1  # Main spectrum plot
        self.ax2 = plot_manager.ax2  # Line inspection plot
        self.ax3 = plot_manager.ax3  # Population diagram
        self.canvas = plot_manager.canvas
        
        # Interaction state
        self.span_selector: Optional[SpanSelector] = None
        self.selected_range: Optional[Tuple[float, float]] = None
        self.mouse_pressed = False
        self.last_click_time = 0
        self.double_click_threshold = 0.5  # seconds
        
        # Callbacks
        self.selection_callbacks: Dict[str, Callable] = {}
        self.click_callbacks: Dict[str, Callable] = {}
        self.zoom_callbacks: Dict[str, Callable] = {}
        
        # Initialize interactions
        self._setup_interactions()
    
    def _setup_interactions(self):
        """Set up all mouse and keyboard interactions"""
        self._setup_span_selector()
        self._setup_mouse_events()
        self._setup_keyboard_events()
        self._setup_plot_navigation()
    
    def _setup_span_selector(self):
        """Set up the span selector for wavelength range selection"""
        self.span_selector = SpanSelector(
            self.ax1,
            self._on_span_select,
            direction='horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='lime'),
            interactive=True,
            drag_from_anywhere=True
        )
    
    def _setup_mouse_events(self):
        """Set up mouse event handlers"""
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        # Connect to draw events to catch navigation changes
        self.canvas.mpl_connect('draw_event', self._on_draw)
    
    def _setup_keyboard_events(self):
        """Set up keyboard event handlers"""
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.canvas.mpl_connect('key_release_event', self._on_key_release)
    
    def _setup_plot_navigation(self):
        """Set up plot navigation callbacks"""
        # Connect to axis limit changes
        self.ax1.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.ax1.callbacks.connect('ylim_changed', self._on_ylim_changed)
        
        # Store the last known xlim to detect changes during draw events
        self._last_xlim = self.ax1.get_xlim() if self.ax1 else None
    
    def _on_span_select(self, xmin: float, xmax: float):
        """Handle span selection on main plot"""
        if xmin == xmax:
            self.clear_current_selection()
            return
        
        # Ensure proper order
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        
        self.selected_range = (xmin, xmax)
        
        # Store selection in plot manager
        if hasattr(self.plot_manager, 'set_active_range'):
            self.plot_manager.set_active_range(xmin, xmax)
        
        # Trigger callbacks
        self._trigger_selection_callbacks('span_select', xmin, xmax)
        
        # Update displays
        self._update_population_diagram_highlights(xmin, xmax)
    
    def _on_mouse_press(self, event):
        """Handle mouse press events"""
        self.mouse_pressed = True
        
        if event.inaxes is None:
            return
        
        # Don't interfere with span selector on main plot
        if event.inaxes == self.ax1 and hasattr(self, 'span_selector') and self.span_selector.active:
            # Let the span selector handle the event on main plot
            return
        
        # Check for double-click on other plots
        current_time = time.time()
        is_double_click = (current_time - self.last_click_time) < self.double_click_threshold
        self.last_click_time = current_time
        
        # Handle different types of clicks
        if event.button == 1:  # Left click
            if is_double_click and event.inaxes != self.ax1:
                self._on_double_click(event)
            elif not is_double_click:  # Single click
                self._on_single_click(event)
        elif event.button == 3:  # Right click
            self._on_right_click(event)
    
    def _on_mouse_release(self, event):
        """Handle mouse release events"""
        self.mouse_pressed = False
    
    def _on_mouse_move(self, event):
        """Handle mouse move events"""
        if event.inaxes is None:
            return
        
        # Update cursor information
        if hasattr(self.plot_manager, 'update_cursor_info'):
            self.plot_manager.update_cursor_info(event.xdata, event.ydata)
    
    def _on_single_click(self, event):
        """Handle single click events"""
        # Don't trigger click callbacks on main plot to avoid interfering with span selector
        if event.inaxes == self.ax2:
            # Click on line inspection plot
            self._trigger_click_callbacks('click', event)
        elif event.inaxes == self.ax3:
            # Click on population diagram
            self._trigger_click_callbacks('click', event)
        # Don't handle clicks on main plot (ax1) to let span selector work
    
    def _on_double_click(self, event):
        """Handle double click events"""
        pass
    
    def _on_right_click(self, event):
        """Handle right click events (context menu)"""
        if event.inaxes == self.ax1:
            # Show context menu for main plot
            self._show_context_menu(event)
    
    def _on_pick(self, event):
        """Handle pick events (clicking on plot elements)"""
        artist = event.artist
        
        # Get the axes that contains this artist
        artist_axes = None
        if hasattr(artist, 'axes'):
            artist_axes = artist.axes
        
        # Handle scatter plot picking in population diagram
        if hasattr(artist, 'get_offsets') and artist_axes == self.ax3:
            # This is a scatter plot in the population diagram
            self._handle_scatter_pick(event)
        
        # Handle line picking
        elif hasattr(artist, '_islat_line_info'):
            line_info = artist._islat_line_info
            self._trigger_click_callbacks('line_picked', line_info)
    
    def _handle_scatter_pick(self, event):
        """Handle clicking on scatter points in population diagram"""
        # Get the indices of picked points
        indices = event.ind
        if not indices:
            return
        
        # Get the first picked point index
        idx = indices[0]
        
        # Get line information from the active molecule's intensity table
        if hasattr(self.plot_manager.islat, 'active_molecule') and self.plot_manager.islat.active_molecule:
            molecule = self.plot_manager.islat.active_molecule
            if hasattr(molecule, 'intensity') and hasattr(molecule.intensity, 'get_table'):
                line_table = molecule.intensity.get_table
                if idx < len(line_table):
                    # Get line data for the clicked point
                    line_data = line_table.iloc[idx]
                    
                    # Create line info dictionary
                    line_info = {
                        'wavelength': line_data['lam'],
                        'intensity': line_data['intens'],
                        'e_up': line_data['e_up'],
                        'a_stein': line_data['a_stein'],
                        'g_up': line_data['g_up'],
                        'up_lev': line_data.get('lev_up', 'N/A'),
                        'low_lev': line_data.get('lev_low', 'N/A'),
                        'tau': line_data.get('tau', 'N/A')
                    }
                    
                    # Display line information in data field
                    self._display_scatter_line_info(line_info)
    
    def _display_scatter_line_info(self, line_info):
        """Display information about clicked scatter point"""
        if hasattr(self.plot_manager.islat, 'GUI') and hasattr(self.plot_manager.islat.GUI, 'data_field'):
            data_field = self.plot_manager.islat.GUI.data_field
            
            # Format and display line information
            info_text = f"Selected Line Information:\n"
            info_text += f"Wavelength: {line_info['wavelength']:.6f} μm\n"
            info_text += f"Intensity: {line_info['intensity']:.3e} Jy\n"
            info_text += f"Upper Energy: {line_info['e_up']:.1f} K\n"
            info_text += f"Einstein A: {line_info['a_stein']:.3e} s⁻¹\n"
            info_text += f"Statistical Weight: {line_info['g_up']}\n"
            info_text += f"Upper Level: {line_info['up_lev']}\n"
            info_text += f"Lower Level: {line_info['low_lev']}\n"
            if line_info['tau'] != 'N/A':
                info_text += f"Optical Depth: {line_info['tau']:.3f}\n"
            
            data_field.insert_text(info_text, console_print=True, clear_after=False)
    
    def _on_scroll(self, event):
        """Handle scroll events for zooming"""        
        pass
    
    def _on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'h':
            # Toggle grid
            self._toggle_grid()
        elif event.key == 'l':
            # Toggle legend
            self._toggle_legend()
    
    def _on_key_release(self, event):
        """Handle key release events"""
        pass
    
    def _on_xlim_changed(self, ax):
        """Handle x-axis limit changes"""
        new_xlim = ax.get_xlim()
        xone = new_xlim[0]
        xtwo = new_xlim[1]
        
        # Update display range in iSLAT (only if changed to prevent infinite loops)
        if hasattr(self.islat, 'display_range'):
            current_range = self.islat.display_range
            new_range = (xone, xtwo)
            # Only update if the values are actually different (with small tolerance for floating point)
            if (not current_range or 
                abs(current_range[0] - new_range[0]) > 1e-10 or 
                abs(current_range[1] - new_range[1]) > 1e-10):
                self.islat.display_range = new_range

        # Trigger zoom callbacks
        self._trigger_zoom_callbacks('xlim_changed', new_xlim)
    
    def _on_ylim_changed(self, ax):
        """Handle y-axis limit changes"""
        new_ylim = ax.get_ylim()
        self._trigger_zoom_callbacks('ylim_changed', new_ylim)
    
    def _on_draw(self, event):
        """Handle draw events to catch navigation changes that don't trigger axis callbacks"""
        # Check if xlim has changed since last draw
        current_xlim = self.ax1.get_xlim()
        
        if self._last_xlim is None:
            # First time, just store the current xlim
            self._last_xlim = current_xlim
            return
            
        # Check if xlim has actually changed (with small tolerance for floating point)
        if (abs(current_xlim[0] - self._last_xlim[0]) > 1e-10 or 
            abs(current_xlim[1] - self._last_xlim[1]) > 1e-10):
            
            # xlim has changed, update display range
            self._last_xlim = current_xlim
            self._on_xlim_changed(self.ax1)
    
    def _update_population_diagram_highlights(self, xmin: float, xmax: float):
        """Update population diagram to highlight lines in selected range"""
        if hasattr(self.plot_manager, 'active_line_range'):
            self.plot_manager.active_line_range = (xmin, xmax)
        
        # Re-render population diagram with highlights
        if (hasattr(self.plot_manager, 'renderer') and 
            hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule):
            self.plot_manager.renderer.render_population_diagram(
                self.islat.active_molecule, wave_range=(xmin, xmax)
            )
    
    def _auto_zoom_to_point(self, x: float, y: float, ax=None):
        """Auto-zoom to a point on the specified plot"""
        if x is None or y is None:
            return
        
        if ax is None:
            ax = self.ax1
        
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Calculate new zoom range (zoom in by factor of 4)
        x_range = (current_xlim[1] - current_xlim[0]) / 4
        y_range = (current_ylim[1] - current_ylim[0]) / 4
        
        new_xlim = (x - x_range/2, x + x_range/2)
        new_ylim = (y - y_range/2, y + y_range/2)
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def _zoom_around_point(self, x: float, y: float, zoom_factor: float, ax):
        """Zoom around a specific point"""
        if x is None or y is None:
            return
        
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Calculate new limits
        x_range = (current_xlim[1] - current_xlim[0]) * zoom_factor
        y_range = (current_ylim[1] - current_ylim[0]) * zoom_factor
        
        new_xlim = (x - x_range/2, x + x_range/2)
        new_ylim = (y - y_range/2, y + y_range/2)
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def _reset_zoom(self):
        """Reset zoom to show all data"""
        if hasattr(self.islat, 'wave_data') and hasattr(self.islat, 'flux_data'):
            if self.islat.wave_data is not None and len(self.islat.wave_data) > 0:
                self.ax1.set_xlim(self.islat.wave_data.min(), self.islat.wave_data.max())
                
                if self.islat.flux_data is not None and len(self.islat.flux_data) > 0:
                    flux_min = np.nanmin(self.islat.flux_data)
                    flux_max = np.nanmax(self.islat.flux_data)
                    margin = (flux_max - flux_min) * 0.1
                    self.ax1.set_ylim(flux_min - margin, flux_max + margin)
                
                self.canvas.draw_idle()
    
    def _toggle_grid(self):
        """Toggle grid on/off"""
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.grid(not ax.get_grid)
        self.canvas.draw_idle()
    
    def _toggle_legend(self):
        """Toggle legend on/off"""
        legend = self.ax1.get_legend()
        if legend:
            legend.set_visible(not legend.get_visible())
        self.canvas.draw_idle()
    
    # Callback management
    def add_selection_callback(self, name: str, callback: Callable):
        """Add a callback for selection events"""
        self.selection_callbacks[name] = callback
    
    def remove_selection_callback(self, name: str):
        """Remove a selection callback"""
        self.selection_callbacks.pop(name, None)
    
    def add_click_callback(self, name: str, callback: Callable):
        """Add a callback for click events"""
        self.click_callbacks[name] = callback
    
    def remove_click_callback(self, name: str):
        """Remove a click callback"""
        self.click_callbacks.pop(name, None)
    
    def add_zoom_callback(self, name: str, callback: Callable):
        """Add a callback for zoom events"""
        self.zoom_callbacks[name] = callback
    
    def remove_zoom_callback(self, name: str):
        """Remove a zoom callback"""
        self.zoom_callbacks.pop(name, None)
    
    def _trigger_selection_callbacks(self, event_type: str, *args):
        """Trigger all selection callbacks"""
        for callback in self.selection_callbacks.values():
            try:
                # For span_select, only pass the coordinates, not the event_type
                if event_type == 'span_select' and len(args) >= 2:
                    callback(args[0], args[1])  # xmin, xmax
                else:
                    callback(*args)
            except Exception as e:
                print(f"Error in selection callback: {e}")
    
    def _trigger_click_callbacks(self, event_type: str, *args):
        """Trigger all click callbacks"""
        for callback in self.click_callbacks.values():
            try:
                # For click events, typically pass the event object or coordinates
                if event_type == 'click' and len(args) == 1:
                    callback(args[0])  # event object
                else:
                    callback(*args)
            except Exception as e:
                print(f"Error in click callback: {e}")
    
    def _trigger_zoom_callbacks(self, event_type: str, *args):
        """Trigger all zoom callbacks"""
        for callback in self.zoom_callbacks.values():
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in zoom callback: {e}")
    
    # Public interface
    def enable_interactions(self):
        """Enable all interactions"""
        if self.span_selector:
            self.span_selector.set_active(True)
    
    def disable_interactions(self):
        """Disable all interactions"""
        if self.span_selector:
            self.span_selector.set_active(False)
    
    def get_current_selection(self) -> Optional[Tuple[float, float]]:
        """Get the current wavelength selection"""
        return self.selected_range
    
    def set_selection(self, xmin: float, xmax: float):
        """Programmatically set a selection"""
        self.selected_range = (xmin, xmax)
        self._on_span_select(xmin, xmax)
    
    def clear_current_selection(self):
        """Clear the current selection"""
        self.selected_range = None
        if hasattr(self.plot_manager, 'clear_selection'):
            self.plot_manager.clear_selection()
    
    def get_interaction_info(self) -> Dict[str, Any]:
        """Get information about current interaction state"""
        return {
            'selected_range': self.selected_range,
            'mouse_pressed': self.mouse_pressed,
            'span_selector_active': self.span_selector.active if self.span_selector else False,
            'num_selection_callbacks': len(self.selection_callbacks),
            'num_click_callbacks': len(self.click_callbacks),
            'num_zoom_callbacks': len(self.zoom_callbacks)
        }
    
    # Additional callback methods expected by MainPlot
    def set_span_select_callback(self, callback: Callable):
        """Set callback for span selection events"""
        self.add_selection_callback('span_select', callback)
    
    def set_click_callback(self, callback: Callable):
        """Set callback for click events"""
        self.add_click_callback('click', callback)
    
    def create_span_selector(self, ax, color):
        """Create a span selector - compatibility method"""
        return self.span_selector
    
    def handle_click_event(self, event):
        """Handle click events - compatibility method"""
        self._on_mouse_press(event)
