import tkinter as tk
from tkinter import filedialog, ttk, font, simpledialog, messagebox
import os
import platform
import threading
import queue

from iSLAT.Modules.Debug.PerformanceLogger import get_performance_summary
from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv

# High-DPI / Retina display support — imported before figures are created so that matplotlib rcParams are configured early.
from iSLAT.Modules.GUI.DisplayConfig import apply_tk_scaling, display_config

from .Widgets.DataField import DataField
from .Widgets.ControlPanel import ControlPanel
from .Widgets.TopBar import TopBar
from .Widgets.FileInteractionPane import FileInteractionPane
from .Tooltips import set_tooltip_theme
from .GUIFunctions import configure_all_button_styles

class GUI:
    def __init__(self, master, molecule_data, wave_data, flux_data, config, islat_class_ref):
        if master is None:
            self.master = tk.Tk()
            # Apply system DPI scaling for HiDPI / Retina displays
            apply_tk_scaling(self.master)
            self.style = ttk.Style()
            self._style_config()
            # self.master = ThemedTk()
            self.master.title("iSLAT - Interactive Spectral-Line Analysis Tool")
            self.master.resizable(True, True)
            # Set minimum size to maintain usability
            self.master.minsize(800, 600)
            #self.master.attributes('-topmost', config.get("start_on_top", True))
            # Configure initial window size based on screen dimensions
            self._configure_initial_size()
        else:
            self.master = master

        self.molecule_data = molecule_data
        self.wave_data = wave_data
        self.flux_data = flux_data
        self.config = config
        self.theme = config["theme"]
        self.islat_class = islat_class_ref
        self.default_font = font.nametofont("TkDefaultFont")

        # Set module-level tooltip theme so every tooltip picks up
        # the current theme's colours automatically.
        set_tooltip_theme(self.theme)
        
        # Apply theme to root window
        # self._apply_theme_to_widget(self.master)

    def _style_config(self):
        # On Windows, prefer a modern built-in ttk theme for cleaner widgets.
        if platform.system() == "Windows":
            available = self.style.theme_names()
            for preferred in ("vista", "winnative", "xpnative"):
                if preferred in available:
                    self.style.theme_use(preferred)
                    break

        self.style.configure("Small.TButton", padding=(0, 5))
    
    def _force_theme_update(self):
        """Force theme update on all widgets in the window."""
        if hasattr(self, 'plot') and hasattr(self.plot, 'apply_theme'):
            #print("applying theme to plot")
            self.plot.apply_theme(self.theme)

    # ------------------------------------------------------------------
    # Reactive system-theme synchronisation
    # ------------------------------------------------------------------
    def _bind_appearance_change(self):
        """Bind to OS appearance changes so the GUI theme updates reactively.

        On macOS (Tk ≥ 8.6.9) the ``<<AppearanceChanged>>`` virtual event
        is fired whenever the user switches between Light and Dark mode in
        System Preferences.  We listen for that event and re-apply the
        theme across the entire GUI.
        """
        if platform.system() != "Darwin":
            return  # Only macOS fires <<AppearanceChanged>>

        # Only auto-sync when the user chose "auto" theme
        theme_key = self.config.get("_theme_key", "")
        if isinstance(theme_key, str) and theme_key.lower() != "auto":
            return

        try:
            self.master.bind("<<AppearanceChanged>>", self._on_appearance_changed)
        except tk.TclError:
            # Older Tk build that doesn't support the virtual event
            pass

    def _on_appearance_changed(self, _event=None):
        """Callback fired by Tk when the macOS appearance changes."""
        from iSLAT.Modules.Plotting.BasePlot import _detect_system_theme, BasePlot

        new_theme_name = _detect_system_theme()

        # Load the new theme dict
        new_theme = BasePlot.load_theme(new_theme_name)

        # Check whether the theme actually changed (avoid redundant redraws)
        old_bg = self.theme.get("background")
        new_bg = new_theme.get("background")
        if old_bg == new_bg:
            return

        # Store the new theme globally
        self.theme = new_theme
        self.config["theme"] = new_theme

        # --- Propagate to every component ---

        # 1. Tooltip colours
        set_tooltip_theme(new_theme)

        # 2. Custom ttk button / menubutton styles
        configure_all_button_styles(new_theme)

        # 3. Matplotlib-based plot (visual-only, no spectrum recalculation)
        if hasattr(self, 'plot') and self.plot is not None:
            self.plot.apply_theme(new_theme)

        # 4. Recurse through every tk/ttk widget in the window.
        self._apply_theme_to_all_widgets(new_theme)

    def _apply_theme_to_all_widgets(self, theme):
        """Walk the entire widget tree and apply *theme* to every widget.

        This mirrors the logic in
        :pymethod:`ResizableFrame._apply_theme_to_widget` but is invoked
        from the top-level ``GUI`` instance so that widgets that are *not*
        children of a ResizableFrame are also themed (e.g. raw ``tk.Frame``
        containers, ``ControlPanel``, ``DataField``, etc.).
        """
        for child in self.master.winfo_children():
            self._apply_theme_to_widget_recursive(child, theme)

    def _apply_theme_to_widget_recursive(self, widget, theme):
        """Apply *theme* colours to *widget* and recurse into children."""
        try:
            # Skip matplotlib canvas widget subtrees entirely – they are
            # already themed by MainPlot.apply_theme() and recursing into
            # their many internal tk children triggers expensive redraws.
            if 'FigureCanvasTkAgg' in type(widget).__name__:
                return
            if hasattr(widget, 'master') and 'FigureCanvasTkAgg' in type(widget.master).__name__:
                return

            widget_class = widget.winfo_class()

            if widget_class in ('Frame', 'LabelFrame'):
                try:
                    widget.configure(bg=theme.get("background", "#181A1B"))
                except tk.TclError:
                    pass
                if widget_class == 'LabelFrame':
                    try:
                        widget.configure(
                            fg=theme.get("foreground", "#F0F0F0"),
                            highlightbackground=theme.get("foreground", "#F0F0F0"),
                        )
                    except tk.TclError:
                        pass

            elif widget_class == 'Button':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                try:
                    widget.configure(
                        bg=btn_theme.get("background", "lightgray"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        activebackground=btn_theme.get("active_background",
                                                       theme.get("selection_color", "#00FF99")),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Label':
                # Skip ColorButton instances – they use bg for the
                # molecule colour and must not be overwritten.
                if not hasattr(widget, 'color'):
                    try:
                        widget.configure(
                            bg=theme.get("background", "#181A1B"),
                            fg=theme.get("foreground", "#F0F0F0"),
                        )
                    except tk.TclError:
                        pass

            elif widget_class == 'Entry':
                try:
                    widget.configure(
                        bg=theme.get("background_accent_color", "#23272A"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        insertbackground=theme.get("foreground", "#F0F0F0"),
                        selectbackground=theme.get("selection_color", "#00FF99"),
                        selectforeground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Text':
                try:
                    widget.configure(
                        bg=theme.get("data_field_background", "#23272A"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        insertbackground=theme.get("foreground", "#F0F0F0"),
                        selectbackground=theme.get("selection_color", "#00FF99"),
                        selectforeground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Listbox':
                try:
                    widget.configure(
                        bg=theme.get("background_accent_color", "#23272A"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        selectbackground=theme.get("selection_color", "#00FF99"),
                        selectforeground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Checkbutton':
                try:
                    widget.configure(
                        bg=theme.get("background", "#181A1B"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        selectcolor=theme.get("background_accent_color", "#23272A"),
                        activebackground=theme.get("background", "#181A1B"),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Menubutton':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                try:
                    widget.configure(
                        bg=btn_theme.get("background", "lightgray"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        activebackground=btn_theme.get("active_background",
                                                       theme.get("selection_color", "#00FF99")),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                        highlightbackground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Menu':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                try:
                    widget.configure(
                        bg=btn_theme.get("background", "lightgray"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        activebackground=btn_theme.get("active_background",
                                                       theme.get("selection_color", "#00FF99")),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Canvas':
                # Skip matplotlib canvas widgets (handled by plot.apply_theme)
                if 'NavigationToolbar2Tk' not in type(widget.master).__name__:
                    try:
                        widget.configure(bg=theme.get("background", "#181A1B"))
                    except tk.TclError:
                        pass

            # Recurse into children
            for child in widget.winfo_children():
                self._apply_theme_to_widget_recursive(child, theme)

        except tk.TclError:
            pass

    def _configure_initial_size(self):
        """Configure initial window size based on screen resolution."""
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Use 80% of screen width and 75% of screen height
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.87)
        
        # Ensure minimum size constraints
        window_width = max(window_width, 800)
        window_height = max(window_height, 600)
        
        # Calculate position to center the window
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        
        self.master.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

    def build_left_panel(self, parent: tk.Frame):
        # Main data field - create this first so we can pass it to other components
        self.data_field = DataField("", parent, theme=self.theme)

        #Control panel for molecule controls
        self.control_panel = ControlPanel(parent, self.islat_class, self.plot, self.data_field, self.default_font)
        
        # Spectrum file selector
        self.file_interaction_pane = FileInteractionPane(parent, self.islat_class, self.theme, self.data_field)
        
        self.file_interaction_pane.grid(padx=(1,0), sticky="nsew",  pady=0)
        self.control_panel.grid(padx=(1,0), sticky="nsew", pady=0)
        self.data_field.grid(padx=(1,0), sticky="nsew", pady=0)

    def update_frame_sizes(self):
        """Update dynamic frame sizes based on current content."""
        if hasattr(self, 'left_resizable'):
            self.left_resizable.update_dynamic_sizes()
        if hasattr(self, 'main_resizable'):
            self.main_resizable.update_dynamic_sizes()

    def create_window(self):
        from iSLAT import __version__ as iSLAT_version
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        self.window = self.master
        self.window.title(f"iSLAT Version {iSLAT_version}")
        
        # Create a main container frame
        main_container = ttk.Frame(self.window)

        # Create frames for left panel and right panel (plot)
        left_main_frame = tk.Frame(main_container)
        left_main_frame.pack(side="left", fill="y", expand=False, padx=0, pady=0)

        right_main_frame = tk.Frame(main_container)
        right_main_frame.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        
        # Create the plot directly in right_frame without extra container
        self.plot = iSLATPlot(right_main_frame, self.wave_data, self.flux_data, self.theme, self.islat_class)

        # Left side: all controls
        self.build_left_panel(left_main_frame)

        # Bottom function buttons
        self.top_bar = TopBar(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.control_panel, self.config)
        self.top_bar.pack(side="top", fill="x", padx=0, pady=0)

        main_container.pack(fill="both", expand=True, padx=0, pady=0)

        self._force_theme_update()

    def start(self, display_spectrum_async=True):
        """
        Start the GUI and enter the main event loop.
        
        Parameters
        ----------
        display_spectrum_async : bool, default True
            If True, display spectrum asynchronously after GUI is shown (faster startup).
            If False, display spectrum synchronously (blocks until complete).
        """
        self.create_window()
    
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Listen for OS appearance changes (macOS dark ↔ light mode)
        self._bind_appearance_change()

        # Check config for start-on-top behavior
        if self.config.get("start_on_top", True):
            self.master.attributes('-topmost', True)
            self.master.after(100, lambda: self.master.attributes('-topmost', False))
        
        self.master.lift()
        
        # Schedule async spectrum display after GUI is shown
        if display_spectrum_async:
            # Use after(1) to run after first event loop iteration (GUI visible first)
            self.master.after(1, self._start_async_spectrum_display)
        
        self.window.mainloop()
    
    def _start_async_spectrum_display(self):
        """Called after GUI is shown - starts async spectrum calculation."""
        if (hasattr(self, 'plot') and self.plot is not None and
            hasattr(self.islat_class, 'wave_data') and hasattr(self.islat_class, 'flux_data')):
            #print("Starting async spectrum display...")
            self.display_spectrum_async(callback=self._on_async_display_complete)
    
    def _on_async_display_complete(self, success):
        """Called when async spectrum display completes."""
        if success:
            # Update file label if available
            if (hasattr(self, "file_interaction_pane") and 
                hasattr(self.islat_class, 'loaded_spectrum_name')):
                self.file_interaction_pane.update_file_label(self.islat_class.loaded_spectrum_name)

    def on_closing(self):
        #if messagebox.askokcancel("Quit", "Do you want to save your work?"):
        if True:            
            try:
                # Save the current molecule parameters
                write_molecules_to_csv(
                    self.islat_class.molecules_dict, 
                )
            except Exception as e:
                print("Error", f"Failed to save molecule parameters: {str(e)}")

        # Close all matplotlib figures to release resources
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

        # Destroy the full-spectrum view's figure (non-pyplot managed)
        try:
            if hasattr(self, 'plot') and hasattr(self.plot, '_full_spectrum_view'):
                self.plot._full_spectrum_view.destroy()
        except Exception:
            pass

        try:
            self.window.destroy()
        except Exception:
            pass

        # Force-exit the process so no background threads keep it alive
        import os
        os._exit(0)

    def get_plot_renderer(self):
        return self.plot.plot_renderer

    # ================================
    # Async Spectrum Display
    # ================================
    def display_spectrum_async(self, callback=None):
        """
        Display spectrum asynchronously to avoid blocking the GUI during startup.
        
        This method shows a loading indicator, runs the intensive spectrum calculations
        in a background thread, and updates the GUI when complete.
        
        Parameters
        ----------
        callback : callable, optional
            Function to call when display is complete
        """
        self._async_result_queue = queue.Queue()
        self._async_callback = callback
        
        # Show loading state immediately
        if hasattr(self, 'plot') and self.plot is not None:
            self.plot.show_loading_indicator("Calculating molecule spectra...")
        
        # Start background calculation thread
        calc_thread = threading.Thread(
            target=self._async_spectrum_calculation,
            daemon=True
        )
        calc_thread.start()
        
        # Start polling for completion (non-blocking)
        self._poll_async_result()
    
    def _async_spectrum_calculation(self):
        """Background thread: perform spectrum calculations using parallel intensity calculation."""
        try:
            # This triggers lazy intensity calculations for all visible molecules
            if (hasattr(self.islat_class, 'molecules_dict') and 
                hasattr(self.islat_class, 'wave_data_original')):
                wave_data = self.islat_class.wave_data_original
                # Use parallel pre-calculation for significant speedup
                self.islat_class.molecules_dict.get_summed_flux_parallel(wave_data, visible_only=True)
            
            self._async_result_queue.put(('success', None))
        except Exception as e:
            self._async_result_queue.put(('error', str(e)))
    
    def _poll_async_result(self):
        """Poll for async calculation completion (runs on main thread)."""
        try:
            status, error = self._async_result_queue.get_nowait()
            
            # Calculation complete - update GUI
            if hasattr(self, 'plot') and self.plot is not None:
                self.plot.hide_loading_indicator()
                
                if status == 'success':
                    #print("Async spectrum calculation complete - updating display...")
                    # Initialize data-dependent plot elements now that calculations are done
                    self.plot.initialize_data()
                    self.plot.update_model_plot()
                    if hasattr(self.plot, 'canvas'):
                        self.plot.canvas.draw()
                    #print("Spectrum displayed successfully (async)")
                    
                    # Print final performance summary including all async operations
                    #print("\n" + "="*80)
                    #print("FINAL PERFORMANCE SUMMARY (including async operations)")
                    #print("="*80)
                    get_performance_summary()
                else:
                    print(f"Warning: Async spectrum calculation failed: {error}")
            
            # Call completion callback if provided
            if self._async_callback:
                self._async_callback(status == 'success')
                
        except queue.Empty:
            # Still calculating - poll again in 50ms
            self.master.after(50, self._poll_async_result)

    @staticmethod
    def file_selector(title : str = None, filetypes=None, initialdir=None, use_abspath=True, allow_multiple=False):
        window_title = title if title else "Select File"
        if use_abspath and initialdir:
            initialdir = os.path.abspath(initialdir)
        elif initialdir is None:
            initialdir = os.getcwd()

        if filetypes is None:
            filetypes = [("All Files", "*.*")]
        elif isinstance(filetypes, str):
            filetypes = [(filetypes, "*.*")]
        else:
            filetypes = filetypes
        
        if allow_multiple:
            file_paths = filedialog.askopenfilenames(
                title=window_title,
                filetypes=filetypes,
                initialdir=initialdir
            )
            return list(file_paths)
        
        file_path = filedialog.askopenfilename(
            title=window_title,
            filetypes=filetypes,
            initialdir=initialdir
        )
        return file_path
    
    @staticmethod
    def add_molecule_name_popup(title : str = "Assign label"):
        """Open a popup to add a new molecule name."""
        molecule_name = simpledialog.askstring(title, "Enter a label for this model (LaTeX and case sensitive):")

        return molecule_name