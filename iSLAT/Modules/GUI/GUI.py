import tkinter as tk
from tkinter import filedialog, ttk
import os

from iSLAT.Modules.Plotting.MainPlot import iSLATPlot

from .Widgets.DataField import DataField
from .Widgets.MoleculeWindow import MoleculeWindow
from .Widgets.ControlPanel import ControlPanel
from .Widgets.TopOptions import TopOptions
from .Widgets.BottomOptions import BottomOptions
from .Widgets.ResizableFrame import ResizableFrame
from .Widgets.FileInteractionPane import FileInteractionPane
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class GUI:
    def __init__(self, master, molecule_data, wave_data, flux_data, config, islat_class_ref):
        if master is None:
            self.master = tk.Tk()
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
        
        # Apply theme to root window
        self._apply_theme_to_widget(self.master)

    def _apply_theme_to_widget(self, widget):
        """Apply theme colors to a tkinter widget and its children."""
        try:
            # Apply theme to the widget itself
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'Toplevel', 'Tk']:
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'LabelFrame':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Label':
                widget.configure(bg=self.theme["background"], fg=self.theme["foreground"])
            elif widget_class == 'Button':
                # Check if this is a marked color selection button
                if hasattr(widget, '_is_color_button') and widget._is_color_button:
                    # This is a color selection button - never theme it
                    pass
                else:
                    # Check other characteristics to preserve color buttons
                    current_bg = widget.cget('bg')
                    current_text = widget.cget('text')
                    
                    # Don't theme color selection buttons (preserve molecule colors)
                    # Color selection buttons typically have hex color backgrounds
                    if (current_bg and current_bg.startswith('#') and len(current_bg) == 7 and 
                        current_text == "" and widget.cget('width') <= 4):
                        # This is likely a color selection button - preserve its color
                        pass
                    # Don't theme delete buttons - they have their own special theme
                    elif current_text == "X":
                        # This is a delete button - it should be themed by its own component
                        pass
                    # Only apply theme if the button has default styling
                    elif widget.cget('bg') in ['SystemButtonFace', '#d9d9d9', '#ececec', 'lightgray']:
                        btn_theme = self.theme["buttons"].get("DefaultBotton", self.theme["buttons"]["DefaultBotton"])
                        widget.configure(
                            bg=btn_theme["background"],
                            fg=self.theme["foreground"],
                            activebackground=btn_theme["active_background"],
                            activeforeground=self.theme["foreground"]
                        )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=self.theme["background_accent_color"], 
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Text':
                widget.configure(
                    bg=self.theme["background_accent_color"], 
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Listbox':
                widget.configure(
                    bg=self.theme["background_accent_color"], 
                    fg=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Checkbutton':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["background_accent_color"]
                )
            elif widget_class == 'Radiobutton':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["background_accent_color"]
                )
            elif widget_class == 'Scale':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["selection_color"],
                    troughcolor=self.theme["background_accent_color"]
                )
            elif widget_class == 'Scrollbar':
                widget.configure(
                    bg=self.theme["background_accent_color"],
                    troughcolor=self.theme["background"],
                    activebackground=self.theme["selection_color"]
                )
            elif widget_class == 'LabelFrame':
                widget.configure(
                    bg=self.theme["background"], 
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Canvas':
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'Menu':
                widget.configure(
                    bg=self.theme["background_accent_color"], 
                    fg=self.theme["foreground"],
                    activebackground=self.theme["selection_color"],
                    activeforeground=self.theme["background"]
                )
            elif widget_class == 'Spinbox':
                widget.configure(
                    bg=self.theme["background_accent_color"], 
                    fg=self.theme["foreground"],
                    buttonbackground=self.theme["background_accent_color"],
                    insertbackground=self.theme["foreground"],
                    selectbackground=self.theme["selection_color"],
                    selectforeground=self.theme["background"]
                )
            elif widget_class == 'Combobox' or widget_class in ['TCombobox']:
                # For ttk widgets, we need to use ttk styles
                try:
                    style = ttk.Style()
                    style.configure("Themed.TCombobox",
                                  fieldbackground=self.theme["background_accent_color"],
                                  background=self.theme["background_accent_color"],
                                  foreground=self.theme["foreground"],
                                  bordercolor=self.theme["background_accent_color"])
                    widget.configure(style="Themed.TCombobox")
                except:
                    pass
            elif widget_class == 'Treeview' or widget_class in ['TTreeview']:
                try:
                    style = ttk.Style()
                    style.configure("Themed.Treeview",
                                  background=self.theme["background_accent_color"],
                                  foreground=self.theme["foreground"],
                                  fieldbackground=self.theme["background_accent_color"],
                                  selectbackground=self.theme["selection_color"],
                                  selectforeground=self.theme["background"])
                    widget.configure(style="Themed.Treeview")
                except:
                    pass
            elif widget_class in ['TScrollbar']:
                try:
                    style = ttk.Style()
                    style.configure("Themed.Vertical.TScrollbar",
                                  background=self.theme["background_accent_color"],
                                  troughcolor=self.theme["background"],
                                  bordercolor=self.theme["background_accent_color"],
                                  arrowcolor=self.theme["foreground"],
                                  darkcolor=self.theme["background_accent_color"],
                                  lightcolor=self.theme["background_accent_color"])
                    style.map("Themed.Vertical.TScrollbar",
                             background=[('active', self.theme["selection_color"]),
                                       ('pressed', self.theme["selection_color"])])
                    widget.configure(style="Themed.Vertical.TScrollbar")
                except:
                    pass
            elif widget_class in ['TFrame']:
                try:
                    style = ttk.Style()
                    style.configure("Themed.TFrame",
                                  background=self.theme["background"])
                    widget.configure(style="Themed.TFrame")
                except:
                    pass
            elif widget_class in ['TLabel']:
                try:
                    style = ttk.Style()
                    style.configure("Themed.TLabel",
                                  background=self.theme["background"],
                                  foreground=self.theme["foreground"])
                    widget.configure(style="Themed.TLabel")
                except:
                    pass
            elif widget_class == 'PanedWindow':
                widget.configure(
                    bg=self.theme["background"],
                    sashrelief='raised'
                )
            
            # Recursively apply theme to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Some widgets might not support certain options
            pass
    
    def _force_theme_update(self):
        """Force theme update on all widgets in the window."""
        if hasattr(self, 'window'):
            self._apply_theme_to_widget(self.window)
        if hasattr(self, 'left_resizable'):
            self.left_resizable.apply_theme(self.theme)
        if hasattr(self, 'main_resizable'):
            self.main_resizable.apply_theme(self.theme)
            
        # Apply theme to all major components - they now handle their own theming
        if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'apply_theme'):
            self.control_panel.apply_theme(self.theme)
            
        if hasattr(self, 'plot') and hasattr(self.plot, 'apply_theme'):
            self.plot.apply_theme(self.theme)
            
        if hasattr(self, 'data_field') and hasattr(self.data_field, 'apply_theme'):
            self.data_field.apply_theme(self.theme)
            
        if hasattr(self, 'top_options') and hasattr(self.top_options, 'apply_theme'):
            self.top_options.apply_theme(self.theme)
            
        if hasattr(self, 'bottom_options') and hasattr(self.bottom_options, 'apply_theme'):
            self.bottom_options.apply_theme(self.theme)
            
        # Apply theme to file interaction pane
        if hasattr(self, 'file_interaction_pane') and hasattr(self.file_interaction_pane, 'apply_theme'):
            self.file_interaction_pane.apply_theme(self.theme)

    def _configure_initial_size(self):
        """Configure initial window size based on screen resolution."""
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Use 80% of screen width and 75% of screen height
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.75)
        
        # Ensure minimum size constraints
        window_width = max(window_width, 800)
        window_height = max(window_height, 600)
        
        # Calculate position to center the window
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        
        self.master.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

    @staticmethod
    def file_selector(title : str = None, filetypes=None, initialdir=None, use_abspath=True):
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
        
        file_path = filedialog.askopenfilename(
            title=window_title,
            filetypes=filetypes,
            initialdir=initialdir
        )
        return file_path

    def build_left_panel(self, parent: tk.Frame):
        # Create a resizable frame container for the left panel
        self.left_resizable = ResizableFrame(parent, orientation='vertical', sash_size=4, theme=self.theme)
        self.left_resizable.pack(fill="both", expand=True, padx=2, pady=2)
        
        # Create individual frames for each component
        top_options_frame = tk.Frame(self.left_resizable)
        control_panel_frame = tk.Frame(self.left_resizable)
        file_selector_frame = tk.Frame(self.left_resizable)
        data_field_frame = tk.Frame(self.left_resizable)
        
        # Apply theme to frames
        for frame in [top_options_frame, control_panel_frame, file_selector_frame, data_field_frame]:
            frame.configure(bg=self.theme["background"])
        
        # Add frames to resizable container with different weights and minimum sizes
        # Enable dynamic sizing for frames that can have variable content
        self.left_resizable.add_frame(top_options_frame, weight=0, minsize=80, dynamic_minsize=True)
        self.left_resizable.add_frame(control_panel_frame, weight=2, minsize=120, dynamic_minsize=True)
        self.left_resizable.add_frame(file_selector_frame, weight=0, minsize=80, dynamic_minsize=True)
        self.left_resizable.add_frame(data_field_frame, weight=4, minsize=200, dynamic_minsize=False)

        # Main data field - create this first so we can pass it to other components
        self.data_field = DataField("Main Data Field", "", data_field_frame, theme=self.theme)
        self.data_field.pack(fill="both", expand=True, padx=5, pady=5)
        
        # DataField now handles its own theming through ResizableFrame inheritance

        # Top control buttons - now we can pass data_field
        self.top_options = TopOptions(top_options_frame, self.islat_class, theme=self.theme, data_field=self.data_field)
        self.top_options.pack(fill="both", expand=True, padx=5, pady=2)
        
        # TopOptions now handles its own theming through ResizableFrame inheritance

        # Control panel for input parameters - ControlPanel now inherits from ResizableFrame
        self.control_panel = ControlPanel(control_panel_frame, self.islat_class)
        self.control_panel.pack(fill="both", expand=True, padx=5, pady=5)
        
        # ControlPanel now handles its own theming through ResizableFrame inheritance

        # Spectrum file selector
        self.file_interaction_pane = FileInteractionPane(file_selector_frame, self.islat_class, self.theme)
        self.file_interaction_pane.pack(fill="both", expand=True, padx=5, pady=5)
    
    def update_frame_sizes(self):
        """Update dynamic frame sizes based on current content."""
        if hasattr(self, 'left_resizable'):
            self.left_resizable.update_dynamic_sizes()
        if hasattr(self, 'main_resizable'):
            self.main_resizable.update_dynamic_sizes()

    def create_window(self):
        self.window = self.master
        self.window.title("iSLAT Version 5.00.00")
        
        # Configure main window for resizable layout
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0, minsize=60)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create a main container frame
        main_container = tk.Frame(self.window)
        main_container.grid(row=0, column=0, sticky="nsew")
        
        # Create horizontal resizable frame for left panel and plot area
        self.main_resizable = ResizableFrame(main_container, orientation='horizontal', sash_size=6, theme=self.theme)
        self.main_resizable.pack(fill="both", expand=True)
        
        # Create frames for left panel and right panel (plot)
        left_main_frame = tk.Frame(self.main_resizable)
        right_main_frame = tk.Frame(self.main_resizable)
        
        # Apply theme to main frames
        left_main_frame.configure(bg=self.theme["background"])
        right_main_frame.configure(bg=self.theme["background"])
        
        # Add frames to horizontal resizable container
        # Reduced weight for left panel to start with less horizontal space
        self.main_resizable.add_frame(left_main_frame, weight=1, minsize=300)
        self.main_resizable.add_frame(right_main_frame, weight=3, minsize=450)

        # Right side: plots
        right_frame = tk.Frame(right_main_frame)
        right_frame.pack(fill="both", expand=True, padx=0, pady=0)
        
        # Configure right frame for responsive plot
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Apply theme to right frame
        self._apply_theme_to_widget(right_frame)
        
        # Create the plot directly in right_frame without extra container
        self.plot = iSLATPlot(right_frame, self.wave_data, self.flux_data, self.theme, self.islat_class)

        # Left side: all controls
        left_frame = tk.Frame(left_main_frame)
        left_frame.pack(fill="both", expand=True)
        
        # Apply theme to left frame
        self._apply_theme_to_widget(left_frame)
        
        self.build_left_panel(left_frame)

        # Bottom function buttons
        self.bottom_options = BottomOptions(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.config)
        self.bottom_options.grid(row=1, column=0, columnspan=2, sticky="ew")
        
        # BottomOptions now handles its own theming through ResizableFrame inheritance

        # Force theme updates to catch any missed widgets
        self.window.after(100, self._force_theme_update)
        # Additional delayed update to catch any widgets created asynchronously
        self.window.after(500, self._force_theme_update)

    def start(self):
        self.create_window()
        
        # Set up cleanup on window close
        def on_closing():
            self.window.destroy()
    
        self.window.protocol("WM_DELETE_WINDOW", on_closing)

        # Check config for start-on-top behavior
        if self.config.get("start_on_top", True):
            self.master.attributes('-topmost', True)
            self.master.after(100, lambda: self.master.attributes('-topmost', False))
        
        self.master.lift()
        #self.master.focus_force()

        self.window.mainloop()
