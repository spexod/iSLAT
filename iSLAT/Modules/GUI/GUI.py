import tkinter as tk
from tkinter import filedialog, ttk, font, simpledialog, messagebox
import os

from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv

from .Widgets.DataField import DataField
from .Widgets.ControlPanel import ControlPanel
from .Widgets.TopBar import TopBar
from .Widgets.FileInteractionPane import FileInteractionPane

class GUI:
    def __init__(self, master, molecule_data, wave_data, flux_data, config, islat_class_ref):
        if master is None:
            self.master = tk.Tk()
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
        
        # Apply theme to root window
        # self._apply_theme_to_widget(self.master)

    def _style_config(self):
        self.style.configure("Small.TButton", padding=(0, 5))

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
                pass
                # self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Some widgets might not support certain options
            pass
    
    def _force_theme_update(self):
        """Force theme update on all widgets in the window."""
            
        if hasattr(self, 'plot') and hasattr(self.plot, 'apply_theme'):
            print("applying theme to plot")
            self.plot.apply_theme(self.theme)

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
        self.file_interaction_pane = FileInteractionPane(parent, self.islat_class, self.theme)
        
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
        self.window = self.master
        self.window.title("iSLAT Version 5.00.00")
        
        # Configure main window for resizable layout
        self.window.grid_rowconfigure(0, weight=0)
        self.window.grid_rowconfigure(1, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        # Create a main container frame
        main_container = ttk.Frame(self.window)
        main_container.grid(row=1, column=0, sticky="nsew")

        # Create frames for left panel and right panel (plot)
        left_main_frame = tk.Frame(main_container)
        left_main_frame.grid(row= 0, column= 0, sticky="nsew")

        right_main_frame = tk.Frame(main_container)
        right_main_frame.grid(row= 0, column= 1, sticky="nsew")
        
        # Configure right frame for responsive plot
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=0)
        main_container.grid_columnconfigure(1, weight=1)
        
        # Create the plot directly in right_frame without extra container
        self.plot = iSLATPlot(right_main_frame, self.wave_data, self.flux_data, self.theme, self.islat_class)

        # Left side: all controls
        self.build_left_panel(left_main_frame)

        # Bottom function buttons
        self.top_bar = TopBar(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.control_panel, self.config)
        self.top_bar.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Force theme updates to catch any missed widgets
        self.window.after(100, self._force_theme_update)
        # Additional delayed update to catch any widgets created asynchronously
        self.window.after(500, self._force_theme_update)

    def start(self):
        self.create_window()
    
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Check config for start-on-top behavior
        if self.config.get("start_on_top", True):
            self.master.attributes('-topmost', True)
            self.master.after(100, lambda: self.master.attributes('-topmost', False))
        
        self.master.lift()

        self.window.mainloop()

    def on_closing(self):
        #if messagebox.askokcancel("Quit", "Do you want to save your work?"):
        if True:
            spectrum_name = getattr(self.islat_class, 'loaded_spectrum_name', 'unknown')
            
            try:
                # Save the current molecule parameters
                write_molecules_to_csv(
                    self.islat_class.molecules_dict, 
                    #loaded_spectrum_name=spectrum_name
                )
            except Exception as e:
                print("Error", f"Failed to save molecule parameters: {str(e)}")
        self.window.destroy()

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