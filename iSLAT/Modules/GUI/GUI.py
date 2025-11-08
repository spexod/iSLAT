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

    def get_plot_renderer(self):
        return self.plot.plot_renderer

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