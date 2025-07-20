import tkinter as tk
from tkinter import ttk
import os
from .ResizableFrame import ResizableFrame

class FileInteractionPane(ResizableFrame):
    def __init__(self, parent, islat_class, theme):
        """
        Initialize the File Interaction Pane widget.
        Now inherits from ResizableFrame for consolidated theming.
        
        Args:
            parent: The parent widget to contain this pane
            islat_class: Reference to the main iSLAT class instance
            theme: Theme dictionary for styling
        """
        # Initialize ResizableFrame with theme
        super().__init__(parent, theme=theme)
        
        self.parent = parent
        self.islat_class = islat_class
        
        # Create the label frame for grouping
        self.label_frame = tk.LabelFrame(self, text="Spectrum File")
        self.label_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure grid layout for the label frame
        self.label_frame.grid_columnconfigure(0, weight=1)  # Label column
        self.label_frame.grid_columnconfigure(1, weight=0)  # Button column
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        
        # Row 0: Spectrum file
        self.file_label = tk.Label(self.label_frame, text=default_text, wraplength=180, anchor="w")
        self.file_label.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.load_spectrum_btn = tk.Button(
            self.label_frame, 
            text="Load Spectrum", 
            command=self.islat_class.load_spectrum
        )
        self.load_spectrum_btn.grid(row=0, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Row 1: Input line list
        self.input_line_list_label = tk.Label(
            self.label_frame, 
            text="Input Line List: None", 
            wraplength=180, 
            anchor="w",
            #font=("TkDefaultFont", 8)
        )
        self.input_line_list_label.grid(row=1, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.input_line_list_btn = tk.Button(
            self.label_frame,
            text="Load Line List",
            command=self._load_input_line_list
        )
        self.input_line_list_btn.grid(row=1, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Row 2: Output line measurements
        self.output_measurements_label = tk.Label(
            self.label_frame, 
            text="Output Measurements: None", 
            wraplength=180, 
            anchor="w",
            #font=("TkDefaultFont", 8)
        )
        self.output_measurements_label.grid(row=2, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.output_line_measurements_btn = tk.Button(
            self.label_frame,
            text="Set Output File",
            command=self._load_output_line_measurements
        )
        self.output_line_measurements_btn.grid(row=2, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Apply theme to all widgets
        self.apply_theme()
    
    
    def update_file_label(self, filename=None):
        """
        Update the file label text.
        
        Args:
            filename: The filename to display. If None, checks islat_class for loaded spectrum name.
        """
        if filename:
            display_text = f"Loaded: {filename}"
        elif hasattr(self.islat_class, 'loaded_spectrum_name') and self.islat_class.loaded_spectrum_name:
            display_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        else:
            display_text = "No file loaded"
        
        self.file_label.configure(text=display_text)
    
    def refresh(self):
        """
        Refresh the file interaction pane to show current state.
        This method can be called when the GUI needs to update its display.
        """
        self.update_file_label()
        self._update_status_labels()
        self.apply_theme()
    
    def _update_status_labels(self):
        """Update all status labels to reflect current loaded files."""
        # Update input line list label
        input_filename = self.get_input_line_list_filename()
        if input_filename:
            self.input_line_list_label.configure(text=f"Input Line List: {input_filename}")
        else:
            self.input_line_list_label.configure(text="Input Line List: None")
        
        # Update output measurements label
        output_filename = self.get_output_line_measurements_filename()
        if output_filename:
            self.output_measurements_label.configure(text=f"Output Measurements: {output_filename}")
        else:
            self.output_measurements_label.configure(text="Output Measurements: None")
    
    def get_loaded_filename(self):
        """
        Get the currently loaded filename.
        
        Returns:
            str: The filename if loaded, None otherwise
        """
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            return self.islat_class.loaded_spectrum_name
        return None
    
    def get_input_line_list_filename(self):
        """
        Get the currently loaded input line list filename.
        
        Returns:
            str: The filename if loaded, None otherwise
        """
        if hasattr(self.islat_class, 'input_line_list') and self.islat_class.input_line_list:
            return os.path.basename(self.islat_class.input_line_list)
        return None
    
    def get_output_line_measurements_filename(self):
        """
        Get the currently loaded output line measurements filename.
        
        Returns:
            str: The filename if loaded, None otherwise
        """
        if hasattr(self.islat_class, 'output_line_measurements') and self.islat_class.output_line_measurements:
            return os.path.basename(self.islat_class.output_line_measurements)
        return None
    
    def get_all_loaded_files(self):
        """
        Get information about all loaded files.
        
        Returns:
            dict: Dictionary with file types as keys and filenames as values
        """
        return {
            'spectrum': self.get_loaded_filename(),
            'input_line_list': self.get_input_line_list_filename(),
            'output_line_measurements': self.get_output_line_measurements_filename()
        }
    
    def _load_input_line_list(self):
        """
        Open file dialog to select input line list file and store in islat_class.
        """
        '''from tkinter import filedialog
        
        # Define appropriate file types for line lists
        filetypes = [
            #('Text Files', '*.txt'),
            #('CSV Files', '*.csv'),
            #('DAT Files', '*.dat'),
            ('All Files', '*.*')
        ]
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Input Line List File",
            filetypes=filetypes,
            initialdir=os.getcwd()
        )
        
        if file_path:
            # Store the file path in the islat_class
            self.islat_class.input_line_list = file_path
            filename = os.path.basename(file_path)
            print(f"Input line list loaded: {filename}")
            
            # Update the status label
            self.input_line_list_label.configure(text=f"Input Line List: {filename}")
        else:
            print("No input line list file selected.")'''
        from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
        filepath, filename = load_input_line_list(self.islat_class.input_line_list)
        self.islat_class.input_line_list = filepath
        self.input_line_list_label.configure(text=f"Input Line List: {filename}")
    
    def _load_output_line_measurements(self):
        """Calls the ifh class to save output line measurements."""
        from iSLAT.Modules.FileHandling.iSLATFileHandling import save_output_line_measurements
        filepath, filename = save_output_line_measurements(self.islat_class.output_line_measurements)
        self.islat_class.output_line_measurements = filepath
        self.output_measurements_label.configure(text=f"Output Measurements: {filename}")