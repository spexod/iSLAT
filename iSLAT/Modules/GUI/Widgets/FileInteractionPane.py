import tkinter as tk
from tkinter import ttk
import os
from .ResizableFrame import ResizableFrame
from ..Tooltips import CreateToolTip

class trim_label(tk.Label):
    def __init__(self, parent, max_len: int = 25, **kwargs):
        super().__init__(parent, **kwargs)
        self.max_len = max_len
        self.trimmed = False
        self.tooltip = None

        if self.cget("text"):
            self.trim_text()

    def trim_text(self):
        text = self.cget("text")
        if len(text) > self.max_len:
            text = text[:self.max_len - 3] + "..."
            self.config(text = text)
            self.trimmed = True
        else:
            self.trimmed = False
    
class FileInteractionPane(ttk.Frame):
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
        super().__init__(parent)
        
        self.max_len = 25

        self.parent = parent
        self.islat_class = islat_class
        
        # Create the label frame for grouping
        self.label_frame = tk.LabelFrame(self, text="Input/Output Files", relief="solid", borderwidth=1)
        self.label_frame.grid(row=0, column=0, sticky="nsew")

        # Let row 0 and column 0 expand inside FileInteractionPane
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
                
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"{self.islat_class.loaded_spectrum_name}"

        # Row 0: Spectrum file
        self.file_label = trim_label(
            self.label_frame, 
            text=default_text,
            anchor="w", 
            bg="white"
        )
        self.file_label.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=2)

        if self.file_label.trimmed:
            self.file_label.tooltip =  CreateToolTip(self.file_label, self.islat_class.loaded_spectrum_name)
        
        self.load_spectrum_btn = ttk.Button(
            self.label_frame, 
            text="Load Spectrum", 
            command=self.islat_class.load_spectrum
        )
        self.load_spectrum_btn.grid(row=0, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Row 1: Input line list
        self.input_line_list_label = trim_label(
            self.label_frame, 
            text="None", 
            anchor="w",
            bg="white"
        )
        self.input_line_list_label.grid(row=1, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.input_line_list_btn = ttk.Button(
            self.label_frame,
            text="Load Line List",
            command=self._load_input_line_list
        )
        self.input_line_list_btn.grid(row=1, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Row 2: Output line measurements
        self.output_measurements_label = tk.Label(
            self.label_frame, 
            text="None", 
            anchor="w",
            bg="white"
        )
        self.output_measurements_label.grid(row=2, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.output_line_measurements_btn = ttk.Button(
            self.label_frame,
            text="Set Output File",
            command=self._load_output_line_measurements
        )
        self.output_line_measurements_btn.grid(row=2, column=1, sticky="e", padx=(5, 5), pady=2)

    def update_label(self, widget, text = None):
        widget.configure(text=text)
        widget.trim_text()

        if widget.trimmed:
            widget.tooltip = CreateToolTip(widget, text)
        else:
            if widget.tooltip:
                widget.tooltip.disable()
            else:
                return
    
    def update_file_label(self, filename=None):
        """
        Update the file label text.
        
        Args:
            filename: The filename to display. If None, checks islat_class for loaded spectrum name.
        """
        if filename:
            display_text = f"{filename}"
        elif hasattr(self.islat_class, 'loaded_spectrum_name') and self.islat_class.loaded_spectrum_name:
            display_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        else:
            display_text = "No file loaded"

        self.update_label(self.file_label, text=display_text)
    
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
        from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
        filepath, filename = load_input_line_list(self.islat_class.input_line_list)
        self.islat_class.input_line_list = filepath

        self.update_label(self.input_line_list_label, filename)
        self.islat_class.GUI.plot.plot_renderer.remove_saved_lines()
    
    def _load_output_line_measurements(self):
        """Calls the ifh class to save output line measurements."""
        from iSLAT.Modules.FileHandling.iSLATFileHandling import save_output_line_measurements
        filepath, filename = save_output_line_measurements(self.islat_class.output_line_measurements)
        self.islat_class.output_line_measurements = filepath

        self.update_label(self.output_measurements_label, text=filename)