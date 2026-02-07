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
    def __init__(self, parent, islat_class, theme, data_field):
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

        # CALCULATE IF DARK/LIGHT MODE TO DETERMINE BG COLOR
        #16 bit rgb values
        r16, g16, b16 = parent.winfo_rgb(parent.cget("bg"))
        #convert to 8bit
        r8 = r16 // 256
        g8 = g16 // 256
        b8 = b16 // 256
        #luminance calculation 
        lum = 0.2126 * r8 + 0.7152 * g8 + 0.0722 * b8
        if lum < 128:
            self.bg = "grey"
        else:
            self.bg = "white"

        self.max_len = 25

        self.parent = parent
        self.islat_class = islat_class
        self.data_field = data_field
        
        # Create the label frame for grouping
        self.label_frame = tk.LabelFrame(self, text="Input/Output Files", relief="solid", borderwidth=1)
        self.label_frame.grid(row=0, column=0, sticky="nsew")

        # Let row 0 and column 0 expand inside FileInteractionPane
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Configure label_frame columns: label expands, buttons don't
        self.label_frame.grid_columnconfigure(0, weight=1)  # Label column expands
        self.label_frame.grid_columnconfigure(1, weight=0)  # Button column fixed
        self.label_frame.grid_columnconfigure(2, weight=0)  # X button column fixed
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"{self.islat_class.loaded_spectrum_name}"

        # Row 0: Spectrum file
        self.file_label = trim_label(
            self.label_frame, 
            text=default_text,
            anchor="w", 
            bg =self.bg
        )
        self.file_label.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=2)

        if self.file_label.trimmed:
            self.file_label.tooltip = CreateToolTip(self.file_label, self.islat_class.loaded_spectrum_name)
        
        self.load_spectrum_btn = ttk.Button(
            self.label_frame, 
            text="Load Spectrum"
        )
        self.load_spectrum_btn.grid(row=0, column=1, columnspan=2, sticky="e", padx=(5, 5), pady=2)
        self.load_spectrum_btn.bind('<Button-1>', self._handle_load_spectrum_click)
        CreateToolTip(
            self.load_spectrum_btn, 
            "Click to load spectrum.\nCtrl/Cmd Click to load saved parameters."
        )
        
        # Row 1: Input line list
        # Check if a default line list is already loaded
        line_list_text = "None"
        if hasattr(self.islat_class, 'input_line_list') and self.islat_class.input_line_list:
            line_list_text = os.path.basename(self.islat_class.input_line_list)
        output_file_text = "None"
        if hasattr(self.islat_class, 'output_line_measurements') and self.islat_class.output_line_measurements:
            output_file_text = os.path.basename(self.islat_class.output_line_measurements)
        
        self.input_line_list_label = trim_label(
            self.label_frame, 
            text=line_list_text, 
            anchor="w",
            bg=self.bg
        )
        self.input_line_list_label.grid(row=1, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.input_line_list_btn = ttk.Button(
            self.label_frame,
            text="Load Line List",
            command=self._load_input_line_list
        )
        self.input_line_list_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=2)
        
        self.input_line_list_clear_btn = ttk.Button(
            self.label_frame,
            text="✕",
            width=2,
            command=self._clear_input_line_list
        )
        self.input_line_list_clear_btn.grid(row=1, column=2, sticky="e", padx=(0, 5), pady=2)
        CreateToolTip(self.input_line_list_clear_btn, "Clear line list")
        
        # Row 2: Output line measurements
        self.output_measurements_label = trim_label(
            self.label_frame, 
            text=output_file_text, 
            anchor="w",
            bg=self.bg
        )
        self.output_measurements_label.grid(row=2, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.output_line_measurements_btn = ttk.Button(
            self.label_frame,
            text="Set Output File",
            command=self._load_output_line_measurements
        )
        self.output_line_measurements_btn.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=2)
        
        self.output_measurements_clear_btn = ttk.Button(
            self.label_frame,
            text="✕",
            width=2,
            command=self._clear_output_line_measurements
        )
        self.output_measurements_clear_btn.grid(row=2, column=2, sticky="e", padx=(0, 5), pady=2)
        CreateToolTip(self.output_measurements_clear_btn, "Clear output file")

    def _handle_load_spectrum_click(self, event):
        """
        Handle load spectrum button clicks with modifier key support.
        Normal click: calls load_spectrum()
        Ctrl/Command click: calls load_spectrum(load_parameters=True) to also load saved parameters
        """
        # Check if Ctrl (Windows/Linux) or Command (Mac) key is pressed
        # On Windows/Linux: Control is state bit 2 (0x4)
        # On Mac: Command is state bit 3 (0x8), but we also check Control
        import platform
        ctrl_pressed = False
        
        if platform.system() == "Darwin":
            # Mac: Check for Command (0x8) or Control (0x4)
            ctrl_pressed = bool(event.state & 0x8) or bool(event.state & 0x4)
        else:
            # Windows/Linux: Check for Control (0x4)
            ctrl_pressed = bool(event.state & 0x4)
        
        if ctrl_pressed:
            # Ctrl/Command click - load spectrum with saved parameters
            self.islat_class.load_spectrum(load_parameters=True)
        else:
            # Normal click
            self.islat_class.load_spectrum()
    
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
        result = load_input_line_list(self.islat_class.input_line_list)
        
        if result is None:
            # User cancelled the dialog
            return
        
        filepath, filename = result
        self.islat_class.input_line_list = filepath

        self.update_label(self.input_line_list_label, filename)
        self.islat_class.GUI.plot.remove_saved_lines()
        self.data_field.insert_text(f"Loaded lines from: {filepath}")
    
    def _load_output_line_measurements(self):
        """Calls the ifh class to save output line measurements."""
        try:
            from iSLAT.Modules.FileHandling.iSLATFileHandling import save_output_line_measurements
            
            if not hasattr(self.islat_class, 'output_line_measurements'):
                self.islat_class.output_line_measurements = None

            #filepath, filename = save_output_line_measurements(self.islat_class.output_line_measurements)
            
            result = save_output_line_measurements(self.islat_class.output_line_measurements)
        
            if result is None:
                # User cancelled the dialog
                return
            
            filepath, filename = result
            self.islat_class.output_line_measurements = filepath
            self.data_field.insert_text(f"Set output line measurements file to: {filepath}")

            try:
                self.update_label(self.output_measurements_label, filename)
            except Exception as g:
                self.data_field.insert_text(f"Error updating output line measurements label: {g}")
                print(f"Error updating output line measurements label: {g}")
        except Exception as e:
            self.data_field.insert_text(f"Error setting output line measurements file: {e}")
            print(f"Error setting output line measurements file: {e}")
    
    def _clear_input_line_list(self):
        """Clear the input line list selection."""
        self.islat_class.input_line_list = None
        self.update_label(self.input_line_list_label, "None")
        if hasattr(self.islat_class, 'GUI') and hasattr(self.islat_class.GUI, 'plot'):
            self.islat_class.GUI.plot.remove_saved_lines()
        self.data_field.insert_text("Cleared input line list")
    
    def _clear_output_line_measurements(self):
        """Clear the output line measurements file selection."""
        self.islat_class.output_line_measurements = None
        self.update_label(self.output_measurements_label, "None")
        self.data_field.insert_text("Cleared output line measurements file")