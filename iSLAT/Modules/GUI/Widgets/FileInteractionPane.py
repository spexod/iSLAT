import tkinter as tk
from tkinter import ttk
import os

class FileInteractionPane:
    def __init__(self, parent, islat_class, theme):
        """
        Initialize the File Interaction Pane widget.
        
        Args:
            parent: The parent widget to contain this pane
            islat_class: Reference to the main iSLAT class instance
            theme: Theme dictionary for styling
        """
        self.parent = parent
        self.islat_class = islat_class
        self.theme = theme
        
        # Create the main frame
        self.frame = tk.LabelFrame(parent, text="Spectrum File")
        self.frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Apply initial theme to the frame
        self._apply_theme_to_frame()
        
        # Configure grid layout for the frame
        self.frame.grid_columnconfigure(0, weight=1)  # Label column
        self.frame.grid_columnconfigure(1, weight=0)  # Button column
        
        # Initialize with default text or show loaded file name if available
        default_text = "No file loaded"
        if hasattr(self.islat_class, 'loaded_spectrum_name'):
            default_text = f"Loaded: {self.islat_class.loaded_spectrum_name}"
        
        # Row 0: Spectrum file
        self.file_label = tk.Label(self.frame, text=default_text, wraplength=180, anchor="w")
        self.file_label.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.load_spectrum_btn = tk.Button(
            self.frame, 
            text="Load Spectrum", 
            command=self.islat_class.load_spectrum
        )
        self.load_spectrum_btn.grid(row=0, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Row 1: Input line list
        self.input_line_list_label = tk.Label(
            self.frame, 
            text="Input Line List: None", 
            wraplength=180, 
            anchor="w",
            #font=("TkDefaultFont", 8)
        )
        self.input_line_list_label.grid(row=1, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.input_line_list_btn = tk.Button(
            self.frame,
            text="Load Line List",
            command=self._load_input_line_list
        )
        self.input_line_list_btn.grid(row=1, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Row 2: Output line measurements
        self.output_measurements_label = tk.Label(
            self.frame, 
            text="Output Measurements: None", 
            wraplength=180, 
            anchor="w",
            #font=("TkDefaultFont", 8)
        )
        self.output_measurements_label.grid(row=2, column=0, sticky="ew", padx=(5, 5), pady=2)
        
        self.output_line_measurements_btn = tk.Button(
            self.frame,
            text="Set Output File",
            command=self._load_output_line_measurements
        )
        self.output_line_measurements_btn.grid(row=2, column=1, sticky="e", padx=(5, 5), pady=2)
        
        # Apply theme to all widgets
        self.apply_theme(self.theme)
    
    def _apply_theme_to_frame(self):
        """Apply theme to the main frame."""
        self.frame.configure(
            bg=self.theme["background"],
            fg=self.theme["foreground"]
        )
    
    def apply_theme(self, theme):
        """
        Apply theme to all widgets in the file interaction pane.
        
        Args:
            theme: Theme dictionary containing color and style information
        """
        self.theme = theme
        
        # Apply theme to the main frame
        self.frame.configure(
            bg=theme["background"],
            fg=theme["foreground"]
        )
        
        # Apply theme to the file label
        self.file_label.configure(
            bg=theme["background"],
            fg=theme["foreground"]
        )
        
        # Apply theme to the additional status labels
        self.input_line_list_label.configure(
            bg=theme["background"],
            fg=theme["foreground"]
        )
        
        self.output_measurements_label.configure(
            bg=theme["background"],
            fg=theme["foreground"]
        )
        
        # Apply theme to the load spectrum button
        btn_theme = theme["buttons"].get("DefaultBotton", theme["buttons"]["DefaultBotton"])
        self.load_spectrum_btn.configure(
            bg=btn_theme["background"],
            fg=theme["foreground"],
            activebackground=btn_theme["active_background"],
            activeforeground=theme["foreground"]
        )
        
        # Apply theme to the input line list button
        self.input_line_list_btn.configure(
            bg=btn_theme["background"],
            fg=theme["foreground"],
            activebackground=btn_theme["active_background"],
            activeforeground=theme["foreground"]
        )
        
        # Apply theme to the output line measurements button
        self.output_line_measurements_btn.configure(
            bg=btn_theme["background"],
            fg=theme["foreground"],
            activebackground=btn_theme["active_background"],
            activeforeground=theme["foreground"]
        )
        
        # Apply theme recursively to all child widgets
        self._apply_theme_to_widget(self.frame)
    
    def _apply_theme_to_widget(self, widget):
        """
        Recursively apply theme to a widget and its children.
        
        Args:
            widget: The widget to apply theme to
        """
        try:
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'Toplevel', 'Tk']:
                widget.configure(bg=self.theme["background"])
            elif widget_class == 'LabelFrame':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Label':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"]
                )
            elif widget_class == 'Button':
                btn_theme = self.theme["buttons"].get("DefaultBotton", self.theme["buttons"]["DefaultBotton"])
                widget.configure(
                    bg=btn_theme["background"],
                    fg=self.theme["foreground"],
                    activebackground=btn_theme["active_background"],
                    activeforeground=self.theme["foreground"]
                )
            elif widget_class == 'Entry':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"]
                )
            elif widget_class == 'Text':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"],
                    insertbackground=self.theme["foreground"]
                )
            elif widget_class == 'Scrollbar':
                widget.configure(
                    bg=self.theme["background"],
                    troughcolor=self.theme["background"],
                    activebackground=self.theme["foreground"]
                )
            
            # Recursively apply theme to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
                
        except tk.TclError:
            # Some widgets might not support certain options
            pass
    
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
        self.apply_theme(self.theme)
    
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