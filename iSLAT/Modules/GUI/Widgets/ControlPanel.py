import tkinter as tk
from tkinter import ttk, colorchooser
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_control_panel_fields_config
from .ResizableFrame import ResizableFrame

class ControlPanel(ResizableFrame):
    def __init__(self, master, islat):
        # Get theme from islat
        theme = getattr(islat, 'config', {}).get('theme', {})
        
        # Initialize the ResizableFrame with theme
        super().__init__(master, theme=theme, borderwidth=2, relief="groove")
        
        self.master = master
        self.islat = islat
        
        # Load field configurations from JSON file using iSLAT file handling
        self._load_field_configurations()
        
        # Pack to the left side and fill vertically
        self.pack(side="left", fill="y")

        # Initialize all UI components
        self._create_all_components()
        
        self._register_callbacks()
        
        # Apply theming after everything is created
        self.after(50, lambda: self.apply_theme(theme))

    def _load_field_configurations(self):
        """Load field configurations from JSON file using iSLAT file handling"""
        try:
            config = load_control_panel_fields_config()
            self.GLOBAL_FIELDS = config.get('global_fields', {})
            self.MOLECULE_FIELDS = config.get('molecule_fields', {})
        except Exception as e:
            print(f"Error loading control panel field configurations: {e}")
            # Use fallback default configurations
            self.GLOBAL_FIELDS = {}
            self.MOLECULE_FIELDS = {}

    def _register_callbacks(self):
        """Register callbacks for UI synchronization only"""
        try:
            if hasattr(self.islat, 'add_active_molecule_change_callback'):
                self.islat.add_active_molecule_change_callback(self._on_active_molecule_change)
            
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
            
            Molecule.add_molecule_parameter_change_callback(self._on_molecule_parameter_change)
            
        except Exception as e:
            print(f"ControlPanel: Error registering callbacks: {e}")

    def _on_active_molecule_change(self, old_molecule, new_molecule):
        """Handle active molecule changes from the iSLAT callback system"""
        # Update the dropdown selection to match the new active molecule
        if hasattr(self, 'molecule_var') and hasattr(self, 'dropdown'):
            # Get the display label for the new molecule
            if hasattr(new_molecule, 'displaylabel'):
                display_label = new_molecule.displaylabel
            elif hasattr(new_molecule, 'name'):
                display_label = new_molecule.name
            elif isinstance(new_molecule, str):
                display_label = new_molecule
            else:
                display_label = str(new_molecule)
            
            # Update dropdown without triggering callback
            self.molecule_var.set(display_label)
        
        self._update_molecule_parameter_fields()
        self._update_color_and_visibility_controls()

    def _on_molecule_parameter_change(self, molecule_name, parameter_name, old_value, new_value):
        """Handle molecule parameter changes to update UI fields"""
        # Only update UI if this is the active molecule
        if (hasattr(self.islat, 'active_molecule') and 
            isinstance(self.islat.active_molecule, str) and 
            self.islat.active_molecule == molecule_name):
            
            # Update the specific parameter field if it exists
            if hasattr(self, '_molecule_parameter_entries') and parameter_name in self._molecule_parameter_entries:
                entry, var = self._molecule_parameter_entries[parameter_name]
                new_value_str = self._get_active_molecule_parameter_value(parameter_name)
                if var.get() != new_value_str:
                    var.set(new_value_str)
            
            # Update color and visibility controls if needed
            if parameter_name in ['color', 'is_visible']:
                self._update_color_and_visibility_controls()

    def _on_global_parameter_change(self, parameter_name, old_value, new_value):
        """Handle global parameter changes to update UI fields"""
        # Update the specific global parameter field if it exists
        if hasattr(self, '_global_parameter_entries') and parameter_name in self._global_parameter_entries:
            entry, var = self._global_parameter_entries[parameter_name]
            if var.get() != str(new_value):
                var.set(str(new_value))

    def _create_all_components(self):
        """Create all control panel components in order"""
        self._create_display_controls(0, 0)
        self._create_wavelength_controls(1, 0)  
        self._create_global_parameter_controls(2, 0)  # Only distance now
        self._create_molecule_specific_controls(3, 0)  # All other params here
        self._create_molecule_selector(9, 0)  # Move down to accommodate molecule params
        self._create_molecule_color_and_visibility_controls(10, 0)  # Add color and visibility controls
        self._reload_molecule_dropdown()

    def _create_simple_entry(self, label_text, initial_value, row, col, on_change_callback, width=8):
        """Create a simple entry field with label and change callback"""
        label = tk.Label(self, text=label_text)
        label.grid(row=row, column=col, padx=5, pady=5)
        
        # Apply theme to the label
        label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )
        
        var = tk.StringVar()
        var.set(str(initial_value))
        
        entry = tk.Entry(self, textvariable=var, width=width)
        entry.grid(row=row, column=col + 1, padx=5, pady=5)
        
        # Apply theme to the entry
        entry.configure(
            bg=self.theme.get("background_accent_color", "#23272A"),
            fg=self.theme.get("foreground", "#F0F0F0"),
            insertbackground=self.theme.get("foreground", "#F0F0F0"),
            selectbackground=self.theme.get("selection_color", "#00FF99"),
            selectforeground=self.theme.get("background", "#181A1B")
        )
        
        def on_change(*args):
            on_change_callback(var.get())
        
        var.trace_add("write", on_change)
        return entry, var

    def _create_bound_parameter_entry(self, label_text, param_name, row, col, width=8):
        """Create an entry bound to a global parameter in molecules_dict"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return None, None
            
        molecules_dict = self.islat.molecules_dict
        current_value = getattr(molecules_dict, param_name, 0)
        
        def update_parameter(value_str):
            try:
                value = float(value_str)
                old_value = getattr(molecules_dict, param_name)
                if abs(old_value - value) > 1e-10:
                    setattr(molecules_dict, param_name, value)
            except (ValueError, AttributeError):
                pass
        
        return self._create_simple_entry(label_text, current_value, row, col, update_parameter, width)

    def _create_display_controls(self, start_row, start_col):
        """Create plot start and range controls for display view"""
        # Plot start
        initial_start = getattr(self.islat, 'display_range', [4.5, 5.5])[0]
        self.plot_start_entry, self.plot_start_var = self._create_simple_entry(
            "Plot start:", initial_start, start_row, start_col, self._update_display_range)
        
        # Plot range  
        display_range = getattr(self.islat, 'display_range', [4.5, 5.5])
        initial_range = display_range[1] - display_range[0]
        self.plot_range_entry, self.plot_range_var = self._create_simple_entry(
            "Plot range:", initial_range, start_row, start_col + 2, self._update_display_range)

    def _create_wavelength_controls(self, start_row, start_col):
        """Create wavelength range controls for model calculation range"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        molecules_dict = self.islat.molecules_dict
        min_wave, max_wave = molecules_dict.global_wavelength_range
        
        self.min_wavelength_entry, self.min_wavelength_var = self._create_simple_entry(
            "Min. Wave:", min_wave, start_row, start_col, self._update_wavelength_range)
        self.max_wavelength_entry, self.max_wavelength_var = self._create_simple_entry(
            "Max. Wave:", max_wave, start_row, start_col + 2, self._update_wavelength_range)

    def _create_global_parameter_controls(self, start_row, start_col):
        """Create global parameter entry fields using MoleculeDict properties"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            label = tk.Label(self, text="Global parameters not available")
            label.grid(row=start_row, column=start_col, columnspan=4, padx=5, pady=5)
            # Apply theme to the label
            label.configure(
                bg=self.theme.get("background", "#181A1B"),
                fg=self.theme.get("foreground", "#F0F0F0")
            )
            return

        # Store references for later updates
        self._global_parameter_entries = {}
        
        # Create fields based on the class-level dictionary
        row_offset = 1
        col_offset = 0
        
        for field_key, field_config in self.GLOBAL_FIELDS.items():
            # Calculate grid position (2 fields per row)
            row = start_row + row_offset + (col_offset // 2)
            col = start_col + (col_offset % 2) * 2
            
            entry, var = self._create_global_parameter_entry(
                field_config['label'], 
                field_config['property'], 
                row, 
                col, 
                field_config['width']
            )
            
            if entry and var:
                self._global_parameter_entries[field_config['property']] = (entry, var)
            
            col_offset += 1

    def _create_global_parameter_entry(self, label_text, property_name, row, col, width=12):
        """Create an entry bound to a global parameter in molecules_dict"""
        
        def update_global_parameter(value_str):
            if value_str in ["N/A", ""]:
                return
                
            if not hasattr(self.islat, 'molecules_dict') or not self.islat.molecules_dict:
                return
                
            molecules_dict = self.islat.molecules_dict
            
            try:
                field_config = None
                for field_key, config in self.GLOBAL_FIELDS.items():
                    if config['property'] == property_name:
                        field_config = config
                        break
                
                if field_config:
                    value = field_config['datatype'](value_str)
                else:
                    value = float(value_str)
                    
                old_value = getattr(molecules_dict, property_name, None)
                
                if field_config and field_config['datatype'] == float:
                    try:
                        old_float_value = float(old_value) if old_value is not None else None
                    except (ValueError, TypeError):
                        old_float_value = None
                    
                    if old_float_value is None or abs(old_float_value - value) > 1e-10:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(molecules_dict, property_name, value)
                else:
                    if old_value != value:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(molecules_dict, property_name, value)
                            
            except (ValueError, AttributeError) as e:
                print(f"Error updating global {property_name}: {e}")
        
        # Get current value from molecules_dict
        current_value = 0.0
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            current_value = getattr(self.islat.molecules_dict, property_name, 0.0)
        
        return self._create_simple_entry(label_text, current_value, row, col, update_global_parameter, width)

    def _create_molecule_specific_controls(self, start_row, start_col):
        """Create controls for molecule-specific parameters that update with active molecule"""
        # Store references for later updates
        self._molecule_parameter_entries = {}
        
        # Create fields based on the class-level dictionary
        row_offset = 1
        col_offset = 0
        
        for field_key, field_config in self.MOLECULE_FIELDS.items():
            # Calculate grid position (2 fields per row)
            row = start_row + row_offset + (col_offset // 2)
            col = start_col + (col_offset % 2) * 2
            
            entry, var = self._create_molecule_parameter_entry(
                field_config['label'], 
                field_config['attribute'], 
                row, 
                col, 
                field_config['width']
            )
            
            if entry and var:
                self._molecule_parameter_entries[field_config['attribute']] = (entry, var)
            
            col_offset += 1

    def _create_molecule_parameter_entry(self, label_text, param_name, row, col, width=12):
        """Create an entry bound to the active molecule's parameter"""
        
        def update_active_molecule_parameter(value_str):
            # Skip updates for special cases
            if value_str in ["N/A", ""]:
                return
                
            if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
                return
                
            # Get the active molecule object
            active_mol = None
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                    active_mol = self.islat.molecules_dict[self.islat.active_molecule]
                elif hasattr(self.islat.active_molecule, 'name'):
                    active_mol = self.islat.active_molecule
            
            if not active_mol:
                return
                
            try:
                # Get field configuration for proper data type handling
                field_config = None
                for field_key, config in self.MOLECULE_FIELDS.items():
                    if config['attribute'] == param_name:
                        field_config = config
                        break
                
                # Convert value based on field configuration
                if field_config:
                    value = field_config['datatype'](value_str)
                else:
                    # Fallback for unknown parameters
                    value = float(value_str) if param_name in ["distance", "stellar_rv", "fwhm", "broad"] else str(value_str)
                    
                # Get old value for comparison
                old_value = getattr(active_mol, param_name, None)
                
                # Only update if the value actually changed
                if field_config and field_config['datatype'] == float:
                    # Convert old_value to float for proper comparison
                    try:
                        old_float_value = float(old_value) if old_value is not None else None
                    except (ValueError, TypeError):
                        old_float_value = None
                    
                    if old_float_value is None or abs(old_float_value - value) > 1e-10:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(active_mol, param_name, value)
                else:
                    if old_value != value:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(active_mol, param_name, value)
                            
            except (ValueError, AttributeError) as e:
                print(f"Error updating {param_name}: {e}")
        
        # Get initial value from active molecule
        initial_value = self._get_active_molecule_parameter_value(param_name)
        
        return self._create_simple_entry(label_text, initial_value, row, col, update_active_molecule_parameter, width)

    def _get_active_molecule_parameter_value(self, param_name):
        """Get the current value of a parameter from the active molecule"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return ""
        
        # Get the active molecule object
        active_mol = None
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                active_mol = self.islat.molecules_dict[self.islat.active_molecule]
            elif hasattr(self.islat.active_molecule, 'name'):
                active_mol = self.islat.active_molecule
        
        if not active_mol:
            return ""
            
        try:
            value = getattr(active_mol, param_name, "")
            
            # Get field configuration for proper formatting
            field_config = None
            for field_key, config in self.MOLECULE_FIELDS.items():
                if config['attribute'] == param_name:
                    field_config = config
                    break
            
            # Format value based on field configuration
            if field_config and isinstance(value, (int, float)):
                return field_config['format'].format(value)
            # Fallback formatting for backward compatibility
            elif param_name in ["distance", "stellar_rv", "fwhm", "broad"] and isinstance(value, (int, float)):
                return f"{value:.2f}"
            
            return str(value)
        except:
            return ""

    def _create_molecule_selector(self, row, column):
        """Create molecule dropdown selector"""
        label = tk.Label(self, text="Molecule:")
        label.grid(row=row, column=column, padx=5, pady=5)
        
        # Apply theme to the label
        label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )

        self.molecule_var = tk.StringVar(self)
        self.dropdown = ttk.Combobox(self, textvariable=self.molecule_var)
        self.dropdown.grid(row=row, column=column + 1, padx=5, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self._on_molecule_selected)
        
        # Apply theming to the control panel after all components are created
        self.after(10, self._apply_theming)

    def _create_molecule_color_and_visibility_controls(self, row, column):
        """Create color button and visibility checkbox for the active molecule"""
        # Visibility checkbox
        visibility_label = tk.Label(self, text="Visible:")
        visibility_label.grid(row=row, column=column, padx=5, pady=5)
        
        # Apply theme to the label
        visibility_label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )
        
        self.visibility_var = tk.BooleanVar()
        self.visibility_checkbox = tk.Checkbutton(
            self, 
            variable=self.visibility_var, 
            command=self._on_visibility_changed
        )
        self.visibility_checkbox.grid(row=row, column=column + 1, padx=5, pady=5)
        
        # Apply theme to checkbutton
        self.visibility_checkbox.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0"),
            activebackground=self.theme.get("background", "#181A1B"),
            activeforeground=self.theme.get("foreground", "#F0F0F0"),
            selectcolor=self.theme.get("background_accent_color", "#23272A")
        )
        
        # Color button
        color_label = tk.Label(self, text="Color:")
        color_label.grid(row=row, column=column + 2, padx=5, pady=5)
        
        # Apply theme to the label
        color_label.configure(
            bg=self.theme.get("background", "#181A1B"),
            fg=self.theme.get("foreground", "#F0F0F0")
        )
        
        # Get default color for initialization
        default_colors = self.theme.get("default_molecule_colors", ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])
        default_color = default_colors[0]
        
        self.color_button = tk.Button(
            self, 
            bg=default_color, 
            width=4,
            command=self._on_color_button_clicked
        )
        self.color_button.grid(row=row, column=column + 3, padx=5, pady=5)
        # Mark this as a color selection button so theming will ignore it
        self.color_button._is_color_button = True
        
        # Initialize with current active molecule data
        self._update_color_and_visibility_controls()

    def _ensure_molecule_color_initialized(self, mol_obj):
        """Ensure molecule has a color assigned, using MoleculeWindow logic"""
        if not hasattr(mol_obj, 'color') or mol_obj.color is None:
            default_colors = self.theme.get("default_molecule_colors", ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])
            
            # Use molecule index for consistent coloring, similar to MoleculeWindow
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                molecules_list = list(self.islat.molecules_dict.keys())
                try:
                    mol_index = molecules_list.index(mol_obj.name)
                except (ValueError, AttributeError):
                    mol_index = 0
            else:
                mol_index = 0
                
            mol_obj.color = default_colors[mol_index % len(default_colors)]

    def _on_visibility_changed(self):
        """Handle visibility checkbox changes for individual molecule plotting"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return
            
        # Get the active molecule object
        active_mol = self._get_active_molecule_object()
        if not active_mol:
            return
            
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        
        new_visibility = self.visibility_var.get()
        molecule_name = active_mol.name
        
        # Simply toggle this molecule's visibility - don't affect other molecules
        self.islat.molecules_dict.bulk_set_visibility(new_visibility, [molecule_name])
        
        # Debug: Verify the visibility was actually set
        print(f"ControlPanel: Set {molecule_name} visibility to {new_visibility}, actual value: {getattr(active_mol, 'is_visible', 'UNDEFINED')}")
        
        # Trigger selective plot refresh to show/hide the molecule
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot') and hasattr(self.islat.GUI.plot, 'on_molecule_visibility_changed'):
            self.islat.GUI.plot.on_molecule_visibility_changed(molecule_name, new_visibility)
            print(f"ControlPanel: Triggered selective plot refresh for visibility change")
        elif hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot') and hasattr(self.islat.GUI.plot, 'update_all_plots'):
            # Fallback to full refresh if selective rendering not available
            self.islat.GUI.plot.update_all_plots()
            print(f"ControlPanel: Triggered full plot refresh for visibility change")

    def _on_color_button_clicked(self):
        """Handle color button clicks to open color chooser"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return
            
        # Get the active molecule object
        active_mol = self._get_active_molecule_object()
        if not active_mol:
            return
            
        # Get molecule name for the color chooser title
        mol_name = getattr(active_mol, 'displaylabel', getattr(active_mol, 'name', 'Molecule'))
        
        # Open color chooser
        color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
        if color_code:
            # Store old color for notification
            old_color = getattr(active_mol, 'color', None)
            
            # Update molecule color
            active_mol.color = color_code
            self.color_button.config(bg=color_code)
            
            # Manually trigger the molecule parameter change notification
            # since color is not a property with a setter
            if hasattr(active_mol, '_notify_my_parameter_change'):
                active_mol._notify_my_parameter_change('color', old_color, color_code)
            
            # Trigger plot refresh to show color change
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot') and hasattr(self.islat.GUI.plot, 'update_all_plots'):
                self.islat.GUI.plot.update_all_plots()
                print(f"ControlPanel: Triggered plot refresh for color change")

    def _get_active_molecule_object(self):
        """Get the active molecule object, similar to MoleculeWindow logic"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return None
            
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                return self.islat.molecules_dict[self.islat.active_molecule]
            elif hasattr(self.islat.active_molecule, 'name'):
                return self.islat.active_molecule
        
        return None

    def _update_color_and_visibility_controls(self):
        """Update color button and visibility checkbox based on active molecule"""
        if not hasattr(self, 'color_button') or not hasattr(self, 'visibility_checkbox'):
            return
            
        # Enable controls for molecules
        self.visibility_checkbox.configure(state='normal')
        self.color_button.configure(state='normal')
        
        # Get the active molecule object
        active_mol = self._get_active_molecule_object()
        if not active_mol:
            return
            
        # Ensure molecule has a color
        self._ensure_molecule_color_initialized(active_mol)
        
        # Update visibility checkbox - simply reflect this molecule's visibility state
        is_visible = getattr(active_mol, 'is_visible', False)
        self.visibility_var.set(is_visible)
        
        # Update color button
        color = getattr(active_mol, 'color', "#FF6B6B")
        self.color_button.config(bg=color)

    def _apply_theming(self):
        """Apply theme to all control panel widgets"""
        # Use the theme from self.theme
        if not self.theme:
            return
            
        # Configure TTK styles for combobox
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Use a theme that supports customization
            
            # Configure combobox
            style.configure("ControlPanel.TCombobox",
                          fieldbackground=self.theme.get("background_accent_color", "#23272A"),
                          background=self.theme.get("background_accent_color", "#23272A"),
                          foreground=self.theme.get("foreground", "#F0F0F0"),
                          bordercolor=self.theme.get("background_accent_color", "#23272A"),
                          selectbackground=self.theme.get("selection_color", "#00FF99"),
                          selectforeground=self.theme.get("background", "#181A1B"))
            
            style.map("ControlPanel.TCombobox",
                     fieldbackground=[('readonly', self.theme.get("background_accent_color", "#23272A"))],
                     selectbackground=[('readonly', self.theme.get("selection_color", "#00FF99"))])
            
            self.dropdown.configure(style="ControlPanel.TCombobox")
            
        except Exception as e:
            print(f"Could not apply TTK theming: {e}")
        
        # Apply inherited theme method
        super().apply_theme()
    
    def apply_theme(self, theme=None):
        """Public method to apply theme to the control panel and all its widgets"""
        # Call parent's apply_theme first
        super().apply_theme(theme)
        
        # Apply specialized TTK styling for Combobox
        self._apply_ttk_styling()
    
    def _apply_ttk_styling(self):
        """Apply specialized TTK styling for control panel widgets"""
        try:
            # Apply TTK styling for Combobox and other TTK widgets
            style = ttk.Style()
            style.theme_use('clam')
            
            # Configure Combobox styling
            style.configure("TCombobox",
                          fieldbackground=self.theme.get("background_accent_color", "#23272A"),
                          background=self.theme.get("background_accent_color", "#23272A"),
                          foreground=self.theme.get("foreground", "#F0F0F0"),
                          bordercolor=self.theme.get("foreground", "#F0F0F0"),
                          arrowcolor=self.theme.get("foreground", "#F0F0F0"),
                          selectbackground=self.theme.get("selection_color", "#00FF99"),
                          selectforeground=self.theme.get("background", "#181A1B"))
            
            style.map("TCombobox",
                     fieldbackground=[('active', self.theme.get("background_accent_color", "#23272A")),
                                    ('focus', self.theme.get("background_accent_color", "#23272A"))],
                     background=[('active', self.theme.get("background_accent_color", "#23272A")),
                               ('focus', self.theme.get("background_accent_color", "#23272A"))],
                     foreground=[('active', self.theme.get("foreground", "#F0F0F0")),
                               ('focus', self.theme.get("foreground", "#F0F0F0"))])
                               
        except Exception as e:
            print(f"Could not apply TTK theming: {e}")
    
    def _apply_theme_to_widget(self, widget):
        """Override to add special handling for color buttons"""
        try:
            widget_class = widget.winfo_class()
            
            # Special handling for color buttons
            if widget_class == 'Button' and hasattr(widget, '_is_color_button') and widget._is_color_button:
                # This is a color selection button - preserve its molecule color, don't theme it
                pass
            else:
                # Use parent's theming logic for all other widgets
                super()._apply_theme_to_widget(widget)
                
        except tk.TclError:
            pass

    def _update_display_range(self, value_str=None):
        """Update display range from either start or range change"""
        try:
            start = float(self.plot_start_var.get())
            range_val = float(self.plot_range_var.get())
            self._set_display_range(start, start + range_val)
        except (ValueError, AttributeError):
            pass

    def _set_display_range(self, start, end):
        if hasattr(self.islat, 'display_range'):
            self.islat.display_range = (start, end)

    def _update_wavelength_range(self, value_str=None):
        """Update wavelength range for model calculations (not display)"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        try:
            min_val = float(self.min_wavelength_var.get())
            max_val = float(self.max_wavelength_var.get())
            
            if min_val < max_val:
                molecules_dict = self.islat.molecules_dict
                molecules_dict.global_wavelength_range = (min_val, max_val)
                if hasattr(self.islat, 'wavelength_range'):
                    self.islat.wavelength_range = (min_val, max_val)
        except (ValueError, AttributeError):
            pass

    def _update_molecule_parameter(self, value_str=None):
        """Update molecule-specific parameter UI fields for the active molecule"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        try:
            # Get the active molecule name
            active_molecule_name = getattr(self.islat, 'active_molecule', None)
            if not active_molecule_name:
                return
            
            molecules_dict = self.islat.molecules_dict
            molecule_obj = molecules_dict.get(active_molecule_name, None)
            if not molecule_obj:
                return
            
            # Update each molecule-specific parameter UI field
            for param_name, (entry, var) in self._molecule_parameter_entries.items():
                if hasattr(molecule_obj, param_name):
                    value = getattr(molecule_obj, param_name)
                    var.set(str(value))
        
        except Exception as e:
            print(f"Error updating molecule parameter UI fields: {e}")

    def _on_molecule_selected(self, event=None):
        """Handle molecule selection - uses iSLAT's active_molecule property"""
        selected_label = self.molecule_var.get()
        
        try:
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                for mol_name, mol_obj in self.islat.molecules_dict.items():
                    display_label = getattr(mol_obj, 'displaylabel', mol_name)
                    if display_label == selected_label:
                        self.islat.active_molecule = mol_name
                        self._update_color_and_visibility_controls()
                        return
                
                first_mol = next(iter(self.islat.molecules_dict.keys()), None)
                if first_mol:
                    self.islat.active_molecule = first_mol
                    self._update_color_and_visibility_controls()
        except Exception as e:
            print(f"Error setting active molecule: {e}")

    def _reload_molecule_dropdown(self):
        """Reload molecule dropdown options"""
        if not hasattr(self, 'dropdown'):
            return
            
        options = []
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            molecule_options = [
                getattr(mol_obj, 'displaylabel', mol_name) 
                for mol_name, mol_obj in self.islat.molecules_dict.items()
            ]
            options = molecule_options
        
        self.dropdown['values'] = options
        
        # Set default value if current selection is invalid
        current_value = self.molecule_var.get()
        if current_value not in options and options:
            self.molecule_var.set(options[0])
            self._on_molecule_selected()

    def refresh_from_molecules_dict(self):
        """Refresh all fields from current molecules_dict values"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        molecules_dict = self.islat.molecules_dict
        
        # Re-register callbacks in case molecules_dict was created after ControlPanel
        try:
            molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
        except:
            pass  # Callback might already be registered
        
        # Update global parameter fields
        self._update_global_parameter_fields()
        
        # Update wavelength range fields
        if (hasattr(self, 'min_wavelength_var') and hasattr(self, 'max_wavelength_var') 
            and hasattr(molecules_dict, 'global_wavelength_range')):
            min_val, max_val = molecules_dict.global_wavelength_range
            self.min_wavelength_var.set(str(min_val))
            self.max_wavelength_var.set(str(max_val))
        
        # Update molecule-specific parameter fields
        self._update_molecule_parameter_fields()
        
        self._reload_molecule_dropdown()
        
        self.apply_theme()

    def cleanup(self):
        try:
            if hasattr(self.islat, 'remove_active_molecule_change_callback'):
                self.islat.remove_active_molecule_change_callback(self._on_active_molecule_change)
        except Exception as e:
            print(f"Error during ControlPanel cleanup: {e}")

    def _update_molecule_parameter_fields(self):
        """Update all molecule-specific parameter fields with values from the active molecule"""
        if not hasattr(self, '_molecule_parameter_entries'):
            return
            
        for param_name, (entry, var) in self._molecule_parameter_entries.items():
            new_value = self._get_active_molecule_parameter_value(param_name)
            current_value = var.get()
            if current_value != new_value:
                var.set(new_value)

    def _update_global_parameter_fields(self):
        """Update all global parameter fields with values from the molecules_dict"""
        if not hasattr(self, '_global_parameter_entries'):
            return
        
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        for property_name, (entry, var) in self._global_parameter_entries.items():
            try:
                new_value = getattr(self.islat.molecules_dict, property_name, 0.0)
                current_value = var.get()
                if str(current_value) != str(new_value):
                    var.set(str(new_value))
            except (AttributeError, TypeError):
                pass
