import traceback
import tkinter as tk
from tkinter import ttk, colorchooser
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_control_panel_fields_config
from ..GUIFunctions import create_wrapper_frame, create_scrollable_frame, ColorButton
from .RegularFrame import RegularFrame
from ..Tooltips import CreateToolTip

class ControlPanel(ttk.Frame):
    def __init__(self, master, islat, plot, data_field, font):

        super().__init__(master)
        
        self.master = master
        self.islat = islat
        self.plot = plot
        self.data_field = data_field
        self.font = font

        self.mol_dict = islat.molecules_dict
        self.updating = False

        self.mol_visibility = {}
        self.column_labels = {
            "On": "turn on/off this\nmodel in the plot ",
            "Molecule": "Select active molecule", 
            "Del.": "remove this model\nfrom the GUI",
            "Color": "change color\nfor this model"
            }
        
        self.label_frame = tk.LabelFrame(self, text="Spectrum and Models Control Panel", relief="solid", borderwidth=1)
        self.label_frame.grid(row=0, column=0, sticky="nsew", pady=0)
        self.label_frame.grid_rowconfigure(0,weight=1)


        bg_frame = tk.Frame(self)
        self.bg_color = bg_frame.cget('bg')
        bg_frame.destroy()
        self.selected_color = "#007BFF"
        
        self.max_name_len = 4
        # Load field configurations from JSON file using iSLAT file handling
        self._load_field_configurations()

        # Initialize all UI components
        self._create_all_components()
        self._register_callbacks()

    def _create_all_components(self):
        """Create all control panel components in order"""
        gen_config_frame = self._create_general_config_frame()
        molecule_param_frame = self._create_molecule_param_frame()
        self.color_vis_frame = self._create_color_and_vis_frame()

        self._create_display_controls(gen_config_frame, 0, 0)
        self._create_wavelength_controls(gen_config_frame, 1, 0)  
        self._create_global_parameter_controls(gen_config_frame, 2, 0)  

        self._create_molecule_specific_controls(molecule_param_frame, 0, 0)  # All other params here

        self._build_color_and_vis_controls(self.color_vis_frame) # my implementation

        self.grid_rowconfigure(1, weight=1)  # Because you placed the wrapper at row 1
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

    def _create_general_config_frame(self):
        wrapper = create_wrapper_frame(self.label_frame, 1, 0, columnspan = 2)

        general_param_frame = ttk.Frame(wrapper)
        general_param_frame.grid(row=0, column=0, sticky="nsew")
        
        return general_param_frame
    
    def _create_molecule_param_frame(self):
       wrapper = create_wrapper_frame(self.label_frame, 0, 1)
       self._create_selected_frame(wrapper, 0, 0)
       molecule_param_frame = create_scrollable_frame(wrapper, height=250, width= 170, horizontal=True, row=1, col=0)

       molecule_param_frame.rowconfigure(0, weight=1)
       molecule_param_frame.columnconfigure(0, weight=1)

       return molecule_param_frame

    def _create_color_and_vis_frame(self):
        wrapper = create_wrapper_frame(self.label_frame, 0, 0, sticky="nsew")

        color_vis_frame = create_scrollable_frame(wrapper, height=250, width = 160, vertical=True)

        return color_vis_frame

    def _create_selected_frame(self, parent, row, col):
        selected_frame = tk.Frame(parent, bg = "darkgrey")
        selected_frame.grid(row=row, column = col, sticky="nsew")

        self.selected_label = tk.Label(selected_frame, background="darkgrey", anchor="center", justify="center")
        self.selected_label.grid(row=0, column=0, sticky="nsew")

        selected_frame.grid_rowconfigure(0, weight=1)
        selected_frame.grid_columnconfigure(0, weight=1)

    def _create_simple_entry(self, parent, label_text, initial_value, row, col, on_change_callback, width=7, param_name = None, tip_text = None):
        """Create a simple entry field with label and change callback"""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=col, padx=1, pady=5)

        if tip_text: 
            CreateToolTip(label, tip_text)
        
        var = tk.StringVar()
        value_str = initial_value
        if not isinstance(value_str, str):
            value_str = self._format_value(initial_value, param_name=param_name)
        
        var.set(value_str)
        
        entry = tk.Entry(
            parent, 
            textvariable=var, 
            width=width, 
            justify="left", 
        )
        
        entry.grid(row=row, column=col + 1, padx=1, sticky="w")
        
        def on_change(*args):
            self.updating = True
            try:
                value = float(var.get())
                
                match param_name:
                    case "temp":
                        if value < 1:
                            self.data_field.insert_text("Cannot set temperature to less than 1")
                            return
                    case "n_mol":
                        if value == 0:
                            self.data_field.insert_text("Cannot set Column Density to 0")
                            return
                    case "radius":
                        if value <= 0: 
                            self.data_field.insert_text("Cannot set radius to 0 or negative")
                            return
                    
                on_change_callback(value)
                value_str = self._format_value(value, param_name)
                var.set(value_str)
                entry.configure(fg="black", font=(self.font.cget("family"), self.font.cget("size"), "roman"))
            except ValueError as e:
                print(f"Error with new value: {e}")
                self.data_field.insert_text(f"Error with new value: {e}")
            finally:
                self.updating = False
        
        def on_write(*args):
            if self.updating: # Updating means that the entry variable is being updated from iSLAT and should not turn grey
                entry.configure(fg="black", font=(self.font.cget("family"), self.font.cget("size"), "roman"))
                return
            try:
                new_entry = float(entry.get())
                old_entry = float(self._get_active_molecule_parameter_value(param_name))
            except (ValueError, TypeError): # If value is global parameter, turn grey
                entry.configure(fg="grey", font=(self.font.cget("family"), self.font.cget("size"), "italic"))
                return
            
            if new_entry == old_entry:
                entry.configure(fg="black", font=(self.font.cget("family"), self.font.cget("size"), "roman"))
            else:
                entry.configure(fg="grey", font=(self.font.cget("family"), self.font.cget("size"), "italic"))

        
        entry.bind("<Return>", on_change)
        var.trace_add("write", on_write)
        
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
        
        return self._create_simple_entry(self, label_text, current_value, row, col, update_parameter, width, param_name=param_name)

    def _create_display_controls(self,parent, start_row, start_col):
        """Create plot start and range controls for display view"""
        plot_start_tip = "Start wavelength\nfor the upper plot\nunits: μm"
        plot_range_tip = "Wavelength range\nfor the upper plot\nunits: μm"

        # Plot start
        initial_start = getattr(self.islat, 'display_range', [4.5, 5.5])[0]
        self.plot_start_entry, self.plot_start_var = self._create_simple_entry( parent,
            "Plot start:", initial_start, start_row, start_col, lambda _: self._update_display_range(), param_name="display_range_start", tip_text=plot_start_tip)
        
        # Plot range  
        display_range = getattr(self.islat, 'display_range', [4.5, 5.5])
        initial_range = round(display_range[1] - display_range[0], 2) # round to 2 decimal places
        self.plot_range_entry, self.plot_range_var = self._create_simple_entry( parent,
            "Plot range:", initial_range, start_row, start_col + 2, lambda _: self._update_display_range(), param_name="display_range_range", tip_text=plot_range_tip)

    def _create_wavelength_controls(self, parent, start_row, start_col):
        """Create wavelength range controls for model calculation range"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        min_wave_tip = "Minimum wavelength\nto calculate the models\nunits: μm"
        max_wave_tip = "Maximum wavelength\nto calculate the models\nunits: μm"
        molecules_dict = self.islat.molecules_dict
        min_wave, max_wave = molecules_dict.global_wavelength_range
        
        self.min_wavelength_entry, self.min_wavelength_var = self._create_simple_entry( parent,
            "Min. Wave:", min_wave, start_row, start_col, self._update_wavelength_range, tip_text=min_wave_tip)
        self.max_wavelength_entry, self.max_wavelength_var = self._create_simple_entry( parent,
            "Max. Wave:", max_wave, start_row, start_col + 2, self._update_wavelength_range, tip_text=max_wave_tip)

    def _create_global_parameter_controls(self, parent, start_row, start_col):
        """Create global parameter entry fields using MoleculeDict properties"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            label = tk.Label(parent, text="Global parameters not available")
            label.grid(row=start_row, column=start_col, columnspan=4, padx=5, pady=5)
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
                parent,
                field_config['label'], 
                field_config['property'], 
                row, 
                col, 
                field_config['width'],
                tip_text=field_config['tip']
            )
            
            if entry and var:
                self._global_parameter_entries[field_config['property']] = (entry, var)
            
            col_offset += 1

        

    def _build_color_and_vis_controls(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        self.mol_frames = {}

        header_frame = tk.Frame(parent)
        header_frame.grid(row=0, column= 0, sticky="ew")

        content_frame = tk.Frame(parent)
        content_frame.grid(row=1, column=0, sticky="nsew")
 
        for col, (label, tip_text) in enumerate(self.column_labels.items()):
            padx = 0
            label_widget = tk.Label(header_frame, text=label)
            if tip_text:
                CreateToolTip(label_widget, tip_text)    
            if label == "Del.":
                padx = (7,0)
            label_widget.grid(row=0, column=col, sticky="ew", padx=padx)
            header_frame.grid_columnconfigure(col, weight=1)
    
        for row, (mol_name, mol_obj) in enumerate(self.mol_dict.items()):
            mol_name = str(mol_name)
            
            current_mol = mol_obj

            mol_frame = tk.Frame(content_frame)
            self.mol_frames[mol_name] = mol_frame
            # mol_frame.grid(row=row, column=0, pady=2, sticky="nsew")
            mol_frame.pack(pady=2)
            mol_frame.grid_rowconfigure(0, weight=1)
            for col in range(len(self.column_labels)):  # adjust number of columns as needed
                mol_frame.grid_columnconfigure(col, weight=1)

            visibility_var = tk.BooleanVar()
            visibility_checkbox = ttk.Checkbutton(
                mol_frame, 
                variable=visibility_var, 
                command=lambda name = mol_name: self._on_visibility_changed(name)
            )

            visibility_checkbox.grid(row=0, column=0, sticky="nsew", pady=2, padx=0)
            if mol_name not in self.mol_visibility:
                self.mol_visibility[mol_name] = visibility_var

            btn_frame = tk.Frame(mol_frame)
            btn_frame.grid(row=0, column=1, pady=2, sticky="nsew")
            mol_btn = tk.Button(btn_frame, 
                                text=mol_name, 
                                width=2,
                                activebackground="white",  # macOS pressed blue
                                activeforeground="#0a84ff",
                                )
            mol_btn.config(command=lambda name=mol_name, frame=mol_frame: self._on_molecule_selected(mol_name=name))
            mol_btn.grid(row=0, column=0)
            if(len(mol_name) > self.max_name_len):
                CreateToolTip(mol_btn, mol_name, bg=self.bg_color)

            delete_btn = tk.Button(
                            mol_frame, 
                            text= "X",
                            command= lambda name = mol_name, frame = mol_frame: self._delete_molecule(mol_name=name, frame=frame)
                            )
            
            delete_btn.grid(row=0, column=2, pady=2,padx=0, sticky="nsew")

            color_button = ColorButton(
                            mol_frame, 
                            color= getattr(current_mol,'color', "Blue"),
                            )
            color_button.add_command(command= lambda btn = color_button, name=mol_name: self._on_color_button_clicked(name, btn))
            color_button.grid(row=0, column = 3, sticky="nsew")

            is_visible = getattr(self.islat.molecules_dict[mol_name], 'is_visible', False)
            visibility_var.set(is_visible)

        self._update_active_molecule_changes()

    def _create_global_parameter_entry(self, parent, label_text, property_name, row, col, width=12, tip_text = None):
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
                self.data_field.insert_text(f"Error updating global {property_name}: {e}")
        
        # Get current value from molecules_dict
        current_value = 0.0
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            current_value = getattr(self.islat.molecules_dict, property_name, 0.0)
        
        return self._create_simple_entry(parent, label_text, current_value, row, col, update_global_parameter, width, param_name=property_name, tip_text=tip_text)

    def _create_molecule_specific_controls(self, parent, start_row, start_col):
        """Create controls for molecule-specific parameters that update with active molecule"""
        # Store references for later updates
        self._molecule_parameter_entries = {}

        parameters_frame = tk.Frame(parent)
        parameters_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create fields based on the class-level dictionary
        row_offset = 1
        col_offset = 0
        col = 0
        row = start_row 
        for field_key, field_config in self.MOLECULE_FIELDS.items():
            
            
            entry, var = self._create_molecule_parameter_entry(
                parameters_frame,
                field_config['label'], 
                field_config['attribute'], 
                row, 
                col, 
                tip_text=field_config['tip']
            )
            
            if entry and var:
                self._molecule_parameter_entries[field_config['attribute']] = (entry, var)
            row +=1

            col_offset += 1

        # default_btn = ttk.Button(parameters_frame, text="default parameters", command= lambda: self._update_molecule_parameter_fields(default=True))
        # default_btn.grid(row=row, column=col, columnspan=2, sticky="s")
        # CreateToolTip(default_btn, "reset current molecule's\nparameters to default values")
        parameters_frame.rowconfigure(row, weight=1)
    
    def reset_parameters_to_default(self):
        pass

    def _delete_molecule(self, mol_name = None, frame = None):
        mol_name = mol_name
        active_mol = self._get_active_molecule_object().name
        default_mol = self.islat.user_settings.get("default_active_molecule", "H2O")
 

        if mol_name == default_mol:
            # print(f"Cannot delete {mol_name}!")
            self.data_field.insert_text(f"Cannot delete default molecule: {mol_name}!")
            return
        
        # print(f"destroying {mol_name}")
        self.data_field.insert_text(f"Deleting {mol_name}", clear_after = True)

        if mol_name == active_mol:
            new_active = self.islat.user_settings.get("default_active_molecule", "H2O")
            print(f"setting {new_active} as active molecule")
            self.data_field.insert_text(f"setting {new_active} as active molecule", clear_after = False)
            self._set_active_molecule(mol_name=new_active)

        frame.destroy()

        self.mol_frames.pop(mol_name, None)
        self.mol_visibility.pop(mol_name, None)
        self.plot.plot_renderer.remove_molecule_lines(mol_name)
        del self.islat.molecules_dict[mol_name]

        self.plot.canvas.draw_idle()

    def _create_molecule_parameter_entry(self, parent, label_text, param_name, row, col, width=7, tip_text = None):
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
                # self.data_field.insert_text(f"Error updating {param_name}: {e}")
        
        # Get initial value from active molecule
        initial_value = self._get_active_molecule_parameter_value(param_name)
        
        return self._create_simple_entry(parent, label_text, initial_value, row, col, update_active_molecule_parameter, width, param_name=param_name, tip_text=tip_text)

    def _load_field_configurations(self):
        """Load field configurations from JSON file using iSLAT file handling"""
        try:
            config = load_control_panel_fields_config()
            self.GLOBAL_FIELDS = config.get('global_fields', {})
            self.MOLECULE_FIELDS = config.get('molecule_fields', {})
        except Exception as e:
            print(f"Error loading control panel field configurations: {e}")
            # self.data_field.insert_text(f"Error loading control panel field configurations: {e}")
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
            # self.data_field.insert_text(f"ControlPanel: Error registering callbacks: {e}")

    def _on_active_molecule_change(self, old_molecule, new_molecule):
        """Handle active molecule changes from the iSLAT callback system"""
        self._update_molecule_parameter_fields()
        self._update_active_molecule_changes()

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
                    self._set_var(var, new_value)
            
            # Update color and visibility controls if needed
            if parameter_name in ['color', 'is_visible']:
                self._update_active_molecule_changes()

    def _on_global_parameter_change(self, parameter_name, old_value, new_value):
        """Handle global parameter changes to update UI fields"""
        # Update the specific global parameter field if it exists
        if hasattr(self, '_global_parameter_entries') and parameter_name in self._global_parameter_entries:
            entry, var = self._global_parameter_entries[parameter_name]
            if var.get() != str(new_value):
                self._set_var(var, str(new_value))

    def _get_active_molecule_parameter_value(self, param_name) -> str:
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
            
        value = getattr(active_mol, param_name, "")
        return self._format_value(value, param_name)
        
    def _format_value(self, value, param_name) -> str:

        if not param_name:
            return f"{value:.2f}"
        
        try:
            # Get field configuration for proper formatting
            field_config = None
            for field_key, config in self.GLOBAL_FIELDS.items():
                if config['property'] == param_name:
                    field_config = config
                    break

            if field_config is None:
                for field_key, config in self.MOLECULE_FIELDS.items():
                    if config['attribute'] == param_name:
                        field_config = config
                        break
            
            # Format value based on field configuration
            if field_config and isinstance(value, (int, float)):
                return field_config['format'].format(value)
            elif isinstance(value, str):
                return value
            return f"{value:.2f}"
        except Exception as e:
            print(f"Error with formatting: {e}")
            return ""

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

    def _on_visibility_changed(self, mol_name):
        """Handle visibility checkbox changes for individual molecule plotting"""
        
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        
        # Get the toggled molecule object
        selected_mol = self.islat.molecules_dict[mol_name]
        if not selected_mol:
            return
            
        new_visibility = self.mol_visibility[mol_name].get()
        molecule_name = selected_mol.name

        # if molecule was toggled on, set as new active molecule (maybe make this a setting)
        if new_visibility:
            self._on_molecule_selected(molecule_name)
        
        # Simply toggle this molecule's visibility - don't affect other molecules
        self.islat.molecules_dict.bulk_set_visibility(new_visibility, [molecule_name])
        
        # Debug: Verify the visibility was actually set
        print(f"ControlPanel: Set {molecule_name} visibility to {new_visibility}, actual value: {getattr(selected_mol, 'is_visible', 'UNDEFINED')}")
        
        # Trigger selective plot refresh to show/hide the molecule
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot') and hasattr(self.islat.GUI.plot, 'on_molecule_visibility_changed'):
            self.islat.GUI.plot.on_molecule_visibility_changed(molecule_name, new_visibility)
            print(f"ControlPanel: Triggered selective plot refresh for visibility change")

    def _on_color_button_clicked(self, mol_name, btn):
        """Handle color button clicks to open color chooser"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        # Get the active molecule object
        selected_mol = self.islat.molecules_dict[mol_name]
        if not selected_mol:
            return
            
        # Get molecule name for the color chooser title
        mol_name = getattr(selected_mol, 'displaylabel', getattr(selected_mol, 'name', 'Molecule'))
        old_color = getattr(selected_mol,'color', "Blue")
        
        # Open color chooser
        color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}", color=old_color)[1]
        if color_code:
            btn.change_color(color_code)
            
            # Update molecule color
            selected_mol.color = color_code
            
            # Manually trigger the molecule parameter change notification
            # since color is not a property with a setter
            if hasattr(selected_mol, '_notify_my_parameter_change'):
                selected_mol._notify_my_parameter_change('color', old_color, color_code)

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

    def _update_active_molecule_changes(self):
        """Update color button and visibility checkbox based on active molecule"""
        active_mol = self._get_active_molecule_object()
        selected_name = active_mol.name
        self.mol_frames[active_mol.name].config(bg=self.selected_color)
        if len(active_mol.name) > self.max_name_len + 4:
            selected_name = active_mol.name[:self.max_name_len] + "..."
            CreateToolTip(self.selected_label, active_mol.name, bg = self.bg_color)
            
        self.selected_label.config(text=f"Selected Molecule: {selected_name}")

    def _update_display_range(self, value_str=None):
        """Update display range bidirectionally between GUI and iSLAT class"""
        # If value_str is a tuple, it's being called from iSLAT to update GUI
        if isinstance(value_str, tuple) and len(value_str) == 2:
            try:

                # Update GUI fields from iSLAT display_range (iSLAT -> GUI)
                start, end = value_str
                range_val = round(end - start, 2)
                
                self._set_var(self.plot_start_var, self._format_value(start, "display_range_start"))
                self._set_var(self.plot_range_var, self._format_value(range_val, "display_range_range"))

            except Exception as e:
                print(f"Error updating display range GUI from iSLAT: {e}")
        else:
            try:
                
                # Update iSLAT from GUI fields (GUI -> iSLAT)
                start = float(self.plot_start_var.get())
                range_val = float(self.plot_range_var.get())
                
                if hasattr(self.islat, 'display_range'):
                    # Temporarily disable the updating flag to allow normal iSLAT property setter
                    new_display_range = (start, start + range_val)
                    # Only update if the values are actually different to avoid unnecessary callbacks
                    if not hasattr(self.islat, '_display_range') or self.islat._display_range != new_display_range:
                        self.islat.display_range = new_display_range
            except (ValueError, AttributeError):
                pass

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
                    self._set_var(var, str(value))
        
        except Exception as e:
            print(f"Error updating molecule parameter UI fields: {e}")

    def _on_molecule_selected(self, mol_name, event=None):
        """Handle molecule selection - uses iSLAT's active_molecule property"""

        old_active_mol = self._get_active_molecule_object().name
        
        try:
            self.mol_frames[old_active_mol].config(bg = self.bg_color)
        except KeyError:
            old_active_mol = self.islat.user_settings.get("default_active_molecule", "H2O")
            self.mol_frames[old_active_mol].config(bg = self.bg_color)
            
        self._set_active_molecule(mol_name= mol_name)

    def _set_active_molecule(self, mol_name):
        selected_label = self.mol_dict[mol_name].displaylabel

        try:
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                for mol_name, mol_obj in self.islat.molecules_dict.items():
                    display_label = getattr(mol_obj, 'displaylabel', mol_name)
                    if display_label == selected_label:
                        self.islat.active_molecule = mol_name
                        return
                
                first_mol = next(iter(self.islat.molecules_dict.keys()), None)
                if first_mol:
                    self.islat.active_molecule = first_mol
        except Exception as e:
            print(f"Error setting active molecule: {e}")

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
            self._set_var(self.min_wavelength_var, f'{min_val:.2e}')
            self._set_var(self.max_wavelength_var, f'{max_val:.2e}')

        # Update display range fields
        self._update_display_range_fields()
        
        # Update molecule-specific parameter fields
        self._update_molecule_parameter_fields()
        
        # Rebuild color and visibility controls for new molecules
        self._rebuild_color_and_vis_controls()

    def _rebuild_color_and_vis_controls(self):
        """Rebuild color and visibility controls when molecules are added/removed"""
        if hasattr(self, 'mol_frames'):
            # Clear existing frames
            for frame in self.mol_frames.values():
                frame.destroy()
            self.mol_frames.clear()
            self.mol_visibility.clear()
        
        # Get the parent frame
        if hasattr(self, 'color_vis_frame'):
            self._build_color_and_vis_controls(self.color_vis_frame)

    def cleanup(self):
        try:
            if hasattr(self.islat, '2_active_molecule_change_callback'):
                self.islat.remove_active_molecule_change_callback(self._on_active_molecule_change)
        except Exception as e:
            print(f"Error during ControlPanel cleanup: {e}")

    def _update_molecule_parameter_fields(self, default = False):
        """Update all molecule-specific parameter fields with values from the active molecule"""
        if not hasattr(self, '_molecule_parameter_entries'):
            return
        
        if default:
            print("resetting to defaults")
            for field_key, field_config in self.MOLECULE_FIELDS.items():
                param_name = field_config['attribute']
                if hasattr(self, '_molecule_parameter_entries') and param_name in self._molecule_parameter_entries:
                    entry, var = self._molecule_parameter_entries[param_name]
                    new_value = float(field_config['default'])
                    if float(var.get()) != new_value:
                        self._set_var(var, new_value)
            return
        else:
            for param_name, (entry, var) in self._molecule_parameter_entries.items():
                new_value = self._get_active_molecule_parameter_value(param_name) 
                current_value = var.get()
                if current_value != new_value:
                    self._set_var(var, new_value)

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
                    self._set_var(var, str(new_value))
            except (AttributeError, TypeError):
                pass

    def _update_display_range_fields(self):
        """Update display range fields with values from the iSLAT display_range property"""
        if not (hasattr(self, 'plot_start_var') and hasattr(self, 'plot_range_var')):
            return
            
        if not hasattr(self.islat, 'display_range') or not self.islat.display_range:
            return
            
        try:
            start, end = self.islat.display_range
            range_val = round(end - start, 2)
            
            # Update only if values are different to avoid unnecessary updates
            current_start = self.plot_start_var.get()
            current_range = self.plot_range_var.get()
            
            if str(current_start) != str(start):
                self._set_var(self.plot_start_var, self._format_value(start, "display_range_start"))
            if str(current_range) != str(range_val):
                self._set_var(self.plot_range_var, self._format_value(range_val, "display_range_range"))
        except (ValueError, AttributeError, TypeError):
            pass

    def _set_var(self, var, value):
        self.updating = True
        var.set(value)
        self.updating = False