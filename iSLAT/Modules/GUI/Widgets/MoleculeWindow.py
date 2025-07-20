"""Old file, kept for reference"""


import tkinter as tk
from tkinter import ttk, colorchooser

class MoleculeWindow:
    def __init__(self, name, parent_frame, molecule_data, output_plot, config, islat):
        self.parent_frame = parent_frame
        self.name = name
        self.plot = output_plot
        self.config = config
        self.theme = config["theme"]
        self.islat = islat

        self.molecules_dict = self.islat.molecules_dict
        self.molecules = {}

        # Ensure molecule colors are initialized before building the table
        self._ensure_molecule_colors_initialized()

        self.build_table()
        self.update_table()
        
        # Apply theme one more time after everything is built to ensure proper theming
        if hasattr(self.parent_frame, 'after'):
            self.parent_frame.after(50, lambda: self.apply_theme())
        elif hasattr(self.parent_frame, 'master') and hasattr(self.parent_frame.master, 'after'):
            self.parent_frame.master.after(50, lambda: self.apply_theme())

    def _apply_theme_to_frame(self):
        """Apply theme colors to the main frame"""
        self.frame.configure(
            bg=self.theme["background"],
            fg=self.theme["foreground"]
        )

    def _apply_theme_to_canvas_and_scrollbar(self):
        """Apply theme colors to canvas and scrollbar"""
        self.canvas.configure(
            bg=self.theme["background"],
            highlightthickness=0
        )
        
        # For TTK Scrollbar, we need to configure the style
        try:
            style = ttk.Style()
            style.theme_use('clam')  # Use a theme that supports customization
            
            # Configure scrollbar colors
            style.configure("Vertical.TScrollbar",
                          background=self.theme["background_accent_color"],
                          troughcolor=self.theme["background"],
                          bordercolor=self.theme["background_accent_color"],
                          arrowcolor=self.theme["foreground"],
                          darkcolor=self.theme["background_accent_color"],
                          lightcolor=self.theme["background_accent_color"])
            
            style.map("Vertical.TScrollbar",
                     background=[('active', self.theme["selection_color"]),
                               ('pressed', self.theme["selection_color"])])
        except:
            # Fallback if TTK styling fails
            pass

    def _apply_theme_to_widget(self, widget):
        """Apply theme to any tkinter widget recursively"""
        try:
            widget_class = widget.winfo_class()
            
            if widget_class in ['Frame', 'LabelFrame']:
                widget.configure(bg=self.theme["background"], fg=self.theme["foreground"])
            elif widget_class == 'Label':
                widget.configure(bg=self.theme["background"], fg=self.theme["foreground"])
            elif widget_class == 'Button':
                # Check if this is a marked color selection button - never theme these
                if hasattr(widget, '_is_color_button') and widget._is_color_button:
                    # This is a color selection button - preserve its molecule color
                    pass
                else:
                    # Check other button characteristics
                    current_bg = widget.cget('bg')
                    current_text = widget.cget('text')
                    
                    # Don't theme color selection buttons (they should keep their molecule color)
                    # And don't theme delete buttons (they have custom theme colors)
                    if current_text == "X":
                        # This is a delete button - apply delete button theme
                        widget.configure(
                            bg=self.theme.get("delete_button_bg_color", "#FF4444"),
                            fg=self.theme.get("delete_button_fg_color", "#FFFFFF"),
                            activebackground=self.theme.get("delete_button_bg_color", "#FF4444"),
                            activeforeground=self.theme.get("delete_button_fg_color", "#FFFFFF")
                        )
                    elif current_bg and current_bg.startswith('#') and len(current_bg) == 7:
                        # This is likely a color selection button - preserve its background color
                        pass
                    else:
                        # Regular button - apply normal theme
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
            elif widget_class == 'Checkbutton':
                widget.configure(
                    bg=self.theme["background"],
                    fg=self.theme["foreground"],
                    activebackground=self.theme["background"],
                    activeforeground=self.theme["foreground"],
                    selectcolor=self.theme["background_accent_color"]
                )
            elif widget_class == 'Canvas':
                widget.configure(bg=self.theme["background"])
                
            # Recursively apply to children
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
        except tk.TclError:
            pass

    def build_table(self):
        # Ensure all molecules have colors assigned
        self._ensure_molecule_colors_initialized()
        
        self.frame = tk.LabelFrame(self.parent_frame, text="Molecules")
        
        # Apply theme to the main frame
        self._apply_theme_to_frame()
        
        # Create a canvas and scrollbar for scrolling
        self.canvas = tk.Canvas(self.frame, height=300)
        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Apply theme to canvas and scrollbar
        self._apply_theme_to_canvas_and_scrollbar()

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Bind mousewheel to canvas (cross-platform)
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_mousewheel_linux(event):
            self.canvas.yview_scroll(-1, "units")
        
        def _on_mousewheel_linux_up(event):
            self.canvas.yview_scroll(1, "units")
            
        # Windows and Mac
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        # Linux
        self.canvas.bind("<Button-4>", _on_mousewheel_linux_up)
        self.canvas.bind("<Button-5>", _on_mousewheel_linux)

        headers = ["Molecule", "Temp", "Radius", "Density", "On", "Color", "Delete"]
        for col, text in enumerate(headers):
            header_label = tk.Label(self.scrollable_frame, text=text)
            header_label.grid(row=0, column=col, padx=2, pady=2)
            # Apply theme to header
            header_label.configure(
                bg=self.theme["background"],
                fg=self.theme["foreground"],
                font=('TkDefaultFont', 9, 'bold')
            )

    def update_table(self):
        # Ensure all molecules have colors assigned
        self._ensure_molecule_colors_initialized()
        
        # Clear existing widgets in the scrollable frame (except headers)
        for widget in self.scrollable_frame.winfo_children()[7:]:  # Skip the 7 header labels
            widget.destroy()
        
        # Clear the molecules dict
        self.molecules.clear()

        # Link Entry widgets and color directly to molecule object attributes
        def make_entry_callback(entry, attr, mol_obj):
            def callback(*args):
                val = entry.get()
                try:
                    if attr == "n_mol":
                        val = float(val)
                    elif attr == "temp":
                        val = float(val)
                    elif attr == "radius":
                        val = float(val)
                    elif attr == "displaylabel":
                        # For label changes, we also need to update any GUI dropdowns
                        old_label = getattr(mol_obj, attr, mol_obj.name)
                        if val != old_label:
                            self.update_control_panel_dropdown()
                    setattr(mol_obj, attr, val)
                    self.update_lines()
                except ValueError:
                    pass  # Ignore invalid input
            return callback

        # Create StringVar objects for two-way binding
        def create_bound_entry(parent, mol_obj, attr, width, format_func=None, grid_args=None):
            """Create an Entry widget bound to a molecule attribute with two-way synchronization."""
            var = tk.StringVar()
            
            # Initialize with current value
            current_val = getattr(mol_obj, attr, "")
            if format_func:
                current_val = format_func(current_val)
            var.set(str(current_val))
            
            entry = tk.Entry(parent, textvariable=var, width=width)
            
            # Apply theme to entry
            entry.configure(
                bg=self.theme["background_accent_color"],
                fg=self.theme["foreground"],
                insertbackground=self.theme["foreground"],
                selectbackground=self.theme["selection_color"],
                selectforeground=self.theme["background"]
            )
            
            if grid_args:
                entry.grid(**grid_args)
            
            # Bind changes to update molecule object
            def on_change(*args):
                val = var.get()
                try:
                    if attr == "n_mol":
                        val = float(val)
                    elif attr == "temp":
                        val = float(val)
                    elif attr == "radius":
                        val = float(val)
                    elif attr == "displaylabel":
                        # For label changes, we also need to update any GUI dropdowns
                        old_label = getattr(mol_obj, attr, mol_obj.name)
                        if val != old_label:
                            self.update_control_panel_dropdown()
                    setattr(mol_obj, attr, val)
                    # Only update lines for numeric parameters to avoid infinite loops
                    if attr in ["temp", "radius", "n_mol"]:
                        self.update_lines()
                        # Also trigger any update_model_spectrum method if it exists
                        if hasattr(self.islat, 'update_model_spectrum'):
                            self.islat.update_model_spectrum()
                except ValueError:
                    pass  # Ignore invalid input
            
            var.trace_add("write", on_change)
            
            # Also bind Return and FocusOut for immediate updates
            entry.bind("<Return>", lambda e: on_change())
            entry.bind("<FocusOut>", lambda e: on_change())
            
            return entry, var

        for i, mol_name in enumerate(self.islat.molecules_dict.keys()):
            mol_obj = self.islat.molecules_dict[mol_name]

            # Make molecule name editable with bound StringVar
            name_entry, name_var = create_bound_entry(
                self.scrollable_frame, mol_obj, "displaylabel", 8,
                format_func=lambda x: getattr(mol_obj, 'displaylabel', mol_name),
                grid_args={"row": i+1, "column": 0, "padx": 2, "pady": 1, "sticky": "w"}
            )

            # Temperature entry with bound StringVar
            temp_entry, temp_var = create_bound_entry(
                self.scrollable_frame, mol_obj, "temp", 6,
                format_func=lambda x: f"{x}",
                grid_args={"row": i+1, "column": 1, "padx": 2, "pady": 1}
            )

            # Radius entry with bound StringVar
            rad_entry, rad_var = create_bound_entry(
                self.scrollable_frame, mol_obj, "radius", 6,
                format_func=lambda x: f"{x}",
                grid_args={"row": i+1, "column": 2, "padx": 2, "pady": 1}
            )

            # Density entry with bound StringVar
            dens_entry, dens_var = create_bound_entry(
                self.scrollable_frame, mol_obj, "n_mol", 6,
                format_func=lambda x: f"{x:.1e}",
                grid_args={"row": i+1, "column": 3, "padx": 2, "pady": 1}
            )

            on_var = tk.BooleanVar(value=mol_obj.is_visible)
            def on_toggle(var=on_var, m=mol_obj):
                m.is_visible = var.get()
                self.update_lines()
            on_btn = tk.Checkbutton(self.scrollable_frame, variable=on_var, command=on_toggle)
            # Apply theme to checkbutton
            on_btn.configure(
                bg=self.theme["background"],
                fg=self.theme["foreground"],
                activebackground=self.theme["background"],
                activeforeground=self.theme["foreground"],
                selectcolor=self.theme["background_accent_color"]
            )
            on_btn.grid(row=i+1, column=4, padx=2, pady=1)

            # Get color from molecule object, or assign default color from theme
            default_colors = self.theme.get("default_molecule_colors", ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])
            
            # Always use molecule's color if it exists, otherwise assign a default
            if hasattr(mol_obj, 'color') and mol_obj.color is not None:
                color = mol_obj.color
            else:
                color = default_colors[i % len(default_colors)]
                mol_obj.color = color  # Set the color on the molecule object
                
            def pick_and_set_color(mol_name=mol_name, mol_obj=mol_obj, btn=None):
                color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
                if color_code:
                    mol_obj.color = color_code
                    btn.config(bg=color_code)
                    self.molecules[mol_name]["color"] = color_code
                    self.update_lines()
                    
            color_btn = tk.Button(self.scrollable_frame, bg=color, width=4)
            color_btn.config(command=lambda m=mol_name, mo=mol_obj, b=color_btn: pick_and_set_color(m, mo, b))
            # Mark this as a color selection button so theming will ignore it
            color_btn._is_color_button = True
            color_btn.grid(row=i+1, column=5, padx=2, pady=1)

            # Add delete button
            delete_btn = tk.Button(self.scrollable_frame, text="X", bg=self.theme["delete_button_bg_color"], fg=self.theme["delete_button_fg_color"], width=3,
                                 command=lambda m=mol_name: self.delete_molecule(m))
            delete_btn.grid(row=i+1, column=6, padx=2, pady=1)

            self.molecules[mol_name] = {
                "name_entry": name_entry,
                "name_var": name_var,
                "temp_entry": temp_entry,
                "temp_var": temp_var,
                "rad_entry": rad_entry,
                "rad_var": rad_var,
                "dens_entry": dens_entry,
                "dens_var": dens_var,
                "on_var": on_var,
                "color": color,
                "color_btn": color_btn,
                "delete_btn": delete_btn
            }

        self.update_lines()
        
        # Apply theme to all widgets in the scrollable frame
        self._apply_theme_to_widget(self.scrollable_frame)

    def refresh_fields_from_molecules(self):
        """Update all GUI fields to reflect current molecule object values."""
        for mol_name, props in self.molecules.items():
            if mol_name in self.islat.molecules_dict:
                mol_obj = self.islat.molecules_dict[mol_name]
                
                # Update StringVar values to reflect current molecule state
                if "name_var" in props:
                    display_label = getattr(mol_obj, 'displaylabel', mol_name)
                    props["name_var"].set(str(display_label))
                
                if "temp_var" in props:
                    props["temp_var"].set(str(mol_obj.temp))
                
                if "rad_var" in props:
                    props["rad_var"].set(str(mol_obj.radius))
                
                if "dens_var" in props:
                    props["dens_var"].set(f"{mol_obj.n_mol:.1e}")
                
                # Update visibility checkbox
                if "on_var" in props:
                    props["on_var"].set(mol_obj.is_visible)
                
                # Update color button - always use molecule's actual color
                if "color_btn" in props and hasattr(mol_obj, 'color') and mol_obj.color:
                    props["color_btn"].config(bg=mol_obj.color)
                    props["color"] = mol_obj.color

    def update_control_panel_dropdown(self):
        """Update the control panel molecule dropdown when molecule labels change."""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            self.islat.GUI.control_panel.reload_molecule_dropdown()

    def delete_molecule(self, mol_name):
        """Delete a molecule from the GUI and the molecules dictionary."""
        if mol_name in self.islat.molecules_dict:
            # Remove from the main molecules dictionary
            del self.islat.molecules_dict[mol_name]
            
            # Clear any model lines for this molecule from the plot
            self.plot.clear_model_lines()
            
            # Update the control panel molecule dropdown if it exists
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
                self.islat.GUI.control_panel.reload_molecule_dropdown()
            
            # Refresh the table to reflect the changes
            self.update_table()
            
            # Update the plot
            self.islat.update_model_spectrum()
            self.plot.update_all_plots()

    def pick_color(self, mol_name):
        color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}")[1]
        if color_code:
            self.molecules[mol_name]["color"] = color_code
            self.update_lines()

    def update_lines(self):
        """Update molecule visibility and recalculate model spectrum."""
        #self.plot.clear_model_lines()
        for mol_name, props in self.molecules.items():
            # Check if molecule still exists in the dictionary (not deleted)
            if mol_name not in self.islat.molecules_dict:
                continue
                
            mol_obj = self.islat.molecules_dict[mol_name]
            
            # Update visibility based on checkbox (other parameters are handled by binding)
            if "on_var" in props:
                mol_obj.is_visible = props["on_var"].get()
            
            # Ensure the molecule retains its own color (don't override with theme color)
            if "color" in props and hasattr(mol_obj, 'color') and mol_obj.color:
                # Update local storage to match molecule's actual color
                props["color"] = mol_obj.color
                # Make sure the color button reflects the molecule's color
                if "color_btn" in props:
                    props["color_btn"].config(bg=mol_obj.color)
        
        # Force cache invalidation for flux calculations
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            # Clear all flux caches since molecules may have changed
            self.islat.molecules_dict._summed_flux_cache.clear()
            self.islat.molecules_dict.fluxes.clear()
        
        # Update the model spectrum and plots using coordinator
        if hasattr(self.islat, 'request_update'):
            self.islat.request_update('model_spectrum')
            self.islat.request_update('plots')
            self.islat.request_update('population_diagram')
            print("Hey!")
        else:
            # Fallback to direct update - ensure all three plots are updated
            self.islat.update_model_spectrum()
            self.plot.update_all_plots()
            # Explicitly trigger population diagram update for active molecule
            if hasattr(self.plot, 'on_active_molecule_changed'):
                self.plot.on_active_molecule_changed()