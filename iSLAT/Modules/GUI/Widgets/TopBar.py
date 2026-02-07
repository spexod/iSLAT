import os
import platform
import traceback
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
from typing import TYPE_CHECKING, Any, Dict

import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import iSLAT.Constants as c
from ..GUIFunctions import create_button, create_menu_btn
from .ResizableFrame import ResizableFrame
from iSLAT.Modules.GUI.Widgets.ChartWindow import MoleculeSelector
from iSLAT.Modules.GUI.PlotGridWindow import PlotGridWindow
from iSLAT.Modules.GUI.FullSpectrumWindow import FullSpectrumWindow
from iSLAT.Modules.FileHandling.iSLATFileHandling import (
    write_molecules_to_csv, generate_csv, line_saves_file_path,
    line_saves_file_name, example_data_folder_path
)
from iSLAT.Modules.FileHandling.OutputFullSpectrum import output_full_spectrum
from iSLAT.Modules.DataProcessing.Slabfit import SlabFit as SlabModel
from iSLAT.Modules.DataProcessing.BatchFittingService import BatchFittingService
from iSLAT.Modules.DataProcessing.DeblendingService import DeblendingService
from iSLAT.Modules.DataProcessing.LineSaveService import LineSaveService

if TYPE_CHECKING:
    from iSLAT.Modules.Plotting.MainPlot import iSLATPlot

class TopBar(ResizableFrame):
    def __init__(
        self, 
        master: tk.Widget, 
        islat: Any, 
        theme: Dict[str, Any], 
        main_plot: 'iSLATPlot', 
        data_field: Any, 
        control_panel: Any,
        config: Dict[str, Any]
    ) -> None:
        # Initialize the ResizableFrame with theme
        super().__init__(master, theme=theme, borderwidth=1, relief="groove")
        
        self.master = master
        self.islat = islat
        self.main_plot = main_plot
        self.data_field = data_field
        self.config = config
        self.control_panel = control_panel

        # Initialize services for non-GUI logic
        self.batch_fitting_service = BatchFittingService(islat)
        self.deblending_service = DeblendingService(islat)
        self.line_save_service = LineSaveService(islat)

        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=0, column=1)

        # Create buttons for options
        self._create_buttons()
        
        # Create and add toolbar 
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=0, column=2, sticky="nsew")
        self.toolbar = self.main_plot.create_toolbar(toolbar_frame)

        self.atomic_toggle: bool = False
        self.line_toggle: bool = False
        
        # Apply initial theme
        # self.apply_theme(theme)
    
    def _create_buttons(self):
        """Create all the button widgets."""
        os_name = platform.system()
        if os_name == "Darwin":
            molecule_drpdwn = create_menu_btn(self.button_frame, self.theme, "Manage Molecules", 0, 0)
        else:
            molecule_drpdwn = create_menu_btn(self.button_frame, self.theme, "Manage Molecules ▼", 0, 0)

        molecule_menu = tk.Menu(molecule_drpdwn, tearoff=0)
        molecule_menu.add_command(label="HITRAN Query", command=self.hitran_query)
        molecule_menu.add_command(label="Default Molecules", command=self.default_molecules)
        molecule_menu.add_command(label="Add Molecules", command=self.add_molecule)
        molecule_menu.add_command(label="Export Models", command=self.export_models)
        molecule_drpdwn.config(menu=molecule_menu)

        if os_name == "Darwin":
            spectrum_drpdwn = create_menu_btn(self.button_frame, self.theme, "Model Parameters", 0, 1)
        else:
            spectrum_drpdwn = create_menu_btn(self.button_frame, self.theme, "Model Parameters ▼", 0, 1)
        spectrum_menu = tk.Menu(spectrum_drpdwn, tearoff=0)
        spectrum_menu.add_command(label="Save Parameters (Ctrl+S)", command=self.save_parameters)
        spectrum_menu.add_command(label="Load Parameters (Ctrl+L)", command=self.load_parameters)
        spectrum_menu.add_command(label="Output Full Spectrum (Ctrl+Shift+F)", command=lambda: output_full_spectrum(self.islat))
        spectrum_menu.add_command(label="Display Full Spectrum (Ctrl+F)", command=lambda: FullSpectrumWindow(self.master, self.islat))
        spectrum_drpdwn.config(menu=spectrum_menu)

        if os_name == "Darwin":
            spec_functions_drpwn = create_menu_btn(self.button_frame, self.theme, "Spectral Functions", 0, 2)
        else:
            spec_functions_drpwn = create_menu_btn(self.button_frame, self.theme, "Spectral Functions ▼", 0, 2)
        spec_functions_menu = tk.Menu(spec_functions_drpwn, tearoff=0)
        spec_functions_menu.add_command(label="Save Line", command=self.save_line)
        spec_functions_menu.add_command(label="Fit Line", command=self.fit_selected_line)
        spec_functions_menu.add_command(label="Fit Saved Lines", command=self.fit_saved_lines)
        spec_functions_menu.add_command(label="Fit Saved Lines To Sample", command=self.fit_saved_lines_to_sample)
        #spec_functions_menu.add_command(label="Find Single Lines", command=self.find_single_lines)
        spec_functions_menu.add_command(label="Single Slab Fit", command=self.single_slab_fit)
        spec_functions_menu.add_command(label="Line de-Blender", command=lambda: self.fit_selected_line(deblend=True))
        
        spec_functions_drpwn.config(menu=spec_functions_menu)

        saved_lines_tip = "Show saved lines\nfrom the 'Input Line List'\nKeybind: S"
        atomic_lines_tip = "Show atomic lines\nusing separation threshold\nset in the 'Line Separ.\nKeybind: A"
        #export_model_tip = "Export current\nmodels into csv files"
        toggle_legend_tip = "Turn legend on/off\nKeybind: L"
        toggle_full_spectrum_tip = "Toggle full spectrum view on/off\nKeybind: F\n\nOpen in new window: Ctrl+F"
        toggle_summed_tip = "Toggle summed model flux on/off\n(gray fill in plot)\nKeybind: M"
        create_button(self.button_frame, self.theme, "Toggle Saved Lines", self.toggle_saved_lines, 0, 3, tip_text=saved_lines_tip)
        create_button(self.button_frame, self.theme, "Toggle Atomic Lines", self.toggle_atomic_lines, 0, 4, tip_text=atomic_lines_tip)
        create_button(self.button_frame, self.theme, "Toggle Full Spectrum", self.toggle_full_spectrum, 0, 5, tip_text=toggle_full_spectrum_tip)
        create_button(self.button_frame, self.theme, "Toggle Total Model", self.toggle_summed_spectrum, 0, 6, tip_text=toggle_summed_tip)
        create_button(self.button_frame, self.theme, "Toggle Legend", self.main_plot.toggle_legend, 0, 7, tip_text=toggle_legend_tip)

    def save_line(self, save_type="selected"):
        """Save the currently selected line to the line saves file."""
        # Use service to extract line info
        selected_line_info, error_msg = self.line_save_service.extract_line_info_from_selection(
            self.main_plot, save_type
        )
        
        if error_msg:
            self.data_field.insert_text(f"{error_msg}\n")
            return
        
        # Get selection bounds
        selected_wave = getattr(self.main_plot, 'selected_wave', None)
        xmin, xmax = self.line_save_service.get_selection_bounds(
            selected_wave,
            self.main_plot.current_selection,
            selected_line_info['lam']
        )
        
        # Format line info for saving
        line_info = self.line_save_service.format_line_for_save(
            selected_line_info,
            self.islat.active_molecule.name,
            xmin,
            xmax
        )
        
        try:
            if not self.islat.output_line_measurements:
                self.data_field.insert_text("No output line measurements file specified.\n")
                return
            ifh.save_line(line_info, file_name=self.islat.output_line_measurements)
            self.data_field.insert_text(f"Saved line at {line_info['lam']:.4f} μm\n")
        except Exception as e:
            self.data_field.insert_text(f"Error saving line: {e}\n")

    def toggle_saved_lines(self):
        """Show saved lines as vertical dashed lines on the plot."""
        loaded_lines = ifh.read_line_saves(file_name=self.islat.input_line_list)
        if loaded_lines.empty:   
            self.data_field.insert_text("No saved lines found.\n")
            return
        try:
            self.line_toggle = not self.line_toggle

            # Check if full spectrum view is active
            if hasattr(self.main_plot, 'is_full_spectrum') and self.main_plot.is_full_spectrum:
                # Use optimized toggle method that only adds/removes line artists
                if hasattr(self.main_plot, 'full_spectrum_plot'):
                    self.main_plot.full_spectrum_plot.toggle_saved_lines(self.line_toggle)
                    if hasattr(self.main_plot, 'full_spectrum_plot_canvas'):
                        self.main_plot.full_spectrum_plot_canvas.draw_idle()
            else:
                # Regular view - toggle on the main plot
                if self.line_toggle:
                    # Plot the saved lines on the main plot
                    self.main_plot.plot_saved_lines(loaded_lines=loaded_lines, data_field=self.data_field)
                else:
                    self.main_plot.remove_saved_lines()
                    # self.data_field.insert_text("Removed lines")
            
        except Exception as e:
            self.data_field.insert_text(f"Error loading saved lines: {e}\n")

    def fit_selected_line(self, deblend=False):
        """Fit the currently selected line using LMFIT."""
        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for fitting.\n")
            return

        try:
            # Compute the fit using the main plot's fitting function
            fit_result = self.main_plot.compute_fit_line(deblend=deblend)
            
            if fit_result and len(fit_result) >= 3:
                lmfit_result, fitted_wave, fitted_flux = fit_result
                
                if lmfit_result is not None and hasattr(lmfit_result, 'params'):
                    # Extract parameters using the FittingEngine methods
                    line_params = self.main_plot.fitting_engine.extract_line_parameters()
                    
                    if deblend:
                        self._display_deblend_results(line_params)
                    else:
                        self._display_single_gaussian_results(line_params)
                else:
                    self.data_field.insert_text("Fit completed but no valid result object returned.\n", clear_after=False)
            else:
                self.data_field.insert_text("Fit failed or insufficient data.\n", clear_after=False)
            
        except Exception as e:
            self.data_field.insert_text(f"Error during fitting: {e}\n", clear_after=False)
            self.data_field.insert_text(f"Traceback: {traceback.format_exc()}\n", clear_after=False)

    def _display_deblend_results(self, line_params):
        """Display and save deblended line fit results."""
        self.data_field.insert_text("\nDe-blended line fit results:\n", clear_after=False)
        
        selection = self.main_plot.current_selection
        if selection and len(selection) >= 2:
            xmin, xmax = selection[0], selection[-1]
        else:
            self.data_field.insert_text("Invalid selection for deblending.\n", clear_after=False)
            return
            
        line_info = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
        
        # Extract deblended components using service
        components = self.deblending_service.extract_deblended_components(
            line_params,
            line_info,
            self.islat.active_molecule.name
        )
        
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        spectrum_base_name = os.path.splitext(spectrum_name)[0] if spectrum_name != "unknown" else "default"
        save_file_name = f"{spectrum_base_name}-{line_saves_file_name}"
        
        # Display each component
        for component in components:
            self.data_field.insert_text(f"\nComponent {component['index']+1}:\n", clear_after=False)
            display_msgs = self.deblending_service.format_component_display(component)
            for msg in display_msgs:
                self.data_field.insert_text(msg, clear_after=False)
        
        # Save components using service
        saved_count = self.deblending_service.save_deblended_components(components, save_file_name)
        
        if not components:
            self.data_field.insert_text("No components found in fit result.\n", clear_after=False)
        else:
            self.data_field.insert_text(f"\nDe-blended line fit completed with {len(components)} components!", clear_after=False)
            
            # Save summary files
            fit_result_summary = self.main_plot.fitting_engine.get_fit_results_summary()
            fit_results_components = self.main_plot.fitting_engine.get_fit_results_components()
            self.deblending_service.save_deblend_summary(
                fit_result_summary,
                fit_results_components,
                spectrum_base_name,
                line_saves_file_path
            )
            
            # Save plot
            figpath = os.path.join(line_saves_file_path, f"{spectrum_base_name}-deblend_plot.pdf")
            self.main_plot.save_fig(figpath, dpi=10)
            
            if saved_count > 0:
                self.data_field.insert_text(f"\nDe-blended line saved in /LINESAVES!", clear_after=False)

    def _display_single_gaussian_results(self, line_params):
        """Display single Gaussian fit results."""
        self.data_field.insert_text("\nGaussian fit results:\n", clear_after=False)
        
        display_msgs = self.deblending_service.format_single_gaussian_display(line_params)
        for msg in display_msgs:
            self.data_field.insert_text(msg, clear_after=False)

    def fit_saved_lines(self, multiple_files=False):
        """
        Fit all saved lines using batch fitting service.
        
        Args:
            multiple_files (bool): If True, allows user to select multiple spectrum files.
                                 If False, fits saved lines to the currently loaded spectrum.
        """
        if not self.islat.input_line_list:
            # Prompt user to load a line list first
            self.data_field.insert_text("No line list loaded. Please select a line list file.\n")
            from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
            result = load_input_line_list()
            
            if result is None:
                self.data_field.insert_text("No line list selected. Operation cancelled.\n")
                return
            
            file_path, file_name = result
            self.islat.input_line_list = file_path
            self.data_field.insert_text(f"Loaded line list: {file_name}\n")
            
            # Update the FileInteractionPane label if available
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'file_pane'):
                self.islat.GUI.file_pane.update_file_labels()
        
        if not self.islat.output_line_measurements:
            self.data_field.insert_text("No output line measurements file configured. Using default\n")
        
        # Progress callback for GUI updates
        def progress_callback(msg):
            self.data_field.insert_text(msg, clear_after=False)
        
        if multiple_files:
            # Ask user to select multiple spectrum files
            spectrum_files = filedialog.askopenfilenames(
                title="Select Spectrum Files to Fit Saved Lines",
                filetypes=[("All files", "*.*")],
                initialdir=example_data_folder_path
            )
            
            if not spectrum_files:
                self.data_field.insert_text("No spectrum files selected.\n")
                return
            
            # Use batch fitting service for multiple files
            plot_grid_list, output_folder = self.batch_fitting_service.fit_lines_to_multiple_spectra(
                saved_lines_file=self.islat.input_line_list,
                spectrum_files=list(spectrum_files),
                config=self.config,
                progress_callback=progress_callback,
                base_output_path=line_saves_file_path
            )
            
            if plot_grid_list:
                # Check user setting for direct PDF save
                save_directly_to_pdf = self.config.get('save_fit_plot_grid_directly_to_PDF', False)
                
                if save_directly_to_pdf:
                    # Use the output folder created during batch processing
                    save_path = output_folder if output_folder else line_saves_file_path
                    self.batch_fitting_service.save_plot_grids_to_pdf(
                        plot_grid_list,
                        save_path,
                        progress_callback=progress_callback
                    )
                else:
                    # Open a new window to display the plot grid
                    PlotGridWindow(self.master, plot_grid_list, theme=self.theme)
        else:
            # Fit saved lines to the currently loaded spectrum
            self._perform_saved_lines_fit()

    def _perform_saved_lines_fit(self, spectrum_name=None, wavedata=None, fluxdata=None, err_data=None, plot_results=True, plot_grid=False):
        """Internal method to perform saved lines fitting on a single spectrum."""
        saved_lines_file = self.islat.input_line_list
        
        if not saved_lines_file:
            self.data_field.insert_text("No input line list file configured.\n")
            return None
        
        # Progress callback for GUI updates
        def progress_callback(msg):
            self.data_field.insert_text(msg, clear_after=False)
        
        # Use batch fitting service
        fit_data = self.batch_fitting_service.fit_lines_to_spectrum(
            saved_lines_file=saved_lines_file,
            spectrum_name=spectrum_name,
            wavedata=wavedata,
            fluxdata=fluxdata,
            err_data=err_data,
            progress_callback=progress_callback
        )
        
        if fit_data:
            fit_results_csv_data, fit_results_data = fit_data
            
            # Display summary
            summary = self.batch_fitting_service.get_fit_summary(fit_results_csv_data)
            self.data_field.insert_text(
                f"Completed fitting {summary['successful_fits']} out of {summary['total_lines']} lines.\n",
                clear_after=False
            )
            self.data_field.insert_text(
                f"Results saved to: {self.islat.output_line_measurements}\n",
                clear_after=False
            )
            
            # Display progress for each line
            progress_msgs = self.batch_fitting_service.format_fit_progress(fit_results_csv_data)
            for msg in progress_msgs:
                self.data_field.insert_text(msg, clear_after=False)
            
            if plot_grid:
                from iSLAT.Modules.Plotting.FitLinesPlotGrid import FitLinesPlotGrid
                if spectrum_name is None:
                    spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
                
                plot = FitLinesPlotGrid(
                    fit_data=fit_data,
                    wave_data=wavedata,
                    flux_data=fluxdata,
                    err_data=err_data,
                    fit_line_uncertainty=self.config.get('fit_line_uncertainty', 3.0),
                    spectrum_name=spectrum_name
                )
                plot.generate_plot()
                return plot
            
            if plot_results:
                self.main_plot.plot_renderer.plot_fitted_saved_lines(fit_results_data, self.main_plot.ax1)
        else:
            self.data_field.insert_text("No lines found or no fits completed successfully.\n", clear_after=False)
        
        return None

    def fit_saved_lines_to_sample(self):
        """Fit saved line list to a number of spectrum files at once."""
        self.fit_saved_lines(multiple_files=True)

    def find_single_lines(self):
        """Find isolated molecular lines (similar to single_finder function in original iSLAT)."""
        lines_to_show = 10

        try:
            self.data_field.clear()
            single_lines = self.main_plot.find_single_lines()
            self.main_plot.plot_single_lines()
            for i, line in enumerate(single_lines[:lines_to_show]):  # Show first lines_to_show lines
                self.data_field.insert_text(f"  Line {i+1}:", clear_after=False)
                for key, value in line.items():
                    self.data_field.insert_text(f"    {key}: {value}", clear_after=False)
                self.data_field.insert_text("\n", clear_after=False)
            
            if len(single_lines) > lines_to_show:
                self.data_field.insert_text(f"  ... and {len(single_lines) - lines_to_show} more lines\n", clear_after=False)
            
        except Exception as e:
            self.data_field.insert_text(f"Error finding single lines: {e}\n")

    def single_slab_fit(self):
        """Run single slab fit analysis."""        
        if self.islat.input_line_list is None:
            self.data_field.insert_text("No input line list specified. Cannot perform slab fit.\n", clear_after=True)
            return

        self.data_field.insert_text("Running single slab fit analysis...\n", clear_after=False)

        try:
            try:
                if not self.islat.output_line_measurements:
                    self.data_field.insert_text(f"No output line measurements file specified.\nUsing default folder: {line_saves_file_path}.", clear_after=False)
                    output_folder = line_saves_file_path
                else:
                    output_folder = os.path.dirname(self.islat.output_line_measurements)
            except Exception as e:
                self.data_field.insert_text(f"Error determining output folder: {e}", clear_after=False)
                self.data_field.insert_text(f"Using default folder: {line_saves_file_path}", clear_after=False)
                output_folder = line_saves_file_path
            # Use the SlabModel class to perform the fit
            slab_model = SlabModel(
                mol_object=self.islat.active_molecule,
                output_folder=output_folder,
                data_field=self.data_field,
                input_file=self.islat.input_line_list,
                flux_col_name=self.islat.user_settings.get("flux_col_name", "Flux_islat"),
                error_col_name=self.islat.user_settings.get("error_col_name", "Err_data")
            )
                
        except Exception as e:
            self.data_field.insert_text(f"Error loading single slab fit: {e}\n")
            return
        
        try:
            fitted_params = slab_model.fit_parameters()
        except Exception as e:
            self.data_field.insert_text(f"Error fitting slab model: {e}\n")
            return
        
        try:
            slab_model.update_molecule_parameters(fitted_params=fitted_params)
        except Exception as e:
            self.data_field.insert_text(f"Error updating molecule parameters: {e}\n")
            return

        try:
            slab_model.save_results(fitted_params=fitted_params)
        except Exception as e:
            self.data_field.insert_text(f"Error saving slab model results: {e}\n")
            return

    def export_models(self):
        """Export current models and data."""
        self.data_field.insert_text("Exporting current models...\n")
        
        # Create a new window for exporting the spectrum
        export_window = tk.Toplevel(self.master)
        export_window.title("Export Spectrum")
        # Always on top
        export_window.attributes("-topmost", True)

        # Create a label in the new window
        label = tk.Label(export_window, text="Select a molecule:")
        label.grid(row=0, column=0)

        # Create a dropdown menu in the new window
        options = list(self.islat.molecules_dict.keys()) + ["SUM", "ALL"]
        dropdown_var = tk.StringVar()
        dropdown = ttk.Combobox(export_window, textvariable=dropdown_var, values=options)
        dropdown.set(options[0])
        dropdown.grid(row=1, column=0)

        # Create a button in the new window
        button = ttk.Button(export_window, text="Generate CSV", command=lambda: generate_csv(molecules_data=self.islat.molecules_dict, mol_name=dropdown_var.get(),data_field=self.data_field, wave_data=self.islat.wave_data))
        button.grid(row=1, column=1)

    def toggle_atomic_lines(self):
        """
        Show atomic lines as vertical dashed lines on the plot.
        """
        try:
            self.atomic_toggle = not self.atomic_toggle

            # Check if full spectrum view is active
            if hasattr(self.main_plot, 'is_full_spectrum') and self.main_plot.is_full_spectrum:
                # Use optimized toggle method that only adds/removes line artists
                if hasattr(self.main_plot, 'full_spectrum_plot'):
                    self.main_plot.full_spectrum_plot.toggle_atomic_lines(self.atomic_toggle)
                    if hasattr(self.main_plot, 'full_spectrum_plot_canvas'):
                        self.main_plot.full_spectrum_plot_canvas.draw_idle()
            else:
                # Regular view
                if self.atomic_toggle:
                    self.main_plot.plot_atomic_lines(data_field=self.data_field)
                else:
                    self.main_plot.remove_atomic_lines()

        except Exception as e:
            self.data_field.insert_text(f"Error displaying atomic lines: {e}\n")
            traceback.print_exc()

    def toggle_summed_spectrum(self):
        """
        Toggle the display of the summed spectral flux on the plot.
        """
        try:
            self.main_plot.toggle_summed_spectrum()
        except Exception as e:
            self.data_field.insert_text(f"Error toggling summed spectrum: {e}\n")
            traceback.print_exc()

    def toggle_full_spectrum(self):
        """
        Toggle the display of the full spectrum on the plot.
        """
        try:
            self.main_plot.toggle_full_spectrum()
        except Exception as e:
            self.data_field.insert_text(f"Error toggling full spectrum: {e}\n")
            traceback.print_exc()

    def hitran_query(self):
        """
        Open the HITRAN molecule selector window.
        """
        try:
            # Use the root window from the islat class for the MoleculeSelector
            root_window = getattr(self.islat, 'root', self.master)
            MoleculeSelector(root_window, self.data_field)
        except Exception as e:
            print(f"Error opening HITRAN query: {e}")
            if self.data_field:
                self.data_field.insert_text(f"Error opening HITRAN query: {e}", console_print=True)
    
    def spectra_browser(self):
        print("Open spectra browser")

    def default_molecules(self):
        self.islat.load_default_molecules()

    def add_molecule(self):
        self.islat.add_molecule_from_hitran()

    def save_parameters(self, file_path = None, auto = False):
        """
        Save current molecule parameters to CSV file.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Sure you want to save? This will overwrite any previous save for this spectrum file."
        )
        if confirmed == "no":
            return
        
        # Get the loaded spectrum name for filename
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        
        from iSLAT.Modules.FileHandling import molsave_file_name

        try:
            # Save the current molecule parameters
            saved_file = write_molecules_to_csv(
                self.islat.molecules_dict, 
                loaded_spectrum_name=spectrum_name,
                file_name=molsave_file_name
            )
                
            # Also save to the general molecules list for session persistence
            #write_molecules_list_csv(self.islat.molecules_dict, loaded_spectrum_name=spectrum_name)
            
            if saved_file:
                # Update the data field to show success message
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Molecule parameters saved to: {saved_file}',
                        clear_after=True
                    )
                print(f"Molecule parameters saved successfully to: {saved_file}")
            else:
                print("Failed to save molecule parameters")
                
        except Exception as e:
            print(f"Error saving parameters: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error saving parameters: {str(e)}',
                    clear_after=True
                )
    
    def load_parameters(self):
        """
        Load molecule parameters from CSV file for the current spectrum.
        Uses the iSLAT class's _load_spectrum_parameters method.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Are you sure you want to load parameters? Make sure to save any unsaved changes!"
        )
        if confirmed == "no":
            return
        
        # Show loading message
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
            self.islat.GUI.data_field.insert_text(
                'Loading saved parameters, this may take a moment...',
                clear_after=True
            )
        
        # Use the iSLAT class method to load parameters
        self.islat._load_spectrum_parameters()
        
        # Update GUI components
        if hasattr(self.islat, 'GUI'):
            if hasattr(self.islat.GUI, 'plot'):
                self.main_plot.update_all_plots()
            if hasattr(self.islat.GUI, 'control_panel'):
                self.islat.GUI.control_panel.refresh_from_molecules_dict()

    def toggle_legend(self):
        #print("Toggled legend on plot")
        self.main_plot.toggle_legend()