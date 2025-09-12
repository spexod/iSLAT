import numpy as np
import platform
import os 
import csv
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, font
import traceback
from typing import TYPE_CHECKING, Any, Dict, Optional
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
from ..GUIFunctions import create_button, create_menu_btn
from iSLAT.Modules.DataProcessing.Slabfit import SlabFit as SlabModel
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer
from .ResizableFrame import ResizableFrame
from iSLAT.Modules.GUI.Widgets.ChartWindow import MoleculeSelector
from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv, generate_csv
from iSLAT.Modules.FileHandling.iSLATFileHandling import save_folder_path, molsave_file_name, line_saves_file_path, line_saves_file_name, fit_save_lines_file_name, example_data_folder_path
import iSLAT.Constants as c

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

        self.atomic_lines = []

        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=0, column=1)

        # Create buttons for options
        self._create_buttons()
        
        # Create and add toolbar 
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=0, column=2, sticky="nsew")
        self.toolbar = self.main_plot.create_toolbar(toolbar_frame)
        
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
            spectrum_drpdwn = create_menu_btn(self.button_frame, self.theme, "Spectrum Parameters", 0, 1)
        else:
            spectrum_drpdwn = create_menu_btn(self.button_frame, self.theme, "Spectrum Parameters ▼", 0, 1)
        spectrum_menu = tk.Menu(spectrum_drpdwn, tearoff=0)
        spectrum_menu.add_command(label="Save Parameters", command=self.save_parameters)
        spectrum_menu.add_command(label="Load Parameters", command=self.load_parameters)
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

        saved_lines_tip = "Show saved lines\nform the 'Input Line List'"
        atomic_lines_tip = "Show atomic lines\nusing seperation threshold\nset in the 'Line Separ."
        export_model_tip = "Export current\nmodels into csv files"
        toggle_legend_tip = "Turn legend on/off"
        create_button(self.button_frame, self.theme, "Toggle Saved Lines", self.show_saved_lines, 0, 3, tip_text=saved_lines_tip)
        create_button(self.button_frame, self.theme, "Toggle Atomic Lines", self.show_atomic_lines, 0, 4, tip_text=atomic_lines_tip)
        create_button(self.button_frame, self.theme, "Toggle Legend", self.main_plot.toggle_legend, 0, 5, tip_text=toggle_legend_tip)


    def save_line(self, save_type="selected"):
        """Save the currently selected line to the line saves file using the new MoleculeLine approach."""
        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for saving.\n")
            return

        if save_type == "strongest":
            # Get the strongest line information from the current selection using new approach
            selected_line_info = self.main_plot.find_strongest_line_from_data()
            if selected_line_info is None:
                self.data_field.insert_text("No valid line found in selection.\n")
                return
        elif save_type == "selected":
            # Use the currently selected region and find the strongest line in it
            selected_line_info = self.main_plot.selected_line
            if selected_line_info is None:
                # Fallback: create basic line info from selection bounds
                xmin, xmax = self.main_plot.current_selection
                center_wave = (xmin + xmax) / 2.0
                
                # Calculate flux integral in the selected range
                err_data = getattr(self.islat, 'err_data', None)
                line_flux, line_err = self.main_plot.flux_integral(
                    self.islat.wave_data, 
                    self.islat.flux_data, 
                    err_data, 
                    xmin, 
                    xmax
                )
                
                selected_line_info = {
                    'lam': center_wave,
                    'wavelength': center_wave,
                    'flux': line_flux,
                    'intensity': line_flux,
                    'e': 0.0,
                    'a': 0.0,
                    'g': 1.0,
                    'inten': line_flux,
                    'up_lev': 'Unknown',
                    'low_lev': 'Unknown',
                    'tau': 0.0
                }
        else:
            self.data_field.insert_text("Invalid save type specified.\n")
            return
            
        # Get selection bounds for xmin/xmax
        if hasattr(self.main_plot, 'selected_wave') and self.main_plot.selected_wave is not None:
            xmin = self.main_plot.selected_wave[0] if len(self.main_plot.selected_wave) > 0 else selected_line_info['lam'] - 0.01
            xmax = self.main_plot.selected_wave[-1] if len(self.main_plot.selected_wave) > 1 else selected_line_info['lam'] + 0.01
        else:
            # Use current selection bounds
            xmin, xmax = self.main_plot.current_selection
            
        # Create line info dictionary with the format expected by the file handler
        line_info = {
            'species': self.islat.active_molecule.name,
            'lev_up': selected_line_info.get('up_lev', ''),
            'lev_low': selected_line_info.get('low_lev', ''),
            'lam': selected_line_info['lam'],
            'tau': selected_line_info.get('tau', 0.0),
            'intens': selected_line_info.get('inten', selected_line_info.get('intensity', 0.0)),
            'a_stein': selected_line_info.get('a', 0.0),
            'e_up': selected_line_info.get('e', 0.0),
            'g_up': selected_line_info.get('g', 1.0),
            'e_low': selected_line_info.get('e_low', 0.0),
            'g_low': selected_line_info.get('g_low', 1.0),
            'xmin': xmin,
            'xmax': xmax,
        }
        
        try:
            if not self.islat.output_line_measurements:
                self.data_field.insert_text("No output line measurements file specified.\n")
                return
            ifh.save_line(line_info, file_name=self.islat.output_line_measurements)
            self.data_field.insert_text(f"Saved line at {line_info['lam']:.4f} μm\n")
        except Exception as e:
            self.data_field.insert_text(f"Error saving line: {e}\n")

    def show_saved_lines(self):
        """Show saved lines as vertical dashed lines on the plot."""
        try:
            # Load saved lines from file
            saved_lines = ifh.read_line_saves(file_name=self.islat.input_line_list)
            if saved_lines.empty:       
                self.data_field.insert_text("No saved lines found.\n")
                return
                
            # Plot the saved lines on the main plot
            self.main_plot.plot_saved_lines(saved_lines)
            
            self.data_field.insert_text(f"Displayed {len(saved_lines)} saved lines on plot.\n")
            
        except Exception as e:
            self.data_field.insert_text(f"Error loading saved lines: {e}\n")

    def fit_selected_line(self, deblend=False):
        """Fit the currently selected line using LMFIT"""

        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for fitting.\n", clear_after=False)
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
                        # For deblending, show detailed results AND save lines automatically
                        self.data_field.insert_text("\nDe-blended line fit results:\n", clear_after=False)
                        
                        selection = self.main_plot.current_selection
                        if selection and len(selection) >= 2:
                            xmin, xmax = selection[0], selection[-1]
                        line_info = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
                        #print(line_info)

                        # Handle multi-component fits - show detailed information
                        component_idx = 0
                        saved_components = 0

                        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
                        
                        spectrum_base_name = os.path.splitext(spectrum_name)[0] if spectrum_name != "unknown" else "default"
                        #save_file = os.path.join(line_saves_file_path, f"{spectrum_base_name}-{line_saves_file_name}")
                        save_file_name = f"{spectrum_base_name}-{line_saves_file_name}"

                        while f'component_{component_idx}' in line_params:
                            comp_params = line_params[f'component_{component_idx}']
                            self.data_field.insert_text(f"\nComponent {component_idx+1}:\n", clear_after=False)
                            
                            # Handle None values in stderr parameters
                            center_err = comp_params.get('center_stderr', 0)
                            center_err_str = f"{center_err:.5f}" if center_err is not None else "0"
                            
                            # Convert FWHM to km/s
                            fwhm_err = line_params.get('fwhm_stderr', 0)
                            fwhm_err_kms = f"{fwhm_err:.1f}" if fwhm_err is not None else "0"
                            
                            area_err = comp_params.get('area_stderr', 0)
                            area_err_str = f"{area_err:.3e}" if area_err is not None else "0"
                            
                            self.data_field.insert_text(f"Centroid (μm) = {comp_params['center']:.5f} +/- {center_err_str}", clear_after=False)
                            self.data_field.insert_text(f"FWHM (km/s) = {comp_params['fwhm']:.1f} +/- {fwhm_err_kms}", clear_after=False)
                            self.data_field.insert_text(f"Area (erg/s/cm2) = {comp_params['area']:.3e} +/- {area_err_str}", clear_after=False)
                            
                            # Automatically save this component
                            try:
                                    '''# Create line info dictionary for each component
                                    line_info = {
                                        'species': self.islat.active_molecule,
                                        'lev_up': f'deblend_comp_{component_idx+1}',
                                        'lev_low': '',
                                        'lam': comp_params['center'],
                                        'tau': comp_params['amplitude'],
                                        'intens': comp_params['area'],
                                        'a_stein': '',
                                        'e_up': '',
                                        'g_up': '',
                                        'xmin': xmin,
                                        'xmax': xmax,
                                        'flux_fit': comp_params['area'],
                                        'fwhm_fit': comp_params['fwhm'],
                                        'centr_fit': comp_params['center']
                                    }'''

                                    current_tripple = line_info[component_idx]
                                    current_line_info = current_tripple[0].get_dict()
                                    current_intens = current_tripple[1]
                                    current_tau = current_tripple[2]

                                    doppler = ((comp_params['center'] - current_line_info["lam"]) / current_line_info["lam"] * c.SPEED_OF_LIGHT_KMS) if current_line_info["lam"] else np.nan

                                    line_save_info = {
                                        'species': self.islat.active_molecule.name,
                                        'lev_up': current_line_info['lev_up'],
                                        'lev_low': current_line_info['lev_low'],
                                        'lam': current_line_info['lam'],
                                        'tau': current_tau,
                                        'intens': current_intens,
                                        'a_stein': current_line_info['a_stein'],
                                        'e_up': current_line_info['e_up'],
                                        'g_up': current_line_info['g_up'],
                                        'Flux_fit': comp_params['area'],
                                        'Err_fit': comp_params['area_stderr'],
                                        'FWHM_fit': comp_params['fwhm'],
                                        'FWHM_err': comp_params['fwhm_stderr'],
                                        'Centr_fit': comp_params['center'],
                                        'Centr_err': comp_params['center_stderr'],
                                        'Doppler': doppler
                                    }

                                    # Save this component
                                    ifh.save_line(line_save_info, file_name=save_file_name)
                                    saved_components += 1
                                    
                            except Exception as save_error:
                                self.data_field.insert_text(f"Error saving component {component_idx+1}: {save_error}", clear_after=False)
                            
                            component_idx += 1
                        
                        if component_idx == 0:
                            self.data_field.insert_text("No components found in fit result.\n", clear_after=False)
                        else:
                            # Show both detailed results
                            self.data_field.insert_text(f"\nDe-blended line fit completed with {component_idx} components!", clear_after=False)
                            fit_result_summary = self.main_plot.fitting_engine.get_fit_results_summary()
                            fit_results_components = self.main_plot.fitting_engine.get_fit_results_components()
                            ifh.save_deblended_fit_stats_and_models(deblended_data = fit_result_summary, components_data=fit_results_components, models_file_name=f"{spectrum_base_name}-deblend_models.csv", stats_file_name=f"{spectrum_base_name}-deblended_fit_statistics.json")
                            figpath = os.path.join(line_saves_file_path, f"{spectrum_base_name}-deblend_plot.pdf")
                            self.main_plot.save_fig(figpath, dpi=10)
                            if saved_components > 0:
                                self.data_field.insert_text(f"\nDe-blended line saved in /LINESAVES!", clear_after=False)
                            
                    else:
                        # Single Gaussian fit - show detailed results
                        self.data_field.insert_text("\nGaussian fit results:\n", clear_after=False)
                        
                        if 'center' in line_params:
                            # Handle None values in stderr parameters
                            center_err = line_params.get('center_stderr', 0)
                            center_err_str = f"{center_err:.5f}" if center_err is not None else "0"
                            
                            fwhm_err = line_params.get('fwhm_stderr', 0)
                            fwhm_err_kms = f"{fwhm_err:.5f}" if fwhm_err is not None else "0"
                            
                            area_err = line_params.get('area_stderr', 0)
                            area_err_str = f"{area_err:.3e}" if area_err is not None else "0"
                            
                            self.data_field.insert_text(f"Centroid (μm) = {line_params['center']:.5f} +/- {center_err_str}", clear_after=False)
                            self.data_field.insert_text(f"FWHM (km/s) = {line_params['fwhm']:.5f} +/- {fwhm_err_kms}", clear_after=False)
                            self.data_field.insert_text(f"Area (erg/s/cm2) = {line_params['area']:.3e} +/- {area_err_str}", clear_after=False)
                        else:
                            self.data_field.insert_text("Could not extract fit parameters.\n", clear_after=False)
                else:
                    self.data_field.insert_text("Fit completed but no valid result object returned.\n", clear_after=False)
            else:
                self.data_field.insert_text("Fit failed or insufficient data.\n", clear_after=False)
            
        except Exception as e:
            self.data_field.insert_text(f"Error during fitting: {e}\n", clear_after=False)
            self.data_field.insert_text(f"Traceback: {traceback.format_exc()}\n", clear_after=False)

    def fit_saved_lines(self, multiple_files=False):
        """
        Fit all saved lines using LineAnalyzer for comprehensive analysis.
        
        Args:
            multiple_files (bool): If True, allows user to select multiple spectrum files.
                                 If False, fits saved lines to the currently loaded spectrum.
        """
        if not self.islat.input_line_list:
            self.data_field.insert_text("No input line list file configured.\n")
            return
        
        if not self.islat.output_line_measurements:
            self.data_field.insert_text("No output line measurements file configured.\n")
            return
        
        if multiple_files:
            # Ask user to select multiple spectrum files
            from tkinter import filedialog
            spectrum_files = filedialog.askopenfilenames(
                title="Select Spectrum Files to Fit Saved Lines",
                filetypes=[("All files", "*.*")],
                initialdir=example_data_folder_path
            )
            
            if not spectrum_files:
                self.data_field.insert_text("No spectrum files selected.\n")
                return
            
            self.data_field.insert_text(f"Fitting saved lines to {len(spectrum_files)} spectrum files...\n")

            for spec_file in spectrum_files:
                try:
                    # Load the spectrum data
                    spectrum_df = ifh.read_spectral_data(spec_file)
                    wavedata=np.array(spectrum_df['wave'].values)
                    fluxdata=np.array(spectrum_df['flux'].values)
                    err_data=np.array(spectrum_df['err'].values) #if 'err' in spectrum_df.columns else None
                    print(f'Err data loaded: {err_data}')
                    print(f"Length of wave data: {len(wavedata)}, flux data: {len(fluxdata)}, err data: {len(err_data)}")
                    # Fit the saved lines to the loaded spectrum
                    self._perform_saved_lines_fit(
                        spectrum_name=os.path.basename(spec_file),
                        wavedata=wavedata,
                        fluxdata=fluxdata,
                        err_data=err_data,
                        plot_results=False
                    )

                    self.data_field.insert_text(f"Completed fitting for: {os.path.basename(spec_file)}\n", clear_after=False)
                    
                except Exception as e:
                    self.data_field.insert_text(f"Error processing {os.path.basename(spec_file)}: {e}\n", clear_after=False)
            
            self.data_field.insert_text("Completed fitting saved lines to all selected spectra.\n")
        else:
            # Fit saved lines to the currently loaded spectrum
            self._perform_saved_lines_fit()

    def _perform_saved_lines_fit(self, spectrum_name=None, wavedata=None, fluxdata=None, err_data=None, plot_results=True):
        """
        Internal method to perform the actual saved lines fitting.
        """
        saved_lines_file = self.islat.input_line_list
        #output_file = self.islat.output_line_measurements if self.islat.output_line_measurements else "fit_results.csv"
        
        if spectrum_name is not None:
            spectrum_base_name = os.path.splitext(spectrum_name)[0]
            output_file = f"{spectrum_base_name}-{os.path.basename(saved_lines_file)}"
            print(f"Output file for spectrum {spectrum_name}: {output_file}")
        else:
            output_file = self.islat.output_line_measurements if self.islat.output_line_measurements else "fit_results.csv"

        if spectrum_name is None:
            spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')

        # Validate that files are properly configured
        if not saved_lines_file:
            self.data_field.insert_text("No input line list file configured.\n")
            return

        self.data_field.insert_text(f"Fitting saved lines from: {saved_lines_file} to spectrum: {spectrum_name}\n")

        # Initialize LineAnalyzer and FittingEngine
        line_analyzer = LineAnalyzer(self.islat)
        fitting_engine = FittingEngine(self.islat)
        
        # Perform comprehensive line analysis
        fit_data = line_analyzer.analyze_saved_lines(
            saved_lines_file,
            fitting_engine,
            output_file,
            wavedata=wavedata,
            fluxdata=fluxdata,
            err_data=err_data,
        )
        
        if fit_data:
            fit_results_csv_data, fit_results_data = fit_data
            successful_fits = sum(1 for result in fit_results_csv_data if result.get('Fit_det', True))
            total_lines = len(fit_results_csv_data)

            self.data_field.insert_text(f"Completed fitting {successful_fits} out of {total_lines} lines.\n", clear_after=False)
            self.data_field.insert_text(f"Results saved to: {self.islat.output_line_measurements}\n", clear_after=False)

            # Update progress for each successful fit
            for i, result in enumerate(fit_results_csv_data):
                if result.get('Fit_det', True):
                    center = result.get('Centr_fit', result.get('lam', 0))
                    snr = result.get('Fit_SN', 0)
                    self.data_field.insert_text(f"Line {i+1} at {center:.4f} μm: Fit successful", clear_after=False)
                else:
                    wavelength = result.get('lam', 0)
                    self.data_field.insert_text(f"Line {i+1} at {wavelength:.4f} μm: Fit failed", clear_after=False)

            if plot_results:
                self.main_plot.plot_renderer.plot_fitted_saved_lines(fit_results_data, self.main_plot.ax1)

        else:
            self.data_field.insert_text("No lines found or no fits completed successfully.\n", clear_after=False)

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
        self.data_field.insert_text("Running single slab fit analysis...\n")
        
        try:
            output_folder = self.islat.output_line_measurements
            # Use the SlabModel class to perform the fit
            slab_model = SlabModel(
                mol_object=self.islat.active_molecule,
                output_folder=output_folder,
                data_field=self.data_field,
                input_file=self.islat.input_line_list,
            )
                
        except Exception as e:
            self.data_field.insert_text(f"Error loading single slab fit: {e}\n")
            return
        
        #try:
        fitted_params = slab_model.fit_parameters()
        '''except Exception as e:
            self.data_field.insert_text(f"Error fitting slab model: {e}\n")
            return'''
        
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
        #options = [molecule[0] for molecule in molecules_data] + ["SUM"] + ["ALL"]
        options = list(self.islat.molecules_dict.keys()) + ["SUM", "ALL"]
        dropdown_var = tk.StringVar()
        dropdown = ttk.Combobox(export_window, textvariable=dropdown_var, values=options)
        dropdown.set(options[0])
        dropdown.grid(row=1, column=0)

        # Create a button in the new window
        button = ttk.Button(export_window, text="Generate CSV", command=lambda: generate_csv(molecules_data=self.islat.molecules_dict, mol_name=dropdown_var.get(),data_field=self.data_field, wave_data=self.islat.wave_data))
        button.grid(row=1, column=1)

    def show_atomic_lines(self):
        """
        Show atomic lines as vertical dashed lines on the plot.
        """
        if self.atomic_lines:
            for (line, text) in self.atomic_lines:
                try:
                    line.remove()
                    text.remove()
                except ValueError:
                    pass
            
            self.atomic_lines.clear()
            self.main_plot.canvas.draw()
            return

        try:
            # Load atomic lines from file using the file handling module
            atomic_lines = ifh.load_atomic_lines()
            
            if atomic_lines.empty:
                self.data_field.insert_text("No atomic lines data found.\n")
                return
            
            # Get the main plot axes
            if hasattr(self.main_plot, 'ax1'):
                ax1 = self.main_plot.ax1
                
                # Get wavelength and other data from the atomic lines DataFrame
                wavelengths = atomic_lines['wave'].values
                species = atomic_lines['species'].values
                line_ids = atomic_lines['line'].values
                
                # Plot vertical lines for each atomic line
                
                for i in range(len(wavelengths)):
                    line = ax1.axvline(wavelengths[i], linestyle='--', color='tomato', alpha=0.7)
                    
                    # Adjust the y-coordinate to place labels within the plot borders
                    ylim = ax1.get_ylim()
                    label_y = ylim[1]
                    
                    # Adjust the x-coordinate to place labels just to the right of the line
                    xlim = ax1.get_xlim()
                    label_x = wavelengths[i] + 0.006 * (xlim[1] - xlim[0])
                    
                    # Add text label for the line
                    label_text = f"{species[i]} {line_ids[i]}"
                    label = ax1.text(label_x, label_y, label_text, fontsize=8, rotation=90, 
                                                        va='top', ha='left', color='tomato')
                    
                    self.atomic_lines.append((line, label))
                
                # Update the plot
                self.main_plot.canvas.draw()
                
                # Update data field
                self.data_field.insert_text(f"Displayed {len(wavelengths)} atomic lines on plot.\n")
                self.data_field.insert_text("Atomic lines retrieved from file.\n")
                
            else:
                self.data_field.insert_text("Main plot not available for atomic lines display.\n")
                
        except Exception as e:
            self.data_field.insert_text(f"Error displaying atomic lines: {e}\n")
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

    def save_parameters(self):
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
        Load molecule parameters from CSV file.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Are you sure you want to load parameters? Make sure to save any unsaved changes!"
        )
        if confirmed == "no":
            return
        
        # Get the loaded spectrum name for filename
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        
        spectrum_base_name = os.path.splitext(spectrum_name)[0] if spectrum_name != "unknown" else "default"
        save_file = os.path.join(save_folder_path, f"{spectrum_base_name}-{molsave_file_name}")

        if not os.path.exists(save_file):
            save_file = os.path.join(save_folder_path, f"{spectrum_base_name}.csv-{molsave_file_name}")
        
        if not os.path.exists(save_file):
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    'No save file found for this spectrum.',
                    clear_after=True
                )
            print(f"No save file found at: {save_file}")
            return
        
        try:
            # Show loading message
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    'Loading saved parameters, this may take a moment...',
                    clear_after=True
                )
                
            # Clear existing molecules
            self.islat.molecules_dict.clear()
            
            mole_save_data = self.islat.get_mole_save_data()
            
            # Initialize molecules from loaded data
            self.islat.init_molecules(mole_save_data)

            # Update GUI components
            if hasattr(self.islat, 'GUI'):
                if hasattr(self.islat.GUI, 'plot'):
                    self.islat.GUI.plot.update_all_plots()
                if hasattr(self.islat.GUI, 'control_panel'):
                    self.islat.GUI.control_panel.refresh_from_molecules_dict()
                if hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Successfully loaded parameters from: {save_file}',
                        clear_after=True
                    )
            
            print(f"Successfully loaded {len(mole_save_data)} molecules from: {save_file}")
            
        except Exception as e:
            print(f"Error loading parameters: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error loading parameters: {str(e)}',
                    clear_after=True
                )

    def toggle_legend(self):
        #print("Toggled legend on plot")
        self.islat.GUI.plot.toggle_legend()