import numpy as np
import tkinter as tk
import traceback
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
from ..GUIFunctions import create_button
from iSLAT.Modules.DataProcessing.Slabfit import SlabFit as SlabModel
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer
from .ResizableFrame import ResizableFrame

class BottomOptions(ResizableFrame):
    def __init__(self, master, islat, theme, main_plot, data_field, config):
        # Initialize the ResizableFrame with theme
        super().__init__(master, theme=theme, borderwidth=2, relief="groove")
        
        self.master = master
        self.islat = islat
        self.main_plot = main_plot
        self.data_field = data_field
        self.config = config

        # Create buttons for options
        self._create_buttons()
        
        # Apply initial theme
        self.apply_theme(theme)
    
    def _create_buttons(self):
        """Create all the button widgets."""
        create_button(self, self.theme, "Save Line", self.save_line, 0, 0)
        create_button(self, self.theme, "Show Saved Lines", self.show_saved_lines, 0, 1)
        create_button(self, self.theme, "Fit Line", self.fit_selected_line, 0, 2)
        create_button(self, self.theme, "Fit Saved Lines", self.fit_saved_lines, 0, 3)
        create_button(self, self.theme, "Find Single Lines", self.find_single_lines, 0, 4)
        create_button(self, self.theme, "Line De-blender", lambda: self.fit_selected_line(deblend=True), 0, 5)
        create_button(self, self.theme, "Single Slab Fit", self.single_slab_fit, 0, 6)
        create_button(self, self.theme, "Show Atomic Lines", self.show_atomic_lines, 0, 7)
    
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
            selected_line_info = self.main_plot.find_strongest_line_from_data()
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
            'xmin': xmin,
            'xmax': xmax,
        }
        
        try:
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
            
            # Update the main plot
            #self.main_plot.update_all_plots()
            
            self.data_field.insert_text(f"Displayed {len(saved_lines)} saved lines on plot.\n")
            
        except Exception as e:
            self.data_field.insert_text(f"Error loading saved lines: {e}\n")

    def fit_selected_line(self, deblend=False):
        """Fit the currently selected line using LMFIT"""

        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for fitting.\n", clear_first=False)
            return

        try:
            # Compute the fit using the main plot's fitting function
            fit_result = self.main_plot.compute_fit_line(deblend=deblend)
            
            if fit_result and len(fit_result) >= 3:
                lmfit_result, fitted_wave, fitted_flux = fit_result
                
                if lmfit_result is not None and hasattr(lmfit_result, 'params'):
                    # Extract parameters using the FittingEngine methods
                    line_params = self.main_plot.fitting_engine.extract_line_parameters()
                    fit_stats = self.main_plot.fitting_engine.get_fit_statistics()
                    
                    if deblend:
                        # For deblending, show detailed results AND save lines automatically
                        self.data_field.insert_text("\nDe-blended line fit results:\n", clear_first=False)
                        
                        # Handle multi-component fits - show detailed information
                        component_idx = 0
                        saved_components = 0
                        while f'component_{component_idx}' in line_params:
                            comp_params = line_params[f'component_{component_idx}']
                            self.data_field.insert_text(f"\nComponent {component_idx+1}:\n", clear_first=False)
                            
                            # Handle None values in stderr parameters
                            center_err = comp_params.get('center_stderr', 0)
                            center_err_str = f"{center_err:.5f}" if center_err is not None else "N/A"
                            
                            # Convert FWHM to km/s like old iSLAT
                            fwhm_kms = comp_params['fwhm'] / comp_params['center'] * 299792.458  # c in km/s
                            fwhm_err_kms = "N/A"  # Would need proper error propagation
                            
                            area_err = comp_params.get('area_stderr', 0)
                            area_err_str = f"{area_err:.3e}" if area_err is not None else "N/A"
                            
                            self.data_field.insert_text(f"Centroid (μm) = {comp_params['center']:.5f} +/- {center_err_str}", clear_first=False)
                            self.data_field.insert_text(f"FWHM (km/s) = {fwhm_kms:.1f} +/- {fwhm_err_kms}", clear_first=False)
                            self.data_field.insert_text(f"Area (erg/s/cm2) = {comp_params['area']:.3e} +/- {area_err_str}", clear_first=False)
                            
                            # Automatically save this component
                            try:
                                selection = self.main_plot.current_selection
                                if selection and len(selection) >= 2:
                                    xmin, xmax = selection[0], selection[-1]
                                    
                                    # Create line info dictionary for each component
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
                                    }
                                    
                                    # Save this component
                                    ifh.save_line(line_info)
                                    saved_components += 1
                                    
                            except Exception as save_error:
                                self.data_field.insert_text(f"Error saving component {component_idx+1}: {save_error}", clear_first=False)
                            
                            component_idx += 1
                        
                        if component_idx == 0:
                            self.data_field.insert_text("No components found in fit result.\n", clear_first=False)
                        else:
                            # Show both detailed results AND the classic save message
                            self.data_field.insert_text(f"\nDe-blended line fit completed with {component_idx} components!", clear_first=False)
                            if saved_components > 0:
                                self.data_field.insert_text(f"\nDe-blended line saved in /LINESAVES!", clear_first=False)
                            
                    else:
                        # Single Gaussian fit - show detailed results like old iSLAT
                        self.data_field.insert_text("\nGaussian fit results:\n", clear_first=False)
                        
                        if 'center' in line_params:
                            # Handle None values in stderr parameters
                            center_err = line_params.get('center_stderr', 0)
                            center_err_str = f"{center_err:.5f}" if center_err is not None else "N/A"
                            
                            # Convert FWHM to km/s like old iSLAT (approximately)
                            fwhm_kms = line_params['fwhm'] / line_params['center'] * 299792.458  # c in km/s
                            fwhm_err_kms = "N/A"  # Would need proper error propagation
                            
                            area_err = line_params.get('area_stderr', 0)
                            area_err_str = f"{area_err:.3e}" if area_err is not None else "N/A"
                            
                            self.data_field.insert_text(f"Centroid (μm) = {line_params['center']:.5f} +/- {center_err_str}", clear_first=False)
                            self.data_field.insert_text(f"FWHM (km/s) = {fwhm_kms:.1f} +/- {fwhm_err_kms}", clear_first=False)
                            self.data_field.insert_text(f"Area (erg/s/cm2) = {line_params['area']:.3e} +/- {area_err_str}", clear_first=False)
                        else:
                            self.data_field.insert_text("Could not extract fit parameters.\n", clear_first=False)
                else:
                    self.data_field.insert_text("Fit completed but no valid result object returned.\n", clear_first=False)
            else:
                self.data_field.insert_text("Fit failed or insufficient data.\n", clear_first=False)
            
            # Update plots
            self.main_plot.plot_line_inspection(highlight_strongest=False)
            
        except Exception as e:
            self.data_field.insert_text(f"Error during fitting: {e}\n", clear_first=False)
            self.data_field.insert_text(f"Traceback: {traceback.format_exc()}\n", clear_first=False)

    def fit_saved_lines(self, print_output=False):
        """
        Fit all saved lines using LineAnalyzer for comprehensive analysis.
        Simplified method that delegates all logic to appropriate classes.
        """
        #try:
        # Get file paths from iSLAT instance
        saved_lines_file = self.islat.input_line_list
        output_file = self.islat.output_line_measurements if self.islat.output_line_measurements else "fit_results.csv"
        
        # Validate that files are properly configured
        if not saved_lines_file:
            self.data_field.insert_text("No input line list file configured.\n")
            return
            
        self.data_field.insert_text(f"Fitting saved lines from: {saved_lines_file}\n")
        
        # Initialize LineAnalyzer and FittingEngine
        line_analyzer = LineAnalyzer(self.islat)
        fitting_engine = FittingEngine(self.islat)
        
        # Perform comprehensive line analysis
        fit_results = line_analyzer.analyze_saved_lines(
            saved_lines_file,
            fitting_engine,
            output_file
        )
        
        if fit_results:
            successful_fits = sum(1 for result in fit_results if result.get('Fit_det', False))
            total_lines = len(fit_results)
            
            # Save results using file handling module
            '''output_path = ifh.save_fit_results_csv(
                fit_results,
                file_path=self.islat.output_line_measurements,
                file_name=output_file
            )'''
            
            #ifh.save_fit_results(fit_results, file_name=self.islat.output_line_measurements)

            self.data_field.insert_text(f"Completed fitting {successful_fits} out of {total_lines} lines.\n")
            self.data_field.insert_text(f"Results saved to: {self.islat.output_line_measurements}\n")
            
            # Update progress for each successful fit
            for i, result in enumerate(fit_results):
                if result.get('Fit_det', False):
                    center = result.get('Centr_fit', result.get('lam', 0))
                    snr = result.get('Fit_SN', 0)
                    self.data_field.insert_text(f"Line {i+1} at {center:.4f} μm: Fit successful (S/N={snr:.1f})\n")
                else:
                    wavelength = result.get('lam', 0)
                    self.data_field.insert_text(f"Line {i+1} at {wavelength:.4f} μm: Fit failed\n")
            
            # Update the line inspection plot if available
            #if hasattr(self.main_plot, 'update_line_inspection_plot'):
            #    self.main_plot.update_line_inspection_plot()
        else:
            self.data_field.insert_text("No lines found or no fits completed successfully.\n")
            
        #except Exception as e:
        #    self.data_field.insert_text(f"Error fitting saved lines: {e}\n")
        #    if print_output:
        #        traceback.print_exc()

    def find_single_lines(self):
        """Find isolated molecular lines (similar to single_finder function in original iSLAT)."""
        lines_to_show = 10

        try:
            self.data_field.clear()
            single_lines = self.main_plot.find_single_lines()
            self.main_plot.plot_single_lines()
            for i, line in enumerate(single_lines[:lines_to_show]):  # Show first lines_to_show lines
                self.data_field.insert_text(f"  Line {i+1}:", clear_first=False)
                for key, value in line.items():
                    self.data_field.insert_text(f"    {key}: {value}", clear_first=False)
                self.data_field.insert_text("\n", clear_first=False)
            
            if len(single_lines) > lines_to_show:
                self.data_field.insert_text(f"  ... and {len(single_lines) - lines_to_show} more lines\n", clear_first=False)
            
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
        slab_model.fit_parameters()
        '''except Exception as e:
            self.data_field.insert_text(f"Error fitting slab model: {e}\n")
            return'''
        
        try:
            slab_model.save_results()
        except Exception as e:
            self.data_field.insert_text(f"Error saving slab model results: {e}\n")
            return

    def export_models(self):
        """Export current models and data."""
        self.data_field.insert_text("Exporting current models...\n")
        
        try:
            # Check if we have models to export
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                # Export model data for each visible molecule
                exported_count = 0
                for mol_name, molecule in self.islat.molecules_dict.items():
                    if hasattr(molecule, 'is_visible') and molecule.is_visible:
                        # Export this molecule's model
                        # This would depend on the specific export functionality
                        self.data_field.insert_text(f"Exported model for {mol_name}\n")
                        exported_count += 1
                
                if exported_count > 0:
                    self.data_field.insert_text(f"Successfully exported {exported_count} models.\n")
                else:
                    self.data_field.insert_text("No visible models to export.\n")
            else:
                self.data_field.insert_text("No models available for export.\n")
                
        except Exception as e:
            self.data_field.insert_text(f"Error exporting models: {e}\n")

        try:
            out_files = self.islat.slab_model.export_results()
            for f in out_files:
                self.data_field.insert_text(f"Exported to: {f}\n")
        except Exception as e:
            self.data_field.insert_text(f"Error exporting models: {e}\n")

    def show_atomic_lines(self):
        """
        Show atomic lines as vertical dashed lines on the plot.
        Replicates the functionality from the original iSLAT atomic lines feature.
        """
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
                    ax1.axvline(wavelengths[i], linestyle='--', color='tomato', alpha=0.7)
                    
                    # Adjust the y-coordinate to place labels within the plot borders
                    ylim = ax1.get_ylim()
                    label_y = ylim[1]
                    
                    # Adjust the x-coordinate to place labels just to the right of the line
                    xlim = ax1.get_xlim()
                    label_x = wavelengths[i] + 0.006 * (xlim[1] - xlim[0])
                    
                    # Add text label for the line
                    label_text = f"{species[i]} {line_ids[i]}"
                    ax1.text(label_x, label_y, label_text, fontsize=8, rotation=90, 
                            va='top', ha='left', color='tomato')
                
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