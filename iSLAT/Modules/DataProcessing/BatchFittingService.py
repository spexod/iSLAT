"""
BatchFittingService - Service for batch fitting operations on spectral data.

This module provides functionality for fitting saved lines across multiple spectrum files,
separated from GUI concerns to enable reuse and testing.
"""

import os
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable

import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import iSLAT.Constants as c
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer

class BatchFittingService:
    """
    Service class for batch fitting operations on spectral line data.
    
    This class handles the logic for fitting saved lines to multiple spectrum files,
    extracting and processing fit results, and generating output files.
    """
    
    def __init__(self, islat_instance):
        """
        Initialize the batch fitting service.
        
        Parameters
        ----------
        islat_instance : iSLAT
            Reference to the main iSLAT instance for accessing data and configuration
        """
        self.islat = islat_instance
        self.line_analyzer = LineAnalyzer(islat_instance)
        self.fitting_engine = FittingEngine(islat_instance)
    
    def fit_lines_to_spectrum(
        self,
        saved_lines_file: str,
        spectrum_name: Optional[str] = None,
        wavedata: Optional[np.ndarray] = None,
        fluxdata: Optional[np.ndarray] = None,
        err_data: Optional[np.ndarray] = None,
        output_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[Optional[List[Dict]], Optional[Tuple]]:
        """
        Fit saved lines to a spectrum.
        
        Parameters
        ----------
        saved_lines_file : str
            Path to the saved lines CSV file
        spectrum_name : str, optional
            Name of the spectrum being fitted
        wavedata : np.ndarray, optional
            Wavelength data. If None, uses islat.wave_data
        fluxdata : np.ndarray, optional
            Flux data. If None, uses islat.flux_data
        err_data : np.ndarray, optional
            Error data. If None, uses islat.err_data
        output_file : str, optional
            Output file name for fit results
        progress_callback : callable, optional
            Callback function for progress updates: callback(message: str)
            
        Returns
        -------
        tuple
            (fit_results_csv_data, fit_results_data) or (None, None) on error
        """
        if not saved_lines_file:
            if progress_callback:
                progress_callback("No input line list file configured.\n")
            return None, None
        
        # Determine output file name
        if output_file is None:
            if spectrum_name is not None:
                spectrum_base_name = os.path.splitext(spectrum_name)[0]
                output_file = f"{spectrum_base_name}-{os.path.basename(saved_lines_file)}"
            else:
                output_file = self.islat.output_line_measurements if self.islat.output_line_measurements else "fit_results.csv"
        
        if spectrum_name is None:
            spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        
        if progress_callback:
            progress_callback(f"Fitting saved lines from: {saved_lines_file} to spectrum: {spectrum_name}\n")
        
        # Perform comprehensive line analysis
        fit_data = self.line_analyzer.analyze_saved_lines(
            saved_lines_file,
            self.fitting_engine,
            output_file,
            wavedata=wavedata,
            fluxdata=fluxdata,
            err_data=err_data,
        )
        
        return fit_data
    
    def fit_lines_to_multiple_spectra(
        self,
        saved_lines_file: str,
        spectrum_files: List[str],
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[Any]:
        """
        Fit saved lines to multiple spectrum files.
        
        Parameters
        ----------
        saved_lines_file : str
            Path to the saved lines CSV file
        spectrum_files : list of str
            List of spectrum file paths to process
        config : dict
            Configuration dictionary with settings like fit_line_uncertainty
        progress_callback : callable, optional
            Callback function for progress updates: callback(message: str)
            
        Returns
        -------
        list
            List of FitLinesPlotGrid objects for each spectrum
        """
        from iSLAT.Modules.Plotting.FitLinesPlotGrid import FitLinesPlotGrid
        
        plot_grid_list = []
        
        if progress_callback:
            progress_callback(f"Fitting saved lines to {len(spectrum_files)} spectrum files...\n")
        
        for spectrum_file in spectrum_files:
            try:
                # Get stellar RV for this spectrum
                save_info = self.islat.get_mole_save_data(os.path.basename(spectrum_file))
                stellar_rv = list(save_info.values())[0].get('StellarRV', 0.0) if save_info else 0.0
                stellar_rv = float(stellar_rv)
                
                # Load the spectrum data
                spectrum_df = ifh.read_spectral_data(spectrum_file)
                wavedata = np.array(spectrum_df['wave'].values)
                wavedata = wavedata - (wavedata / c.SPEED_OF_LIGHT_KMS * stellar_rv)  # Apply stellar RV correction
                fluxdata = np.array(spectrum_df['flux'].values)
                err_data = np.array(spectrum_df['err'].values)
                
                spectrum_name = os.path.basename(spectrum_file)
                
                # Fit the saved lines to the loaded spectrum
                fit_data = self.fit_lines_to_spectrum(
                    saved_lines_file=saved_lines_file,
                    spectrum_name=spectrum_name,
                    wavedata=wavedata,
                    fluxdata=fluxdata,
                    err_data=err_data,
                    progress_callback=None  # Don't spam progress for individual files
                )
                
                if fit_data:
                    # Generate plot grid
                    plot_grid = FitLinesPlotGrid(
                        fit_data=fit_data,
                        wave_data=wavedata,
                        flux_data=fluxdata,
                        err_data=err_data,
                        fit_line_uncertainty=config.get('fit_line_uncertainty', 3.0),
                        spectrum_name=spectrum_name
                    )
                    plot_grid.generate_plot()
                    plot_grid_list.append(plot_grid)
                
                if progress_callback:
                    progress_callback(f"Completed fitting for: {spectrum_name}\n")
                    
            except Exception as load_error:
                if progress_callback:
                    progress_callback(f"Error loading spectrum {os.path.basename(spectrum_file)}: {load_error}\n")
                continue
        
        if progress_callback:
            progress_callback("Completed fitting saved lines to all selected spectra.\n")
        
        return plot_grid_list
    
    def save_plot_grids_to_pdf(
        self,
        plot_grid_list: List[Any],
        save_directory: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> List[str]:
        """
        Save plot grids directly to PDF files.
        
        Parameters
        ----------
        plot_grid_list : list
            List of FitLinesPlotGrid objects
        save_directory : str
            Directory to save PDF files
        progress_callback : callable, optional
            Callback function for progress updates
            
        Returns
        -------
        list of str
            List of saved PDF file paths
        """
        import matplotlib.pyplot as plt
        
        os.makedirs(save_directory, exist_ok=True)
        saved_files = []
        
        for plot_grid in plot_grid_list:
            pdf_filename = f"{plot_grid.spectrum_name}_fit_grid.pdf"
            pdf_path = os.path.join(save_directory, pdf_filename)
            
            try:
                plot_grid.fig.savefig(pdf_path, dpi=200, bbox_inches='tight')
                saved_files.append(pdf_path)
                
                if progress_callback:
                    progress_callback(f"Saved fit plot grid to: {pdf_path}\n")
                    
            except Exception as save_error:
                if progress_callback:
                    progress_callback(f"Error saving PDF {pdf_filename}: {save_error}\n")
            finally:
                # Close the figure to free memory
                plt.close(plot_grid.fig)
        
        return saved_files
    
    def get_fit_summary(
        self,
        fit_results_csv_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Generate a summary of fit results.
        
        Parameters
        ----------
        fit_results_csv_data : list of dict
            List of fit result dictionaries
            
        Returns
        -------
        dict
            Summary statistics including successful fits, total lines, etc.
        """
        successful_fits = sum(1 for result in fit_results_csv_data if result.get('Fit_det', True))
        total_lines = len(fit_results_csv_data)
        
        return {
            'successful_fits': successful_fits,
            'total_lines': total_lines,
            'success_rate': successful_fits / total_lines if total_lines > 0 else 0.0
        }
    
    def format_fit_progress(
        self,
        fit_results_csv_data: List[Dict]
    ) -> List[str]:
        """
        Format fit results as progress messages.
        
        Parameters
        ----------
        fit_results_csv_data : list of dict
            List of fit result dictionaries
            
        Returns
        -------
        list of str
            List of progress message strings
        """
        messages = []
        
        for i, result in enumerate(fit_results_csv_data):
            if result.get('Fit_det', True):
                center = result.get('Centr_fit', result.get('lam', 0))
                messages.append(f"Line {i+1} at {center:.4f} μm: Fit successful")
            else:
                wavelength = result.get('lam', 0)
                messages.append(f"Line {i+1} at {wavelength:.4f} μm: Fit failed")
        
        return messages