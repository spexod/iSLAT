"""
BatchFittingService - Service for batch fitting operations on spectral data.

This module provides functionality for fitting saved lines across multiple spectrum files,
separated from GUI concerns to enable reuse and testing.
"""

import os
import sys
import numpy as np
import pandas as pd
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable

import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import iSLAT.Constants as c
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer

class ProgressTracker:
    """
    Thread-safe progress tracker for parallel batch fitting.
    
    Redraws all progress bars in-place when any spectrum updates,
    providing a clean, non-scrolling display.
    """
    
    _update_interval = 10  # Update display every N lines processed

    def __init__(self, spectrum_names: List[str]):
        self._lock = threading.Lock()
        self._spectrum_names = list(spectrum_names)  # Preserve order
        self._progress = {name: (0, 1) for name in spectrum_names}  # name -> (current, total)
        self._lines_printed = 0
        self._last_update = {name: 0 for name in spectrum_names}
    
    def update(self, spectrum_name: str, current: int, total: int):
        """Update progress for a specific spectrum and redraw all bars."""
        with self._lock:
            self._progress[spectrum_name] = (current, total)
            
            # Only redraw on interval or completion to reduce flicker
            last = self._last_update.get(spectrum_name, 0)
            if current < total and (current - last) < self._update_interval:
                return
            self._last_update[spectrum_name] = current
            
            # Move cursor up to overwrite previous output
            if self._lines_printed > 0:
                sys.stdout.write(f"\033[{self._lines_printed}A")
            
            # Redraw all progress bars
            lines = []
            for name in self._spectrum_names:
                curr, tot = self._progress[name]
                percent = int((curr / tot) * 100) if tot > 0 else 0
                bar_length = 20
                filled = int(bar_length * curr / tot) if tot > 0 else 0
                bar = '█' * filled + '░' * (bar_length - filled)
                # Pad to consistent width to overwrite previous content
                line = f"  {name}: [{bar}] {curr}/{tot} ({percent}%)"
                lines.append(line.ljust(80))
            
            output = '\n'.join(lines)
            sys.stdout.write(output + '\n')
            sys.stdout.flush()
            self._lines_printed = len(lines)

class BatchFittingService:
    """
    Service class for batch fitting operations on spectral line data.
    
    This class handles the logic for fitting saved lines to multiple spectrum files,
    extracting and processing fit results, and generating output files.
    """
    
    # Class-level settings for batch fitting behavior
    PARALLEL_BATCH_FITTING: bool = True
    BATCH_FITTING_MAX_WORKERS: int = None  # None uses CPU count - 1
    
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
        # Cache for saved lines to avoid re-reading
        self._saved_lines_cache: Dict[str, pd.DataFrame] = {}
        # Thread-safe printing lock
        self._print_lock = threading.Lock()
        # Progress tracker for parallel processing
        self._progress_tracker: Optional[ProgressTracker] = None
        # Current output folder for batch run
        self._current_output_folder: Optional[str] = None
    
    def _create_run_folder(self, base_path: str, line_list_name: str) -> str:
        """
        Create a unique subfolder for this batch run.
        
        Parameters
        ----------
        base_path : str
            Base directory for output
        line_list_name : str
            Name of the line list file (used in folder name)
            
        Returns
        -------
        str
            Path to the created folder
        """
        # Create folder name with line list and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        line_list_base = os.path.splitext(os.path.basename(line_list_name))[0]
        folder_name = f"batch_{line_list_base}_{timestamp}"
        folder_path = os.path.join(base_path, folder_name)
        
        os.makedirs(folder_path, exist_ok=True)
        return folder_path
    
    def _get_saved_lines(self, saved_lines_file: str) -> pd.DataFrame:
        """Get saved lines from cache or read from file."""
        if saved_lines_file not in self._saved_lines_cache:
            try:
                self._saved_lines_cache[saved_lines_file] = pd.read_csv(saved_lines_file)
            except Exception as e:
                print(f"Error reading saved lines file: {e}")
                return pd.DataFrame()
        return self._saved_lines_cache[saved_lines_file]
    
    def clear_cache(self):
        """Clear the saved lines cache."""
        self._saved_lines_cache.clear()
    
    def fit_lines_to_spectrum(
        self,
        saved_lines_file: str,
        spectrum_name: Optional[str] = None,
        wavedata: Optional[np.ndarray] = None,
        fluxdata: Optional[np.ndarray] = None,
        err_data: Optional[np.ndarray] = None,
        output_file: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        saved_lines_df: Optional[pd.DataFrame] = None,
        line_progress_callback: Optional[Callable[[int, int], None]] = None
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
        line_progress_callback : callable, optional
            Callback for per-line progress: callback(current_line, total_lines)
            
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
            progress_callback=line_progress_callback,
            output_path=self._current_output_folder
        )
        
        return fit_data
    
    def fit_lines_to_multiple_spectra(
        self,
        saved_lines_file: str,
        spectrum_files: List[str],
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None,
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
        defer_plots: bool = True,
        base_output_path: Optional[str] = None
    ) -> Tuple[List[Any], Optional[str]]:
        """
        Fit saved lines to multiple spectrum files with optional parallel processing.
        
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
        parallel : bool, optional
            If True, use parallel processing. If None, uses class default.
        max_workers : int, optional
            Maximum number of parallel workers. If None, uses class default.
        defer_plots : bool, optional
            If True, generate plots after all fitting is complete (default: True)
        base_output_path : str, optional
            Base directory for output. If provided, creates a unique subfolder.
            
        Returns
        -------
        tuple
            (List of FitLinesPlotGrid objects, output_folder path or None)
        """
        from iSLAT.Modules.Plotting.FitLinesPlotGrid import FitLinesPlotGrid
        
        # Use class defaults if not specified
        if parallel is None:
            parallel = self.PARALLEL_BATCH_FITTING
        if max_workers is None:
            max_workers = self.BATCH_FITTING_MAX_WORKERS
        
        # Create unique output folder for this run
        if base_output_path:
            self._current_output_folder = self._create_run_folder(base_output_path, saved_lines_file)
            print(f"\n{'='*60}")
            print(f"Batch Fit: {len(spectrum_files)} spectra")
            print(f"Line list: {os.path.basename(saved_lines_file)}")
            print(f"Output folder: {self._current_output_folder}")
            print(f"{'='*60}")
        else:
            self._current_output_folder = None
        
        if progress_callback:
            progress_callback(f"Fitting saved lines to {len(spectrum_files)} spectra...\n")
        
        # Pre-cache saved lines to avoid repeated file reads
        saved_lines_df = self._get_saved_lines(saved_lines_file)
        if saved_lines_df.empty:
            if progress_callback:
                progress_callback("No saved lines found in file.\n")
            return []
        
        # Collect fit results and spectrum data for deferred plot generation
        fit_results = []
        
        if parallel and len(spectrum_files) > 1:
            # Use parallel processing
            if max_workers is None:
                import os as os_module
                max_workers = max(1, os_module.cpu_count() - 1)
            
            # Initialize thread-safe progress tracker with all spectrum names
            spectrum_names = [os.path.basename(f) for f in spectrum_files]
            self._progress_tracker = ProgressTracker(spectrum_names)
            
            # Process spectra in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all fitting tasks
                future_to_file = {
                    executor.submit(
                        self._fit_single_spectrum,
                        spectrum_file,
                        saved_lines_file,
                        saved_lines_df,
                        config
                    ): spectrum_file
                    for spectrum_file in spectrum_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    spectrum_file = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            fit_results.append(result)
                    except Exception as e:
                        if progress_callback:
                            progress_callback(f"Error: {os.path.basename(spectrum_file)}: {e}\n")
            
            # Clear progress tracker
            self._progress_tracker = None
        else:
            # Sequential processing
            for spectrum_file in spectrum_files:
                try:
                    result = self._fit_single_spectrum(
                        spectrum_file,
                        saved_lines_file,
                        saved_lines_df,
                        config
                    )
                    if result:
                        fit_results.append(result)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Error: {os.path.basename(spectrum_file)}: {e}\n")
        
        # Generate plots (can be parallelized or deferred)
        plot_grid_list = []
        
        for result in fit_results:
            try:
                plot_grid = FitLinesPlotGrid(
                    fit_data=result['fit_data'],
                    wave_data=result['wavedata'],
                    flux_data=result['fluxdata'],
                    err_data=result['err_data'],
                    fit_line_uncertainty=config.get('fit_line_uncertainty', 3.0),
                    spectrum_name=result['spectrum_name']
                )
                plot_grid.generate_plot()
                plot_grid_list.append(plot_grid)
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error generating plot for {result['spectrum_name']}: {e}\n")
        
        if progress_callback:
            progress_callback(f"Completed: {len(fit_results)}/{len(spectrum_files)} spectra fitted.\n")
        
        # Print summary
        print(f"\nFitting complete: {len(fit_results)}/{len(spectrum_files)} spectra processed")
        
        # Store output folder for return
        output_folder = self._current_output_folder
        
        # Clear cache after batch processing
        self.clear_cache()
        
        return plot_grid_list, output_folder
    
    def _fit_single_spectrum(
        self,
        spectrum_file: str,
        saved_lines_file: str,
        saved_lines_df: pd.DataFrame,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fit saved lines to a single spectrum file.
        
        This is a worker method designed for parallel execution.
        
        Parameters
        ----------
        spectrum_file : str
            Path to the spectrum file
        saved_lines_file : str
            Path to the saved lines CSV file
        saved_lines_df : pd.DataFrame
            Pre-loaded saved lines dataframe (for caching)
        config : dict
            Configuration dictionary
        progress_callback : callable, optional
            Callback for progress updates
        
        Returns
        -------
        dict or None
            Dictionary containing fit results and spectrum data for plot generation
        """
        try:
            # Get stellar RV for this spectrum
            save_info = self.islat.get_mole_save_data(os.path.basename(spectrum_file))
            stellar_rv = list(save_info.values())[0].get('StellarRV', 0.0) if save_info else 0.0
            stellar_rv = float(stellar_rv)
            
            # Load the spectrum data
            spectrum_df = ifh.read_spectral_data(spectrum_file)
            wavedata = spectrum_df['wave'].to_numpy()  # More efficient than np.array(df['col'].values)
            wavedata = wavedata - (wavedata / c.SPEED_OF_LIGHT_KMS * stellar_rv)
            fluxdata = spectrum_df['flux'].to_numpy()
            err_data = spectrum_df['err'].to_numpy()
            
            spectrum_name = os.path.basename(spectrum_file)
            
            # Create progress callback based on whether parallel tracker is active
            if self._progress_tracker:
                # Use milestone-based progress for parallel execution
                def line_progress(current, total):
                    self._progress_tracker.update(spectrum_name, current, total)
            else:
                # Use simple carriage-return progress for sequential execution
                def line_progress(current, total):
                    percent = int((current / total) * 100)
                    bar_length = 20
                    filled = int(bar_length * current / total)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    sys.stdout.write(f"\r  {spectrum_name}: [{bar}] {current}/{total} lines ({percent}%)")
                    sys.stdout.flush()
                    if current == total:
                        print()  # New line when complete
            
            # Fit the saved lines to the loaded spectrum
            fit_data = self.fit_lines_to_spectrum(
                saved_lines_file=saved_lines_file,
                spectrum_name=spectrum_name,
                wavedata=wavedata,
                fluxdata=fluxdata,
                err_data=err_data,
                progress_callback=None,
                line_progress_callback=line_progress
            )
            
            if fit_data:
                return {
                    'spectrum_name': spectrum_name,
                    'fit_data': fit_data,
                    'wavedata': wavedata,
                    'fluxdata': fluxdata,
                    'err_data': err_data
                }
            return None
            
        except Exception as e:
            print(f"Error fitting spectrum {spectrum_file}: {e}")
            return None
    
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
                
                # Always print to console when PDF is saved
                print(f"Saved fit plot grid to: {pdf_path}")
                
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