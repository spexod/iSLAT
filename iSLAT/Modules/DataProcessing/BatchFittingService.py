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
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable

import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import iSLAT.Constants as c
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer

def batch_extract_component_parameters(params, prefix='', rest_wavelength=None, sig_det_lim=2):
        """
        Extract parameters for a single Gaussian component from fit results.
        
        Parameters
        ----------
        params : lmfit.Parameters
            Fitted parameters object
        prefix : str, optional
            Parameter prefix (e.g., 'g1_' for multi-component fits, '' for single component)
        rest_wavelength : float, optional
            Rest wavelength for Doppler shift calculation
        sig_det_lim : float, optional
            Detection significance limit (default: 2)
            
        Returns
        -------
        dict
            Dictionary containing extracted component parameters
        """
        center = params[f'{prefix}center'].value
        center_err = params[f'{prefix}center'].stderr if params[f'{prefix}center'].stderr else 0.0
        amplitude = params[f'{prefix}amplitude'].value
        amplitude_err = params[f'{prefix}amplitude'].stderr if params[f'{prefix}amplitude'].stderr else 0.0
        height = params[f'{prefix}height'].value if f'{prefix}height' in params else amplitude
        height_err = params[f'{prefix}height'].stderr if f'{prefix}height' in params and params[f'{prefix}height'].stderr else amplitude_err
        sigma = params[f'{prefix}sigma'].value
        sigma_freq = c.SPEED_OF_LIGHT_MICRONS / (center**2) * sigma
        sigma_freq_err = c.SPEED_OF_LIGHT_MICRONS / (center**2) * params[f'{prefix}sigma'].stderr if params[f'{prefix}sigma'].stderr else 0.0
        
        fwhm = params[f'{prefix}fwhm'].value / center * c.SPEED_OF_LIGHT_KMS if f'{prefix}fwhm' in params else 2.355 * sigma
        fwhm_err = params[f'{prefix}fwhm'].stderr / center * c.SPEED_OF_LIGHT_KMS if f'{prefix}fwhm' in params and params[f'{prefix}fwhm'].stderr else 0.0
        
        gauss_area = height * sigma_freq * np.sqrt(2 * np.pi) * (1.e-23)  # to get line flux in erg/s/cm2
        if height_err is not None and height != 0 and sigma_freq != 0:
            gauss_area_err = np.absolute(gauss_area * np.sqrt(
                (height_err / height) ** 2 +
                (sigma_freq_err / sigma_freq) ** 2))  # get area error
        else:
            gauss_area_err = np.nan
        
        # Calculate fit signal-to-noise and detection for component
        fit_det = abs(gauss_area) > sig_det_lim * gauss_area_err if not np.isnan(gauss_area_err) else False
        
        # Calculate Doppler shift if rest wavelength provided
        doppler = ((center - rest_wavelength) / rest_wavelength * c.SPEED_OF_LIGHT_KMS) if rest_wavelength else np.nan
        
        component_params = {
            'center': center,
            'center_stderr': center_err,
            'amplitude': amplitude,
            'amplitude_stderr': amplitude_err,
            'sigma': sigma,
            'sigma_freq': sigma_freq,
            'sigma_freq_stderr': sigma_freq_err,
            'fwhm': fwhm,
            'fwhm_stderr': fwhm_err,
            'area': gauss_area,
            'area_stderr': gauss_area_err,
            'fit_detected': fit_det,
            'doppler_shift': doppler
        }
        
        return component_params

def _fit_task_worker(task_data: Dict[str, Any]) -> Tuple[int, int, Dict, Any, Any, Any]:
    """
    Module-level worker function for ProcessPoolExecutor.
    
    ProcessPoolExecutor requires a top-level function (can't pickle instance methods).
    This function recreates the minimal fitting logic needed for each task.
    """
    from lmfit.models import GaussianModel
    import numpy as np
    
    # Unpack task data
    spectrum_idx = task_data['spectrum_idx']
    line_idx = task_data['line_idx']
    wave_data = task_data['wave_data']
    flux_data = task_data['flux_data']
    err_data = task_data['err_data']
    xmin = task_data['xmin']
    xmax = task_data['xmax']
    center_wave = task_data['center_wave']
    line_info = task_data['line_info']
    sig_det_lim = task_data.get('sig_det_lim', 2.0)
    
    # Extract fitting region
    fit_mask = (wave_data >= xmin) & (wave_data <= xmax)
    x_fit = wave_data[fit_mask]
    y_fit = flux_data[fit_mask]
    
    # Perform Gaussian fit (simplified version of FittingEngine logic)
    fit_result = None
    fitted_wave = None
    fitted_flux = None
    
    try:
        model = GaussianModel()
        params = model.guess(y_fit, x=x_fit)
        
        # Apply error weights if available
        weights = None
        if err_data is not None:
            err_fit = err_data[fit_mask] if len(err_data) > len(x_fit) else err_data
            if len(err_fit) == len(y_fit) and len(err_fit) > 0:
                max_err = np.max(err_fit)
                if max_err > 0:
                    err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
                    weights = 1.0 / err_fit_safe
        
        if weights is not None:
            fit_result = model.fit(y_fit, params, x=x_fit, weights=weights, nan_policy='omit')
        else:
            fit_result = model.fit(y_fit, params, x=x_fit, nan_policy='omit')
        
        fitted_wave = x_fit
        fitted_flux = fit_result.eval(x=fitted_wave)
    except Exception:
        pass
    
    # Calculate flux integral
    wavelength_mask = (wave_data >= xmin) & (wave_data <= xmax)
    flux_data_integral = 0.0
    err_data_integral = 0.0
    
    if np.any(wavelength_mask):
        lam_range = wave_data[wavelength_mask]
        flux_range = flux_data[wavelength_mask]
        if len(lam_range) >= 2:
            freq_range = c.SPEED_OF_LIGHT_MICRONS / lam_range[::-1]
            flux_data_integral = -np.trapezoid(flux_range[::-1], x=freq_range[::-1]) * 1e-23
            if err_data is not None:
                err_range = err_data[wavelength_mask]
                err_data_integral = -np.trapezoid(err_range[::-1], x=freq_range[::-1]) * 1e-23
    
    # Format result entry (simplified version)
    result_entry = dict(line_info)
    result_entry['xmin'] = xmin
    result_entry['xmax'] = xmax
    result_entry['Flux_data'] = flux_data_integral
    result_entry['Err_data'] = err_data_integral
    
    line_sn = np.round(flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0, decimals=1)
    result_entry['Line_SN'] = line_sn

    if np.absolute(flux_data_integral) > sig_det_lim * err_data_integral:
        line_det = True
    else:
        line_det = False

    result_entry['Line_Det'] = line_det
    result_entry['Flux_islat'] = flux_data_integral # Default to data values
    result_entry['Err_islat'] = err_data_integral   # Will be overwritten if fit succeeds

    # Process fit results if successful
    if fit_result and fit_result.success:
        # Extract fit parameters using the helper method
        component_params = batch_extract_component_parameters(fit_result.params, '', line_info.get('lam'), sig_det_lim)
        
        # Get individual parameters for backward compatibility
        center = component_params['center']
        center_err = component_params['center_stderr']
        gauss_area = component_params['area']
        gauss_area_err = component_params['area_stderr']
        fwhm = component_params['fwhm']
        fwhm_err = component_params['fwhm_stderr']
        fit_det = component_params['fit_detected']
        doppler = component_params['doppler_shift']

        fit_sn = gauss_area / gauss_area_err if gauss_area_err > 0 else 0.0
        
        # Update result with fit information
        result_entry.update({
            'Fit_SN': np.round(fit_sn, decimals=1),
            'Fit_det': bool(fit_det),
            'Flux_fit': np.float64(f'{gauss_area:.{3}e}'),
            'Err_fit': np.float64(f'{gauss_area_err:.{3}e}'),
            'FWHM_fit': np.round(fwhm, decimals=5) if fit_det else np.nan,
            'FWHM_err': np.round(fwhm_err, decimals=5) if fit_det else np.nan,
            'Centr_fit': np.round(center, decimals=5) if fit_det else np.nan,
            'Centr_err': np.round(center_err, decimals=5) if fit_det else np.nan,
            'Doppler': np.round(doppler, decimals=1) if fit_det else np.nan,
            'Red-chisq': np.float64(f'{fit_result.redchi:.{3}e}' if fit_result.redchi is not None else np.nan),
            'Fit_success': bool(True)
        })
        
        # Update iSLAT flux values if fit is good and detected
        if fit_det:
            result_entry['Flux_islat'] = np.float64(f'{gauss_area:.{3}e}')
            result_entry['Err_islat'] = np.float64(f'{gauss_area_err:.{3}e}')
    else:
        # Fit failed - fill with NaN values but keep data measurements
        result_entry.update({
            #'Fit_SN': np.round(flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0, decimals=1),
            'Fit_SN': 0.0,
            'Fit_det': False,
            'Flux_fit': np.nan, #np.float64(f'{flux_data_integral:.{3}e}'),  # Use data flux for failed fit
            'Err_fit': np.nan, #np.float64(f'{err_data_integral:.{3}e}'),
            'FWHM_fit': np.nan,
            'FWHM_err': np.nan,
            'Centr_fit': np.nan,
            'Centr_err': np.nan,
            'Doppler': np.nan,
            'Red-chisq': np.nan,
            'Fit_success': bool(False)
        })

    '''if fit_result is not None and hasattr(fit_result, 'params'):
        result_entry['Centr_fit'] = fit_result.params.get('center', center_wave)
        result_entry['Flux_fit'] = fit_result.params.get('amplitude', 0.0) * fit_result.params.get('sigma', 1.0) * np.sqrt(2 * np.pi)
        result_entry['FWHM_fit'] = fit_result.params.get('sigma', 0.0) * 2.35482
        result_entry['Fit_det'] = True
    else:
        result_entry['Centr_fit'] = center_wave
        result_entry['Flux_fit'] = 0.0
        result_entry['FWHM_fit'] = 0.0
        result_entry['Fit_det'] = False'''
    
    '''# Determine detection based on SNR
    if err_data_integral != 0:
        snr = abs(flux_data_integral / err_data_integral)
        result_entry['SNR'] = snr
        result_entry['Det'] = snr >= sig_det_lim
    else:
        result_entry['SNR'] = 0.0
        result_entry['Det'] = False'''

    return (spectrum_idx, line_idx, result_entry, fit_result, fitted_wave, fitted_flux)

@dataclass
class FittingTask:
    """Represents a single line fitting task."""
    spectrum_name: str
    spectrum_idx: int
    line_idx: int
    line_row: pd.Series
    wave_data: np.ndarray
    flux_data: np.ndarray
    err_data: np.ndarray
    xmin: float
    xmax: float
    center_wave: float
    line_info: Dict[str, Any]


class FittingWorkQueue:
    """
    Manages a flattened work queue of all (spectrum, line) fitting tasks.
    
    Distributes all fitting tasks across a single thread pool for optimal
    CPU utilization without nested parallelism or thread oversubscription.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the work queue.
        
        Parameters
        ----------
        max_workers : int, optional
            Maximum number of worker threads. If None, uses CPU count - 1.
        """
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._progress = {}  # spectrum_name -> (completed, total)
        self._lines_printed = 0
        self._update_interval = 5  # Update display every N tasks completed
        self._tasks_since_update = 0
    
    def initialize_progress(self, spectrum_line_counts: Dict[str, int]):
        """Initialize progress tracking for all spectra."""
        with self._lock:
            self._progress = {name: (0, count) for name, count in spectrum_line_counts.items()}
            self._lines_printed = 0
            self._tasks_since_update = 0
    
    def update_progress(self, spectrum_name: str):
        """Increment progress for a spectrum and redraw if needed."""
        with self._lock:
            curr, total = self._progress.get(spectrum_name, (0, 1))
            self._progress[spectrum_name] = (curr + 1, total)
            self._tasks_since_update += 1
            
            # Only redraw periodically to reduce overhead
            if self._tasks_since_update < self._update_interval:
                # Check if any spectrum just completed
                if curr + 1 < total:
                    return
            
            self._tasks_since_update = 0
            self._redraw_progress()
    
    def _redraw_progress(self):
        """Redraw all progress bars (called with lock held)."""
        # Move cursor up to overwrite previous output
        if self._lines_printed > 0:
            sys.stdout.write(f"\033[{self._lines_printed}A")
        
        lines = []
        for name, (curr, total) in self._progress.items():
            percent = int((curr / total) * 100) if total > 0 else 0
            bar_length = 20
            filled = int(bar_length * curr / total) if total > 0 else 0
            bar = '█' * filled + '░' * (bar_length - filled)
            line = f"  {name}: [{bar}] {curr}/{total} ({percent}%)"
            lines.append(line.ljust(80))
        
        output = '\n'.join(lines)
        sys.stdout.write(output + '\n')
        sys.stdout.flush()
        self._lines_printed = len(lines)
    
    def finalize_progress(self):
        """Ensure final progress state is displayed."""
        with self._lock:
            self._redraw_progress()


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
    USE_PROCESS_POOL: bool = True  # True = ProcessPoolExecutor (true parallelism), False = ThreadPoolExecutor (shared memory)
    
    def __init__(self):
        """
        Initialize the batch fitting service.
        """
        self.line_analyzer = LineAnalyzer()
        self.fitting_engine = FittingEngine()
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
    
    def _fit_single_task(
        self,
        task: FittingTask,
        sig_det_lim: float = 2.0
    ) -> Tuple[int, int, Dict[str, Any], Any, np.ndarray, np.ndarray]:
        """
        Fit a single line task - worker method for flattened parallel execution.
        
        Parameters
        ----------
        task : FittingTask
            The fitting task containing all necessary data
        sig_det_lim : float
            Detection limit for signal-to-noise ratio
            
        Returns
        -------
        tuple
            (spectrum_idx, line_idx, result_entry, fit_result, fitted_wave, fitted_flux)
        """
        fit_mask = (task.wave_data >= task.xmin) & (task.wave_data <= task.xmax)
        x_fit = task.wave_data[fit_mask]
        y_fit = task.flux_data[fit_mask]
        
        fit_result, fitted_wave, fitted_flux = self.fitting_engine.fit_gaussian_line(
            wave_data=x_fit,
            flux_data=y_fit,
            xmin=task.xmin,
            xmax=task.xmax,
            initial_guess=None,
            deblend=False,
            err_data=task.err_data
        )
        
        flux_data_integral, err_data_integral = self.line_analyzer.flux_integral(
            task.wave_data, task.flux_data, err=task.err_data, 
            lam_min=task.xmin, lam_max=task.xmax
        )
        
        result_entry = self.fitting_engine.format_fit_results_for_csv(
            fit_result, flux_data_integral, err_data_integral,
            task.xmin, task.xmax, task.center_wave, task.line_info, sig_det_lim
        )
        
        return (task.spectrum_idx, task.line_idx, result_entry, fit_result, fitted_wave, fitted_flux)
    
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
        line_progress_callback: Optional[Callable[[int, int], None]] = None,
        parallel_lines: Optional[bool] = None
    ) -> Tuple[Optional[List[Dict]], Optional[Tuple]]:
        """
        Fit saved lines to a spectrum.
        
        Parameters
        ----------
        saved_lines_file : str
            Path to the saved lines CSV file
        spectrum_name : str, optional
            Name of the spectrum being fitted
        wavedata : np.ndarray
            Wavelength data
        fluxdata : np.ndarray
            Flux data
        err_data : np.ndarray
            Error data
        output_file : str, optional
            Output file name for fit results
        progress_callback : callable, optional
            Callback function for progress updates: callback(message: str)
        line_progress_callback : callable, optional
            Callback for per-line progress: callback(current_line, total_lines)
        parallel_lines : bool, optional
            If True, parallelize line fitting within this spectrum.
            If None, uses LineAnalyzer class default.
            
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
                output_file = "fit_results.csv"
        
        if spectrum_name is None:
            spectrum_name = 'unknown'
        
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
            output_path=self._current_output_folder,
            parallel=parallel_lines
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
        base_output_path: Optional[str] = None,
        get_mole_save_data: Optional[Callable] = None
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
        get_mole_save_data : callable, optional
            Function that accepts a spectrum filename and returns molecule save data dict.
            
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
            return [], None
        
        # Use the flattened work queue approach for optimal parallelism
        if parallel:
            fit_results = self._fit_multiple_spectra_flattened(
                spectrum_files, saved_lines_file, saved_lines_df, config, max_workers, progress_callback,
                get_mole_save_data=get_mole_save_data
            )
        else:
            # Sequential processing
            fit_results = []
            for spectrum_file in spectrum_files:
                try:
                    result = self._fit_single_spectrum(
                        spectrum_file,
                        saved_lines_file,
                        saved_lines_df,
                        config,
                        get_mole_save_data=get_mole_save_data
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
    
    def _fit_multiple_spectra_flattened(
        self,
        spectrum_files: List[str],
        saved_lines_file: str,
        saved_lines_df: pd.DataFrame,
        config: Dict[str, Any],
        max_workers: Optional[int],
        progress_callback: Optional[Callable[[str], None]] = None,
        get_mole_save_data: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Fit multiple spectra using a flattened work queue for optimal CPU utilization.
        
        Instead of parallelizing at either spectrum or line level, this method
        creates a single queue of all (spectrum, line) fitting tasks and distributes
        them across all available workers.
        
        Parameters
        ----------
        spectrum_files : list of str
            List of spectrum file paths
        saved_lines_file : str
            Path to saved lines file
        saved_lines_df : pd.DataFrame
            Pre-loaded saved lines DataFrame
        config : dict
            Configuration dictionary
        max_workers : int, optional
            Maximum workers for thread pool
        progress_callback : callable, optional
            Callback for progress updates
            
        Returns
        -------
        list of dict
            List of result dictionaries for each spectrum
        """
        # Set up worker count
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
        
        total_lines = len(saved_lines_df)
        sig_det_lim = 2.0
        
        # Phase 1: Load all spectrum data (I/O bound, done sequentially)
        spectrum_data = []
        spectrum_names = []
        
        for spectrum_idx, spectrum_file in enumerate(spectrum_files):
            try:
                stellar_rv = 0.0
                if get_mole_save_data is not None:
                    save_info = get_mole_save_data(os.path.basename(spectrum_file))
                    stellar_rv = list(save_info.values())[0].get('StellarRV', 0.0) if save_info else 0.0
                    stellar_rv = float(stellar_rv)
                
                spectrum_df = ifh.read_spectral_data(spectrum_file)
                wavedata = spectrum_df['wave'].to_numpy()
                wavedata = wavedata - (wavedata / c.SPEED_OF_LIGHT_KMS * stellar_rv)
                fluxdata = spectrum_df['flux'].to_numpy()
                err_data = spectrum_df['err'].to_numpy()
                
                spectrum_name = os.path.basename(spectrum_file)
                spectrum_names.append(spectrum_name)
                
                spectrum_data.append({
                    'spectrum_idx': spectrum_idx,
                    'spectrum_name': spectrum_name,
                    'spectrum_file': spectrum_file,
                    'wavedata': wavedata,
                    'fluxdata': fluxdata,
                    'err_data': err_data
                })
            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error loading {os.path.basename(spectrum_file)}: {e}\n")
        
        if not spectrum_data:
            return []
        
        # Phase 2: Create all fitting tasks (flattened across all spectra and lines)
        all_tasks = []
        
        for spec_data in spectrum_data:
            for line_idx, line_row in saved_lines_df.iterrows():
                center_wave = float(line_row['lam']) if 'lam' in line_row else 0.0
                xmin = float(line_row['xmin']) if 'xmin' in line_row and not pd.isna(line_row['xmin']) else center_wave - 0.01
                xmax = float(line_row['xmax']) if 'xmax' in line_row and not pd.isna(line_row['xmax']) else center_wave + 0.01
                
                line_info = {
                    'species': line_row.get('species', 'Unknown'),
                    'lev_up': line_row.get('lev_up', ''),
                    'lev_low': line_row.get('lev_low', ''),
                    'lam': center_wave,
                    'a_stein': float(line_row.get('a_stein', 0.0)) if not pd.isna(line_row.get('a_stein', 0.0)) else 0.0,
                    'e_up': float(line_row.get('e_up', 0.0)) if not pd.isna(line_row.get('e_up', 0.0)) else 0.0,
                    'e_low': float(line_row.get('e_low', 0.0)) if not pd.isna(line_row.get('e_low', 0.0)) else 0.0,
                    'g_up': float(line_row.get('g_up', 1.0)) if not pd.isna(line_row.get('g_up', 1.0)) else 1.0,
                    'g_low': float(line_row.get('g_low', 1.0)) if not pd.isna(line_row.get('g_low', 1.0)) else 1.0,
                }
                line_info.update({key: line_row[key] for key in line_row.index if key not in line_info})
                
                task = FittingTask(
                    spectrum_name=spec_data['spectrum_name'],
                    spectrum_idx=spec_data['spectrum_idx'],
                    line_idx=line_idx,
                    line_row=line_row,
                    wave_data=spec_data['wavedata'],
                    flux_data=spec_data['fluxdata'],
                    err_data=spec_data['err_data'],
                    xmin=xmin,
                    xmax=xmax,
                    center_wave=center_wave,
                    line_info=line_info
                )
                all_tasks.append(task)
        
        total_tasks = len(all_tasks)
        
        # Initialize work queue with progress tracking
        work_queue = FittingWorkQueue(max_workers=max_workers)
        work_queue.initialize_progress({name: total_lines for name in spectrum_names})
        
        # Pre-allocate result storage: results[spectrum_idx][line_idx] = result
        results_by_spectrum = {
            spec_data['spectrum_idx']: {
                'spectrum_name': spec_data['spectrum_name'],
                'wavedata': spec_data['wavedata'],
                'fluxdata': spec_data['fluxdata'],
                'err_data': spec_data['err_data'],
                'fit_results_csv': [None] * total_lines,
                'fit_results_data': [None] * total_lines,
                'fitted_waves': [None] * total_lines,
                'fitted_fluxes': [None] * total_lines
            }
            for spec_data in spectrum_data
        }
        
        # Phase 3: Execute all tasks in parallel
        # Choose executor based on class setting
        if self.USE_PROCESS_POOL:
            # ProcessPoolExecutor for true parallelism (bypasses GIL)
            # Convert tasks to picklable dictionaries
            task_dicts = []
            for task in all_tasks:
                task_dicts.append({
                    'spectrum_idx': task.spectrum_idx,
                    'line_idx': task.line_idx,
                    'wave_data': task.wave_data,
                    'flux_data': task.flux_data,
                    'err_data': task.err_data,
                    'xmin': task.xmin,
                    'xmax': task.xmax,
                    'center_wave': task.center_wave,
                    'line_info': task.line_info,
                    'sig_det_lim': sig_det_lim
                })
            
            ExecutorClass = ProcessPoolExecutor
            task_list = task_dicts
            worker_func = _fit_task_worker
            # Map task_dict back to FittingTask for progress tracking
            task_lookup = {id(td): all_tasks[i] for i, td in enumerate(task_dicts)}
        else:
            # ThreadPoolExecutor for shared memory (lower overhead)
            ExecutorClass = ThreadPoolExecutor
            task_list = all_tasks
            worker_func = lambda t: self._fit_single_task(t, sig_det_lim)
            task_lookup = None
        
        with ExecutorClass(max_workers=max_workers) as executor:
            if self.USE_PROCESS_POOL:
                futures = {executor.submit(worker_func, task): task for task in task_list}
            else:
                futures = {executor.submit(self._fit_single_task, task, sig_det_lim): task for task in task_list}
            
            for future in as_completed(futures):
                submitted_task = futures[future]
                # Get the original FittingTask for progress tracking
                if self.USE_PROCESS_POOL:
                    # Find matching task by index
                    idx = task_dicts.index(submitted_task)
                    task = all_tasks[idx]
                else:
                    task = submitted_task
                
                try:
                    spec_idx, line_idx, result_entry, fit_result, fitted_wave, fitted_flux = future.result()
                    
                    # Store result in appropriate spectrum bucket
                    spec_results = results_by_spectrum[spec_idx]
                    spec_results['fit_results_csv'][line_idx] = result_entry
                    spec_results['fit_results_data'][line_idx] = fit_result
                    spec_results['fitted_waves'][line_idx] = fitted_wave
                    spec_results['fitted_fluxes'][line_idx] = fitted_flux
                    
                    # Update progress
                    work_queue.update_progress(task.spectrum_name)
                    
                except Exception as e:
                    # Continue processing other tasks
                    pass
        
        # Ensure final progress is displayed
        work_queue.finalize_progress()
        
        # Phase 4: Aggregate and save results for each spectrum
        fit_results = []
        
        for spec_idx, spec_results in results_by_spectrum.items():
            # Filter out None entries
            fit_results_csv = [r for r in spec_results['fit_results_csv'] if r is not None]
            
            # Add rotation diagram values
            if fit_results_csv and any(entry.get('a_stein', 0) > 0 for entry in fit_results_csv):
                self.line_analyzer.add_rotation_diagram_values(fit_results_csv)
            
            # Save results
            spectrum_name = spec_results['spectrum_name']
            spectrum_base_name = os.path.splitext(spectrum_name)[0]
            output_file = f"{spectrum_base_name}-{os.path.basename(saved_lines_file)}"
            
            if self._current_output_folder and fit_results_csv:
                ifh.save_fit_results(fit_results_csv, file_path=self._current_output_folder, file_name=output_file)
            
            # Package for plot generation
            fit_results.append({
                'spectrum_name': spectrum_name,
                'fit_data': (fit_results_csv, (
                    spec_results['fit_results_data'],
                    spec_results['fitted_waves'],
                    spec_results['fitted_fluxes']
                )),
                'wavedata': spec_results['wavedata'],
                'fluxdata': spec_results['fluxdata'],
                'err_data': spec_results['err_data']
            })
        
        return fit_results

    def _fit_single_spectrum(
        self,
        spectrum_file: str,
        saved_lines_file: str,
        saved_lines_df: pd.DataFrame,
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[str], None]] = None,
        get_mole_save_data: Optional[Callable] = None
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
        get_mole_save_data : callable, optional
            Function that accepts a spectrum filename and returns molecule save data dict.
        
        Returns
        -------
        dict or None
            Dictionary containing fit results and spectrum data for plot generation
        """
        try:
            # Get stellar RV for this spectrum
            stellar_rv = 0.0
            if get_mole_save_data is not None:
                save_info = get_mole_save_data(os.path.basename(spectrum_file))
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
            # Disable line-level parallelism when we're already running spectra in parallel
            # to avoid nested parallelism and thread oversubscription
            parallel_lines = not bool(self._progress_tracker)
            
            fit_data = self.fit_lines_to_spectrum(
                saved_lines_file=saved_lines_file,
                spectrum_name=spectrum_name,
                wavedata=wavedata,
                fluxdata=fluxdata,
                err_data=err_data,
                progress_callback=None,
                line_progress_callback=line_progress,
                parallel_lines=parallel_lines
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