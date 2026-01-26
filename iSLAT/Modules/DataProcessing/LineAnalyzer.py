"""
LineAnalyzer - Line detection, identification, and analysis functionality
"""

import numpy as np
from scipy.signal import find_peaks
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import pandas as pd
import iSLAT.Constants as c
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable, Tuple, List, Dict, Any

class LineAnalyzer:
    """
    Centralized line analysis engine for automatic line detection and characterization.
    
    This class provides comprehensive line analysis capabilities including
    detection, identification, measurement, and statistical analysis.
    """
    
    # Class-level settings for parallel line fitting
    PARALLEL_LINE_FITTING: bool = True
    LINE_FITTING_MAX_WORKERS: int = None  # None uses CPU count - 1
    
    def __init__(self, islat_instance):
        """
        Initialize the line analyzer.
        
        Parameters
        ----------
        islat_instance : iSLAT
            Reference to the main iSLAT instance for accessing data and configuration
        """
        self.islat = islat_instance
        self.detected_lines = []
        self.line_measurements = {}
        
        # Detection parameters
        self.min_snr = 0.1
        self.min_line_width = 0.001  # microns
        self.max_line_width = 0.1    # microns
        self.continuum_window = 0.05  # microns for local continuum estimation
    
    def find_single_lines(self, wave_data, flux_data, specsep=0.01, line_threshold=0.1, isolation_threshold=0.1):
        """
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        specsep : float, optional
            Spectral separation distance to check for neighboring lines (default: 0.01 microns)
        line_threshold : float, optional
            Minimum line intensity as fraction of maximum intensity (default: 0.1)
        isolation_threshold : float, optional
            Local threshold for determining if a line is isolated (default: 0.1)
            
        Returns
        -------
        detected_lines : list
            List of dictionaries containing isolated line information
        """
        #continuum, noise_std = self.estimate_continuum_and_noise(wave_data, flux_data)
        
        detected_lines = []
        
        max_intensity = np.max(flux_data)

        # Detect both emission and absorption lines in one pass
        # For emission lines: look for peaks above continuum
        excess_flux = flux_data - specsep
        #emission_threshold = self.min_snr * noise_std
        emission_threshold = max_intensity * line_threshold
        
        emission_peaks, _ = find_peaks(
            excess_flux,
            height=emission_threshold,
            width=1,
            distance=3
        )
        
        # For absorption lines: look for peaks below continuum
        deficit_flux = flux_data + specsep
        absorption_peaks, _ = find_peaks(
            deficit_flux,
            height=emission_threshold,
            width=1,
            distance=3
        )
        
        # Combine all detected peaks
        all_peaks = []
        
        # Add emission lines
        for peak_idx in emission_peaks:
            line_strength = excess_flux[peak_idx]
            all_peaks.append({
                'peak_idx': peak_idx,
                'wavelength': wave_data[peak_idx],
                'line_strength': line_strength,
                'line_type': 'emission'
            })
        
        # Add absorption lines
        for peak_idx in absorption_peaks:
            line_strength = deficit_flux[peak_idx]
            all_peaks.append({
                'peak_idx': peak_idx,
                'wavelength': wave_data[peak_idx],
                'line_strength': line_strength,
                'line_type': 'absorption'
            })
        
        # Sort by wavelength
        all_peaks.sort(key=lambda x: x['wavelength'])
        
        # Apply single_finder logic for isolation filtering
        if not all_peaks:
            self.detected_lines = detected_lines
            return detected_lines
        
        # Find maximum line strength for global threshold
        max_strength = max(peak['line_strength'] for peak in all_peaks)
        max_threshold = max_strength * line_threshold
        
        # Filter for isolated lines using single_finder logic
        for peak in all_peaks:
            include = True  # Boolean for determining if line is isolated
            peak_wavelength = peak['wavelength']
            peak_strength = peak['line_strength']
            
            # Only consider lines above the global threshold
            if peak_strength >= max_threshold:
                # Define local search range
                sub_xmin = peak_wavelength - specsep
                sub_xmax = peak_wavelength + specsep
                local_threshold = peak_strength * isolation_threshold
                
                # Check all other peaks in the local range
                for other_peak in all_peaks:
                    other_wavelength = other_peak['wavelength']
                    other_strength = other_peak['line_strength']
                    
                    # Skip the peak itself
                    if other_wavelength == peak_wavelength:
                        continue
                    
                    # Check if other peak is within the local range
                    if sub_xmin < other_wavelength < sub_xmax:
                        # If nearby line is strong enough, this line is not isolated
                        if other_strength >= local_threshold:
                            include = False
                            break
                
                # If line passes isolation test, characterize and add it
                if include:
                    line_info = self._characterize_line(
                        #wave_data, flux_data, continuum, peak['peak_idx'],
                        wave_data, flux_data, peak['peak_idx'],
                    )
                    
                    if line_info is not None:
                        # Add isolation parameters to line info
                        line_info['is_isolated'] = True
                        line_info['isolation_checked'] = True
                        line_info['specsep_used'] = specsep
                        line_info['line_threshold_used'] = line_threshold
                        line_info['isolation_threshold_used'] = isolation_threshold
                        detected_lines.append(line_info)
        
        # Sort final results by wavelength
        detected_lines.sort(key=lambda x: x['wavelength'])
        
        self.detected_lines = detected_lines
        return detected_lines
    
    def _characterize_line(self, wave_data, flux_data, peak_idx):
        """
        Characterize a detected line.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        continuum : array_like
            Continuum estimate
        peak_idx : int
            Index of the line peak
        line_type : str
            'emission' or 'absorption'
            
        Returns
        -------
        line_info : dict or None
            Dictionary with line characteristics
        """
        try:
            wavelength = wave_data[peak_idx]
            peak_flux = flux_data[peak_idx]
            
            # Create line information dictionary
            line_info = {
                'wavelength': wavelength,
                'peak_flux': peak_flux,
                'peak_index': peak_idx
            }
            
            return line_info
            
        except Exception as e:
            print(f"Error characterizing line at index {peak_idx}: {str(e)}")
            return None

    def add_rotation_diagram_values(self, fit_results_data):
        """
        Add rotation diagram values to fit results data.
        
        Parameters
        ----------
        fit_results_data : list of dict
            List of fit result dictionaries to update
        """
        for entry in fit_results_data:
            try:
                if (entry.get('a_stein', 0) > 0 and 
                    entry.get('g_up', 0) > 0 and 
                    entry.get('Flux_fit', 0) != 0):
                    
                    freq = c.SPEED_OF_LIGHT_MICRONS / entry['lam']
                    flux_fit = float(entry['Flux_fit'])

                    rd_y = np.log(4 * np.pi * flux_fit / (entry['a_stein'] * c.PLANCK_CONSTANT * freq * entry['g_up']))
                    entry['RD_y'] = np.round(rd_y, decimals=3)
                else:
                    entry['RD_y'] = np.nan
            except (ValueError, TypeError, ZeroDivisionError):
                entry['RD_y'] = np.nan

    @staticmethod
    def flux_integral(lam, flux, err, lam_min, lam_max) -> Tuple[float, float]:
        wavelength_mask = (lam >= lam_min) & (lam <= lam_max)

        if not np.any (wavelength_mask):
            return 0.0, 0.0

        lam_range = lam[wavelength_mask]
        flux_range = flux[wavelength_mask]

        if len (lam_range) < 2:
            return 0.0, 0.0

        # Convert to frequency space for proper integration
        freq_range = c.SPEED_OF_LIGHT_MICRONS / lam_range[::-1]

        # Integrate in frequency space (reverse order for proper frequency ordering)
        line_flux_meas = np.trapezoid(flux_range[::-1], x=freq_range[::-1])
        line_flux_meas = -line_flux_meas * 1e-23  # Convert Jy*Hz to erg/s/cm^2

        # Calculate error propagation if error data provided
        if err is not None:
            err_range = err[wavelength_mask]
            line_err_meas = np.trapezoid(err_range[::-1], x=freq_range[::-1])
            line_err_meas = -line_err_meas * 1e-23
        else:
            line_err_meas = 0.0

        return line_flux_meas, line_err_meas

    def _fit_single_line(
        self,
        line_index: int,
        line_row: pd.Series,
        fitting_engine,
        calc_wave_data: np.ndarray,
        calc_flux_data: np.ndarray,
        err_data: np.ndarray,
        sig_det_lim: float
    ) -> Tuple[int, Dict[str, Any], Any, np.ndarray, np.ndarray]:
        """
        Fit a single line - worker method for parallel execution.
        
        Parameters
        ----------
        line_index : int
            Index of the line in the saved lines DataFrame
        line_row : pd.Series
            Row from saved lines DataFrame
        fitting_engine : FittingEngine
            Fitting engine instance
        calc_wave_data : np.ndarray
            Wavelength data array
        calc_flux_data : np.ndarray
            Flux data array
        err_data : np.ndarray
            Error data array
        sig_det_lim : float
            Detection limit for signal-to-noise ratio
            
        Returns
        -------
        tuple
            (line_index, result_entry, fit_result, fitted_wave, fitted_flux)
        """
        # Extract line information from saved file
        center_wave = float(line_row['lam']) if 'lam' in line_row else 0.0
        xmin = float(line_row['xmin']) if 'xmin' in line_row and not pd.isna(line_row['xmin']) else center_wave - 0.01
        xmax = float(line_row['xmax']) if 'xmax' in line_row and not pd.isna(line_row['xmax']) else center_wave + 0.01
        
        # Create line info from saved data
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

        # Add any other rows that were present in the saved file
        line_info.update({key: line_row[key] for key in line_row.index if key not in line_info})

        fit_mask = (calc_wave_data >= xmin) & (calc_wave_data <= xmax)
        x_fit = calc_wave_data[fit_mask]
        y_fit = calc_flux_data[fit_mask]

        fit_result, fitted_wave, fitted_flux = fitting_engine.fit_gaussian_line(
            wave_data=x_fit,
            flux_data=y_fit,
            xmin=xmin,
            xmax=xmax,
            initial_guess=None,
            deblend=False,
            err_data=err_data
        )

        flux_data_integral, err_data_integral = self.flux_integral(
            calc_wave_data, calc_flux_data, err=err_data, lam_min=xmin, lam_max=xmax
        )

        # Format results using existing method
        result_entry = fitting_engine.format_fit_results_for_csv(
            fit_result, flux_data_integral, err_data_integral,
            xmin, xmax, center_wave, line_info, sig_det_lim
        )
        
        return (line_index, result_entry, fit_result, fitted_wave, fitted_flux)

    def analyze_saved_lines(
        self,
        saved_lines_file: str,
        fitting_engine,
        output_file: Optional[str] = None,
        wavedata: Optional[np.ndarray] = None,
        fluxdata: Optional[np.ndarray] = None,
        err_data: Optional[np.ndarray] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        output_path: Optional[str] = None,
        parallel: Optional[bool] = None,
        max_workers: Optional[int] = None
    ):
        """
        Comprehensive analysis of saved lines using data already present in the saved lines file.
        Performs fitting and rotation diagram calculations with optional parallel processing.
        
        Parameters
        ----------
        saved_lines_file : str
            Path to saved lines CSV file
        fitting_engine : FittingEngine
            Fitting engine instance to use for formatting results
        output_file : str, optional
            Output file name for results
        output_path : str, optional
            Output directory path for results. If None, uses default.
        progress_callback : callable, optional
            Callback function for progress updates: callback(current, total)
        parallel : bool, optional
            If True, use parallel processing. If None, uses class default.
        max_workers : int, optional
            Maximum number of parallel workers. If None, uses class default.
            
        Returns
        -------
        tuple
            (fit_results_csv_data, (fit_results_data, fitted_waves, fitted_fluxes))
        """
        # Use class defaults if not specified
        if parallel is None:
            parallel = self.PARALLEL_LINE_FITTING
        if max_workers is None:
            max_workers = self.LINE_FITTING_MAX_WORKERS
        
        # Read saved lines
        try:
            saved_lines = pd.read_csv(saved_lines_file)
            if saved_lines.empty:
                return [], ([], [], [])
        except Exception as e:
            print(f"Error reading saved lines file: {e}")
            return [], ([], [], [])
        
        sig_det_lim = 2  # Detection limit for signal-to-noise ratio

        calc_wave_data = self.islat.wave_data if wavedata is None else wavedata
        calc_flux_data = self.islat.flux_data if fluxdata is None else fluxdata
        calc_err_data = self.islat.err_data if err_data is None else err_data
        
        total_lines = len(saved_lines)
        
        # Pre-allocate result lists with None to maintain order
        fit_results_csv_data = [None] * total_lines
        fit_results_data = [None] * total_lines
        fitted_waves = [None] * total_lines
        fitted_fluxes = [None] * total_lines
        
        completed_count = 0
        
        if parallel and total_lines > 1:
            # Parallel processing
            import os as os_module
            if max_workers is None:
                max_workers = max(1, os_module.cpu_count() - 1)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all fitting tasks
                futures = {
                    executor.submit(
                        self._fit_single_line,
                        i,
                        row,
                        fitting_engine,
                        calc_wave_data,
                        calc_flux_data,
                        calc_err_data,
                        sig_det_lim
                    ): i
                    for i, row in saved_lines.iterrows()
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        line_idx, result_entry, fit_result, fitted_wave, fitted_flux = future.result()
                        fit_results_csv_data[line_idx] = result_entry
                        fit_results_data[line_idx] = fit_result
                        fitted_waves[line_idx] = fitted_wave
                        fitted_fluxes[line_idx] = fitted_flux
                        
                        completed_count += 1
                        if progress_callback:
                            progress_callback(completed_count, total_lines)
                    except Exception as e:
                        # Log but continue processing other lines
                        pass
        else:
            # Sequential processing
            for i, line_row in saved_lines.iterrows():
                line_idx, result_entry, fit_result, fitted_wave, fitted_flux = self._fit_single_line(
                    i, line_row, fitting_engine, calc_wave_data, calc_flux_data, calc_err_data, sig_det_lim
                )
                fit_results_csv_data[line_idx] = result_entry
                fit_results_data[line_idx] = fit_result
                fitted_waves[line_idx] = fitted_wave
                fitted_fluxes[line_idx] = fitted_flux
                
                if progress_callback:
                    progress_callback(i + 1, total_lines)
        
        # Filter out any None entries (from failed fits)
        fit_results_csv_data = [r for r in fit_results_csv_data if r is not None]
        
        # Add rotation diagram values if we have successful fits with molecular data
        if fit_results_csv_data and any(entry.get('a_stein', 0) > 0 for entry in fit_results_csv_data):
            self.add_rotation_diagram_values(fit_results_csv_data)
        
        # Save results if output file specified
        if output_file and fit_results_csv_data:
            if output_path:
                ifh.save_fit_results(fit_results_csv_data, file_path=output_path, file_name=output_file)
            else:
                ifh.save_fit_results(fit_results_csv_data, file_name=output_file)

        return fit_results_csv_data, (fit_results_data, fitted_waves, fitted_fluxes)