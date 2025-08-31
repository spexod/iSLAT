"""
LineAnalyzer - Line detection, identification, and analysis functionality
"""

import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import median_filter
import json
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import pandas as pd
import iSLAT.Constants as c

class LineAnalyzer:
    """
    Centralized line analysis engine for automatic line detection and characterization.
    
    This class provides comprehensive line analysis capabilities including
    detection, identification, measurement, and statistical analysis.
    """
    
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
        self.continuum_level = None
        self.noise_level = None
        
        # Load atomic/molecular line databases
        self.atomic_lines = ifh.load_atomic_lines()
        
        # Detection parameters
        self.min_snr = 0.1
        self.min_line_width = 0.001  # microns
        self.max_line_width = 0.1    # microns
        self.continuum_window = 0.05  # microns for local continuum estimation
    
    def set_detection_parameters(self, min_snr=None, min_width=None, max_width=None):
        """
        Set parameters for automatic line detection.
        
        Parameters
        ----------
        min_snr : float, optional
            Minimum signal-to-noise ratio for detection
        min_width : float, optional
            Minimum line width in microns
        max_width : float, optional
            Maximum line width in microns
        """
        if min_snr is not None:
            self.min_snr = min_snr
        if min_width is not None:
            self.min_line_width = min_width
        if max_width is not None:
            self.max_line_width = max_width
    
    def estimate_continuum_and_noise(self, wave_data, flux_data, method='median_filter'):
        """
        Estimate continuum level and noise characteristics.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        method : str, optional
            Method for continuum estimation ('median_filter', 'percentile', 'linear')
            
        Returns
        -------
        continuum : array_like
            Estimated continuum flux
        noise_std : float
            Estimated noise standard deviation
        """
        if method == 'median_filter':
            # Use a wide median filter to estimate continuum
            filter_size = max(5, len(flux_data) // 50)
            continuum = median_filter(flux_data, size=filter_size)
            
        elif method == 'percentile':
            # Use moving percentile to estimate continuum
            window_size = max(10, len(flux_data) // 20)
            continuum = np.zeros_like(flux_data)
            
            for i in range(len(flux_data)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(flux_data), i + window_size // 2)
                continuum[i] = np.percentile(flux_data[start_idx:end_idx], 75)
                
        elif method == 'linear':
            # Simple linear continuum between endpoints
            continuum = np.linspace(flux_data[0], flux_data[-1], len(flux_data))
            
        else:
            raise ValueError(f"Unknown continuum estimation method: {method}")
        
        # Estimate noise from continuum-subtracted residuals
        residuals = flux_data - continuum
        noise_std = np.std(residuals)
        
        self.continuum_level = continuum
        self.noise_level = noise_std
        
        return continuum, noise_std
    
    def detect_lines_automatic(self, wave_data, flux_data, specsep=0.01, line_threshold=0.1, isolation_threshold=0.1):
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
        continuum, noise_std = self.estimate_continuum_and_noise(wave_data, flux_data)
        
        detected_lines = []
        
        # Detect both emission and absorption lines in one pass
        # For emission lines: look for peaks above continuum
        excess_flux = flux_data - continuum
        emission_threshold = self.min_snr * noise_std
        
        emission_peaks, _ = find_peaks(
            excess_flux,
            height=emission_threshold,
            width=1,
            distance=3
        )
        
        # For absorption lines: look for peaks below continuum
        deficit_flux = continuum - flux_data
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
                        wave_data, flux_data, continuum, peak['peak_idx'],
                        line_type=peak['line_type']
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
    
    def _characterize_line(self, wave_data, flux_data, continuum, peak_idx, line_type):
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
            continuum_flux = continuum[peak_idx]
            
            # Calculate line strength
            if line_type == 'emission':
                line_strength = peak_flux - continuum_flux
            else:  # absorption
                line_strength = continuum_flux - peak_flux
            
            # Estimate line width using peak_widths
            if line_type == 'emission':
                signal = flux_data - continuum
            else:
                signal = continuum - flux_data
            
            widths, width_heights, left_ips, right_ips = peak_widths(
                signal, [peak_idx], rel_height=0.5
            )
            
            # Convert width from pixels to wavelength units
            if len(widths) > 0 and widths[0] > 0:
                width_pixels = widths[0]
                # Approximate conversion assuming uniform wavelength spacing
                dwav_dpix = (wave_data[-1] - wave_data[0]) / len(wave_data)
                line_width = width_pixels * dwav_dpix
            else:
                line_width = np.nan
            
            # Check if line width is within acceptable range
            if (np.isnan(line_width) or 
                line_width < self.min_line_width or 
                line_width > self.max_line_width):
                return None
            
            # Calculate signal-to-noise ratio
            snr = line_strength / self.noise_level if self.noise_level > 0 else np.inf
            
            # Create line information dictionary
            line_info = {
                'wavelength': wavelength,
                'peak_flux': peak_flux,
                'continuum_flux': continuum_flux,
                'line_strength': line_strength,
                'line_width': line_width,
                'snr': snr,
                #'type': line_type,
                'peak_index': peak_idx
            }
            
            return line_info
            
        except Exception as e:
            print(f"Error characterizing line at index {peak_idx}: {str(e)}")
            return None
    
    def _search_atomic_lines(self, wavelength, tolerance):
        """Search atomic line database for matches."""
        if self.atomic_lines.empty:
            return []
        
        matches = []
        
        # Check if wavelength column exists (handle different column names)
        wavelength_col = None
        for col in ['wavelength', 'wave', 'lambda', 'wl']:
            if col in self.atomic_lines.columns:
                wavelength_col = col
                break
        
        if wavelength_col is None:
            return []
        
        # Find matches within tolerance
        wave_diff = np.abs(self.atomic_lines[wavelength_col] - wavelength)
        match_mask = wave_diff <= tolerance
        
        if np.any(match_mask):
            matched_lines = self.atomic_lines[match_mask].copy()
            matched_lines['wavelength_diff'] = wave_diff[match_mask]
            matched_lines = matched_lines.sort_values('wavelength_diff')
            
            for _, match in matched_lines.iterrows():
                match_info = {
                    'species': match.get('species', 'Unknown'),
                    'transition': match.get('transition', ''),
                    'wavelength_ref': match[wavelength_col],
                    'wavelength_diff': match['wavelength_diff'],
                    'source': 'atomic',
                    'strength': match.get('strength', np.nan)
                }
                matches.append(match_info)
        
        return matches
    
    def export_line_analysis(self, filename=None):
        """
        Export line analysis results to a file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, generates automatic filename.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"line_analysis_{timestamp}.json"
        
        # Prepare export data
        export_data = {
            'detection_parameters': {
                'min_snr': self.min_snr,
                'min_line_width': self.min_line_width,
                'max_line_width': self.max_line_width,
                'continuum_window': self.continuum_window
            },
            'detected_lines': self.detected_lines,
            'line_measurements': self.line_measurements,
            'noise_level': self.noise_level
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        export_data = convert_numpy(export_data)
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Line analysis results exported to {filename}")
    
    def get_analysis_summary(self):
        """
        Get a summary of all line analysis results.
        
        Returns
        -------
        dict
            Summary of detected lines, measurements, and statistics
        """
        summary = {
            'total_lines_detected': len(self.detected_lines),
            'emission_lines': len([l for l in self.detected_lines if l.get('line_type') == 'emission']),
            'absorption_lines': len([l for l in self.detected_lines if l.get('line_type') == 'absorption']),
            'continuum_level': self.continuum_level,
            'noise_level': self.noise_level,
            'analysis_timestamp': datetime.now().isoformat(),
            'detected_lines': self.detected_lines,
            'line_measurements': self.line_measurements
        }
        
        return summary

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

    def analyze_saved_lines(self, saved_lines_file, fitting_engine, output_file=None):
        """
        Comprehensive analysis of saved lines using data already present in the saved lines file.
        Performs fitting and rotation diagram calculations without external spectral arrays.
        
        Parameters
        ----------
        saved_lines_file : str
            Path to saved lines CSV file
        fitting_engine : FittingEngine
            Fitting engine instance to use for formatting results
        output_file : str, optional
            Output file path for results
            
        Returns
        -------
        list of dict
            Complete analysis results with fit parameters and rotation diagram values
        """
        # Read saved lines
        try:
            saved_lines = pd.read_csv(saved_lines_file)
            if saved_lines.empty:
                return []
        except Exception as e:
            print(f"Error reading saved lines file: {e}")
            return []
            
        fit_results_csv_data = []
        fit_results_data = []
        fitted_fluxes = []
        fitted_waves = []
        sig_det_lim = 2  # Detection limit for signal-to-noise ratio
        
        for i, line_row in saved_lines.iterrows():
            try:
                # Extract all line information from saved file
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
                    'g_up': float(line_row.get('g_up', 1.0)) if not pd.isna(line_row.get('g_up', 1.0)) else 1.0,
                    #'intens': float(line_row.get('intens', 0.0)) if not pd.isna(line_row.get('intens', 0.0)) else 0.0,
                    #'tau': float(line_row.get('tau', 0.0)) if not pd.isna(line_row.get('tau', 0.0)) else 0.0
                }

                fit_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
                x_fit = self.islat.wave_data[fit_mask]
                y_fit = self.islat.flux_data[fit_mask]

                fit_result, fitted_wave, fitted_flux = fitting_engine.fit_gaussian_line(
                    wave_data=x_fit,
                    flux_data=y_fit,
                    xmin=xmin,
                    xmax=xmax,
                    initial_guess=None,
                    deblend=False
                )
                
                # Create synthetic data arrays for formatting compatibility
                wave_range = np.linspace(xmin, xmax, 50)
                flux_range = np.ones(50)  # Placeholder flux
                #wave_range = fitted_wave
                #flux_range = fitted_flux
                fit_results_data.append(fit_result)
                fitted_waves.append(fitted_wave)
                fitted_fluxes.append(fitted_flux)
                err_range = np.ones(50) * 0.1  # Placeholder error
                
                # Format results using existing method
                result_entry = fitting_engine.format_fit_results_for_csv(
                    fit_result, wave_range, flux_range, err_range,
                    xmin, xmax, center_wave, line_info, sig_det_lim
                )
                
                fit_results_csv_data.append(result_entry)
                
            except Exception as e:
                print(f"Error analyzing line {i+1}: {e}")
                continue
        
        # Add rotation diagram values if we have successful fits with molecular data
        if fit_results_csv_data and any(entry.get('a_stein', 0) > 0 for entry in fit_results_csv_data):
            self.add_rotation_diagram_values(fit_results_csv_data)
        
        # Save results if output file specified
        if output_file and fit_results_csv_data:
            ifh.save_fit_results(fit_results_csv_data, file_name=output_file)

        return fit_results_csv_data, (fit_results_data, fitted_waves, fitted_fluxes)