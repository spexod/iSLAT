"""
LineAnalyzer - Line detection, identification, and analysis functionality

This class handles all line analysis operations including:
- Automatic line detection
- Line identification and matching
- Equivalent width calculations
- Line strength measurements
- Multi-line analysis and statistics
"""

import numpy as np
from datetime import datetime
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import trapezoid, simpson
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
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
        self.molecular_lines = {}  # Will be populated as needed
        
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
    
    def detect_lines_automatic(self, wave_data, flux_data, detection_type='emission'):
        """
        Automatically detect spectral lines in the data.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        detection_type : str, optional
            Type of lines to detect ('emission', 'absorption', 'both')
            
        Returns
        -------
        detected_lines : list
            List of dictionaries containing line information
        """
        # Estimate continuum and noise if not already done
        if self.continuum_level is None:
            continuum, noise_std = self.estimate_continuum_and_noise(wave_data, flux_data)
        else:
            continuum = self.continuum_level
            noise_std = self.noise_level
        
        #print("Wave data, and flux data", wave_data, flux_data)
        
        detected_lines = []
        
        if detection_type in ['emission', 'both']:
            emission_lines = self._detect_emission_lines(wave_data, flux_data, continuum, noise_std)
            detected_lines.extend(emission_lines)
        
        if detection_type in ['absorption', 'both']:
            absorption_lines = self._detect_absorption_lines(wave_data, flux_data, continuum, noise_std)
            detected_lines.extend(absorption_lines)
        
        #print(f"Detected {len(detected_lines)} lines ({len(emission_lines)} emission, {len(absorption_lines)} absorption)")

        # Sort by wavelength
        detected_lines.sort(key=lambda x: x['wavelength'])
        
        self.detected_lines = detected_lines
        return detected_lines
    
    def _detect_emission_lines(self, wave_data, flux_data, continuum, noise_std):
        """Detect emission lines above continuum."""
        # Look for peaks above continuum + noise threshold
        excess_flux = flux_data - continuum
        threshold = self.min_snr * noise_std
        
        #print("Excess flux:", excess_flux, "\nThreshold for peak detection:", threshold)

        # Find peaks
        peaks, properties = find_peaks(
            excess_flux,
            height=threshold,
            width=1,  # Minimum width in pixels
            distance=3  # Minimum separation in pixels
        )
        
        emission_lines = []
        
        for i, peak_idx in enumerate(peaks):
            line_info = self._characterize_line(
                wave_data, flux_data, continuum, peak_idx,
                line_type='emission'
            )
            
            if line_info is not None:
                emission_lines.append(line_info)
        
        return emission_lines
    
    def _detect_absorption_lines(self, wave_data, flux_data, continuum, noise_std):
        """Detect absorption lines below continuum."""
        # Look for negative peaks (absorption)
        deficit_flux = continuum - flux_data
        threshold = self.min_snr * noise_std
        
        # Find peaks in the deficit (absorption lines)
        peaks, properties = find_peaks(
            deficit_flux,
            height=threshold,
            width=1,
            distance=3
        )
        
        absorption_lines = []
        
        for i, peak_idx in enumerate(peaks):
            line_info = self._characterize_line(
                wave_data, flux_data, continuum, peak_idx,
                line_type='absorption'
            )
            
            if line_info is not None:
                absorption_lines.append(line_info)
        
        return absorption_lines
    
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
                'type': line_type,
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
    
    def _search_molecular_lines(self, wavelength, tolerance):
        """Search molecular line databases for matches."""
        # This would be expanded to search loaded molecular databases
        # For now, return empty list
        return []
    
    def analyze_line_ratios(self, line_pairs):
        """
        Analyze ratios between pairs of spectral lines.
        
        Parameters
        ----------
        line_pairs : list
            List of tuples containing wavelength pairs for ratio analysis
            
        Returns
        -------
        ratios : dict
            Dictionary containing line ratio information
        """
        ratios = {}
        
        for pair in line_pairs:
            wave1, wave2 = pair
            
            # Get measurements for both lines
            if wave1 in self.line_measurements and wave2 in self.line_measurements:
                meas1 = self.line_measurements[wave1]
                meas2 = self.line_measurements[wave2]
                
                # Calculate various ratios
                ratio_info = {}
                
                # Flux ratio
                if 'integrated_flux' in meas1 and 'integrated_flux' in meas2:
                    if meas2['integrated_flux'] != 0:
                        ratio_info['flux_ratio'] = meas1['integrated_flux'] / meas2['integrated_flux']
                    else:
                        ratio_info['flux_ratio'] = np.inf
                
                # Equivalent width ratio
                if 'equivalent_width' in meas1 and 'equivalent_width' in meas2:
                    if meas2['equivalent_width'] != 0:
                        ratio_info['ew_ratio'] = meas1['equivalent_width'] / meas2['equivalent_width']
                    else:
                        ratio_info['ew_ratio'] = np.inf
                
                # Peak strength ratio
                if 'line_strength' in meas1 and 'line_strength' in meas2:
                    if meas2['line_strength'] != 0:
                        ratio_info['strength_ratio'] = meas1['line_strength'] / meas2['line_strength']
                    else:
                        ratio_info['strength_ratio'] = np.inf
                
                ratios[f"{wave1:.3f}/{wave2:.3f}"] = ratio_info
        
        return ratios
    
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
            
        fit_results_data = []
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

                fit_result, _, _ = fitting_engine.fit_gaussian_line(
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
                err_range = np.ones(50) * 0.1  # Placeholder error
                
                # Format results using existing method
                result_entry = fitting_engine.format_fit_results_for_csv(
                    fit_result, wave_range, flux_range, err_range,
                    xmin, xmax, center_wave, line_info, sig_det_lim
                )
                
                fit_results_data.append(result_entry)
                
            except Exception as e:
                print(f"Error analyzing line {i+1}: {e}")
                continue
        
        # Add rotation diagram values if we have successful fits with molecular data
        if fit_results_data and any(entry.get('a_stein', 0) > 0 for entry in fit_results_data):
            self.add_rotation_diagram_values(fit_results_data)
        
        # Save results if output file specified
        if output_file and fit_results_data:
            ifh.save_fit_results(fit_results_data, file_name=output_file)

        return fit_results_data