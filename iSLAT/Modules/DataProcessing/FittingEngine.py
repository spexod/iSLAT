"""
FittingEngine - LMFIT operations and model fitting functionality

This class handles all fitting operations including:
- Gaussian line fitting
- Multi-component fitting (deblending)
- Slab model fitting
- Chi-squared calculations
- Parameter estimation and uncertainty analysis
"""

import numpy as np
from datetime import datetime
import json
from lmfit.models import GaussianModel, PseudoVoigtModel
from lmfit import Parameters, minimize, fit_report
from scipy.optimize import fmin
from scipy.signal import find_peaks
#from iSLAT.Modules.DataTypes import Chi2Spectrum, FluxMeasurement
from iSLAT.Modules.DataProcessing.Slabfit import SlabFit
import json

class FittingEngine:
    """
    Centralized fitting engine for all LMFIT operations and model fitting.
    
    This class provides a unified interface for various fitting operations
    including line fitting, deblending, and slab model fitting.
    """
    
    def __init__(self, islat_instance):
        """
        Initialize the fitting engine.
        
        Parameters
        ----------
        islat_instance : iSLAT
            Reference to the main iSLAT instance for accessing data and configuration
        """
        self.islat = islat_instance
        self.last_fit_result = None
        self.last_fit_params = None
        self.fit_uncertainty = 1.0  # Default uncertainty factor
        
        # Line detection strategy configuration
        self.line_detection_strategy = self._get_line_detection_strategy()
        self.user_selected_centers = []  # For manual line selection strategy
        
    def _get_line_detection_strategy(self):
        """
        Get the line detection strategy from iSLAT user settings.
        
        Returns
        -------
        str
            Line detection strategy: 'molecular_table', 'peak_detection', or 'user_selection'
        """
        try:
            # Get from iSLAT user settings (no direct file I/O)
            if hasattr(self.islat, 'user_settings') and self.islat.user_settings:
                return self.islat.user_settings.get('line_detection_strategy', 'user_selection')
            
            # Default fallback - use user_selection to match original iSLATOld behavior
            return 'user_selection'
            
        except Exception as e:
            print(f"Warning: Could not load line detection strategy from settings: {e}")
            return 'user_selection'
    
    def set_line_detection_strategy(self, strategy):
        """
        Set the line detection strategy using iSLAT's settings management.
        
        Parameters
        ----------
        strategy : str
            Strategy to use: 'molecular_table', 'peak_detection', or 'user_selection'
        """
        valid_strategies = ['molecular_table', 'peak_detection', 'user_selection']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Must be one of: {valid_strategies}")
        
        self.line_detection_strategy = strategy
        
        # Update iSLAT user settings (let iSLAT handle persistence)
        if hasattr(self.islat, 'user_settings'):
            self.islat.user_settings['line_detection_strategy'] = strategy
            
            # If iSLAT has a method to save settings, use it
            if hasattr(self.islat, 'save_user_settings'):
                try:
                    self.islat.save_user_settings()
                except Exception as e:
                    print(f"Warning: Could not save line detection strategy: {e}")
    
    def set_user_selected_centers(self, centers):
        """
        Set manually selected line centers for user_selection strategy.
        
        Parameters
        ----------
        centers : list or array_like
            List of wavelength centers manually selected by user
        """
        if centers is None:
            self.user_selected_centers = []
        else:
            # Convert to list to ensure hashable types (avoid numpy array issues)
            self.user_selected_centers = list(np.atleast_1d(centers))
        
    def set_fit_uncertainty(self, uncertainty):
        """Set the uncertainty factor for fitting operations."""
        self.fit_uncertainty = uncertainty
    
    def fit_gaussian_line(self, wave_data, flux_data, xmin=None, xmax=None, 
                         initial_guess=None, deblend=False):
        """
        Fit a Gaussian model to spectral line data.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        xmin, xmax : float, optional
            Wavelength range for fitting. If None, uses full range
        initial_guess : dict, optional
            Initial parameter guesses {'center': float, 'amplitude': float, 'sigma': float}
        deblend : bool, optional
            If True, attempt to fit multiple components
            
        Returns
        -------
        fit_result : lmfit.ModelResult
            Fitting result object
        fitted_wave : array_like
            Wavelength array for fitted model
        fitted_flux : array_like
            Fitted flux values
        """
        # Apply wavelength range constraints
        if xmin is not None and xmax is not None:
            mask = (wave_data >= xmin) & (wave_data <= xmax)
            fit_wave = wave_data[mask]
            fit_flux = flux_data[mask]
        else:
            fit_wave = wave_data
            fit_flux = flux_data
            
        if len(fit_wave) < 3:
            raise ValueError("Insufficient data points for fitting")
        
        if deblend:
            return self._fit_multi_gaussian(fit_wave, fit_flux, initial_guess, xmin, xmax)
        else:
            return self._fit_single_gaussian(fit_wave, fit_flux, initial_guess)
    
    def _fit_single_gaussian(self, wave_data, flux_data, initial_guess=None):
        """Fit a single Gaussian component."""
        model = GaussianModel()
        
        # Generate initial parameter estimates
        if initial_guess is None:
            initial_guess = self._estimate_gaussian_params(wave_data, flux_data)
        
        params = model.make_params(
            center=initial_guess.get('center', wave_data[np.argmax(flux_data)]),
            amplitude=initial_guess.get('amplitude', np.max(flux_data)),
            sigma=initial_guess.get('sigma', (wave_data[-1] - wave_data[0]) / 10)
        )
        
        # Set reasonable bounds
        params['center'].set(min=wave_data.min(), max=wave_data.max())
        params['amplitude'].set(min=0)
        params['sigma'].set(min=1e-6, max=(wave_data[-1] - wave_data[0]))
        
        # Perform fit
        result = model.fit(flux_data, params, x=wave_data)
        
        # Generate fitted curve
        fitted_wave = np.linspace(wave_data.min(), wave_data.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def _fit_multi_gaussian(self, wave_data, flux_data, initial_guess=None, xmin=None, xmax=None):
        """Fit multiple Gaussian components for deblending (matching old iSLAT behavior)."""
        # Estimate number of components and line centers based on detection strategy
        n_components, line_centers = self._estimate_n_components(wave_data, flux_data, xmin, xmax)
        
        if n_components == 1:
            # If only one component detected, use single Gaussian fit
            return self._fit_single_gaussian(wave_data, flux_data, initial_guess)
        
        # Get error data for weights (like old iSLAT)
        weights = None
        if hasattr(self.islat, 'err_data') and self.islat.err_data is not None:
            if xmin is not None and xmax is not None:
                err_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
                err_fit = self.islat.err_data[err_mask]
                if len(err_fit) == len(flux_data) and len(err_fit) > 0:
                    # Avoid division by zero - replace zero or negative errors with a small value
                    max_err = np.max(err_fit)
                    if max_err <= 0:
                        # If all errors are zero or negative, don't use weights
                        weights = np.ones_like(flux_data)
                    else:
                        err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
                        weights = 1.0 / err_fit_safe
        
        if weights is None:
            weights = np.ones_like(flux_data)
        
        # Get tolerance settings from user_settings like old iSLAT
        centrtolerance = self.islat.user_settings.get('centrtolerance', 0.0001)
        fwhmtolerance = self.islat.user_settings.get('fwhmtolerance', 5)
        
        # Get FWHM from the molecule dict (can be updated at runtime)
        fwhm = self.islat.molecules_dict.global_fwhm  # km/s
        mean_wavelength = np.mean([xmin or wave_data.min(), xmax or wave_data.max()])
        fwhm_um = mean_wavelength / 299792.458 * fwhm  # Convert km/s to μm
        sig = fwhm_um / 2.35482  # Convert FWHM to sigma
        sig_tol = (mean_wavelength / 299792.458 * fwhmtolerance) / 2.35482
        
        # Calculate initial amplitude estimate like old iSLAT
        # Total integrated flux divided by number of lines
        total_flux = np.trapz(flux_data, wave_data)
        #total_flux = np.trapz(flux_data, wave_data)
        garea_fg = total_flux / len(line_centers) * 1e11  # Scaling factor like old iSLAT
        
        # Set up parameter bounds based on tolerances like old iSLAT
        fwhm_vary = sig_tol > 0
        centr_vary = centrtolerance > 0
        
        # Create composite model exactly like old iSLAT
        model = None
        params = Parameters()
        
        for i in range(n_components):
            prefix = f'g{i+1}_'  # Use 1-based indexing like old iSLAT
            if model is None:
                model = GaussianModel(prefix=prefix)
            else:
                model += GaussianModel(prefix=prefix)
            
            # Use detected line centers
            center = line_centers[i] if i < len(line_centers) else mean_wavelength
            
            # Set parameters exactly like old iSLAT
            params.add(f'{prefix}center', 
                      value=center, 
                      vary=centr_vary,
                      min=center - centrtolerance, 
                      max=center + centrtolerance)
            params.add(f'{prefix}sigma', 
                      value=sig, 
                      vary=fwhm_vary,
                      min=sig - sig_tol if sig_tol > 0 else sig * 0.1, 
                      max=sig + sig_tol if sig_tol > 0 else sig * 10)
            params.add(f'{prefix}amplitude', 
                      value=garea_fg, 
                      min=0)
        
        # Perform fit with same method as old iSLAT
        result = model.fit(flux_data, params, x=wave_data, weights=weights, 
                          method='leastsq', nan_policy='omit')
        
        # Generate fitted curve
        fitted_wave = np.linspace(wave_data.min(), wave_data.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def _estimate_gaussian_params(self, wave_data, flux_data):
        """Estimate initial Gaussian parameters from data."""
        max_idx = np.argmax(flux_data)
        center = wave_data[max_idx]
        amplitude = flux_data[max_idx]
        
        # Estimate sigma from FWHM
        half_max = amplitude / 2
        indices = np.where(flux_data >= half_max)[0]
        if len(indices) > 1:
            fwhm = wave_data[indices[-1]] - wave_data[indices[0]]
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma = (wave_data[-1] - wave_data[0]) / 10
        
        return {'center': center, 'amplitude': amplitude, 'sigma': sigma}
    
    def _estimate_n_components(self, wave_data, flux_data, xmin=None, xmax=None):
        """
        Estimate number of Gaussian components needed based on detection strategy.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        xmin, xmax : float, optional
            Wavelength range for analysis
            
        Returns
        -------
        int
            Number of components
        list
            List of line centers (wavelengths)
        """
        if self.line_detection_strategy == 'molecular_table':
            return self._estimate_components_from_molecular_table(wave_data, flux_data, xmin, xmax)
        elif self.line_detection_strategy == 'user_selection':
            return self._estimate_components_from_user_selection(wave_data, flux_data, xmin, xmax)
        else:  # 'peak_detection' (default)
            return self._estimate_components_from_peak_detection(wave_data, flux_data, xmin, xmax)
    
    def _estimate_components_from_peak_detection(self, wave_data, flux_data, xmin=None, xmax=None):
        """Original peak detection method (default FittingEngine behavior)."""
        peaks, _ = find_peaks(flux_data, height=np.max(flux_data) * 0.1)
        n_components = max(1, min(len(peaks), 3))  # Limit to 3 components
        
        # Extract peak centers
        if len(peaks) > 0:
            peak_centers = wave_data[peaks].tolist()
        else:
            peak_centers = [wave_data[np.argmax(flux_data)]]
            
        return n_components, peak_centers
    
    def _estimate_components_from_molecular_table(self, wave_data, flux_data, xmin=None, xmax=None):
        """Use molecular line tables like MainPlotOld does."""
        try:
            # Get line data from active molecule
            if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
                if xmin is None:
                    xmin = wave_data.min()
                if xmax is None:
                    xmax = wave_data.max()
                
                # Try the new MoleculeLine approach first
                line_data = None
                try:
                    lines_with_intensity = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
                    if lines_with_intensity:
                        # Convert to DataFrame-like format for compatibility
                        line_centers = np.array([line.lam for line, _, _ in lines_with_intensity])
                        intensities = np.array([intensity for _, intensity, _ in lines_with_intensity])
                        line_data = {'lam': line_centers, 'intens': intensities}
                except Exception as e:
                    print(f"Warning: Could not use new MoleculeLine approach: {e}")
                
                # Fallback to pandas DataFrame approach
                if line_data is None:
                    try:
                        line_data_df = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
                        if not line_data_df.empty:
                            line_centers = np.array(line_data_df['lam'])
                            intensities = np.array(line_data_df['intens'])
                            line_data = {'lam': line_centers, 'intens': intensities}
                    except Exception as e:
                        print(f"Warning: Could not get line data: {e}")
                
                if line_data is not None:
                    # Get centers and intensities
                    line_centers = np.array(line_data['lam'])
                    intensities = np.array(line_data['intens'])
                    
                    # Remove duplicate/very close lines (like MainPlotOld)
                    if len(line_centers) > 1:
                        line_centers = np.sort(line_centers)
                        min_sep = 1e-4  # μm, same as MainPlotOld
                        filtered_centers = [line_centers[0]]
                        filtered_intensities = [intensities[0]]
                        
                        for i, lc in enumerate(line_centers[1:], 1):
                            if np.all(np.abs(lc - np.array(filtered_centers)) > min_sep):
                                filtered_centers.append(lc)
                                filtered_intensities.append(intensities[i])
                        
                        line_centers = np.array(filtered_centers)
                        intensities = np.array(filtered_intensities)
                    
                    # Sort line centers by intensity (descending) like MainPlotOld
                    if len(line_centers) > 1:
                        sort_idx = np.argsort(-intensities)
                        line_centers = line_centers[sort_idx]
                        intensities = intensities[sort_idx]
                    
                    # Check if we have sufficient data points for multi-gaussian fitting
                    max_gaussians = len(line_centers)
                    while max_gaussians > 0:
                        num_params = 3 * max_gaussians  # center, sigma, amplitude per Gaussian
                        if len(wave_data) >= num_params * 2:  # Same check as MainPlotOld
                            break
                        max_gaussians -= 1
                    
                    if max_gaussians == 0:
                        max_gaussians = 1  # Always fit at least one component
                    
                    # Use only the most important (strongest) lines
                    use_centers = line_centers[:max_gaussians].tolist()
                    n_components = max_gaussians
                    
                    return n_components, use_centers
        except Exception as e:
            print(f"Warning: Could not use molecular table detection: {e}")
        
        # Fallback to peak detection
        return self._estimate_components_from_peak_detection(wave_data, flux_data, xmin, xmax)
    
    def _estimate_components_from_user_selection(self, wave_data, flux_data, xmin=None, xmax=None):
        """
        Use pre-selected line positions like iSLATOld does.
        
        In iSLATOld, this method gets line centers from the molecular data in the selected range,
        similar to molecular_table but with manual user confirmation through the selection process.
        """
        # First try to use manually set user-selected centers
        if self.user_selected_centers:
            # Filter user centers that are within the current wavelength range
            if xmin is None:
                xmin = wave_data.min()
            if xmax is None:
                xmax = wave_data.max()
                
            valid_centers = [center for center in self.user_selected_centers 
                            if xmin <= center <= xmax]
            
            if valid_centers:
                n_components = len(valid_centers)
                return n_components, valid_centers
        
        # If no manual centers set, mimic iSLATOld behavior:
        # Use molecular line data from the selected region (like onselect_lines['lam'])
        try:
            if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
                if xmin is None:
                    xmin = wave_data.min()
                if xmax is None:
                    xmax = wave_data.max()
                
                # Try the new MoleculeLine approach first
                line_data = None
                try:
                    lines_with_intensity = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
                    if lines_with_intensity:
                        # Convert to DataFrame-like format for compatibility
                        line_centers = np.array([line.lam for line, _, _ in lines_with_intensity])
                        intensities = np.array([intensity for _, intensity, _ in lines_with_intensity])
                        line_data = {'lam': line_centers, 'intens': intensities}
                except Exception as e:
                    print(f"Warning: Could not use new MoleculeLine approach: {e}")
                
                # Fallback to pandas DataFrame approach  
                if line_data is None:
                    try:
                        line_data_df = self.islat.active_molecule.intensity.get_table_in_range(xmin, xmax)
                        if not line_data_df.empty:
                            line_centers = np.array(line_data_df['lam'])
                            intensities = np.array(line_data_df['intens'])
                            line_data = {'lam': line_centers, 'intens': intensities}
                    except Exception as e:
                        print(f"Warning: Could not get line data: {e}")
                
                if line_data is not None:
                    # Apply line_threshold filtering like iSLATOld
                    line_threshold = 0.03  # Default from iSLATOld
                    try:
                        if hasattr(self.islat, 'user_settings') and self.islat.user_settings:
                            line_threshold = self.islat.user_settings.get('line_threshold', 0.03)
                    except:
                        pass
                    
                    # Filter lines above threshold (like iSLATOld does)
                    line_centers = np.array(line_data['lam'])
                    intensities = np.array(line_data['intens'])
                    max_intensity = intensities.max()
                    threshold_intensity = max_intensity * line_threshold
                    
                    # Create boolean mask for filtering instead of treating dict as DataFrame
                    strong_mask = intensities >= threshold_intensity
                    
                    if np.any(strong_mask):
                        filtered_centers = line_centers[strong_mask]
                        n_components = len(filtered_centers)
                        return n_components, filtered_centers.tolist()
        except Exception as e:
            print(f"Warning: Could not use user selection detection: {e}")
        
        print("Warning: No user-selected centers in current range, falling back to peak detection")
        return self._estimate_components_from_peak_detection(wave_data, flux_data, xmin, xmax)
    
    def _estimate_component_params(self, wave_data, flux_data, component_idx, line_centers):
        """
        Estimate parameters for a specific component in multi-component fit.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        component_idx : int
            Index of component to estimate
        line_centers : list
            List of line centers from detection strategy
            
        Returns
        -------
        dict
            Parameter estimates for this component
        """
        if component_idx < len(line_centers):
            # Use detected line center
            center = line_centers[component_idx]
            
            # Better amplitude estimation like MainPlotOld: (y_max - y_min) * scaling_factor
            amplitude = (flux_data.max() - flux_data.min()) * 0.1  # Same as MainPlotOld
            
            # Better sigma estimation: use fixed value like MainPlotOld
            sigma = 0.001  # Same initial value as MainPlotOld
            
        else:
            # Fallback to region-based estimation (original method)
            wave_range = wave_data[-1] - wave_data[0]
            region_size = wave_range / max(len(line_centers), 1)
            region_start = wave_data[0] + component_idx * region_size
            region_end = region_start + region_size
            
            # Find peak in this region
            mask = (wave_data >= region_start) & (wave_data <= region_end)
            if np.any(mask):
                region_flux = flux_data[mask]
                region_wave = wave_data[mask]
                max_idx = np.argmax(region_flux)
                center = region_wave[max_idx]
                amplitude = region_flux[max_idx]
            else:
                center = region_start + region_size / 2
                amplitude = np.max(flux_data) / max(len(line_centers), 1)
            
            sigma = region_size / 4  # Conservative estimate
        
        return {'center': center, 'amplitude': amplitude, 'sigma': sigma}
    
    def fit_voigt_profile(self, wave_data, flux_data, xmin=None, xmax=None):
        """
        Fit a Voigt profile to spectral line data.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        xmin, xmax : float, optional
            Wavelength range for fitting
            
        Returns
        -------
        fit_result : lmfit.ModelResult
            Fitting result object
        fitted_wave : array_like
            Wavelength array for fitted model
        fitted_flux : array_like
            Fitted flux values
        """
        # Apply wavelength range constraints
        if xmin is not None and xmax is not None:
            mask = (wave_data >= xmin) & (wave_data <= xmax)
            fit_wave = wave_data[mask]
            fit_flux = flux_data[mask]
        else:
            fit_wave = wave_data
            fit_flux = flux_data
        
        model = PseudoVoigtModel()
        
        # Initial parameter estimates
        initial_guess = self._estimate_gaussian_params(fit_wave, fit_flux)
        
        params = model.make_params(
            center=initial_guess['center'],
            amplitude=initial_guess['amplitude'],
            sigma=initial_guess['sigma'],
            fraction=0.5  # Start with equal Gaussian and Lorentzian components
        )
        
        # Set bounds
        params['center'].set(min=fit_wave.min(), max=fit_wave.max())
        params['amplitude'].set(min=0)
        params['sigma'].set(min=1e-6, max=(fit_wave[-1] - fit_wave[0]))
        params['fraction'].set(min=0, max=1)
        
        # Perform fit
        result = model.fit(fit_flux, params, x=fit_wave)
        
        # Generate fitted curve
        fitted_wave = np.linspace(fit_wave.min(), fit_wave.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def perform_slab_fit(self, target_file, molecule_name, 
                        start_temp=500, start_n_mol=1e17, start_radius=1.0):
        """
        Perform slab model fitting for a given target spectrum.
        
        Parameters
        ----------
        target_file : str
            Path to target spectrum file
        molecule_name : str
            Name of molecule to fit
        start_temp : float, optional
            Initial temperature guess (K)
        start_n_mol : float, optional
            Initial column density guess (cm^-2)
        start_radius : float, optional
            Initial radius guess (AU)
            
        Returns
        -------
        fit_result : dict
            Dictionary containing fitted parameters and statistics
        """
        try:
            # Get molecule data path
            mol_data = None
            for mol_info in self.islat.default_molecule_csv_data():
                if mol_info['name'] == molecule_name:
                    mol_data = mol_info
                    break
            
            if mol_data is None:
                raise ValueError(f"Molecule {molecule_name} not found in available molecules")
            
            # Create slab fit instance
            slab_fitter = SlabFit(
                target=target_file,
                save_folder="EXAMPLE-data",
                mol=molecule_name,
                molpath=mol_data['file'],
                dist=self.islat.molecules_dict.global_dist,
                fwhm=self.islat.molecules_dict.global_fwhm,
                min_lamb=self.islat.wavelength_range[0],
                max_lamb=self.islat.wavelength_range[1],
                pix_per_fwhm=10,
                intrinsic_line_width=self.islat.molecules_dict.global_intrinsic_line_width,
                cc=3e8,
                data_field=getattr(self.islat.gui, 'data_field', None)
            )
            
            # Initialize and perform fit
            slab_fitter.initialize()
            result = slab_fitter.fit(start_temp, start_n_mol, start_radius)
            
            return result
            
        except Exception as e:
            print(f"Error in slab fitting: {str(e)}")
            return None
    
    def calculate_chi_squared(self, wave_obs, flux_obs, flux_error, 
                            wave_model, flux_model):
        """
        Calculate chi-squared statistic between observed and model data.
        
        Parameters
        ----------
        wave_obs : array_like
            Observed wavelength data
        flux_obs : array_like
            Observed flux data
        flux_error : array_like
            Flux uncertainties
        wave_model : array_like
            Model wavelength data
        flux_model : array_like
            Model flux data
            
        Returns
        -------
        chi2 : float
            Chi-squared statistic
        reduced_chi2 : float
            Reduced chi-squared statistic
        """
        # Interpolate model to observed wavelength grid
        flux_model_interp = np.interp(wave_obs, wave_model, flux_model)
        
        # Calculate chi-squared
        chi2 = np.sum(((flux_obs - flux_model_interp) / flux_error) ** 2)
        
        # Calculate reduced chi-squared (adjust for number of fitted parameters)
        n_params = len(self.last_fit_params) if self.last_fit_params else 0
        dof = len(wave_obs) - n_params
        reduced_chi2 = chi2 / dof if dof > 0 else chi2
        
        return chi2, reduced_chi2
    
    def get_fit_report(self):
        """
        Get a detailed report of the last fitting operation.
        
        Returns
        -------
        report : str
            Formatted fit report
        """
        if self.last_fit_result is None:
            return "No fitting results available."
        
        return fit_report(self.last_fit_result)
    
    def get_fit_statistics(self):
        """
        Get statistical information about the last fit.
        
        Returns
        -------
        stats : dict
            Dictionary containing fit statistics
        """
        if self.last_fit_result is None:
            return {}
        
        result = self.last_fit_result
        
        stats = {
            'chi_squared': result.chisqr,
            'reduced_chi_squared': result.redchi,
            'aic': result.aic,
            'bic': result.bic,
            'n_data': result.ndata,
            'n_variables': result.nvarys,
            'n_function_evals': result.nfev,
            'success': result.success,
            'method': result.method
        }
        
        return stats
    
    def is_multi_component_fit(self):
        """
        Check if the last fit result contains multiple components.
        
        Returns
        -------
        bool
            True if multi-component fit, False if single component
        """
        if self.last_fit_result is None or self.last_fit_params is None:
            return False
        
        # Check for component prefixes (g0_, g1_, etc.)
        component_prefixes = set()
        for param_name in self.last_fit_params:
            if '_' in param_name:
                prefix = param_name.split('_')[0] + '_'
                if prefix.startswith('g') and prefix[1:-1].isdigit():
                    component_prefixes.add(prefix)
        
        return len(component_prefixes) > 1

    def get_component_prefixes(self):
        """
        Get the component prefixes from the last fit result.
        
        Returns
        -------
        list
            List of component prefixes (e.g., ['g0_', 'g1_'])
        """
        if self.last_fit_result is None or self.last_fit_params is None:
            return []
        
        component_prefixes = set()
        for param_name in self.last_fit_params:
            if '_' in param_name:
                prefix = param_name.split('_')[0] + '_'
                if prefix.startswith('g') and prefix[1:-1].isdigit():
                    component_prefixes.add(prefix)
        
        return sorted(list(component_prefixes))
    
    def evaluate_fit_components(self, x_data):
        """
        Evaluate individual components of a multi-component fit.
        
        Parameters
        ----------
        x_data : array_like
            X values to evaluate at
            
        Returns
        -------
        dict
            Dictionary mapping component names to flux arrays
        """
        if self.last_fit_result is None:
            return {}
        
        if not self.is_multi_component_fit():
            # Single component - return the full fit
            return {'total': self.last_fit_result.eval(x=x_data)}
        
        # Multi-component fit
        try:
            components = self.last_fit_result.eval_components(x=x_data)
            return components
        except Exception as e:
            print(f"Error evaluating fit components: {e}")
            return {'total': self.last_fit_result.eval(x=x_data)}

    def extract_line_parameters(self):
        """
        Extract line parameters from the last fitting result.
        
        Returns
        -------
        line_params : dict
            Dictionary containing line parameters (center, amplitude, width, etc.)
        """
        if self.last_fit_result is None or self.last_fit_params is None:
            return {}
        
        params = self.last_fit_params
        line_params = {}
        
        # Extract parameters for single Gaussian fit
        if 'center' in params:
            line_params['center'] = params['center'].value
            line_params['center_stderr'] = params['center'].stderr if params['center'].stderr is not None else None
            line_params['amplitude'] = params['amplitude'].value
            line_params['amplitude_stderr'] = params['amplitude'].stderr if params['amplitude'].stderr is not None else None
            line_params['sigma'] = params['sigma'].value
            line_params['sigma_stderr'] = params['sigma'].stderr if params['sigma'].stderr is not None else None
            
            # Calculate derived parameters
            line_params['fwhm'] = 2.355 * params['sigma'].value  # 2*sqrt(2*ln(2))
            line_params['area'] = np.sqrt(2 * np.pi) * params['amplitude'].value * params['sigma'].value
            
            # Calculate area error using error propagation
            if (params['amplitude'].stderr is not None and 
                params['sigma'].stderr is not None):
                area_err = np.sqrt(2 * np.pi) * np.sqrt(
                    (params['sigma'].value * params['amplitude'].stderr)**2 +
                    (params['amplitude'].value * params['sigma'].stderr)**2
                )
                line_params['area_stderr'] = area_err
            else:
                line_params['area_stderr'] = None
            
        # Extract parameters for multi-component fits (check both 0-based and 1-based prefixes)
        component_idx = 0
        
        # First try 1-based prefixes (like old iSLAT: g1_, g2_, etc.)
        while f'g{component_idx+1}_center' in params:
            prefix = f'g{component_idx+1}_'
            component_params = {}
            
            component_params['center'] = params[f'{prefix}center'].value
            component_params['center_stderr'] = params[f'{prefix}center'].stderr if params[f'{prefix}center'].stderr is not None else None
            component_params['amplitude'] = params[f'{prefix}amplitude'].value
            component_params['amplitude_stderr'] = params[f'{prefix}amplitude'].stderr if params[f'{prefix}amplitude'].stderr is not None else None
            component_params['sigma'] = params[f'{prefix}sigma'].value
            component_params['sigma_stderr'] = params[f'{prefix}sigma'].stderr if params[f'{prefix}sigma'].stderr is not None else None
            
            # Calculate derived parameters for component
            component_params['fwhm'] = 2.355 * params[f'{prefix}sigma'].value
            component_params['area'] = (np.sqrt(2 * np.pi) * 
                                      params[f'{prefix}amplitude'].value * 
                                      params[f'{prefix}sigma'].value)
            
            # Calculate area error for component
            if (params[f'{prefix}amplitude'].stderr is not None and 
                params[f'{prefix}sigma'].stderr is not None):
                area_err = np.sqrt(2 * np.pi) * np.sqrt(
                    (params[f'{prefix}sigma'].value * params[f'{prefix}amplitude'].stderr)**2 +
                    (params[f'{prefix}amplitude'].value * params[f'{prefix}sigma'].stderr)**2
                )
                component_params['area_stderr'] = area_err
            else:
                component_params['area_stderr'] = None
            
            line_params[f'component_{component_idx}'] = component_params
            component_idx += 1
        
        # If no 1-based prefixes found, try 0-based prefixes (g0_, g1_, etc.)
        if component_idx == 0:
            while f'g{component_idx}_center' in params:
                prefix = f'g{component_idx}_'
                component_params = {}
                
                component_params['center'] = params[f'{prefix}center'].value
                component_params['center_stderr'] = params[f'{prefix}center'].stderr if params[f'{prefix}center'].stderr is not None else None
                component_params['amplitude'] = params[f'{prefix}amplitude'].value
                component_params['amplitude_stderr'] = params[f'{prefix}amplitude'].stderr if params[f'{prefix}amplitude'].stderr is not None else None
                component_params['sigma'] = params[f'{prefix}sigma'].value
                component_params['sigma_stderr'] = params[f'{prefix}sigma'].stderr if params[f'{prefix}sigma'].stderr is not None else None
                
                # Calculate derived parameters for component
                component_params['fwhm'] = 2.355 * params[f'{prefix}sigma'].value
                component_params['area'] = (np.sqrt(2 * np.pi) * 
                                          params[f'{prefix}amplitude'].value * 
                                          params[f'{prefix}sigma'].value)
                
                # Calculate area error for component
                if (params[f'{prefix}amplitude'].stderr is not None and 
                    params[f'{prefix}sigma'].stderr is not None):
                    area_err = np.sqrt(2 * np.pi) * np.sqrt(
                        (params[f'{prefix}sigma'].value * params[f'{prefix}amplitude'].stderr)**2 +
                        (params[f'{prefix}amplitude'].value * params[f'{prefix}sigma'].stderr)**2
                    )
                    component_params['area_stderr'] = area_err
                else:
                    component_params['area_stderr'] = None
                
                line_params[f'component_{component_idx}'] = component_params
                component_idx += 1
        
        return line_params
    
    def save_fit_results(self, filename=None):
        """
        Save fit results to a JSON file.
        
        Parameters
        ----------
        filename : str, optional
            Output filename. If None, generates timestamp-based name
            
        Returns
        -------
        str
            Path to saved file
        """
        if self.last_fit_result is None:
            raise ValueError("No fit results available to save")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fit_results_{timestamp}.json"
            
        # Extract comprehensive fit information
        fit_data = {
            'timestamp': datetime.now().isoformat(),
            'fit_method': 'gaussian_line_fitting',
            'parameters': self.extract_line_parameters(),
            'statistics': self.get_fit_statistics(),
            'fit_report': self.get_fit_report()
        }
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(fit_data, f, indent=2, default=str)
            
        return filename

    def format_fit_results_for_csv(self, fit_result, wave_data, flux_data, error_data, 
                                   xmin, xmax, rest_wavelength, line_info, sig_det_lim=2):
        """
        Format fit results into a standardized dictionary for CSV output.
        
        Parameters
        ----------
        fit_result : lmfit.ModelResult
            The fitting result object
        wave_data, flux_data, error_data : array_like
            Spectral data arrays
        xmin, xmax : float
            Fitting range boundaries
        rest_wavelength : float
            Rest wavelength of the line
        line_info : dict
            Molecular line information
        sig_det_lim : float, optional
            Detection significance limit (default: 2)
            
        Returns
        -------
        dict
            Formatted result dictionary
        """
        from scipy.integrate import trapezoid
        
        # Calculate line flux integral (direct calculation since we have the data)
        flux_data_integral = trapezoid(flux_data, wave_data)
        if error_data is not None:
            err_data_integral = np.sqrt(trapezoid(error_data**2, wave_data))
        else:
            err_data_integral = abs(flux_data_integral) * 0.1
        
        # Calculate signal-to-noise ratios for data
        line_sn = flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0
        line_det = abs(flux_data_integral) > sig_det_lim * err_data_integral
        
        # Prepare base result dictionary with data measurements
        result_entry = {
            'species': line_info.get('species', 'Unknown'),
            'lev_up': line_info.get('lev_up', ''),
            'lev_low': line_info.get('lev_low', ''),
            'lam': line_info.get('lam', rest_wavelength),
            'tau': line_info.get('tau', 0.0),
            'intens': line_info.get('intens', 0.0),
            'a_stein': line_info.get('a_stein', 0.0),
            'e_up': line_info.get('e_up', 0.0),
            'g_up': line_info.get('g_up', 1.0),
            'xmin': xmin,
            'xmax': xmax,
            'Flux_data': np.float64(f'{flux_data_integral:.{3}e}'),
            'Err_data': np.float64(f'{err_data_integral:.{3}e}'),
            'Line_SN': np.round(line_sn, decimals=1),
            'Line_det': bool(line_det),
            'Flux_islat': np.float64(f'{flux_data_integral:.{3}e}'),  # Default to data values
            'Err_islat': np.float64(f'{err_data_integral:.{3}e}')     # Will be overwritten if fit succeeds
        }
        
        # Process fit results if successful
        if fit_result and hasattr(fit_result, 'params') and fit_result.success:
            # Extract fit parameters
            center = fit_result.params['center'].value
            center_err = fit_result.params['center'].stderr if fit_result.params['center'].stderr else 0.0
            amplitude = fit_result.params['amplitude'].value
            sigma = fit_result.params['sigma'].value
            
            # Calculate derived quantities
            fwhm = 2.355 * sigma  # Convert sigma to FWHM
            area = amplitude * sigma * np.sqrt(2 * np.pi)  # Gaussian area
            area_err = area * 0.1  # Approximate 10% error for area
            
            # Calculate fit signal-to-noise and detection
            fit_sn = area / area_err if area_err > 0 else 0.0
            fit_det = abs(area) > sig_det_lim * area_err
            
            # Calculate Doppler shift
            doppler = (center - rest_wavelength) / rest_wavelength * 299792.458  # km/s
            
            # Update result with fit information
            result_entry.update({
                'Fit_SN': np.round(fit_sn, decimals=1),
                'Fit_det': bool(fit_det),
                'Flux_fit': np.float64(f'{area:.{3}e}'),
                'Err_fit': np.float64(f'{area_err:.{3}e}'),
                'FWHM_fit': np.round(fwhm, decimals=1) if fit_det else np.nan,
                'FWHM_err': np.round(fwhm * 0.1, decimals=1) if fit_det else np.nan,
                'Centr_fit': np.round(center, decimals=5) if fit_det else np.nan,
                'Centr_err': np.round(center_err, decimals=5) if fit_det else np.nan,
                'Doppler': np.round(doppler, decimals=1) if fit_det else np.nan,
                'Red-chisq': np.round(fit_result.redchi, decimals=2)
            })
            
            # Update iSLAT flux values if fit is good and detected
            if fit_det:
                result_entry['Flux_islat'] = np.float64(f'{area:.{3}e}')
                result_entry['Err_islat'] = np.float64(f'{area_err:.{3}e}')
        else:
            # Fit failed - fill with NaN values but keep data measurements
            result_entry.update({
                'Fit_SN': np.round(flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0, decimals=1),
                'Fit_det': False,
                'Flux_fit': np.float64(f'{flux_data_integral:.{3}e}'),  # Use data flux for failed fit
                'Err_fit': np.float64(f'{err_data_integral:.{3}e}'),
                'FWHM_fit': np.nan,
                'FWHM_err': np.nan,
                'Centr_fit': np.nan,
                'Centr_err': np.nan,
                'Doppler': np.nan,
                'Red-chisq': np.nan
            })
        
        return result_entry
