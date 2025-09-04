"""
FittingEngine - LMFIT operations and model fitting functionality
"""

import numpy as np
from lmfit.models import GaussianModel
from lmfit import Parameters
from iSLAT.Modules.DataProcessing.Slabfit import SlabFit
import iSLAT.Constants as c

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
    
    def fit_gaussian_line(self, wave_data, flux_data, xmin=None, xmax=None, 
                         initial_guess=None, deblend=False):
        """
        Fit a Gaussian model to spectral line data.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data (should already be filtered to desired range)
        flux_data : array_like
            Flux data (should already be filtered to desired range)
        xmin, xmax : float, optional
            Wavelength range for fitting (used for multi-gaussian detection strategy)
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
        # Data is assumed to already be filtered by caller
        fit_wave = wave_data
        fit_flux = flux_data
            
        if len(fit_wave) < 3:
            raise ValueError("Insufficient data points for fitting")
        
        # Set xmin/xmax from data if not provided (for multi-gaussian detection)
        if xmin is None:
            xmin = fit_wave.min()
        if xmax is None:
            xmax = fit_wave.max()
        
        if deblend:
            return self._fit_multi_gaussian(fit_wave, fit_flux, initial_guess, xmin, xmax)
        else:
            return self._fit_single_gaussian(fit_wave, fit_flux, initial_guess, xmin, xmax)

    def _fit_single_gaussian(self, wave_data, flux_data, initial_guess=None, xmin=None, xmax=None):
        """Fit a single Gaussian component."""
        # Data is already filtered to the fit range by the caller
        x_fit = wave_data
        flux_fit = flux_data
        
        # Use gaussian model from LMFIT
        model = GaussianModel()
        
        # Get initial guess for parameters (let LMFIT do the guessing)
        params = model.guess(flux_fit, x=x_fit)
        
        # Get error data for weights if available
        weights = None
        if hasattr(self.islat, 'err_data') and self.islat.err_data is not None:
            # Need to get error data for the same range
            if xmin is not None and xmax is not None:
                err_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
                err_fit = self.islat.err_data[err_mask]
                # Following LMFIT docs: use 1/error as weights, avoiding division by zero
                if len(err_fit) == len(flux_fit) and len(err_fit) > 0:
                    max_err = np.max(err_fit)
                    if max_err > 0:
                        err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
                        weights = 1.0 / err_fit_safe
        
        # Make the fit, using error data as weights and ignoring nans
        if weights is not None:
            result = model.fit(flux_fit, params, x=x_fit, weights=weights, nan_policy='omit')
        else:
            result = model.fit(flux_fit, params, x=x_fit, nan_policy='omit')
        
        print(result.fit_report())

        # Generate fitted curve on original wavelength grid
        fitted_wave = wave_data
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def flux_integral(self, lam, flux, err, lam_min, lam_max):
        # Use vectorized operations for efficiency
        wavelength_mask = (lam >= lam_min) & (lam <= lam_max)
        
        if not np.any(wavelength_mask):
            return 0.0, 0.0
            
        lam_range = lam[wavelength_mask]
        flux_range = flux[wavelength_mask]
        
        if len(lam_range) < 2:
            return 0.0, 0.0
        
        # Convert to frequency space for proper integration
        freq_range = c.SPEED_OF_LIGHT_KMS / lam_range
        
        # Integrate in frequency space (reverse order for proper frequency ordering)
        line_flux_meas = np.trapz(flux_range[::-1], x=freq_range[::-1])
        line_flux_meas = -line_flux_meas * 1e-23  # Convert Jy*Hz to erg/s/cm^2
        
        # Calculate error propagation if error data provided
        if err is not None:
            err_range = err[wavelength_mask]
            line_err_meas = np.trapz(err_range[::-1], x=freq_range[::-1])
            line_err_meas = -line_err_meas * 1e-23
        else:
            line_err_meas = 0.0
            
        return line_flux_meas, line_err_meas

    def _fit_multi_gaussian(self, wave_data, flux_data, initial_guess=None, xmin=None, xmax=None):
        """Fit multiple Gaussian components for deblending"""
        # Estimate number of components and line centers based on detection strategy
        n_components, line_centers = self._estimate_components_from_user_selection(wave_data, flux_data, xmin, xmax)
        print(f"Detected {n_components} components")
        
        if n_components == 1:
            # If only one component detected, use single Gaussian fit
            return self._fit_single_gaussian(wave_data, flux_data, initial_guess)
        
        # Get error data for weights
        weights = None
        if hasattr(self.islat, 'err_data') and self.islat.err_data is not None:
            if xmin is not None and xmax is not None:
                err_mask = (self.islat.wave_data >= xmin) & (self.islat.wave_data <= xmax)
                err_fit = self.islat.err_data[err_mask]
                if len(err_fit) == len(flux_data) and len(err_fit) > 0:
                    # Avoid division by zero - replace zero or negative errors with a small value
                    max_err = np.max(err_fit)
                    '''if max_err <= 0:
                        # If all errors are zero or negative, don't use weights
                        weights = np.ones_like(flux_data)
                    else:
                        err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
                        weights = 1.0 / err_fit_safe'''
                    err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
                    weights = 1.0 / err_fit_safe
        
        if weights is None:
            err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
            weights = 1.0 / err_fit_safe
        
        # Get tolerance settings from user_settings
        centrtolerance = self.islat.user_settings.get('centrtolerance', 0.0001)
        fwhmtolerance = self.islat.user_settings.get('fwhmtolerance', 5)
        
        # Get FWHM from the molecule dict (can be updated at runtime)
        #fwhm = self.islat.molecules_dict.global_fwhm  # km/s
        fwhm = self.islat.active_molecule.fwhm  # km/s
        print(f"Using FWHM: {fwhm} km/s")
        mean_wavelength = np.mean([xmin or wave_data.min(), xmax or wave_data.max()])
        #fwhm_um = mean_wavelength / 299792.458 * fwhm  # Convert km/s to μm
        fwhm_um = mean_wavelength / c.SPEED_OF_LIGHT_KMS * fwhm  # Convert km/s to μm
        sig = fwhm_um / 2.35482  # Convert FWHM to sigma
        sig_tol = (mean_wavelength / c.SPEED_OF_LIGHT_KMS * fwhmtolerance) / 2.35482
        
        # Calculate initial amplitude estimate
        # Total integrated flux divided by number of lines
        total_flux = np.trapz(flux_data, wave_data)
        #total_flux = np.trapz(flux_data, wave_data)
        garea_fg = total_flux / len(line_centers) * 1e11  # Scaling factor
        
        # Set up parameter bounds based on tolerances
        fwhm_vary = sig_tol > 0
        centr_vary = centrtolerance > 0
        
        # Create composite model exactly
        model = None
        params = Parameters()
        
        for i in range(n_components):
            prefix = f'g{i+1}_'  # Use 1-based indexing
            if model is None:
                current_model = GaussianModel(prefix=prefix)
                model = current_model
                params = current_model.guess(flux_data, x=wave_data)
            else:
                current_model = GaussianModel(prefix=prefix)
                model += current_model
            
            # Use detected line centers
            center = line_centers[i] if i < len(line_centers) else mean_wavelength

            center_dict = {
                'value': center,
                'vary': centr_vary,
                'min': center - centrtolerance,
                'max': center + centrtolerance
            }

            sigma_dict = {
                'value': sig,
                'vary': fwhm_vary,
                'min': sig - sig_tol if sig_tol > 0 else sig * 0.1,
                'max': sig + sig_tol if sig_tol > 0 else sig * 10
            }

            amplitude_dict = {
                'value': garea_fg,
                'min': 0
            }

            params.update(current_model.make_params(center = center_dict, sigma = sigma_dict, amplitude = amplitude_dict))

        # Perform fit with
        result = model.fit(flux_data, params, x=wave_data, weights=weights, 
                          method='leastsq', nan_policy='omit')
        
        print(result.fit_report())

        # Generate fitted curve
        fitted_wave = np.linspace(wave_data.min(), wave_data.max(), 1000)
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params
        
        return result, fitted_wave, fitted_flux
    
    def _estimate_components_from_user_selection(self, wave_data, flux_data, xmin=None, xmax=None):
        """
        Use pre-selected line positions
        """
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
                
                if line_data is not None:
                    line_threshold = 0.03 # default
                    try:
                        if hasattr(self.islat, 'user_settings') and self.islat.user_settings:
                            line_threshold = self.islat.user_settings.get('line_threshold', 0.03)
                    except:
                        pass
                    
                    # Filter lines above threshold
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
                dist=self.islat.active_molecule.dist,
                fwhm=self.islat.active_molecule.fwhm,
                min_lamb=self.islat.molecules_dict.wavelength_range[0],
                max_lamb=self.islat.molecules_dict.wavelength_range[1],
                pix_per_fwhm=10,
                intrinsic_line_width=self.islat.active_molecule.broad,
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

    def _extract_component_parameters(self, params, prefix='', rest_wavelength=None, sig_det_lim=2):
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

    def extract_line_parameters(self, rest_wavelength=None, sig_det_lim=2):
        """
        Extract line parameters from the last fitting result.
        
        Parameters
        ----------
        rest_wavelength : float, optional
            Rest wavelength for Doppler shift calculation
        sig_det_lim : float, optional
            Detection significance limit (default: 2)
        
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
            line_params.update(self._extract_component_parameters(params, '', rest_wavelength, sig_det_lim))
            
        # Extract parameters for multi-component fits
        component_idx = 0
        
        while f'g{component_idx+1}_center' in params:
            prefix = f'g{component_idx+1}_'
            component_params = self._extract_component_parameters(params, prefix, rest_wavelength, sig_det_lim)
            line_params[f'component_{component_idx}'] = component_params
            component_idx += 1
        
        return line_params
    
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
        #line_sn = flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0
        line_sn = np.round(flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0, decimals=1)
        #line_det = abs(flux_data_integral) > sig_det_lim * err_data_integral

        if np.absolute(flux_data_integral) > sig_det_lim * err_data_integral:
            line_det = True
        else:
            line_det = False

        # Prepare base result dictionary with data measurements
        result_entry = {
            'species': line_info.get('species', 'Unknown'),
            'lev_up': line_info.get('lev_up', ''),
            'lev_low': line_info.get('lev_low', ''),
            'lam': line_info.get('lam', rest_wavelength),
            #'tau': line_info.get('tau', 0.0),
            #'intens': line_info.get('intens', 0.0),
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
        if fit_result and fit_result.success:
            # Extract fit parameters using the helper method
            component_params = self._extract_component_parameters(fit_result.params, '', rest_wavelength, sig_det_lim)
            
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
                'Red-chisq': np.round(fit_result.redchi, decimals=2)
            })
            
            # Update iSLAT flux values if fit is good and detected
            if fit_det:
                result_entry['Flux_islat'] = np.float64(f'{gauss_area:.{3}e}')
                result_entry['Err_islat'] = np.float64(f'{gauss_area_err:.{3}e}')
        else:
            # Fit failed - fill with NaN values but keep data measurements
            result_entry.update({
                #'Fit_SN': np.round(flux_data_integral / err_data_integral if err_data_integral > 0 else 0.0, decimals=1),
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