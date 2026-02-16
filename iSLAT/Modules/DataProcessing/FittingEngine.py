"""
FittingEngine - LMFIT operations and model fitting functionality
"""

import numpy as np
from lmfit.models import GaussianModel
from lmfit import Parameters
from iSLAT.Modules.DataProcessing.Slabfit import SlabFit
import iSLAT.Constants as c
import sys

class FittingEngine:
    """
    Centralized fitting engine for all LMFIT operations and model fitting.
    
    This class provides a unified interface for various fitting operations
    including line fitting, deblending, and slab model fitting.
    """
    
    # Class-level setting to control verbose fit output
    VERBOSE_FIT_OUTPUT: bool = False
    
    def __init__(self):
        """
        Initialize the fitting engine.
        """
        self.last_fit_result = None
        self.last_fit_params = None
    
    def fit_gaussian_line(self, wave_data, flux_data, xmin=None, xmax=None, 
                         initial_guess=None, deblend=False, err_data=None,
                         wave_data_full=None, err_data_full=None,
                         user_settings=None, active_molecule_fwhm=None,
                         lines_with_intensity=None, line_threshold=None):
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
        err_data : array_like, optional
            Error data (filtered to same range as wave/flux) for single Gaussian
        wave_data_full : array_like, optional
            Full (unfiltered) wavelength array. Required for deblend=True.
        err_data_full : array_like, optional
            Full (unfiltered) error array. Required for deblend=True.
        user_settings : dict, optional
            User settings dict with 'centrtolerance', 'fwhmtolerance', 'line_threshold'.
            Required for deblend=True.
        active_molecule_fwhm : float, optional
            Active molecule FWHM in km/s. Required for deblend=True.
        lines_with_intensity : list, optional
            List of (MoleculeLine, intensity, extra) tuples for component estimation.
            Required for deblend=True.
        line_threshold : float, optional
            Threshold fraction for filtering weak lines (default from user_settings or 0.03).
            
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
            
        #if len(fit_wave) < 3:
        #    raise ValueError("Insufficient data points for fitting")
        
        # Set xmin/xmax from data if not provided (for multi-gaussian detection)
        if xmin is None:
            xmin = fit_wave.min()
        if xmax is None:
            xmax = fit_wave.max()
        
        if deblend:
            return self._fit_multi_gaussian(
                fit_wave, fit_flux, initial_guess, xmin, xmax,
                wave_data_full=wave_data_full, err_data_full=err_data_full,
                user_settings=user_settings, active_molecule_fwhm=active_molecule_fwhm,
                lines_with_intensity=lines_with_intensity, line_threshold=line_threshold
            )
        else:
            return self._fit_single_gaussian(fit_wave, fit_flux, initial_guess, xmin, xmax, err_data)

    def _fit_single_gaussian(self, wave_data, flux_data, initial_guess=None, xmin=None, xmax=None, err_data=None):
        """Fit a single Gaussian component."""
        # Data is already filtered to the fit range by the caller
        x_fit = wave_data
        flux_fit = flux_data
        calc_err_data = err_data #if err_data is not None else self.islat.err_data
        
        try:
            # Use gaussian model from LMFIT
            model = GaussianModel()
            
            # Get initial guess for parameters (let LMFIT do the guessing)
            params = model.guess(flux_fit, x=x_fit)
            
            # Get error data for weights if available
            weights = None
            if calc_err_data is not None:
                # Need to get error data for the same range
                if xmin is not None and xmax is not None:
                    #err_mask = (wave_data >= xmin) & (wave_data <= xmax)
                    err_fit = calc_err_data#[err_mask]
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
            
            if self.VERBOSE_FIT_OUTPUT:
                sys.stdout.write(result.fit_report() + '\n')

            # Generate fitted curve on original wavelength grid
            fitted_wave = wave_data
            fitted_flux = result.eval(x=fitted_wave)
            
            self.last_fit_result = result
            self.last_fit_params = result.params
            
            return result, fitted_wave, fitted_flux
        except Exception as e:
            if self.VERBOSE_FIT_OUTPUT:
                print(f"Error during single Gaussian fit: {e}")
            return None, None, None

    def _fit_multi_gaussian(self, wave_data, flux_data, initial_guess=None, xmin=None, xmax=None,
                            wave_data_full=None, err_data_full=None,
                            user_settings=None, active_molecule_fwhm=None,
                            lines_with_intensity=None, line_threshold=None):
        """Fit multiple Gaussian components for deblending.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data (filtered to fit range)
        flux_data : array_like
            Flux data (filtered to fit range)
        initial_guess : dict, optional
            Initial parameter guesses
        xmin, xmax : float, optional
            Wavelength range boundaries
        wave_data_full : array_like, optional
            Full (unfiltered) wavelength array for error masking
        err_data_full : array_like, optional
            Full (unfiltered) error array
        user_settings : dict, optional
            User settings with 'centrtolerance', 'fwhmtolerance', 'line_threshold'
        active_molecule_fwhm : float, optional
            Active molecule FWHM in km/s
        lines_with_intensity : list, optional
            List of (MoleculeLine, intensity, extra) tuples
        line_threshold : float, optional
            Threshold fraction for filtering weak lines
        """
        # Estimate number of components and line centers based on detection strategy
        n_components, line_centers = self._estimate_components_from_user_selection(
            wave_data, flux_data, xmin, xmax,
            lines_with_intensity=lines_with_intensity,
            line_threshold=line_threshold
        )
        print(f"Detected {n_components} components")
        
        if n_components == 1:
            # If only one component detected, use single Gaussian fit
            return self._fit_single_gaussian(wave_data, flux_data, initial_guess)
        
        # Get error data for weights
        weights = None
        if err_data_full is not None and wave_data_full is not None:
            if xmin is not None and xmax is not None:
                err_mask = (wave_data_full >= xmin) & (wave_data_full <= xmax)
                err_fit = err_data_full[err_mask]
                if len(err_fit) == len(flux_data) and len(err_fit) > 0:
                    # Avoid division by zero - replace zero or negative errors with a small value
                    max_err = np.max(err_fit)
                    err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
                    weights = 1.0 / err_fit_safe
        
        if weights is None:
            err_fit_safe = np.where(err_fit <= 0, max_err * 0.01, err_fit)
            weights = 1.0 / err_fit_safe
        
        # Get tolerance settings from user_settings
        if user_settings is None:
            user_settings = {}
        centrtolerance = user_settings.get('centrtolerance', 0.0001)
        fwhmtolerance = user_settings.get('fwhmtolerance', 5)
        
        # Get FWHM from the active molecule (can be updated at runtime)
        fwhm = active_molecule_fwhm if active_molecule_fwhm is not None else 10.0  # km/s default
        print(f"Using FWHM: {fwhm} km/s")
        mean_wavelength = np.mean([xmin or wave_data.min(), xmax or wave_data.max()])
        #fwhm_um = mean_wavelength / 299792.458 * fwhm  # Convert km/s to μm
        fwhm_um = mean_wavelength / c.SPEED_OF_LIGHT_KMS * fwhm  # Convert km/s to μm
        sig = fwhm_um / 2.35482  # Convert FWHM to sigma
        sig_tol = (mean_wavelength / c.SPEED_OF_LIGHT_KMS * fwhmtolerance) / 2.35482
        
        # Calculate initial amplitude estimate
        # Total integrated flux divided by number of lines
        total_flux = np.trapezoid(flux_data, wave_data)
        #total_flux = np.trapezoid(flux_data, wave_data)
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
        
        if self.VERBOSE_FIT_OUTPUT:
            sys.stdout.write(result.fit_report() + '\n')
        self.fit_results_summary = result.summary()
        
        # Generate fitted curve from x min and x max and wave data
        fitted_wave = np.linspace(xmin, xmax, 1000)
        #fitted_wave = wave_data
        fitted_flux = result.eval(x=fitted_wave)
        
        self.last_fit_result = result
        self.last_fit_params = result.params

        wave_mask = (wave_data >= xmin) & (wave_data <= xmax)
        self.fit_results_components = result.eval_components(x=wave_data[wave_mask])

        return result, fitted_wave, fitted_flux
    
    def get_fit_results_summary(self):
        """Return a summary string of the last fit results."""
        if not hasattr(self, 'fit_results_summary') or self.fit_results_summary is None:
            print("No fit summary available.")
            return
        return self.fit_results_summary
    
    def get_fit_results_components(self):
        """Return the evaluated components of the last multi-component fit."""
        if not hasattr(self, 'fit_results_components') or self.fit_results_components is None:
            print("No fit components available.")
            return
        return self.fit_results_components

    def _estimate_components_from_user_selection(self, wave_data, flux_data, xmin=None, xmax=None,
                                                  lines_with_intensity=None, line_threshold=None):
        """
        Use pre-selected line positions to estimate components.
        
        Parameters
        ----------
        wave_data : array_like
            Wavelength data
        flux_data : array_like
            Flux data
        xmin, xmax : float, optional
            Wavelength range boundaries
        lines_with_intensity : list, optional
            List of (MoleculeLine, intensity, extra) tuples from active molecule
        line_threshold : float, optional
            Threshold fraction for filtering weak lines (default: 0.03)
        """
        # Use molecular line data from the selected region (like onselect_lines['lam'])
        try:
            if lines_with_intensity is not None:
                if xmin is None:
                    xmin = wave_data.min()
                if xmax is None:
                    xmax = wave_data.max()
                
                # Convert to arrays
                line_data = None
                try:
                    line_centers = np.array([line.lam for line, _, _ in lines_with_intensity])
                    intensities = np.array([intensity for _, intensity, _ in lines_with_intensity])
                    line_data = {'lam': line_centers, 'intens': intensities}
                except Exception as e:
                    print(f"Warning: Could not process lines_with_intensity: {e}")
                
                if line_data is not None:
                    if line_threshold is None:
                        line_threshold = 0.03  # default
                    
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
                        start_temp=500, start_n_mol=1e17, start_radius=1.0,
                        mol_data=None, dist=None, fwhm=None,
                        wavelength_range=None, broad=None, data_field=None):
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
        mol_data : dict, optional
            Molecule info dict with 'file' key for the data path
        dist : float, optional
            Distance in pc
        fwhm : float, optional
            FWHM in km/s
        wavelength_range : tuple, optional
            (min_lamb, max_lamb) wavelength range
        broad : float, optional
            Intrinsic line width
        data_field : object, optional
            GUI data field widget
            
        Returns
        -------
        fit_result : dict
            Dictionary containing fitted parameters and statistics
        """
        try:
            if mol_data is None:
                raise ValueError(f"mol_data must be provided for molecule {molecule_name}")
            
            if wavelength_range is None:
                wavelength_range = (0, 100)  # fallback
            
            # Create slab fit instance
            slab_fitter = SlabFit(
                target=target_file,
                save_folder="EXAMPLE-data",
                mol=molecule_name,
                molpath=mol_data['file'],
                dist=dist,
                fwhm=fwhm,
                min_lamb=wavelength_range[0],
                max_lamb=wavelength_range[1],
                pix_per_fwhm=10,
                intrinsic_line_width=broad,
                cc=3e8,
                data_field=data_field
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
    
    def format_fit_results_for_csv(self, fit_result, flux_data_integral, err_data_integral,
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
            'e_low': line_info.get('e_low', 0.0),
            'g_up': line_info.get('g_up', 1.0),
            'g_low': line_info.get('g_low', 1.0),
            'xmin': xmin,
            'xmax': xmax,
        }

        result_entry.update({key: line_info[key] for key in line_info if key not in result_entry})  # Include all line info fields

        result_entry.update({
            'Flux_data': np.float64(f'{flux_data_integral:.{3}e}'),
            'Err_data': np.float64(f'{err_data_integral:.{3}e}'),
            'Line_SN': np.round(line_sn, decimals=1),
            'Line_det': bool(line_det),
            'Flux_islat': np.float64(f'{flux_data_integral:.{3}e}'),  # Default to data values
            'Err_islat': np.float64(f'{err_data_integral:.{3}e}')     # Will be overwritten if fit succeeds
        })

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
        
        return result_entry