from typing import Optional, Dict, Any, Tuple, Union, Callable
import numpy as np
from collections import defaultdict
import threading

# Lazy imports with thread safety
_spectrum_module = None
_intensity_module = None
_import_lock = threading.Lock()

def _get_spectrum_module():
    """Thread-safe lazy import of spectrum module"""
    global _spectrum_module
    if _spectrum_module is None:
        with _import_lock:
            if _spectrum_module is None:  # Double-check pattern
                from .Spectrum import Spectrum
                _spectrum_module = Spectrum
    return _spectrum_module

def _get_intensity_module():
    """Thread-safe lazy import of intensity module"""
    global _intensity_module
    if _intensity_module is None:
        with _import_lock:
            if _intensity_module is None:  # Double-check pattern
                from .Intensity import Intensity
                _intensity_module = Intensity
    return _intensity_module

import iSLAT.Constants as c
from .MoleculeLineList import MoleculeLineList

from iSLAT.Modules.FileHandling import absolute_data_files_path

class Molecule:
    """
    Optimized Molecule class with enhanced caching and performance improvements.
    """
    __slots__ = (
        'name', 'filepath', 'displaylabel', 'color', '_is_visible', '_rv_shift',
        'user_save_data', 'hitran_data', 'initial_molecule_parameters',
        'lines', 'intensity', 'spectrum',
        '_temp', '_radius', '_n_mol', '_distance', '_fwhm', '_broad',
        '_temp_val', '_radius_val', '_n_mol_val', '_distance_val', '_fwhm_val', '_broad_val',
        '_lines_filepath',
        't_kin', 'scale_exponent', 'scale_number', 'radius_init', 'n_mol_init',
        '_wavelength_range', '_model_pixel_res', '_model_line_width',
        '_intensity_cache', '_spectrum_cache', '_flux_cache',
        '_param_hash_cache', '_dirty_flags', '_cache_stats'
    )
    
    _molecule_parameter_change_callbacks = []
    _shared_calculation_cache = {}
    _cache_lock = threading.Lock()
    
    INTENSITY_AFFECTING_PARAMS = {'temp', 'n_mol', 'broad', 'rv_shift', 'wavelength_range'}
    SPECTRUM_AFFECTING_PARAMS = {'radius', 'distance', 'fwhm', 'rv_shift', 'wavelength_range', 'model_pixel_res'}
    FLUX_AFFECTING_PARAMS = INTENSITY_AFFECTING_PARAMS | SPECTRUM_AFFECTING_PARAMS
    
    @classmethod
    def add_molecule_parameter_change_callback(cls, callback):
        """Add a callback function to be called when individual molecule parameters change"""
        cls._molecule_parameter_change_callbacks.append(callback)
    
    @classmethod
    def remove_molecule_parameter_change_callback(cls, callback):
        """Remove a callback function for molecule parameter changes"""
        if callback in cls._molecule_parameter_change_callbacks:
            cls._molecule_parameter_change_callbacks.remove(callback)
    
    @classmethod
    def _notify_molecule_parameter_change(cls, molecule_name, parameter_name, old_value, new_value):
        """Notify all callbacks that a molecule parameter has changed"""
        for callback in cls._molecule_parameter_change_callbacks:
            try:
                callback(molecule_name, parameter_name, old_value, new_value)
            except Exception as e:
                print(f"Error in molecule parameter change callback: {e}")
    
    def _notify_my_parameter_change(self, parameter_name, old_value, new_value):
        if old_value == new_value:
            return
        
        self._invalidate_caches_for_parameter(parameter_name)
        self.__class__._notify_molecule_parameter_change(self.name, parameter_name, old_value, new_value)
    
    def _invalidate_caches_for_parameter(self, parameter_name):
        if parameter_name in self.INTENSITY_AFFECTING_PARAMS:
            self._dirty_flags['intensity'] = True
            self._dirty_flags['spectrum'] = True
            self._dirty_flags['flux'] = True
        elif parameter_name in self.SPECTRUM_AFFECTING_PARAMS:
            self._dirty_flags['spectrum'] = True
            self._dirty_flags['flux'] = True
        else:
            self._dirty_flags['flux'] = True
        
        if parameter_name in self.FLUX_AFFECTING_PARAMS:
            self._flux_cache.clear()
    
    def __init__(self, **kwargs):
        self._initialize_caching_system()
        
        if 'hitran_data' in kwargs and kwargs['hitran_data'] is not None:
            #print("Generating new molecule from default parameters.")
            self.user_save_data = None
            self.hitran_data = kwargs['hitran_data']
        elif 'user_save_data' in kwargs and kwargs['user_save_data'] is not None:
            #print("Generating new molecule from user saved data.")
            self.user_save_data = kwargs['user_save_data']
            self.hitran_data = None
        else:
            self.user_save_data = None
            self.hitran_data = None

        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', {})

        # Load parameters from appropriate source
        if self.user_save_data is not None:
            self._load_from_user_save_data(kwargs)
        else:
            self._load_from_kwargs(kwargs)

        self.lines = None
        #self._lines_filepath = absolute_data_files_path / self.filepath
        
        # Calculate derived parameters
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        
        # Set final parameter values
        self._temp = float(getattr(self, '_temp_val', None) or self.t_kin)
        self._radius = float(getattr(self, '_radius_val', None) or self.radius_init)
        self._n_mol = float(getattr(self, '_n_mol_val', None) or self.n_mol_init)
        self._distance = float(getattr(self, '_distance_val', None) or c.DEFAULT_DISTANCE)
        self._fwhm = float(getattr(self, '_fwhm_val', None) or c.DEFAULT_FWHM)
        self._broad = float(getattr(self, '_broad_val', 1.0) or c.INTRINSIC_LINE_WIDTH)
        self._rv_shift = float(getattr(self, '_rv_shift', 0.0) or c.DEFAULT_STELLAR_RV)

        self._wavelength_range = kwargs.get('wavelength_range', c.WAVELENGTH_RANGE)
        self._model_pixel_res = kwargs.get('model_pixel_res', c.MODEL_PIXEL_RESOLUTION)
        self._model_line_width = kwargs.get('model_line_width', c.MODEL_LINE_WIDTH)

        self.intensity = None
        self.spectrum = None
        
        self._calculate_initial_parameter_hashes()
    
    def _initialize_caching_system(self):
        self._intensity_cache = {'data': None, 'hash': None}
        self._spectrum_cache = {'data': None, 'hash': None}
        self._flux_cache = {}
        self._param_hash_cache = {}
        self._dirty_flags = {
            'intensity': True,
            'spectrum': True, 
            'flux': True
        }
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0
        }
    
    def _calculate_initial_parameter_hashes(self):
        self._param_hash_cache = {
            'intensity': self._compute_intensity_hash(),
            'spectrum': self._compute_spectrum_hash(),
            'full': self._compute_full_parameter_hash()
        }
    
    def _compute_intensity_hash(self):
        return hash((self._temp, self._n_mol, self._broad))
    
    def _compute_spectrum_hash(self):
        # Get all spectrum-affecting parameters and create hash tuple
        param_values = []
        for param in self.SPECTRUM_AFFECTING_PARAMS:
            if param == 'wavelength_range':
                param_values.append(tuple(self._wavelength_range))
            else:
                param_values.append(getattr(self, f'_{param}'))
        
        # Include intensity hash for dependencies
        param_values.append(self._compute_intensity_hash())
        
        return hash(tuple(param_values))

    def _compute_full_parameter_hash(self):
        return hash((self._compute_spectrum_hash()))

    def _load_from_user_save_data(self, kwargs):
        """Load parameters from user save data"""
        usd = self.user_save_data
        self.name = usd.get('Molecule Name', kwargs.get('name', 'Unknown Molecule'))
        self.filepath = usd.get('File Path', kwargs.get('filepath', None))
        self.displaylabel = usd.get('Molecule Label', self.name)
        self._temp_val = usd.get('Temp', kwargs.get('temp', None))
        self._radius_val = usd.get('Rad', kwargs.get('radius', None))
        self._n_mol_val = usd.get('N_Mol', kwargs.get('n_mol', None))
        self.color = usd.get('Color', kwargs.get('color', None))
        self._is_visible = usd.get('Vis', kwargs.get('is_visible', True))

        # Get instance values from user save data or kwargs
        self._distance_val = usd.get('Dist', kwargs.get('distance', c.DEFAULT_DISTANCE))
        self._fwhm_val = usd.get('FWHM', kwargs.get('fwhm', c.DEFAULT_FWHM))
        self._broad_val = usd.get('Broad', kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH))
        self._rv_shift = usd.get('RV Shift', kwargs.get('rv_shift', c.DEFAULT_STELLAR_RV))

        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)

    def _load_from_kwargs(self, kwargs):
        """Load parameters from kwargs"""
        self.name = kwargs.get('name', kwargs.get('displaylabel', kwargs.get('filepath', 'Unknown Molecule')))
        self.filepath = kwargs.get('filepath', (self.hitran_data if hasattr(self, 'hitran_data') else None))
        self.displaylabel = kwargs.get('displaylabel', kwargs.get('name', 'Unknown Molecule'))
        self._temp_val = kwargs.get('temp', self.initial_molecule_parameters.get('t_kin', 300.0))
        self._radius_val = kwargs.get('radius', self.initial_molecule_parameters.get('radius_init', 1.0))
        self._n_mol_val = kwargs.get('n_mol', self.initial_molecule_parameters.get('n_mol', None))
        self.color = kwargs.get('color', None)
        self._is_visible = kwargs.get('is_visible', True)

        # Get instance values from kwargs or defaults
        self._distance_val = kwargs.get('Dist', kwargs.get('distance', c.DEFAULT_DISTANCE))
        self._fwhm_val = kwargs.get('FWHM', kwargs.get('fwhm', c.DEFAULT_FWHM))
        self._broad_val = kwargs.get('Broad', kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH))
        self._rv_shift = kwargs.get('rv_shift', kwargs.get('RV Shift', c.DEFAULT_STELLAR_RV))

        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)
        
    def _ensure_lines_loaded(self):
        if self.lines is None:
            if self.filepath:
                #print("Loading lines from filepath:", self._lines_filepath)
                self.lines = MoleculeLineList(molecule_id=self.name, filename=self.filepath)
            else:
                print("Creating empty line list")
                self.lines = MoleculeLineList(molecule_id=self.name)
    
    def _ensure_intensity_calculated(self):
        if self._dirty_flags['intensity'] or self._intensity_cache['data'] is None:
            self._calculate_intensity_with_caching()
    
    def _ensure_spectrum_calculated(self):
        self._ensure_intensity_calculated()
        if self._dirty_flags['spectrum'] or self._spectrum_cache['data'] is None:
            self._calculate_spectrum_with_caching()
    
    def _calculate_intensity_with_caching(self):
        current_hash = self._compute_intensity_hash()
        
        if (self._intensity_cache['hash'] == current_hash and 
            self._intensity_cache['data'] is not None):
            self._cache_stats['hits'] += 1
            return
        
        self._ensure_lines_loaded()
        
        if self.intensity is None:
            Intensity = _get_intensity_module()
            self.intensity = Intensity(self.lines)
        
        print(f"Calculating intensity for {self.name}: T={self._temp}K, N_mol={self._n_mol:.2e}, dv={self._broad}")
        
        self.intensity.calc_intensity(
            t_kin=self._temp,
            n_mol=self._n_mol,
            dv=self._broad
        )

        intensity_data = {
            'intensity_array': self.intensity._intensity.copy() if self.intensity._intensity is not None else None,
            'tau_array': self.intensity._tau.copy() if self.intensity._tau is not None else None
        }
        
        self._intensity_cache = {
            'data': intensity_data,
            'hash': current_hash
        }
        
        with self._cache_lock:
            if len(self._shared_calculation_cache) > 100:
                oldest_keys = list(self._shared_calculation_cache.keys())[:10]
                for key in oldest_keys:
                    del self._shared_calculation_cache[key]
        
        self._dirty_flags['intensity'] = False
        self._param_hash_cache['intensity'] = current_hash
        self._cache_stats['misses'] += 1
    
    def _calculate_spectrum_with_caching(self):
        current_hash = self._compute_spectrum_hash()
        
        if (self._spectrum_cache['hash'] == current_hash and 
            self._spectrum_cache['data'] is not None):
            self._cache_stats['hits'] += 1
            return
            
        self._ensure_intensity_calculated()
        
        # Always recreate spectrum when parameters change
        Spectrum = _get_spectrum_module()
        mean_wavelength = (self.wavelength_range[0] + self.wavelength_range[1]) / 2.0
        delta_lambda = mean_wavelength * (self._fwhm / c.SPEED_OF_LIGHT_KMS)
        spectral_resolution = mean_wavelength / delta_lambda if delta_lambda > 0 else self.model_line_width
        
        # Use consistent wavelength range for all molecules to ensure identical grids
        if hasattr(self, '_wavelength_range') and self._wavelength_range is not None:
            global_min, global_max = self._wavelength_range
            print(f"getting wavelength range: {self._wavelength_range}")
            # Use the exact global range
            spectrum_lam_min = global_min
            spectrum_lam_max = global_max
        else:
            # Fallback to standard range for backward compatibility
            spectrum_lam_min = self.wavelength_range[0]
            spectrum_lam_max = self.wavelength_range[1]
        
        self.spectrum = Spectrum(
            lam_min=spectrum_lam_min,
            lam_max=spectrum_lam_max,
            dlambda=self._model_pixel_res,
            R=spectral_resolution,
            distance=self._distance
        )
        
        area = np.pi * (self._radius) ** 2  # Area in AU^2 as expected by add_intensity
        self.spectrum.add_intensity(self.intensity, area)
        
        self._spectrum_cache = {
            'data': self.spectrum,
            'hash': current_hash
        }
        
        self._dirty_flags['spectrum'] = False
        self._param_hash_cache['spectrum'] = current_hash
        self._cache_stats['misses'] += 1
    
    def calculate_intensity(self):
        self._ensure_intensity_calculated()
    
    def get_parameter_hash(self, cache_type='full'):
        """Get parameter hash for given cache type"""
        if cache_type in self._param_hash_cache:
            return self._param_hash_cache[cache_type]
        return self._compute_full_parameter_hash()

    def _clear_specific_cache(self, cache_type):
        """Clear specific cache and update stats"""
        if cache_type == 'flux':
            self._flux_cache.clear()
            self._dirty_flags['flux'] = True
        elif cache_type == 'intensity':
            self._intensity_cache = {'data': None, 'hash': None}
            self._dirty_flags['intensity'] = True
            self._dirty_flags['spectrum'] = True  # Spectrum depends on intensity
            self._dirty_flags['flux'] = True      # Flux depends on spectrum
        elif cache_type == 'spectrum':
            self._spectrum_cache = {'data': None, 'hash': None}
            self._dirty_flags['spectrum'] = True
            self._dirty_flags['flux'] = True      # Flux depends on spectrum

    def get_flux(self, wavelength_array=None, return_wavelengths=False, interpolate_to_input=False):
        """Get flux data with RV shift properly applied
        
        The RV shift works by:
        1. Calculate spectrum on the unshifted wavelength grid
        2. If RV shift != 0, interpolate the flux as if it came from a shifted source
           back to the unshifted wavelength grid
        3. This ensures all molecules return flux on the same wavelength grid
        
        Parameters
        ----------
        wavelength_array : np.ndarray, optional
            Input wavelength array for additional interpolation
        return_wavelengths : bool, default False
            If True, return tuple of (wavelengths, flux)
        interpolate_to_input : bool, default False
            If True and wavelength_array is provided, interpolate to match input grid
            If False, return spectrum's native grid (with RV shift applied)
            
        Returns
        -------
        np.ndarray or tuple
            Flux array or (wavelengths, flux) tuple
        """
        if wavelength_array is not None:
            try:
                cache_key = hash(wavelength_array.tobytes()) if hasattr(wavelength_array, 'tobytes') else str(wavelength_array)
            except (TypeError, ValueError):
                cache_key = f"fallback_{id(wavelength_array)}"
        else:
            cache_key = "no_wavelength_input"
        
        # Add interpolation flag to cache key
        cache_key = f"{cache_key}_interp_{interpolate_to_input}"
        
        current_param_hash = self._compute_full_parameter_hash()
        cache_entry = self._flux_cache.get(cache_key)
        
        # Check both dirty flags and validate cache entry for cache validity
        if (not self._dirty_flags['flux'] and 
            self._validate_flux_cache_entry(cache_entry, current_param_hash)):
            self._cache_stats['hits'] += 1
            if return_wavelengths:
                # Return copies to prevent accidental modification of cached data
                return cache_entry['wavelengths'].copy(), cache_entry['flux'].copy()
            return cache_entry['flux'].copy()
        
        # Ensure we have a valid spectrum
        self._ensure_spectrum_calculated()
        
        if self.spectrum is None:
            print(f"Warning: No spectrum available for molecule {self.name}")
            empty_array = np.array([])
            if return_wavelengths:
                return empty_array, empty_array
            return empty_array
        
        # Get the spectrum data - this is the native unshifted grid
        lam_grid = self.spectrum._lamgrid
        flux_grid = self.spectrum.flux_jy  # Use Jy units for consistency with observed data
        
        if lam_grid is None or flux_grid is None:
            print(f"Warning: Invalid spectrum data for molecule {self.name}")
            empty_array = np.array([])
            if return_wavelengths:
                return empty_array, empty_array
            return empty_array
        
        # Apply RV shift correction to get flux on the unshifted grid
        # Create the shifted wavelength grid where the flux "came from"
        if self._rv_shift != 0:
            # Only print debug info for significant RV shifts or when debugging is enabled
            if abs(self._rv_shift) > 0.1:  # Only log if RV shift > 0.1 km/s
                print(f"Applying RV shift of {self._rv_shift} km/s for molecule {self.name}")
            
            shifted_source_lam_grid = lam_grid + (lam_grid / c.SPEED_OF_LIGHT_KMS * self._rv_shift)

            # Interpolate the flux from the shifted source back to the unshifted grid
            # This simulates observing a shifted source and correcting it back to rest frame
            rv_corrected_flux = np.interp(lam_grid, shifted_source_lam_grid, flux_grid, left=0, right=0)
        else:
            rv_corrected_flux = flux_grid

        # Now decide on final output grid
        if interpolate_to_input and wavelength_array is not None:
            # Interpolate the RV-corrected flux to the input wavelength array
            interpolated_flux = np.interp(wavelength_array, lam_grid, rv_corrected_flux, left=0, right=0)
            
            result_wavelengths = wavelength_array.copy()  # Copy to prevent modification
            result_flux = interpolated_flux
        else:
            # Return on the native grid with RV shift applied
            result_wavelengths = lam_grid.copy()  # Copy to prevent modification
            result_flux = rv_corrected_flux
        
        # Cache the result (store copies to prevent modification)
        self._flux_cache[cache_key] = {
            'wavelengths': result_wavelengths.copy(),
            'flux': result_flux.copy(),
            'param_hash': current_param_hash
        }
        
        flux_cache_limit = 12
        flux_oldest_amount = 4

        # Limit cache size
        if len(self._flux_cache) > flux_cache_limit:
            oldest_keys = list(self._flux_cache.keys())[:flux_oldest_amount]
            for key in oldest_keys:
                del self._flux_cache[key]
        
        # Mark flux as clean since we just calculated it
        self._dirty_flags['flux'] = False
        self._cache_stats['misses'] += 1
        
        if return_wavelengths:
            return result_wavelengths, result_flux
        return result_flux
    
    # Define properties using a factory function approach
    def _make_property(attr_name, converter=float, special_setter=None):
        """Factory function to create properties"""
        private_attr = f'_{attr_name}'
        
        def getter(self):
            return getattr(self, private_attr)
        
        def setter(self, value):
            if converter:
                value = converter(value)
            
            old_value = getattr(self, private_attr)
            setattr(self, private_attr, value)
            
            if special_setter:
                special_setter(self, value)
            
            self._notify_my_parameter_change(attr_name, old_value, value)
        
        return property(getter, setter)

    # Standard numeric parameters
    temp = _make_property('temp', converter=float, special_setter=lambda self, value: setattr(self, 't_kin', value))
    radius = _make_property('radius', converter=float)
    distance = _make_property('distance', converter=float)
    fwhm = _make_property('fwhm', converter=float, special_setter=lambda self, value: setattr(self, 'spectrum', None))
    model_pixel_res = _make_property('model_pixel_res', converter=float)
    broad = _make_property('broad', converter=float)
    rv_shift = _make_property('rv_shift', converter=float)
    n_mol = _make_property('n_mol', converter=float)
    model_line_width = _make_property('model_line_width', converter=float)
    
    # Special case properties
    @property
    def wavelength_range(self):
        return self._wavelength_range

    @wavelength_range.setter
    def wavelength_range(self, value):
        old_value = self._wavelength_range
        self._wavelength_range = value
        self._notify_my_parameter_change('wavelength_range', old_value, self._wavelength_range)
    
    @property
    def is_visible(self):
        if isinstance(self._is_visible, str):
            # Convert string representations to proper boolean
            return self._is_visible.lower() in ('true', '1', 'yes', 'on')
        return bool(self._is_visible)
    
    @is_visible.setter
    def is_visible(self, value):
        old_value = self._is_visible
        if isinstance(value, str):
            self._is_visible = value.lower() in ('true', '1', 'yes', 'on')
        else:
            self._is_visible = bool(value)
        self._notify_my_parameter_change('is_visible', old_value, self._is_visible)

    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], skip_notification: bool = False):
        if not parameter_dict:
            return
            
        # Make a copy to avoid modifying the input dict
        params_to_update = parameter_dict.copy()
        old_values = {}
        affected_params = set()
        
        # Batch cache invalidation flags
        needs_intensity_invalidation = False
        needs_spectrum_invalidation = False
        needs_flux_invalidation = False
        
        # Properties with special converters - default to float for all others
        special_converters = {
            'is_visible': lambda x: x.lower() in ('true', '1', 'yes', 'on') if isinstance(x, str) else bool(x)
        }
        
        # Properties with special setters
        special_setters = {
            'temp': lambda self, value: setattr(self, 't_kin', value),
            'fwhm': lambda self, value: setattr(self, 'spectrum', None)
        }
        
        # Batch process parameters with type conversion
        for param_name, value in params_to_update.items():
            # Apply type conversion - use special converter if available, otherwise float
            try:
                if param_name in special_converters:
                    value = special_converters[param_name](value)
                else:
                    value = float(value)
            except (ValueError, TypeError):
                continue  # Skip invalid values
            
            private_attr = f'_{param_name}'
            
            # Try private attribute first (most common case)
            if hasattr(self, private_attr):
                old_values[param_name] = getattr(self, private_attr)
                setattr(self, private_attr, value)
                affected_params.add(param_name)
                
                # Apply special setters
                if param_name in special_setters:
                    special_setters[param_name](self, value)
                
                # Determine cache invalidation needs
                if param_name in self.INTENSITY_AFFECTING_PARAMS:
                    needs_intensity_invalidation = True
                    needs_spectrum_invalidation = True
                    needs_flux_invalidation = True
                elif param_name in self.SPECTRUM_AFFECTING_PARAMS:
                    needs_spectrum_invalidation = True
                    needs_flux_invalidation = True
                else:
                    needs_flux_invalidation = True
                    
            # Fallback to public attribute
            elif hasattr(self, param_name):
                old_values[param_name] = getattr(self, param_name)
                setattr(self, param_name, value)
                affected_params.add(param_name)
                needs_flux_invalidation = True  # Conservative invalidation for unknown params
        
        # Batch cache invalidation - more efficient than individual calls
        if needs_intensity_invalidation:
            self._dirty_flags['intensity'] = True
        if needs_spectrum_invalidation:
            self._dirty_flags['spectrum'] = True
        if needs_flux_invalidation:
            self._dirty_flags['flux'] = True
            self._flux_cache.clear()
        
        # Batch notifications if required
        if not skip_notification and affected_params:
            params_to_notify = affected_params            
            for param_name in params_to_notify:
                old_value = old_values.get(param_name)
                # Fast attribute access
                new_value = getattr(self, f'_{param_name}', None) or getattr(self, param_name, None)
                if old_value != new_value:
                    self._notify_my_parameter_change(param_name, old_value, new_value)
    
    def force_recalculate(self):
        self.clear_all_caches()
        self._ensure_intensity_calculated()
        self._ensure_spectrum_calculated()
    
    def get_cache_stats(self):
        return self._cache_stats.copy()
    
    def clear_all_caches(self):
        self._intensity_cache = {'data': None, 'hash': None}
        self._spectrum_cache = {'data': None, 'hash': None}
        self._flux_cache.clear()
        self._dirty_flags = {'intensity': True, 'spectrum': True, 'flux': True}
        self._cache_stats['invalidations'] += 1
    
    def is_cache_valid(self, cache_type='full'):
        if cache_type == 'intensity':
            return not self._dirty_flags['intensity'] and self._intensity_cache['data'] is not None
        elif cache_type == 'spectrum':
            return not self._dirty_flags['spectrum'] and self._spectrum_cache['data'] is not None
        elif cache_type == 'flux':
            return not self._dirty_flags['flux'] and len(self._flux_cache) > 0
        else:
            return (self.is_cache_valid('intensity') and 
                   self.is_cache_valid('spectrum') and 
                   self.is_cache_valid('flux'))
    
    def _validate_flux_cache_entry(self, cache_entry, expected_param_hash):
        """Validate that a flux cache entry is still valid"""
        if cache_entry is None:
            return False
        
        # Check parameter hash
        if cache_entry.get('param_hash') != expected_param_hash:
            return False
        
        # Validate that arrays exist and are not empty
        wavelengths = cache_entry.get('wavelengths')
        flux = cache_entry.get('flux')
        
        if (wavelengths is None or flux is None or 
            len(wavelengths) == 0 or len(flux) == 0):
            return False
        
        # Check that arrays have the same length
        if len(wavelengths) != len(flux):
            return False
        
        return True

    def __str__(self):
        return f"Molecule(name={self.name}, temp={self._temp}, radius={self._radius}, n_mol={self._n_mol})"