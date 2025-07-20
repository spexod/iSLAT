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

class Molecule:
    """
    Optimized Molecule class with enhanced caching and performance improvements.
    """
    __slots__ = (
        'name', 'filepath', 'displaylabel', 'color', 'is_visible', 'stellar_rv',
        'user_save_data', 'hitran_data', 'initial_molecule_parameters',
        'lines', 'intensity', 'spectrum',
        '_temp', '_radius', '_n_mol', '_distance', '_fwhm', '_broad',
        '_temp_val', '_radius_val', '_n_mol_val', '_distance_val', '_fwhm_val', '_broad_val',
        '_lines_filepath',
        't_kin', 'scale_exponent', 'scale_number', 'radius_init', 'n_mol_init',
        'wavelength_range', 'model_pixel_res', 'model_line_width',
        'plot_lam', 'plot_flux',
        '_intensity_cache', '_spectrum_cache', '_flux_cache', '_wave_data_cache',
        '_param_hash_cache', '_dirty_flags', '_cache_stats'
    )
    
    _molecule_parameter_change_callbacks = []
    _shared_calculation_cache = {}
    _cache_lock = threading.Lock()
    
    INTENSITY_AFFECTING_PARAMS = {'temp', 'n_mol', 'broad'}
    SPECTRUM_AFFECTING_PARAMS = {'distance', 'fwhm', 'stellar_rv'}
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
            # DO NOT clear _wave_data_cache - it uses parameter hashes for validation
            # and should persist across parameter changes to enable cache restoration
    
    def __init__(self, **kwargs):
        self._initialize_caching_system()
        
        self.plot_lam = None
        self.plot_flux = None
        
        if 'hitran_data' in kwargs:
            print("Generating new molecule from default parameters.")
            self.user_save_data = None
            self.hitran_data = kwargs['hitran_data']
        elif 'user_save_data' in kwargs:
            print("Generating new molecule from user saved data.")
            self.user_save_data = kwargs['user_save_data']
            self.hitran_data = None
        else:
            self.user_save_data = None
            self.hitran_data = None

        self.initial_molecule_parameters = kwargs.get('initial_molecule_parameters', {})

        self._temp_val = None
        self._radius_val = None  
        self._n_mol_val = None
        self._distance_val = None
        self._fwhm_val = None
        self._broad_val = None

        if self.user_save_data is not None:
            self._load_from_user_save_data(kwargs)
        else:
            self._load_from_kwargs(kwargs)

        self.lines = None
        self._lines_filepath = self.filepath
        
        self.n_mol_init = float(self.scale_number * (10 ** self.scale_exponent))
        
        self._temp = float(self._temp_val if self._temp_val is not None else self.t_kin)
        self._radius = float(self._radius_val if self._radius_val is not None else self.radius_init)
        self._n_mol = float(self._n_mol_val if self._n_mol_val is not None else self.n_mol_init)
        self._distance = float(self._distance_val if self._distance_val is not None else c.DEFAULT_DISTANCE)
        self._fwhm = float(self._fwhm_val if self._fwhm_val is not None else c.DEFAULT_FWHM)
        self._broad = float(self._broad_val if self._broad_val is not None else c.INTRINSIC_LINE_WIDTH)

        self.wavelength_range = kwargs.get('wavelength_range', c.WAVELENGTH_RANGE)
        self.model_pixel_res = kwargs.get('model_pixel_res', c.MODEL_PIXEL_RESOLUTION)
        self.model_line_width = kwargs.get('model_line_width', c.MODEL_LINE_WIDTH)

        self.intensity = None
        self.spectrum = None
        
        self._calculate_initial_parameter_hashes()
    
    def _initialize_caching_system(self):
        self._intensity_cache = {'data': None, 'hash': None}
        self._spectrum_cache = {'data': None, 'hash': None}
        self._flux_cache = {}
        self._wave_data_cache = {}
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
        return hash((self._distance, self._fwhm, self.stellar_rv, self._compute_intensity_hash()))
    
    def _compute_full_parameter_hash(self):
        return hash((self._compute_spectrum_hash(), self._radius))

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
        self.is_visible = usd.get('Vis', kwargs.get('is_visible', True))
        
        # Get instance values from user save data or kwargs
        self._distance_val = usd.get('Dist', kwargs.get('distance', c.DEFAULT_DISTANCE))
        self._fwhm_val = usd.get('FWHM', kwargs.get('fwhm', c.DEFAULT_FWHM))
        self._broad_val = usd.get('Broad', kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH))
        self.stellar_rv = kwargs.get('stellar_rv', c.DEFAULT_STELLAR_RV)
        
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
        self.is_visible = kwargs.get('is_visible', True)
        
        # Get instance values from kwargs or defaults
        self._distance_val = kwargs.get('distance', c.DEFAULT_DISTANCE)
        self._fwhm_val = kwargs.get('fwhm', c.DEFAULT_FWHM)
        self._broad_val = kwargs.get('_broad', c.INTRINSIC_LINE_WIDTH)
        self.stellar_rv = kwargs.get('stellar_rv', c.DEFAULT_STELLAR_RV)
        
        # Set kinetic temperature and molecule-specific parameters
        self.t_kin = self.initial_molecule_parameters.get('t_kin', self._temp_val if self._temp_val is not None else 300.0)
        self.scale_exponent = self.initial_molecule_parameters.get('scale_exponent', 1.0)
        self.scale_number = self.initial_molecule_parameters.get('scale_number', 1.0)
        self.radius_init = self.initial_molecule_parameters.get('radius_init', self._radius_val if self._radius_val is not None else 1.0)
        
    def _ensure_lines_loaded(self):
        if self.lines is None:
            if self._lines_filepath:
                print("Loading lines from filepath:", self._lines_filepath)
                self.lines = MoleculeLineList(molecule_id=self.name, filename=self._lines_filepath)
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
        
        cache_key = (self.name, current_hash)
        with self._cache_lock:
            if cache_key in self._shared_calculation_cache:
                cached_data = self._shared_calculation_cache[cache_key]
                self._intensity_cache = {
                    'data': cached_data,
                    'hash': current_hash
                }
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
            self._shared_calculation_cache[cache_key] = intensity_data
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
        delta_lambda = mean_wavelength * (self._fwhm / 299792.458)
        spectral_resolution = mean_wavelength / delta_lambda if delta_lambda > 0 else self.model_line_width
        
        self.spectrum = Spectrum(
            lam_min=self.wavelength_range[0],
            lam_max=self.wavelength_range[1],
            dlambda=self.model_pixel_res,
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
    
    def _calculate_parameter_hash(self):
        """Calculate a hash of current parameters for cache validation"""
        param_tuple = (
            self._temp, self._radius, self._n_mol, self._distance, 
            self._fwhm, self._broad, self.wavelength_range, 
            self.model_pixel_res, self.model_line_width,
            getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV)  # Include stellar RV in hash
        )
        self._parameter_hash = hash(param_tuple)
    
    def _invalidate_parameter_hash(self):
        """Invalidate the parameter hash"""
        self._parameter_hash = None

    def calculate_intensity(self):
        self._ensure_intensity_calculated()
    
    def get_parameter_hash(self, cache_type='full'):
        if cache_type in self._param_hash_cache:
            return self._param_hash_cache[cache_type]
        return self._compute_full_parameter_hash()

    def _get_current_parameter_hash(self):
        """Get hash of current parameters for intensity calculation"""
        param_tuple = (
            getattr(self, '_temp', self.t_kin),
            getattr(self, '_n_mol', self.n_mol_init),
            getattr(self, '_broad', c.INTRINSIC_LINE_WIDTH),  # Use broad for intensity dv parameter
            getattr(self, '_fwhm', c.DEFAULT_FWHM),  # Include FWHM for spectrum resolution
            getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV),  # Include stellar RV
            # Include line data hash if available
            hash(str(self.lines.molecule_id)) if self.lines else 0
        )
        return hash(param_tuple)

    def _clear_flux_caches(self):
        self._flux_cache.clear()
        # DO NOT clear _wave_data_cache - it uses parameter hashes for validation
        # and should persist across parameter changes to enable cache restoration
        self.plot_lam = None
        self.plot_flux = None

    def get_flux(self, wavelength_array):
        """Get flux for given wavelength array with improved caching"""
        try:
            cache_key = hash(wavelength_array.tobytes()) if hasattr(wavelength_array, 'tobytes') else str(wavelength_array)
        except (TypeError, ValueError):
            cache_key = f"fallback_{id(wavelength_array)}"
        
        current_param_hash = self._compute_full_parameter_hash()
        cache_entry = self._flux_cache.get(cache_key)
        
        if (cache_entry is not None and 
            cache_entry.get('param_hash') == current_param_hash):
            self._cache_stats['hits'] += 1
            return cache_entry['flux']
        
        # Ensure we have a valid spectrum
        self._ensure_spectrum_calculated()
        
        if self.spectrum is None:
            print(f"Warning: No spectrum available for molecule {self.name}")
            return np.zeros_like(wavelength_array)
        
        # Get the spectrum data
        lam_grid = self.spectrum._lamgrid
        flux_grid = self.spectrum.flux_jy  # Use Jy units for consistency with observed data
        
        if lam_grid is None or flux_grid is None:
            print(f"Warning: Invalid spectrum data for molecule {self.name}")
            return np.zeros_like(wavelength_array)
        
        # Interpolate to the requested wavelength grid
        interpolated_flux = np.interp(wavelength_array, lam_grid, flux_grid, left=0, right=0)
        
        # Cache the result
        self._flux_cache[cache_key] = {
            'flux': interpolated_flux,
            'param_hash': current_param_hash
        }
        
        flux_cache_limit = 12
        flux_oldest_amount = 4

        # Limit cache size
        if len(self._flux_cache) > flux_cache_limit:
            oldest_keys = list(self._flux_cache.keys())[:flux_oldest_amount]
            for key in oldest_keys:
                del self._flux_cache[key]
        
        self._cache_stats['misses'] += 1
        return interpolated_flux
    
    def prepare_plot_data(self, wave_data):
        """Prepare plot data for given wavelength array, using caching for efficiency"""
        wave_data_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else str(wave_data)
        current_param_hash = self._compute_full_parameter_hash()
        
        # Create composite cache key from both wavelength and parameters
        cache_key = (wave_data_hash, current_param_hash)
        
        if cache_key in self._wave_data_cache:
            cached_entry = self._wave_data_cache[cache_key]
            self.plot_lam = cached_entry['lam']
            self.plot_flux = cached_entry['flux']
            self._cache_stats['hits'] += 1
            return (self.plot_lam, self.plot_flux)
        
        flux = self.get_flux(wave_data)
        
        self.plot_lam = wave_data.copy()
        self.plot_flux = flux
        
        self._wave_data_cache[cache_key] = {
            'lam': self.plot_lam,
            'flux': self.plot_flux
        }
        
        if len(self._wave_data_cache) > 20:
            oldest_keys = list(self._wave_data_cache.keys())[:5]
            for key in oldest_keys:
                del self._wave_data_cache[key]
        
        self._cache_stats['misses'] += 1
        return (self.plot_lam, self.plot_flux)

    @property
    def temp(self):
        """Temperature getter"""
        return self._temp
    
    @temp.setter
    def temp(self, value):
        old_value = self._temp
        self._temp = float(value)
        self.t_kin = self._temp
        self._notify_my_parameter_change('temp', old_value, self._temp)
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        old_value = self._radius
        self._radius = float(value)
        self._notify_my_parameter_change('radius', old_value, self._radius)
    
    @property
    def n_mol(self):
        return self._n_mol
    
    @n_mol.setter
    def n_mol(self, value):
        if value is None:
            value = getattr(self, 'n_mol_init', 1e17)
        old_value = self._n_mol
        self._n_mol = float(value)
        self._notify_my_parameter_change('n_mol', old_value, self._n_mol)
    
    @property
    def distance(self):
        return self._distance
    
    @distance.setter
    def distance(self, value):
        old_value = self._distance
        self._distance = float(value)
        self._notify_my_parameter_change('distance', old_value, self._distance)
    
    @property
    def fwhm(self):
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value):
        old_value = self._fwhm
        self._fwhm = float(value)
        self.spectrum = None
        self._notify_my_parameter_change('fwhm', old_value, self._fwhm)

    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], skip_notification: bool = False):
        old_values = {}
        affected_params = set()
        
        for param_name, value in parameter_dict.items():
            if hasattr(self, f'_{param_name}'):
                old_values[param_name] = getattr(self, f'_{param_name}')
                setattr(self, f'_{param_name}', float(value))
                affected_params.add(param_name)
            elif param_name == 'intrinsic_line_width':
                old_values[param_name] = self._broad
                self._broad = float(value)
                affected_params.add('broad')
            elif hasattr(self, param_name):
                old_values[param_name] = getattr(self, param_name)
                setattr(self, param_name, value)
                affected_params.add(param_name)
        
        for param in affected_params:
            self._invalidate_caches_for_parameter(param)
        
        if not skip_notification:
            for param_name, value in parameter_dict.items():
                old_value = old_values.get(param_name)
                if old_value != value:
                    self._notify_my_parameter_change(param_name, old_value, value)

    @property
    def star_rv(self):
        return getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV)
    
    @star_rv.setter
    def star_rv(self, value):
        old_value = getattr(self, 'stellar_rv', c.DEFAULT_STELLAR_RV)
        self.stellar_rv = float(value)
        self._notify_my_parameter_change('stellar_rv', old_value, self.stellar_rv)
    
    @property
    def broad(self):
        return self._broad
    
    @broad.setter
    def broad(self, value):
        old_value = self._broad
        self._broad = float(value)
        self._notify_my_parameter_change('broad', old_value, self._broad)

    @property
    def intrinsic_line_width(self):
        return self.broad
    
    @intrinsic_line_width.setter
    def intrinsic_line_width(self, value):
        self.broad = value
    
    def _update_spectrum(self):
        """Update the spectrum with current intensity and area"""
        if hasattr(self, 'spectrum') and hasattr(self, 'intensity'):
            # Ensure intensity is valid first
            if not self._intensity_valid:
                self.calculate_intensity()
                
            # Clear previous intensity data
            self.spectrum._I_list = []
            self.spectrum._lam_list = []
            self.spectrum._dA_list = []
            
            # Add updated intensity with current radius
            self.spectrum.add_intensity(
                intensity=self.intensity,
                dA=self.radius ** 2 * np.pi  # Use property to get correct value
            )
            
            # Mark spectrum as valid and clear flux caches
            self._spectrum_valid = True
            self._clear_flux_caches()
    
    def _recreate_spectrum(self):
        """Recreate the spectrum when distance or other fundamental parameters change"""
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
        self._wave_data_cache.clear()
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

    def __str__(self):
        return f"Molecule(name={self.name}, temp={self._temp}, radius={self._radius}, n_mol={self._n_mol})"