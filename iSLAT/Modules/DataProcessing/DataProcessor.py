import numpy as np
from typing import Dict, Tuple, Optional, List, Any

class DataProcessor:
    """Computations and caching - handles all data processing and caching operations"""
    
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.islat = plot_manager.islat
        
        # Cache system
        self._flux_cache: Dict[str, np.ndarray] = {}
        self._summed_flux_cache: Dict[str, np.ndarray] = {}
        self._interpolated_cache: Dict[str, np.ndarray] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_dependencies: Dict[str, List[str]] = {}
        
        # Cache validation
        self._last_wave_data_hash: Optional[str] = None
        self._last_molecules_hash: Optional[str] = None
        
    def clear_all_caches(self):
        """Clear all cached data"""
        self._flux_cache.clear()
        self._summed_flux_cache.clear()
        self._interpolated_cache.clear()
        self._cache_timestamps.clear()
        self._cache_dependencies.clear()
        self._last_wave_data_hash = None
        self._last_molecules_hash = None
    
    def invalidate_cache_for_molecule(self, molecule_name: str):
        """Invalidate all cached data related to a specific molecule"""
        keys_to_remove = []
        for key in self._flux_cache.keys():
            if molecule_name in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._flux_cache.pop(key, None)
            self._interpolated_cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        # Clear summed flux cache as it depends on individual molecules
        self._summed_flux_cache.clear()
    
    def _create_cache_key(self, *args) -> str:
        """Create a cache key from arguments"""
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(str(hash(arg.tobytes())))
            else:
                key_parts.append(str(arg))
        return "_".join(key_parts)
    
    def _get_wave_data_hash(self, wave_data: np.ndarray) -> str:
        """Get hash for wave data"""
        if wave_data is None:
            return "none"
        return str(hash(wave_data.tobytes()))
    
    def _get_molecules_hash(self) -> str:
        """Get hash representing current molecule state"""
        if not hasattr(self.islat, 'molecules_dict'):
            return "no_molecules"
        
        # Create hash based on molecule names and their visibility
        molecule_state = []
        for name, mol in self.islat.molecules_dict.items():
            state = f"{name}_{mol.is_visible}_{mol.temp}_{mol.radius}_{mol.n_mol}"
            molecule_state.append(state)
        
        return str(hash("_".join(sorted(molecule_state))))
    
    def _is_cache_valid(self, wave_data: np.ndarray) -> bool:
        """Check if current caches are still valid"""
        current_wave_hash = self._get_wave_data_hash(wave_data)
        current_mol_hash = self._get_molecules_hash()
        
        wave_changed = self._last_wave_data_hash != current_wave_hash
        molecules_changed = self._last_molecules_hash != current_mol_hash
        
        if wave_changed or molecules_changed:
            if wave_changed:
                self.clear_all_caches()
            elif molecules_changed:
                self._flux_cache.clear()
                self._summed_flux_cache.clear()
            
            self._last_wave_data_hash = current_wave_hash
            self._last_molecules_hash = current_mol_hash
            return False
        
        return True
    
    def get_molecule_flux(self, molecule_name: str, wave_data: np.ndarray) -> Optional[np.ndarray]:
        """Get flux for a specific molecule with caching"""
        if not hasattr(self.islat, 'molecules_dict') or molecule_name not in self.islat.molecules_dict:
            return None
        
        # Check cache validity
        self._is_cache_valid(wave_data)
        
        # Create cache key
        cache_key = self._create_cache_key(molecule_name, self._get_wave_data_hash(wave_data))
        
        # Check cache
        if cache_key in self._flux_cache:
            return self._flux_cache[cache_key]
        
        # Calculate flux
        molecule = self.islat.molecules_dict[molecule_name]
        try:
            molecule.prepare_plot_data(wave_data)
            flux = molecule.plot_flux.copy()
            
            # Cache result
            self._flux_cache[cache_key] = flux
            return flux
            
        except Exception as e:
            print(f"Error calculating flux for {molecule_name}: {e}")
            return np.zeros_like(wave_data)
    
    def get_summed_flux(self, wave_data: np.ndarray, visible_only: bool = True) -> np.ndarray:
        """Get summed flux for all molecules with caching"""
        if not hasattr(self.islat, 'molecules_dict') or len(self.islat.molecules_dict) == 0:
            return np.zeros_like(wave_data)
        
        # Check cache validity
        self._is_cache_valid(wave_data)
        
        # Create cache key
        visible_molecules = [name for name, mol in self.islat.molecules_dict.items() 
                           if not visible_only or mol.is_visible]
        cache_key = self._create_cache_key(
            self._get_wave_data_hash(wave_data),
            "_".join(sorted(visible_molecules)),
            "visible" if visible_only else "all"
        )
        
        # Check cache
        if cache_key in self._summed_flux_cache:
            return self._summed_flux_cache[cache_key]
        
        # Calculate summed flux
        summed_flux = np.zeros_like(wave_data)
        
        for molecule_name in visible_molecules:
            molecule_flux = self.get_molecule_flux(molecule_name, wave_data)
            if molecule_flux is not None:
                summed_flux += molecule_flux
        
        # Cache result
        self._summed_flux_cache[cache_key] = summed_flux
        return summed_flux
    
    def compute_summed_flux(self, wave_data: np.ndarray, molecules, visible_only: bool = True) -> np.ndarray:
        """
        Compute summed flux for molecules - interface compatible with MainPlot.
        
        Args:
            wave_data: Wavelength array
            molecules: Collection of molecules (ignored, uses internal molecules_dict)
            visible_only: Whether to include only visible molecules
            
        Returns:
            Summed flux array
        """
        return self.get_summed_flux(wave_data, visible_only)

    def get_visible_molecules(self) -> List[Any]:
        """Get list of visible molecules"""
        if not hasattr(self.islat, 'molecules_dict'):
            return []
        
        return [mol for mol in self.islat.molecules_dict.values() if mol.is_visible]
    
    def interpolate_flux_to_range(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                                 xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate flux data to a specific wavelength range"""
        # Create cache key
        cache_key = self._create_cache_key(
            self._get_wave_data_hash(wave_data),
            str(hash(flux_data.tobytes())) if flux_data is not None else "none",
            f"{xmin}_{xmax}"
        )
        
        # Check cache
        if cache_key in self._interpolated_cache:
            return self._interpolated_cache[cache_key]
        
        # Create wavelength mask
        mask = (wave_data >= xmin) & (wave_data <= xmax)
        
        if not np.any(mask):
            # No data in range
            result = (np.array([]), np.array([]))
            self._interpolated_cache[cache_key] = result
            return result
        
        # Extract data in range
        wave_range = wave_data[mask]
        flux_range = flux_data[mask] if flux_data is not None else np.zeros_like(wave_range)
        
        result = (wave_range, flux_range)
        self._interpolated_cache[cache_key] = result
        return result
    
    def find_flux_extrema(self, wave_data: np.ndarray, flux_data: np.ndarray, 
                         xmin: float, xmax: float) -> Dict[str, float]:
        """Find flux extrema in a wavelength range"""
        wave_range, flux_range = self.interpolate_flux_to_range(wave_data, flux_data, xmin, xmax)
        
        if len(flux_range) == 0:
            return {'max': 0.0, 'min': 0.0, 'max_wave': 0.0, 'min_wave': 0.0}
        
        max_idx = np.argmax(flux_range)
        min_idx = np.argmin(flux_range)
        
        return {
            'max': flux_range[max_idx],
            'min': flux_range[min_idx],
            'max_wave': wave_range[max_idx],
            'min_wave': wave_range[min_idx]
        }
    
    def calculate_flux_statistics(self, wave_data: np.ndarray, flux_data: np.ndarray,
                                 xmin: float, xmax: float) -> Dict[str, float]:
        """Calculate statistics for flux in a wavelength range"""
        wave_range, flux_range = self.interpolate_flux_to_range(wave_data, flux_data, xmin, xmax)
        
        if len(flux_range) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'median': 0.0,
                'integral': 0.0, 'rms': 0.0, 'peak_to_peak': 0.0
            }
        
        # Calculate statistics
        mean_flux = np.mean(flux_range)
        std_flux = np.std(flux_range)
        median_flux = np.median(flux_range)
        
        # Numerical integration (trapezoidal rule)
        if len(wave_range) > 1:
            integral = np.trapz(flux_range, wave_range)
        else:
            integral = 0.0
        
        # RMS
        rms = np.sqrt(np.mean(flux_range**2))
        
        # Peak-to-peak
        peak_to_peak = np.max(flux_range) - np.min(flux_range)
        
        return {
            'mean': mean_flux,
            'std': std_flux,
            'median': median_flux,
            'integral': integral,
            'rms': rms,
            'peak_to_peak': peak_to_peak,
            'n_points': len(flux_range)
        }
    
    def process_line_data_for_range(self, molecule, xmin: float, xmax: float) -> Optional[Dict[str, np.ndarray]]:
        """Process line data for a specific molecule in a wavelength range"""
        if molecule is None:
            return None
        
        try:
            # Get line data
            lines_df = molecule.intensity.get_table
            
            if lines_df.empty:
                return None
            
            # Filter by wavelength range
            subset = lines_df[(lines_df['lam'] >= xmin) & (lines_df['lam'] <= xmax)]
            
            if subset.empty:
                return None
            
            return {
                'wavelengths': subset['lam'].values,
                'intensities': subset['intens'].values,
                'upper_energies': subset['e_up'].values,
                'einstein_coeffs': subset['a_stein'].values,
                'upper_degeneracies': subset['g_up'].values,
                'lower_energies': subset['e_low'].values,
                'lower_degeneracies': subset['g_low'].values
            }
            
        except Exception as e:
            print(f"Error processing line data: {e}")
            return None
    
    def get_strongest_line_in_range(self, molecule, xmin: float, xmax: float) -> Optional[Dict[str, float]]:
        """Find the strongest line in a wavelength range"""
        line_data = self.process_line_data_for_range(molecule, xmin, xmax)
        
        if line_data is None or len(line_data['intensities']) == 0:
            return None
        
        # Find strongest line
        max_idx = np.argmax(line_data['intensities'])
        
        return {
            'wavelength': line_data['wavelengths'][max_idx],
            'intensity': line_data['intensities'][max_idx],
            'upper_energy': line_data['upper_energies'][max_idx],
            'einstein_coeff': line_data['einstein_coeffs'][max_idx],
            'upper_degeneracy': line_data['upper_degeneracies'][max_idx]
        }
    
    def refresh_molecule_data(self, wave_data: np.ndarray):
        """Refresh all molecule data and caches"""
        if not hasattr(self.islat, 'molecules_dict'):
            return
        
        # Force cache invalidation
        self._last_molecules_hash = None
        self._is_cache_valid(wave_data)
        
        # Update all molecules
        for molecule in self.islat.molecules_dict.values():
            try:
                molecule.prepare_plot_data(wave_data)
            except Exception as e:
                print(f"Error refreshing data for {molecule.name}: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state"""
        return {
            'flux_cache_size': len(self._flux_cache),
            'summed_flux_cache_size': len(self._summed_flux_cache),
            'interpolated_cache_size': len(self._interpolated_cache),
            'last_wave_hash': self._last_wave_data_hash,
            'last_molecules_hash': self._last_molecules_hash,
            'total_cache_keys': len(self._cache_timestamps)
        }
