from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import os

from .Molecule import Molecule
import iSLAT.Constants as default_parms

class MoleculeDict(dict):
    """
    A dictionary to store Molecule objects with their names as keys.
    Provides efficient operations on collections of molecules with caching and lazy evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.fluxes: Dict[str, np.ndarray] = {}
        self._summed_flux_cache: Dict[int, np.ndarray] = {}
        self._cache_wave_data_hash: Optional[int] = None
        
        self._visible_molecules: set = set()
        self._dirty_molecules: set = set()
        
        self._global_dist: float = default_parms.DEFAULT_DISTANCE
        #self._global_star_rv: float = default_parms.DEFAULT_STELLAR_RV
        #self._global_fwhm: float = default_parms.DEFAULT_FWHM
        self._global_intrinsic_line_width: float = default_parms.INTRINSIC_LINE_WIDTH
        self._global_wavelength_range: Tuple[float, float] = default_parms.WAVELENGTH_RANGE
        self._global_model_line_width: float = default_parms.MODEL_LINE_WIDTH
        self._global_model_pixel_res: float = default_parms.MODEL_PIXEL_RESOLUTION
        
        self._global_parameter_change_callbacks: List[Callable] = []
        
        from .Molecule import Molecule
        Molecule.add_molecule_parameter_change_callback(self._on_molecule_parameter_changed)

    def add_molecule(self, mol_entry: Dict[str, Any], intrinsic_line_width: Optional[float] = None, 
                     wavelength_range: Optional[Tuple[float, float]] = None, 
                     model_pixel_res: Optional[float] = None, model_line_width: Optional[float] = None, 
                     distance: Optional[float] = None, hitran_data: Optional[Any] = None) -> Molecule:
        """Add a new molecule to the dictionary using molecule entry data."""
        mol_name = mol_entry["name"]

        # Use global parameters if not specifically provided
        effective_intrinsic_line_width = intrinsic_line_width if intrinsic_line_width is not None else self._global_intrinsic_line_width
        effective_wavelength_range = wavelength_range if wavelength_range is not None else self._global_wavelength_range
        effective_model_pixel_res = model_pixel_res if model_pixel_res is not None else self._global_model_pixel_res
        effective_model_line_width = model_line_width if model_line_width is not None else self._global_model_line_width
        effective_distance = distance if distance is not None else self._global_dist

        # Create a Molecule instance
        molecule = Molecule(
            name=mol_name,
            filepath=mol_entry["file"],
            displaylabel=mol_entry["label"],
            color=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Color"),
            initial_molecule_parameters=getattr(self, 'initial_molecule_parameters', {}).get(mol_name, {}),
            wavelength_range=effective_wavelength_range,
            broad=effective_intrinsic_line_width,
            model_pixel_res=effective_model_pixel_res,
            model_line_width=effective_model_line_width,
            distance=effective_distance,
            fwhm=self._global_fwhm,
            stellar_rv=self._global_star_rv,
            radius=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Rad", None),
            temp=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Temp", None),
            n_mol=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("N_Mol", None),
            is_visible=getattr(self, 'save_file_data', {}).get(mol_name, {}).get("Vis", True),
            hitran_data=hitran_data
        )

        # Store the molecule in the dictionary
        self[mol_name] = molecule

        print(f"Molecule Initialized: {mol_name}")
        
        # Update fluxes if the molecule has plot data
        if hasattr(molecule, 'plot_flux'):
            self.fluxes[mol_name] = molecule.plot_flux
            
        return molecule

    def add_molecules(self, *molecules) -> None:
        """Add multiple molecules to the dictionary with optional bulk intensity calculation."""
        molecules = molecules[0]
        added_molecules = []
        
        for mol in molecules:
            if isinstance(mol, Molecule):
                self[mol.name] = mol
                added_molecules.append(mol)
            else:
                raise TypeError("Expected a Molecule instance.")
        
        # If multiple molecules were added, trigger bulk intensity calculation
        if len(added_molecules) > 1:
            print(f"Triggering bulk intensity calculation for {len(added_molecules)} molecules...")
            self._bulk_calculate_intensities([mol.name for mol in added_molecules])
    
    def clear(self):
        """Clear the dictionary of all molecules."""
        super().clear()
        self.fluxes.clear()
        print("MoleculeDict cleared.")

    def update_molecule_fluxes(self, wave_data: Optional[np.ndarray] = None) -> None:
        """Update stored fluxes for all molecules with current wave_data - with caching"""
        if wave_data is None:
            return
            
        # Create cache key for wave_data using int hash for efficiency
        wave_data_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else hash(str(wave_data))
        
        # Only update if wave_data changed
        if self._cache_wave_data_hash == wave_data_hash:
            return
            
        for mol_name, molecule in self.items():
            if hasattr(molecule, 'prepare_plot_data'):
                molecule.prepare_plot_data(wave_data)
                if hasattr(molecule, 'plot_flux'):
                    self.fluxes[mol_name] = molecule.plot_flux
        
        # Clear summed flux cache when wave data changes
        self._summed_flux_cache.clear()
        self._cache_wave_data_hash = wave_data_hash
    
    def get_summed_flux(self, wave_data: np.ndarray, visible_only: bool = True) -> np.ndarray:
        """Optimized summed flux calculation using vectorized operations and advanced caching.
        
        This consolidated method uses the best available optimization techniques including:
        - Vectorized numpy operations and broadcasting
        - Advanced caching with intelligent cache management
        - Memory-efficient float32 operations
        - Robust error handling with fallbacks
        
        Parameters
        ----------
        wave_data: np.ndarray
            Wavelength data for flux calculation
        visible_only: bool, default True
            If True, only sum flux from visible molecules
            
        Returns
        -------
        np.ndarray
            Summed flux array
        """
        if wave_data is None:
            return np.array([])
            
        visible_molecules = list(self.get_visible_molecules() if visible_only else self.keys())
        
        if not visible_molecules:
            return np.zeros_like(wave_data, dtype=np.float32)
        
        # Try cache lookup first
        try:
            wave_hash = hash(wave_data.tobytes())
            cache_key = hash((wave_hash, frozenset(visible_molecules)))
            if cache_key in self._summed_flux_cache:
                return self._summed_flux_cache[cache_key]
        except (TypeError, ValueError):
            cache_key = None
        
        # For large datasets, use parallel processing
        if len(visible_molecules) >= 8 and len(wave_data) > 1000:
            try:
                return self._parallel_flux_calculation_internal(wave_data, visible_molecules, cache_key)
            except Exception as e:
                print(f"Parallel processing failed, using vectorized fallback: {e}")
        
        # Pre-allocate flux array matrix for vectorized operations
        n_molecules = len(visible_molecules)
        if len(wave_data) == 0:
            return np.array([])
            
        flux_matrix = np.zeros((n_molecules, len(wave_data)), dtype=np.float32)
        
        # Prepare plot data for all molecules with error handling
        valid_molecules = []
        for i, mol_name in enumerate(visible_molecules):
            if mol_name in self:
                molecule = self[mol_name]
                try:
                    molecule.prepare_plot_data(wave_data)
                    if hasattr(molecule, 'plot_flux') and molecule.plot_flux is not None:
                        # Ensure plot_flux has the right shape
                        flux = molecule.plot_flux
                        if len(flux) == len(wave_data):
                            flux_matrix[i] = flux.astype(np.float32)
                            valid_molecules.append(i)
                        else:
                            print(f"Warning: flux shape mismatch for {mol_name}: {len(flux)} vs {len(wave_data)}")
                except Exception as e:
                    print(f"Warning: Failed to prepare plot data for molecule {mol_name}: {e}")
                    continue
        
        # Vectorized summation along molecule axis
        if valid_molecules:
            summed_flux = np.sum(flux_matrix[valid_molecules], axis=0)
        else:
            summed_flux = np.zeros_like(wave_data, dtype=np.float32)
        
        # Cache result with intelligent cache management
        if cache_key is not None:
            if len(self._summed_flux_cache) > 50:
                oldest_key = next(iter(self._summed_flux_cache))
                del self._summed_flux_cache[oldest_key]
            self._summed_flux_cache[cache_key] = summed_flux
            
        return summed_flux

    def _parallel_flux_calculation_internal(self, wave_data: np.ndarray, visible_molecules: list, cache_key) -> np.ndarray:
        """Internal parallel flux calculation method."""
        max_workers = min(len(visible_molecules), mp.cpu_count())
        
        def calculate_molecule_flux(mol_name):
            """Worker function to calculate flux for a single molecule"""
            if mol_name in self:
                molecule = self[mol_name]
                try:
                    molecule.prepare_plot_data(wave_data)
                    if hasattr(molecule, 'plot_flux') and molecule.plot_flux is not None:
                        return molecule.plot_flux.astype(np.float32)
                except Exception as e:
                    print(f"Warning: Failed to calculate flux for {mol_name}: {e}")
            return np.zeros_like(wave_data, dtype=np.float32)
        
        # Use ThreadPoolExecutor for I/O bound tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(calculate_molecule_flux, mol_name): mol_name 
                      for mol_name in visible_molecules}
            
            # Collect results as they complete
            summed_flux = np.zeros_like(wave_data, dtype=np.float32)
            
            for future in as_completed(futures):
                mol_name = futures[future]
                try:
                    flux = future.result()
                    if flux is not None and len(flux) == len(wave_data):
                        summed_flux += flux
                except Exception as e:
                    print(f"Failed to calculate flux for {mol_name}: {e}")
        
        return summed_flux

    def batch_flux_calculation(self, wave_data_list: List[np.ndarray], visible_only: bool = True) -> List[np.ndarray]:
        """Calculate fluxes for multiple wavelength arrays using vectorized operations.
        
        Parameters
        ----------
        wave_data_list: List[np.ndarray]
            List of wavelength arrays to calculate fluxes for
        visible_only: bool, default True
            If True, only calculate flux from visible molecules
            
        Returns
        -------
        List[np.ndarray]
            List of summed flux arrays corresponding to input wavelength arrays
        """
        if not wave_data_list:
            return []
            
        visible_molecules = list(self.get_visible_molecules() if visible_only else self.keys())
        
        if not visible_molecules:
            return [np.zeros_like(wd, dtype=np.float32) if wd is not None else np.array([]) 
                   for wd in wave_data_list]
        
        results = []
        
        # Process each wavelength array
        for wave_data in wave_data_list:
            if wave_data is None:
                results.append(np.array([]))
            else:
                # Use existing optimized method
                flux = self.get_summed_flux(wave_data, visible_only)
                results.append(flux)
        
        return results

    def bulk_parameter_update_vectorized(self, parameter_dict: Dict[str, np.ndarray], 
                                       molecule_names: Optional[List[str]] = None) -> Dict[str, int]:
        """Vectorized bulk parameter updates for multiple molecules.
        
        This method efficiently updates parameters across multiple molecules using
        vectorized numpy operations where possible.
        
        Parameters
        ----------
        parameter_dict: Dict[str, np.ndarray]
            Dictionary mapping parameter names to arrays of values. Each array should
            have the same length as the number of molecules being updated.
        molecule_names: Optional[List[str]], default None
            List of molecule names to update. If None, updates all molecules.
            
        Returns
        -------
        Dict[str, int]
            Dictionary with statistics: {'updated': count, 'failed': count}
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        # Filter to existing molecules
        valid_molecules = [name for name in molecule_names if name in self]
        n_molecules = len(valid_molecules)
        
        if n_molecules == 0:
            return {'updated': 0, 'failed': 0}
        
        # Validate parameter arrays
        param_lengths = set()
        for param_name, values in parameter_dict.items():
            values = np.asarray(values)
            param_lengths.add(len(values))
            
        if len(param_lengths) > 1:
            raise ValueError("All parameter arrays must have the same length")
        
        array_length = param_lengths.pop()
        if array_length != n_molecules and array_length != 1:
            raise ValueError(f"Parameter arrays must have length {n_molecules} or 1, got {array_length}")
        
        updated_count = 0
        failed_count = 0
        affected_molecules = []
        
        # Vectorized parameter updates
        for i, mol_name in enumerate(valid_molecules):
            try:
                molecule = self[mol_name]
                old_params = {}
                changed = False
                
                for param_name, values in parameter_dict.items():
                    values = np.asarray(values)
                    new_value = values[i] if len(values) == n_molecules else values[0]
                    old_value = getattr(molecule, param_name, None)
                    old_params[param_name] = old_value
                    
                    if old_value != new_value:
                        # Update parameter using appropriate method
                        if hasattr(molecule, f'_{param_name}'):
                            setattr(molecule, f'_{param_name}', float(new_value))
                        elif param_name == 'intrinsic_line_width':
                            molecule._broad = float(new_value)
                        elif param_name == 'stellar_rv':
                            molecule.stellar_rv = float(new_value)
                        elif hasattr(molecule, param_name):
                            setattr(molecule, param_name, new_value)
                        
                        changed = True
                
                if changed:
                    # Invalidate caches for this molecule
                    molecule._invalidate_caches_for_parameters(list(parameter_dict.keys()))
                    affected_molecules.append(mol_name)
                    updated_count += 1
                    
            except Exception as e:
                print(f"Failed to update parameters for molecule {mol_name}: {e}")
                failed_count += 1
        
        # Clear global caches for affected molecules
        if affected_molecules:
            self._summed_flux_cache.clear()
            for mol_name in affected_molecules:
                self.fluxes.pop(mol_name, None)
        
        print(f"Vectorized bulk update: {updated_count} molecules updated, {failed_count} failed")
        
        return {'updated': updated_count, 'failed': failed_count}

    def get_parameter_arrays_vectorized(self, parameter_names: List[str], 
                                      molecule_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get parameter arrays for multiple molecules using vectorized operations.
        
        Parameters
        ----------
        parameter_names: List[str]
            List of parameter names to extract
        molecule_names: Optional[List[str]], default None
            List of molecule names to extract from (None for all)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping parameter names to numpy arrays of values
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        valid_molecules = [name for name in molecule_names if name in self]
        n_molecules = len(valid_molecules)
        
        if n_molecules == 0:
            return {param: np.array([]) for param in parameter_names}
        
        # Pre-allocate result arrays
        result = {}
        for param_name in parameter_names:
            result[param_name] = np.empty(n_molecules, dtype=np.float64)
        
        # Vectorized extraction
        for i, mol_name in enumerate(valid_molecules):
            molecule = self[mol_name]
            for param_name in parameter_names:
                value = getattr(molecule, param_name, np.nan)
                result[param_name][i] = value if value is not None else np.nan
        
        return result

    def batch_intensity_calculation(self, parameter_combinations: List[Dict[str, float]], 
                                   molecule_names: Optional[List[str]] = None,
                                   method: str = "curve_growth") -> Dict[str, np.ndarray]:
        """Calculate intensities for multiple parameter combinations across multiple molecules.
        
        This method leverages the vectorized intensity calculations for efficient
        batch processing of intensity calculations.
        
        Parameters
        ----------
        parameter_combinations: List[Dict[str, float]]
            List of parameter dictionaries, each containing 't_kin', 'n_mol', 'dv'
        molecule_names: Optional[List[str]], default None
            List of molecule names to calculate intensities for (None for all)
        method: str, default "curve_growth"
            Intensity calculation method
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping molecule names to 2D intensity arrays 
            with shape (n_combinations, n_lines)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        valid_molecules = [name for name in molecule_names if name in self]
        
        if not parameter_combinations or not valid_molecules:
            return {}
        
        results = {}
        
        for mol_name in valid_molecules:
            molecule = self[mol_name]
            if hasattr(molecule, 'intensity') and hasattr(molecule.intensity, 'bulk_parameter_update_vectorized'):
                try:
                    # Use vectorized intensity calculation
                    intensity_array = molecule.intensity.bulk_parameter_update_vectorized(
                        parameter_combinations, method=method
                    )
                    results[mol_name] = intensity_array
                    
                except Exception as e:
                    print(f"Failed vectorized intensity calculation for {mol_name}: {e}")
                    # Fallback to individual calculations
                    try:
                        n_combos = len(parameter_combinations)
                        n_lines = len(molecule.lines.lines) if hasattr(molecule, 'lines') else 0
                        
                        if n_lines > 0:
                            intensity_matrix = np.zeros((n_combos, n_lines))
                            
                            for i, params in enumerate(parameter_combinations):
                                molecule.intensity.calc_intensity(
                                    t_kin=params['t_kin'],
                                    n_mol=params['n_mol'], 
                                    dv=params['dv'],
                                    method=method
                                )
                                if molecule.intensity.intensity is not None:
                                    intensity_matrix[i] = molecule.intensity.intensity
                            
                            results[mol_name] = intensity_matrix
                        
                    except Exception as fallback_e:
                        print(f"Fallback intensity calculation failed for {mol_name}: {fallback_e}")
        
        print(f"Batch intensity calculation completed for {len(results)} molecules")
        return results

    def _bulk_calculate_intensities(self, molecule_names: List[str], max_workers: Optional[int] = None) -> dict:
        """Bulk calculate intensities for multiple molecules using parallel processing.
        
        This method efficiently calculates intensities for multiple molecules simultaneously,
        providing significant performance improvements when loading many molecules.
        
        Parameters
        ----------
        molecule_names: List[str]
            List of molecule names to calculate intensities for
        max_workers: Optional[int], default None
            Maximum number of worker threads (None for auto-detect)
            
        Returns
        -------
        dict
            Dictionary with calculation statistics and results
        """
        if not molecule_names:
            return {'success': 0, 'failed': 0, 'molecules': []}
        
        valid_molecules = [name for name in molecule_names if name in self]
        if not valid_molecules:
            return {'success': 0, 'failed': 0, 'molecules': []}
        
        print(f"Starting bulk intensity calculation for {len(valid_molecules)} molecules...")
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(len(valid_molecules), mp.cpu_count())
        
        def calculate_molecule_intensity(mol_name):
            """Worker function to calculate intensity for a single molecule"""
            try:
                molecule = self[mol_name]
                
                # Check if intensity is already cached and valid
                if (hasattr(molecule, '_intensity_cache') and 
                    molecule._intensity_cache.get('data') is not None and
                    hasattr(molecule, '_dirty_flags') and 
                    not molecule._dirty_flags.get('intensity', True)):
                    
                    # Intensity is already cached and valid - no need to recalculate
                    return {
                        'success': True, 
                        'molecule': mol_name, 
                        'error': None,
                        'intensity_calculated': False,  # Already cached
                        'has_data': True
                    }
                
                # Need to calculate intensity
                if hasattr(molecule, '_ensure_intensity_calculated'):
                    molecule._ensure_intensity_calculated()
                else:
                    # Fallback: direct intensity calculation
                    t_kin = getattr(molecule, 'temp', 300.0)
                    n_mol = getattr(molecule, 'n_mol', 1e17)
                    dv = getattr(molecule, 'intrinsic_line_width', 1.0)
                    
                    if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                        molecule.intensity.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=dv)
                
                # Mark caches as valid after calculation
                if hasattr(molecule, '_dirty_flags'):
                    molecule._dirty_flags['intensity'] = False
                    molecule._dirty_flags['spectrum'] = False
                
                # Ensure spectrum is also calculated and properly cached
                if hasattr(molecule, '_ensure_spectrum_calculated'):
                    molecule._ensure_spectrum_calculated()
                
                # Verify that the intensity was properly calculated and cached
                intensity_data = None
                if hasattr(molecule, '_intensity_cache') and molecule._intensity_cache.get('data'):
                    intensity_data = molecule._intensity_cache['data']
                elif hasattr(molecule, 'intensity') and molecule.intensity:
                    if hasattr(molecule.intensity, '_intensity') and molecule.intensity._intensity is not None:
                        intensity_data = {
                            'intensity_array': molecule.intensity._intensity,
                            'tau_array': getattr(molecule.intensity, '_tau', None)
                        }
                
                if intensity_data is not None:
                    return {
                        'success': True, 
                        'molecule': mol_name, 
                        'error': None,
                        'intensity_calculated': True,
                        'has_data': True
                    }
                else:
                    return {
                        'success': False, 
                        'molecule': mol_name, 
                        'error': 'Intensity calculation completed but no data available',
                        'intensity_calculated': True,
                        'has_data': False
                    }
                    
            except Exception as e:
                return {
                    'success': False, 
                    'molecule': mol_name, 
                    'error': str(e),
                    'intensity_calculated': False,
                    'has_data': False
                }
        
        # Use parallel processing for bulk calculations
        success_count = 0
        failed_count = 0
        failed_molecules = []
        
        if len(valid_molecules) >= 2:
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(calculate_molecule_intensity, mol_name): mol_name 
                          for mol_name in valid_molecules}
                
                for future in as_completed(futures):
                    mol_name = futures[future]
                    try:
                        result = future.result()
                        if result['success']:
                            success_count += 1
                            print(f"✓ {mol_name}: Intensity calculated and cached")
                        else:
                            failed_count += 1
                            failed_molecules.append({'molecule': mol_name, 'error': result['error']})
                            print(f"✗ {mol_name}: {result['error']}")
                    except Exception as e:
                        failed_count += 1
                        failed_molecules.append({'molecule': mol_name, 'error': str(e)})
                        print(f"✗ {mol_name}: Execution error - {str(e)}")
        else:
            # Single molecule - no need for parallel processing
            result = calculate_molecule_intensity(valid_molecules[0])
            if result['success']:
                success_count = 1
            else:
                failed_count = 1
                failed_molecules.append({'molecule': valid_molecules[0], 'error': result['error']})
        
        # Clear caches after bulk calculation
        self._clear_flux_caches()
        
        print(f"Bulk intensity calculation completed: {success_count} successful, {failed_count} failed")
        if failed_molecules:
            print("Failed molecules:")
            for failure in failed_molecules:
                print(f"  - {failure['molecule']}: {failure['error']}")
        
        return {
            'success': success_count,
            'failed': failed_count,
            'molecules': valid_molecules,
            'failures': failed_molecules
        }

    def bulk_recalculate(self, molecule_names: Optional[List[str]] = None,
                        parameter_overrides: Optional[dict] = None,
                        use_parallel: bool = None,
                        max_workers: Optional[int] = None) -> dict:
        """
        Unified bulk recalculation method that consolidates sequential and parallel approaches.
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
            parameter_overrides: Optional parameter overrides for all molecules
            use_parallel: Whether to use parallel processing (auto-detect if None)
            max_workers: Maximum number of worker threads for parallel processing
        
        Returns:
            Dictionary with calculation statistics
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        if not molecule_names:
            return {'success': 0, 'failed': 0, 'molecules': []}
        
        # Auto-detect parallel processing preference
        if use_parallel is None:
            use_parallel = len(molecule_names) >= 3  # Parallel beneficial for 3+ molecules
        
        # Apply parameter overrides if provided
        if parameter_overrides:
            print(f"Applying parameter overrides: {parameter_overrides}")
            for mol_name in molecule_names:
                if mol_name in self:
                    molecule = self[mol_name]
                    for param_name, value in parameter_overrides.items():
                        if hasattr(molecule, param_name):
                            setattr(molecule, param_name, value)
                        elif hasattr(molecule, f'_{param_name}'):
                            setattr(molecule, f'_{param_name}', value)
                    
                    # Invalidate caches to force recalculation
                    if hasattr(molecule, '_intensity_valid'):
                        molecule._intensity_valid = False
                    if hasattr(molecule, '_spectrum_valid'):
                        molecule._spectrum_valid = False
        
        # Choose recalculation strategy
        if use_parallel and len(molecule_names) >= 2:
            return self._bulk_recalculate_parallel_worker(molecule_names, max_workers)
        else:
            return self._bulk_recalculate_sequential_worker(molecule_names)

    def _bulk_recalculate_sequential_worker(self, molecule_names: List[str]) -> dict:
        """Sequential bulk recalculation worker."""
        print(f"Recalculating {len(molecule_names)} molecules sequentially...")
        start_time = time.time()
        success_count = 0
        failed_molecules = []
        
        for mol_name in molecule_names:
            try:
                if mol_name in self:
                    molecule = self[mol_name]
                    # Force invalidation and recalculation
                    molecule._intensity_valid = False
                    molecule._spectrum_valid = False
                    molecule._clear_flux_caches()
                    molecule._invalidate_parameter_hash()
                    
                    # Trigger recalculation
                    if hasattr(molecule, 'calculate_intensity'):
                        molecule.calculate_intensity()
                    
                    success_count += 1
                else:
                    failed_molecules.append({'molecule': mol_name, 'error': 'Molecule not found'})
            except Exception as e:
                failed_molecules.append({'molecule': mol_name, 'error': str(e)})
        
        # Clear global caches
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Sequential recalculation completed in {elapsed_time:.2f}s")
        print(f"Successfully recalculated {success_count}/{len(molecule_names)} molecules")
        
        return {
            'success': success_count,
            'failed': len(failed_molecules),
            'molecules': molecule_names,
            'failures': failed_molecules
        }

    def _bulk_recalculate_parallel_worker(self, molecule_names: List[str], 
                                         max_workers: Optional[int] = None) -> dict:
        """Parallel bulk recalculation worker."""
        if max_workers is None:
            max_workers = min(len(molecule_names), mp.cpu_count())
        
        print(f"Recalculating {len(molecule_names)} molecules using {max_workers} worker threads...")
        
        def recalculate_molecule(mol_name):
            """Worker function for recalculating a single molecule"""
            try:
                if mol_name in self:
                    molecule = self[mol_name]
                    # Force invalidation and recalculation
                    molecule._intensity_valid = False
                    molecule._spectrum_valid = False
                    molecule._clear_flux_caches()
                    molecule._invalidate_parameter_hash()
                    
                    # Trigger recalculation
                    if hasattr(molecule, 'calculate_intensity'):
                        molecule.calculate_intensity()
                    
                    return True, mol_name, None
                else:
                    return False, mol_name, "Molecule not found"
            except Exception as e:
                return False, mol_name, str(e)
        
        start_time = time.time()
        success_count = 0
        failed_molecules = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(recalculate_molecule, mol_name): mol_name
                for mol_name in molecule_names
            }
            
            for future in as_completed(future_to_name):
                mol_name = future_to_name[future]
                try:
                    success, result_name, error = future.result()
                    if success:
                        success_count += 1
                    else:
                        failed_molecules.append({'molecule': result_name, 'error': error})
                except Exception as e:
                    failed_molecules.append({'molecule': mol_name, 'error': str(e)})
        
        # Clear global caches
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Parallel recalculation completed in {elapsed_time:.2f}s")
        print(f"Successfully recalculated {success_count}/{len(molecule_names)} molecules")
        
        return {
            'success': success_count,
            'failed': len(failed_molecules),
            'molecules': molecule_names,
            'failures': failed_molecules
        }
    
    def get_visible_molecules(self, return_objects: bool = False) -> Union[set, List['Molecule']]:
        """Get set of visible molecule names or objects for fast operations.
        
        Parameters
        ----------
        return_objects : bool, default False
            If True, returns a list of visible molecule objects.
            If False, returns a set of visible molecule names.
            
        Returns
        -------
        Union[set, List['Molecule']]
            Set of visible molecule names (if return_objects=False) or
            List of visible molecule objects (if return_objects=True)
        """
        # Lazy update of visible molecules set
        current_visible = {name for name, mol in self.items() if bool(mol.is_visible)}
        print(f"Current visible molecules: {current_visible}")
        self._visible_molecules = current_visible
        
        if return_objects:
            return [self[name] for name in current_visible]
        else:
            return current_visible
    
    def bulk_set_visibility(self, is_visible: bool, molecule_names: Optional[List[str]] = None) -> None:
        """Fast visibility update using sets."""
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        # Use set operations for efficiency
        molecule_set = set(molecule_names) & set(self.keys())
        
        for mol_name in molecule_set:
            molecule = self[mol_name]
            molecule.is_visible = is_visible
            
            # Update visibility tracking set
            if is_visible:
                self._visible_molecules.add(mol_name)
            else:
                self._visible_molecules.discard(mol_name)
        
        self._clear_flux_caches()
        print(f"Updated visibility for {len(molecule_set)} molecules")
    
    # ================================
    # Molecule Loading
    # ================================
    def load_molecules(self, molecules_data: List[Dict[str, Any]], 
                       initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                       strategy: str = "auto",
                       max_workers: Optional[int] = None, 
                       batch_size: Optional[int] = None,
                       force_multiprocessing: bool = False) -> Dict[str, Any]:
        """
        Unified molecule loading method that consolidates all loading strategies.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
            strategy: Loading strategy ("auto", "sequential", "parallel", "batched")
            max_workers: Maximum number of worker processes/threads (None for auto-detect)
            batch_size: Number of molecules to process per batch (None for auto-detect)
            force_multiprocessing: If True, forces multiprocessing even for small datasets
        
        Returns:
            Dictionary with loading statistics and results
        """
        if not molecules_data:
            return {"success": 0, "failed": 0, "errors": [], "molecules": []}
        
        print(f"Starting molecule loading for {len(molecules_data)} molecules...")
        start_time = time.time()
        
        # Determine optimal loading strategy
        n_molecules = len(molecules_data)
        if strategy == "auto":
            if force_multiprocessing and self._should_use_multiprocessing(molecules_data, max_workers):
                strategy = "parallel"
            elif n_molecules >= 10:
                strategy = "batched"
            else:
                strategy = "sequential"
        
        print(f"Using loading strategy: {strategy}")
        
        # Set default batch size based on strategy
        if batch_size is None:
            if strategy == "batched":
                batch_size = max(3, min(6, n_molecules // 3))
            else:
                batch_size = n_molecules  # Process all at once
        
        # Filter valid molecules
        valid_molecules_data = []
        for mol_data in molecules_data:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            if mol_name and mol_name not in self:
                valid_molecules_data.append(mol_data)
        
        if not valid_molecules_data:
            print("No valid molecule configurations found!")
            return {'success': 0, 'failed': 0, 'molecules': [], 'errors': ['No valid molecules to load']}
        
        # Execute loading strategy
        results = {'success': 0, 'failed': 0, 'molecules': [], 'errors': []}
        
        if strategy == "parallel":
            print(f"Using parallel processing for {len(valid_molecules_data)} molecules...")
            results = self._load_molecules_parallel_worker(valid_molecules_data, initial_molecule_parameters, max_workers)
        elif strategy == "batched":
            print(f"Using batched processing for {len(valid_molecules_data)} molecules...")
            results = self._load_molecules_batched_worker(valid_molecules_data, initial_molecule_parameters, batch_size, max_workers)
        else:  # sequential
            print(f"Using sequential processing for {len(valid_molecules_data)} molecules...")
            results = self._load_molecules_sequential_worker(valid_molecules_data, initial_molecule_parameters)
        
        # Bulk intensity calculation for loaded molecules
        if results['success'] > 0:
            loaded_molecules = results['molecules']
            if loaded_molecules:
                print(f"Performing bulk intensity calculations for {len(loaded_molecules)} molecules...")
                intensity_results = self._bulk_calculate_intensities(loaded_molecules)
                results['intensity_calculation'] = intensity_results
                
                if intensity_results['failed'] > 0:
                    print(f"Warning: Intensity calculation failed for {intensity_results['failed']} molecules")
        
        # Clear caches after loading
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Molecule loading completed in {elapsed_time:.3f}s")
        print(f"Success: {results['success']}, Failed: {results['failed']}")
        
        return results

    def _load_molecules_parallel_worker(self, molecules_data: List[Dict[str, Any]], 
                                       initial_molecule_parameters: Dict[str, Dict[str, Any]],
                                       max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Consolidated parallel loading worker using multiprocessing."""
        if max_workers is None:
            max_workers = min(len(molecules_data), mp.cpu_count())
        
        # Prepare global parameters
        global_params = {
            "wavelength_range": self._global_wavelength_range,
            "distance": self._global_dist,
            "fwhm": getattr(self, '_global_fwhm', None),
            "stellar_rv": getattr(self, '_global_star_rv', None),
            "intrinsic_line_width": self._global_intrinsic_line_width,
            "model_pixel_res": self._global_model_pixel_res,
            "model_line_width": self._global_model_line_width
        }
        
        # Prepare worker arguments
        worker_args = [
            (mol_data, global_params, initial_molecule_parameters)
            for mol_data in molecules_data
        ]
        
        results = {"success": 0, "failed": 0, "molecules": [], "errors": []}
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_args = {
                    executor.submit(self._create_molecule_worker, args): args[0]
                    for args in worker_args
                }
                
                for future in as_completed(future_to_args):
                    mol_data = future_to_args[future]
                    try:
                        success, result, mol_name = future.result()
                        
                        if success:
                            self[mol_name] = result
                            results["molecules"].append(mol_name)
                            results["success"] += 1
                            print(f"✓ Successfully loaded molecule: {mol_name}")
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"{mol_name}: {result}")
                            print(f"✗ Failed to load molecule '{mol_name}': {result}")
                            
                    except Exception as e:
                        mol_name = mol_data.get("Molecule Name", "Unknown")
                        results["failed"] += 1
                        results["errors"].append(f"{mol_name}: {str(e)}")
                        print(f"✗ Error processing molecule '{mol_name}': {e}")
        
        except Exception as e:
            print(f"Error in parallel molecule loading: {e}")
            # Fall back to sequential loading
            return self._load_molecules_sequential_worker(molecules_data, initial_molecule_parameters)
        
        return results

    def _load_molecules_batched_worker(self, molecules_data: List[Dict[str, Any]], 
                                      initial_molecule_parameters: Dict[str, Any],
                                      batch_size: int,
                                      max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Consolidated batched loading worker using threading."""
        results = {'success': 0, 'failed': 0, 'molecules': [], 'errors': []}
        results_lock = threading.Lock()
        
        # Process in batches to manage memory and I/O
        for i in range(0, len(molecules_data), batch_size):
            batch = molecules_data[i:i + batch_size]
            batch_names = [mol.get('Molecule Name', mol.get('name', 'Unknown')) for mol in batch]
            print(f"Loading batch {i//batch_size + 1}: {batch_names}")
            
            # Use fewer workers than molecules to prevent I/O contention
            if max_workers is None:
                max_workers = min(len(batch), max(1, mp.cpu_count() // 2))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_data = {
                    executor.submit(self._load_single_molecule, mol_data, initial_molecule_parameters): mol_data
                    for mol_data in batch
                }
                
                for future in as_completed(future_to_data):
                    mol_data = future_to_data[future]
                    mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
                    try:
                        success = future.result(timeout=30)
                        with results_lock:
                            if success:
                                results['success'] += 1
                                results['molecules'].append(mol_name)
                                print(f"✓ Loaded {mol_name}")
                            else:
                                results['failed'] += 1
                                results['errors'].append(f"Failed to load {mol_name}")
                                print(f"✗ Failed to load {mol_name}")
                    except Exception as e:
                        print(f"✗ Error loading {mol_name}: {e}")
                        with results_lock:
                            results['failed'] += 1
                            results['errors'].append(f"Error loading {mol_name}: {str(e)}")
        
        return results

    def _load_molecules_sequential_worker(self, molecules_data: List[Dict[str, Any]], 
                                         initial_molecule_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidated sequential loading worker."""
        results = {'success': 0, 'failed': 0, 'molecules': [], 'errors': []}
        
        for i, mol_data in enumerate(molecules_data):
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            print(f"Loading molecule {i+1}/{len(molecules_data)}: {mol_name}")
            try:
                success = self._load_single_molecule(mol_data, initial_molecule_parameters)
                if success:
                    results['success'] += 1
                    results['molecules'].append(mol_name)
                    print(f"✓ Loaded {mol_name}")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to load {mol_name}")
                    print(f"✗ Failed to load {mol_name}")
            except Exception as e:
                print(f"✗ Error loading {mol_name}: {e}")
                results['failed'] += 1
                results['errors'].append(f"Error loading {mol_name}: {str(e)}")
        
        return results

    def _load_single_molecule(self, mol_data: Dict[str, Any], 
                                       initial_molecule_parameters: Dict[str, Any]) -> bool:
        """
        """
        try:
            # Extract molecule name with multiple fallback options
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            if not mol_name:
                print("Error: Missing molecule name in molecule data")
                return False
            
            # Skip if molecule already exists
            if mol_name in self:
                print(f"Warning: Molecule '{mol_name}' already exists, skipping")
                return True
            
            # Get initial parameters for this molecule with fallback
            mol_initial_params = initial_molecule_parameters.get(
                mol_name, 
                initial_molecule_parameters.get('default', {})
            )
            
            # Determine data source and extract parameters efficiently
            use_user_save_data = "Molecule Name" in mol_data
            hitran_data = mol_data.get("hitran_data") if "hitran_data" in mol_data else None
            
            # Extract file path with multiple options
            filepath = mol_data.get("file") or mol_data.get("File Path")
            if not filepath and not hitran_data:
                print(f"Warning: No file path or HITRAN data for molecule '{mol_name}'")
            
            # Extract display label with fallback
            displaylabel = (mol_data.get("label") or 
                           mol_data.get("Molecule Label") or 
                           mol_name)
            
            # Extract molecule-specific parameters with type conversion
            def safe_extract_float(key, default=None):
                """Safely extract and convert float values"""
                value = mol_data.get(key, default)
                if value is not None:
                    try:
                        return float(value)
                    except (ValueError, TypeError):
                        return default
                return default
            
            temp = safe_extract_float("Temp")
            radius = safe_extract_float("Rad") 
            n_mol = safe_extract_float("N_Mol")
            is_visible = mol_data.get("Vis", True)
            color = mol_data.get("Color")
            
            # Create molecule with optimized parameter passing
            molecule = Molecule(
                # Data sources
                user_save_data=mol_data if use_user_save_data else None,
                hitran_data=hitran_data,
                
                # Basic identification
                name=mol_name,
                filepath=filepath,
                displaylabel=displaylabel,
                
                # Global parameters (from MoleculeDict)
                wavelength_range=self._global_wavelength_range,
                distance=self._global_dist,
                fwhm=getattr(self, '_global_fwhm', None),
                stellar_rv=getattr(self, '_global_star_rv', None),
                broad=self._global_intrinsic_line_width,
                model_pixel_res=self._global_model_pixel_res,
                model_line_width=self._global_model_line_width,
                
                # Molecule-specific parameters
                temp=temp,
                radius=radius,
                n_mol=n_mol,
                color=color,
                is_visible=is_visible,
                
                # Initial parameters
                initial_molecule_parameters=mol_initial_params
            )
            
            # Add to dictionary
            self[mol_name] = molecule
            
            # Update fluxes if available
            if hasattr(molecule, 'plot_flux') and molecule.plot_flux is not None:
                self.fluxes[mol_name] = molecule.plot_flux
            
            print(f"Successfully loaded molecule: {mol_name}")
            return True
            
        except Exception as e:
            # Comprehensive error reporting
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            print(f"Error loading molecule '{mol_name}': {e}")
            
            return False

    def get_ndarray_of_attributes(self, attribute_name: str) -> np.ndarray:
        """Get a numpy array of a specific attribute for all molecules."""
        # Use optimized pre-allocated array approach
        values = np.empty(len(self), dtype=np.float64)
        for i, mol in enumerate(self.values()):
            values[i] = getattr(mol, attribute_name, np.nan)
        return values
    
    def get_ndarray_of_line_attributes(self, attribute_name: str) -> np.ndarray:
        """Get a numpy array of a specific line attribute for all molecules."""
        return np.array([mol.lines.get_ndarray_of_attribute(attribute_name) for mol in self.values() if hasattr(mol, 'lines')])

    # Enhanced bulk parameter update methods
    def bulk_update_parameter(self, parameter_name: str, value: Any, molecule_names: Optional[List[str]] = None) -> None:
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        affected_molecules = []
        
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                old_value = getattr(molecule, parameter_name, None)
                
                if hasattr(molecule, f'_{parameter_name}'):
                    setattr(molecule, f'_{parameter_name}', float(value))
                elif parameter_name == 'intrinsic_line_width':
                    molecule._broad = float(value)
                elif parameter_name == 'stellar_rv':
                    molecule.stellar_rv = float(value)
                elif hasattr(molecule, parameter_name):
                    setattr(molecule, parameter_name, value)
                
                if old_value != value:
                    molecule._invalidate_caches_for_parameter(parameter_name)
                    affected_molecules.append(mol_name)
        
        if affected_molecules:
            self._summed_flux_cache.clear()
            for mol_name in affected_molecules:
                self.fluxes.pop(mol_name, None)
        
        print(f"Bulk updated {parameter_name} to {value} for {len(affected_molecules)} molecules")
    
    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], molecule_names: Optional[List[str]] = None) -> None:
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        affected_molecules = []
        
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                old_params = {param: getattr(molecule, param, None) for param in parameter_dict.keys()}
                molecule.bulk_update_parameters(parameter_dict, skip_notification=True)
                
                if any(old_params[param] != parameter_dict[param] for param in parameter_dict.keys()):
                    affected_molecules.append(mol_name)
        
        if affected_molecules:
            self._summed_flux_cache.clear()
            for mol_name in affected_molecules:
                self.fluxes.pop(mol_name, None)
        
        print(f"Bulk updated parameters for {len(affected_molecules)} molecules")
    
    def force_recalculate_all(self, molecule_names: Optional[List[str]] = None) -> None:
        """
        Force recalculation of intensity and spectrum for specified molecules.
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        recalculated_count = 0
        
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                # Force invalidation of all caches
                molecule._intensity_valid = False
                molecule._spectrum_valid = False
                molecule._clear_flux_caches()
                molecule._invalidate_parameter_hash()
                recalculated_count += 1
        
        # Clear global caches
        self._clear_flux_caches()
        
        print(f"Forced recalculation for {recalculated_count} molecules")
    
    def _batch_invalidate_caches(self, updated_molecules: List[Tuple], parameter_name: str) -> None:
        """
        Efficiently invalidate caches for batch-updated molecules.
        
        Args:
            updated_molecules: List of (molecule, param_name, old_value, new_value) tuples
            parameter_name: Name of the parameter that was updated
        """
        # Determine what needs to be invalidated based on parameter type
        invalidate_intensity = parameter_name in ['temp', 'n_mol', 'fwhm', 'intrinsic_line_width']
        invalidate_spectrum = True  # Most parameters affect spectrum
        
        for molecule, param_name, old_value, new_value in updated_molecules:
            if invalidate_intensity:
                molecule._intensity_valid = False
            
            if invalidate_spectrum:
                molecule._spectrum_valid = False
            
            molecule._clear_flux_caches()
            molecule._invalidate_parameter_hash()
            
            # Send notification for this molecule
            molecule._notify_my_parameter_change(param_name, old_value, new_value)
        
        # Clear global caches once
        self._clear_flux_caches()
    
    def _clear_flux_caches(self) -> None:
        """Clear all flux-related caches"""
        self.fluxes.clear()
        self._summed_flux_cache.clear()
        self._cache_wave_data_hash = None
    
    def _on_molecule_parameter_changed(self, molecule_name: str, parameter_name: str, old_value: Any, new_value: Any) -> None:
        if old_value == new_value:
            return
            
        self._summed_flux_cache.clear()
        
        if molecule_name in self.fluxes:
            del self.fluxes[molecule_name]
            
        self._dirty_molecules.add(molecule_name)
    
    def _batch_invalidate_caches_multiple(self, updated_molecules: List, parameter_names: List[str]) -> None:
        """
        Efficiently invalidate caches for molecules with multiple parameter updates.
        
        Args:
            updated_molecules: List of updated molecule objects
            parameter_names: List of parameter names that were updated
        """
        # Determine what needs to be invalidated
        invalidate_intensity = any(param in ['temp', 'n_mol', 'fwhm', 'intrinsic_line_width'] 
                                 for param in parameter_names)
        
        for molecule in updated_molecules:
            if invalidate_intensity:
                molecule._intensity_valid = False
            
            molecule._spectrum_valid = False
            molecule._clear_flux_caches()
            molecule._invalidate_parameter_hash()
        
        # Clear global caches once
        self._clear_flux_caches()
    
    def get_parameter_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all parameters for all molecules.
        
        Returns:
            Dictionary with molecule names as keys and parameter dictionaries as values
        """
        summary = {}
        
        for mol_name, molecule in self.items():
            summary[mol_name] = {
                'temp': molecule.temp,
                'radius': molecule.radius,
                'n_mol': molecule.n_mol,
                'distance': molecule.distance,
                'fwhm': molecule.fwhm,
                'intrinsic_line_width': molecule.intrinsic_line_width,
                'is_visible': molecule.is_visible,
                'stellar_rv': molecule.star_rv,
                'wavelength_range': molecule.wavelength_range,
                'model_pixel_res': molecule.model_pixel_res,
                'model_line_width': molecule.model_line_width
            }
        
        return summary
    
    # Multiprocessing support for molecule loading
    @staticmethod
    def _create_molecule_worker(args):
        """
        Worker function for creating molecules in parallel.
        
        Args:
            args: Tuple of (mol_data, global_params, init_params)
        
        Returns:
            Tuple of (success, molecule_or_error, mol_name)
        """
        try:
            mol_data, global_params, init_params = args
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            
            if not mol_name:
                return False, "Missing molecule name", None
            
            # Import Molecule here to avoid pickling issues
            from .Molecule import Molecule
            
            # Create molecule with provided parameters
            molecule = Molecule(
                user_save_data=mol_data if "Molecule Name" in mol_data else None,
                hitran_data=mol_data.get("hitran_data") if "hitran_data" in mol_data else None,
                name=mol_name,
                filepath=mol_data.get("file") or mol_data.get("File Path"),
                displaylabel=mol_data.get("label") or mol_data.get("Molecule Label", mol_name),
                wavelength_range=global_params.get("wavelength_range"),
                distance=global_params.get("distance"),
                fwhm=global_params.get("fwhm"),
                stellar_rv=global_params.get("stellar_rv"),
                broad=global_params.get("intrinsic_line_width"),
                model_pixel_res=global_params.get("model_pixel_res"),
                model_line_width=global_params.get("model_line_width"),
                temp=mol_data.get("Temp"),
                radius=mol_data.get("Rad"),
                n_mol=mol_data.get("N_Mol"),
                color=mol_data.get("Color"),
                is_visible=mol_data.get("Vis", True),
                initial_molecule_parameters=init_params.get(mol_name, {})
            )
            
            return True, molecule, mol_name
            
        except Exception as e:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            return False, str(e), mol_name
    
    def _should_use_multiprocessing(self, molecules_data: List[Dict[str, Any]], 
                                   max_workers: Optional[int] = None) -> bool:
        """
        Determine whether to use multiprocessing based on workload characteristics.
        
        With ultra-fast loading, multiprocessing overhead may outweigh benefits for small datasets.
        """
        num_molecules = len(molecules_data)
        
        # Don't use multiprocessing for very few molecules
        if num_molecules < 3:
            return False
            
        # Estimate total workload based on file sizes (if available)
        total_estimated_lines = 0
        large_files_count = 0
        
        for mol_data in molecules_data:
            # Try to estimate file size/complexity
            file_path = mol_data.get("hitran_data") or mol_data.get("File Path")
            if file_path and os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    # Rough estimate: 80 bytes per line on average for .par files
                    estimated_lines = file_size // 80
                    total_estimated_lines += estimated_lines
                    
                    # Consider files with >50k lines as "large"
                    if estimated_lines > 50000:
                        large_files_count += 1
                except:
                    # If we can't estimate, assume medium size
                    total_estimated_lines += 25000
        
        # Use multiprocessing if:
        # 1. We have multiple large files (>50k lines each), OR
        # 2. Total estimated lines > 100k AND more than 2 molecules
        if large_files_count >= 2:
            return True
        elif total_estimated_lines > 100000 and num_molecules > 2:
            return True
        else:
            return False

    def bulk_filter_molecules(self, condition_func: Callable[[Dict[str, float]], bool]) -> List[str]:
        """Filter molecules based on a condition function using vectorized operations."""
        matching_molecules = []
        
        # Get all parameter values efficiently
        param_dict = {}
        for param_name in self._parameter_arrays:
            param_dict[param_name] = self.get_parameter_array(param_name)
        
        # Apply condition to each molecule
        for i, mol_name in enumerate(self._molecule_names_array):
            mol_params = {param: values[i] for param, values in param_dict.items()}
            if condition_func(mol_params):
                matching_molecules.append(mol_name)
        
        return matching_molecules
    
    def bulk_conditional_update(self, condition_func: Callable[[Dict[str, float]], bool],
                              parameter_updates: Dict[str, float]) -> None:
        """Update parameters for molecules that meet a condition."""
        matching_molecules = self.bulk_filter_molecules(condition_func)
        if matching_molecules:
            self.bulk_update_parameters(parameter_updates, matching_molecules)
            print(f"Conditionally updated {len(matching_molecules)} molecules")
    
    def add_global_parameter_change_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when global parameters change"""
        if callback not in self._global_parameter_change_callbacks:
            self._global_parameter_change_callbacks.append(callback)
    
    def remove_global_parameter_change_callback(self, callback: Callable) -> None:
        """Remove a callback function"""
        if callback in self._global_parameter_change_callbacks:
            self._global_parameter_change_callbacks.remove(callback)

    def _notify_global_parameter_change(self, parameter_name: str, old_value: Any, new_value: Any) -> None:
        """Notify all registered callbacks of a global parameter change"""
        for callback in self._global_parameter_change_callbacks:
            try:
                callback(parameter_name, old_value, new_value)
            except Exception as e:
                print(f"Error in global parameter change callback: {e}")

    # Global parameter properties with bulk update capabilities
    @property
    def global_distance(self) -> float:
        """Global distance parameter that affects all molecules"""
        return self._global_dist
    
    @global_distance.setter
    def global_distance(self, value: float) -> None:
        """Set global distance and update all molecules"""
        old_value = self._global_dist
        if abs(old_value - value) > 1e-10:
            self._global_dist = value
            self.bulk_update_parameters({'distance': value})
            self._notify_global_parameter_change('distance', old_value, value)
    
    @property
    def global_wavelength_range(self) -> Tuple[float, float]:
        """Global wavelength range parameter"""
        return self._global_wavelength_range
    
    @global_wavelength_range.setter
    def global_wavelength_range(self, value: Tuple[float, float]) -> None:
        """Set global wavelength range"""
        old_value = self._global_wavelength_range
        if value != old_value:
            self._global_wavelength_range = value
            self._notify_global_parameter_change('wavelength_range', old_value, value)

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            #self.cleanup_memory_mapped_storage()
            if hasattr(self, '_flux_memmap_file'):
                try:
                    import os
                    if hasattr(self, '_flux_memmap'):
                        del self._flux_memmap
                    if os.path.exists(self._flux_memmap_file):
                        os.unlink(self._flux_memmap_file)
                    print("Cleaned up memory-mapped storage")
                except Exception as e:
                    print(f"Error cleaning up memory-mapped storage: {e}")
        except:
            pass

    def get_molecule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about molecules in the dictionary."""
        if not self:
            return {"count": 0}
        
        # Update parameter arrays for efficient computation
        #self.update_parameter_arrays()
        
        stats = {
            "count": len(self),
            "visible_count": len(self.get_visible_molecules()),
            "parameter_stats": {}
        }
        
        # Compute statistics for each parameter using numpy
        for param_name, param_array in self._parameter_arrays.items():
            if len(param_array) > 0:
                stats["parameter_stats"][param_name] = {
                    "mean": float(np.mean(param_array)),
                    "std": float(np.std(param_array)),
                    "min": float(np.min(param_array)),
                    "max": float(np.max(param_array)),
                    "median": float(np.median(param_array))
                }
        
        return stats