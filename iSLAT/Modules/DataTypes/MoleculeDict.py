from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import os

# Performance logging
from iSLAT.Modules.Debug.PerformanceLogger import perf_log, log_timing, PerformanceSection

from .Molecule import Molecule
import iSLAT.Constants as default_parms


def _safe_float(data: dict, key: str, default=None):
    """Safely convert a dictionary value to float."""
    value = data.get(key, default)
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default


class MoleculeDict(dict):
    """
    A dictionary to store Molecule objects with their names as keys.
    Provides efficient operations on collections of molecules with unified processing.
    """
    
    def __init__(self, *args, **kwargs):
        # Core caches
        self._summed_flux_cache: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}  # Added param hash
        self._visible_molecules: set = set()
        
        # Global parameters
        self._global_parms = {
            'dist': kwargs.pop('global_distance', default_parms.DEFAULT_DISTANCE),
            'stellar_rv': kwargs.pop('global_stellar_rv', default_parms.DEFAULT_STELLAR_RV),
            'wavelength_range': kwargs.pop('global_wavelength_range', default_parms.WAVELENGTH_RANGE),
            'model_line_width': kwargs.pop('global_model_line_width', default_parms.MODEL_LINE_WIDTH),
            'model_pixel_res': kwargs.pop('global_model_pixel_res', default_parms.MODEL_PIXEL_RESOLUTION),
            #'model_pixel_res': f"{kwargs.pop('global_model_pixel_res', default_parms.MODEL_PIXEL_RESOLUTION):.2e}",
            'intensity_calculation_method': kwargs.pop('global_intensity_calculation_method', "curve_growth")
        }
        
        super().__init__(*args, **kwargs)

        # Create individual properties for backward compatibility
        for param, value in self._global_parms.items():
            setattr(self, f'_global_{param}', value)
        
        # Flag to enable matched spectral sampling (interpolate model to data wavelengths)
        self._match_spectral_sampling = False
        
        self._global_parameter_change_callbacks: List[Callable] = []
        
        from .Molecule import Molecule
        Molecule.add_molecule_parameter_change_callback(self._on_molecule_parameter_changed)
    
    def clear(self):
        """Clear the dictionary of all molecules."""
        super().clear()
        self._clear_all_caches()
        self._visible_molecules = set()  # Clear visibility cache
        print("MoleculeDict cleared.")
    
    def get_visible_molecules(self, return_objects: bool = False) -> Union[set, List['Molecule']]:
        """Get visible molecule names or objects."""
        # Build visible set - use property accessor for proper string-to-bool conversion
        current_visible = {name for name, mol in self.items() if mol.is_visible}
        
        # Only update cache if changed
        if current_visible != self._visible_molecules:
            self._visible_molecules = current_visible
        
        if return_objects:
            return [self[name] for name in current_visible]
        return current_visible
    
    def bulk_set_visibility(self, is_visible: bool, molecule_names: Optional[List[str]] = None) -> None:
        """Update visibility for multiple molecules efficiently."""
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        molecule_set = set(molecule_names) & set(self.keys())
        
        changed_molecules = []
        for mol_name in molecule_set:
            old_visibility = self[mol_name].is_visible
            self[mol_name].is_visible = is_visible
            if old_visibility != is_visible:
                changed_molecules.append(mol_name)
        
        # Only clear caches if visibility actually changed
        if changed_molecules:
            self._selective_cache_invalidation(changed_molecules)
            
        # debug print
        #print(f"Updated visibility for {len(molecule_set)} molecules ({len(changed_molecules)} changed)")
    
    def get_summed_flux(self, wave_data: np.ndarray, visible_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get summed flux from molecules with consistent wavelength grids.
        
        Parameters
        ----------
        wave_data : np.ndarray
            Input wavelength array. When _match_spectral_sampling is True, the model
            flux will be interpolated to match this wavelength grid pixel-by-pixel.
        visible_only : bool, default True
            Whether to include only visible molecules
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Combined wavelengths and summed flux arrays. When _match_spectral_sampling
            is enabled, wavelengths will match wave_data exactly.
        """
        if wave_data is None:
            return np.array([]), np.array([])
            
        molecules = list(self.get_visible_molecules() if visible_only else self.keys())
        if not molecules:
            return np.array([]), np.array([])
        
        # Include match_spectral_sampling flag in cache key for proper invalidation
        cache_key = self._get_flux_cache_key(wave_data, molecules)
        if self._match_spectral_sampling:
            cache_key = hash((cache_key, 'matched_sampling', wave_data.tobytes() if hasattr(wave_data, 'tobytes') else str(wave_data)))
        current_param_hash = self._compute_molecules_parameter_hash(molecules)
        
        # Check cache validity
        if cache_key in self._summed_flux_cache:
            cached_wavelengths, cached_flux, cached_param_hash = self._summed_flux_cache[cache_key]
            
            # Validate cache entry
            if (cached_param_hash == current_param_hash and
                self._validate_summed_cache_entry(cached_wavelengths, cached_flux)):
                # Return copies to prevent accidental modification
                return cached_wavelengths.copy(), cached_flux.copy()
        
        # When match_spectral_sampling is enabled, interpolate each molecule's flux
        # to the input wave_data grid before summing
        use_interpolation = self._match_spectral_sampling
        
        # Initialize output arrays
        if use_interpolation:
            # Use input wavelength grid
            combined_wavelengths = wave_data.copy()
            combined_flux = np.zeros_like(wave_data, dtype=np.float64)
        else:
            combined_wavelengths = None
            combined_flux = None
        
        for mol_name in molecules:
            if mol_name not in self:
                continue
                
            molecule = self[mol_name]
            # Ensure molecule uses global wavelength range
            molecule._wavelength_range = self._global_wavelength_range
            
            try:
                if use_interpolation:
                    # Get flux interpolated to input wavelength array
                    mol_wavelengths, mol_flux = molecule.get_flux(
                        wavelength_array=wave_data,
                        return_wavelengths=True, 
                        interpolate_to_input=True
                    )
                else:
                    # Get flux on native model grid
                    mol_wavelengths, mol_flux = molecule.get_flux(return_wavelengths=True, interpolate_to_input=False)
                
                if mol_wavelengths is not None and mol_flux is not None and len(mol_wavelengths) > 0:
                    # Ensure finite values
                    if not np.all(np.isfinite(mol_flux)):
                        mol_flux = np.nan_to_num(mol_flux, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if use_interpolation:
                        # Direct summation on matched grid
                        combined_flux += mol_flux
                    else:
                        if combined_wavelengths is None:
                            # First molecule - use its grid as the reference
                            combined_wavelengths = mol_wavelengths.copy()
                            combined_flux = mol_flux.copy()
                        else:
                            # All subsequent molecules should have identical grids - directly sum
                            if len(mol_wavelengths) != len(combined_wavelengths):
                                raise ValueError(f"Grid size mismatch for {mol_name}: {len(mol_wavelengths)} vs {len(combined_wavelengths)}. This should not happen with consistent spectrum calculation.")
                            
                            # Direct summation
                            combined_flux += mol_flux
                            
            except Exception as e:
                print(f"Warning: Failed to get flux for molecule {mol_name}: {e}")
        
        # Handle case where no valid molecules were found
        if combined_wavelengths is None:
            return np.array([]), np.array([])
        
        # Cache result with parameter hash for validation
        self._cache_summed_flux_result(cache_key, combined_wavelengths, combined_flux, current_param_hash)
        
        return combined_wavelengths, combined_flux
    
    def _get_flux_cache_key(self, wave_data: np.ndarray, molecules: List[str]) -> int:
        """Generate cache key for flux calculations."""
        try:
            # Include input wavelength array
            wave_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else str(wave_data)
            
            # Include molecule set and their visibility
            molecule_set = frozenset(molecules)
            
            # Include RV shifts for each molecule
            rv_shifts = frozenset((name, self[name].rv_shift) for name in molecules if name in self)
            
            # Include global parameters that affect flux calculation
            global_params = (
                tuple(self._global_wavelength_range),
                self._global_model_pixel_res,
                self._global_dist
            )
            
            return hash((wave_hash, molecule_set, rv_shifts, global_params))
        except (TypeError, ValueError):
            # Fallback for unhashable types
            return hash((str(wave_data), str(sorted(molecules))))
    
    def _compute_molecules_parameter_hash(self, molecules: List[str]) -> int:
        """Compute combined parameter hash for a set of molecules."""
        param_hashes = []
        for mol_name in molecules:
            if mol_name in self:
                molecule = self[mol_name]
                if hasattr(molecule, 'get_parameter_hash'):
                    param_hashes.append(molecule.get_parameter_hash('full'))
        return hash(tuple(param_hashes))
    
    def _validate_summed_cache_entry(self, wavelengths: np.ndarray, flux: np.ndarray) -> bool:
        """Validate that a summed flux cache entry is still valid."""
        if wavelengths is None or flux is None:
            return False
        
        if len(wavelengths) == 0 or len(flux) == 0:
            return False
        
        if len(wavelengths) != len(flux):
            return False
        
        # Check for finite values
        if not np.all(np.isfinite(wavelengths)) or not np.all(np.isfinite(flux)):
            return False
        
        return True
    
    def _cache_summed_flux_result(self, cache_key: int, wavelengths: np.ndarray, 
                                 flux: np.ndarray, param_hash: int) -> None:
        """Cache the summed flux result with proper memory management."""
        # Store copies to prevent modification
        self._summed_flux_cache[cache_key] = (
            wavelengths.copy(),
            flux.copy(),
            param_hash
        )
        
        # Limit cache size - use more intelligent eviction
        cache_limit = 50
        if len(self._summed_flux_cache) > cache_limit:
            # Remove oldest 25% of entries
            oldest_keys = list(self._summed_flux_cache.keys())[:-int(cache_limit * 0.75)]
            for key in oldest_keys:
                del self._summed_flux_cache[key]

    # ================================
    # Parallel Intensity Calculations
    # ================================
    def calculate_intensities_parallel(self, molecule_names: Optional[List[str]] = None,
                                       max_workers: Optional[int] = None,
                                       show_progress: bool = False) -> Dict[str, Any]:
        """
        Calculate intensities for multiple molecules in parallel using threads.
        
        This method triggers lazy loading and intensity calculations for all 
        specified molecules concurrently, significantly speeding up startup 
        and recalculation operations.
        
        Note: Uses ThreadPoolExecutor because NumPy releases the GIL during
        heavy computations, allowing true parallelism for numeric operations.
        
        Parameters
        ----------
        molecule_names : List[str], optional
            List of molecule names to calculate. If None, calculates all visible molecules.
        max_workers : int, optional
            Maximum number of parallel workers. Defaults to min(CPU cores, 6).
        show_progress : bool, default False
            Whether to print progress information.
            
        Returns
        -------
        dict
            Dictionary with 'success' count, 'failed' count, 'times' dict, and 'errors' list.
        """
        section = PerformanceSection("MoleculeDict.calculate_intensities_parallel")
        section.start()
        
        if molecule_names is None:
            molecule_names = list(self.get_visible_molecules())
        
        if not molecule_names:
            section.end()
            return {'success': 0, 'failed': 0, 'times': {}, 'errors': []}
        
        # Filter to existing molecules
        valid_molecules = [name for name in molecule_names if name in self]
        
        if not valid_molecules:
            section.end()
            return {'success': 0, 'failed': 0, 'times': {}, 'errors': []}
        
        if show_progress:
            print(f"[PARALLEL] Starting intensity calculations for {len(valid_molecules)} molecules...")
        
        # Determine worker count
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 6, len(valid_molecules))
        
        results = {
            'success': 0,
            'failed': 0,
            'times': {},
            'errors': []
        }
        
        section.mark("submit_jobs")
        
        def calculate_single_molecule(mol_name: str) -> Tuple[str, bool, float, Optional[str]]:
            """Calculate intensity for a single molecule (runs in thread)."""
            start = time.perf_counter()
            try:
                molecule = self[mol_name]
                # Set wavelength range
                molecule._wavelength_range = self._global_wavelength_range
                # Trigger lazy calculation
                molecule._ensure_intensity_calculated()
                elapsed = time.perf_counter() - start
                return (mol_name, True, elapsed, None)
            except Exception as e:
                elapsed = time.perf_counter() - start
                return (mol_name, False, elapsed, str(e))
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(calculate_single_molecule, name): name 
                      for name in valid_molecules}
            
            for future in as_completed(futures):
                mol_name, success, elapsed, error = future.result()
                results['times'][mol_name] = elapsed
                
                if success:
                    results['success'] += 1
                    if show_progress:
                        print(f"  [DONE] {mol_name}: {elapsed:.3f}s")
                else:
                    results['failed'] += 1
                    results['errors'].append(f"{mol_name}: {error}")
                    if show_progress:
                        print(f"  [FAIL] {mol_name}: {error}")
        
        section.mark("all_complete")
        #elapsed = section.end()
        
        total_time = sum(results['times'].values())
        if show_progress:
            print(f"[PARALLEL] Complete: {results['success']} succeeded, {results['failed']} failed")
            print(f"[PARALLEL] Total calculation time: {total_time:.2f}s (wall time: {elapsed:.2f}s)")
            print(f"[PARALLEL] Speedup: {total_time / max(elapsed, 0.001):.1f}x")
        
        return results

    def get_summed_flux_parallel(self, wave_data: np.ndarray, visible_only: bool = True,
                                 max_workers: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get summed flux with parallel pre-calculation of intensities.
        
        This is an optimized version of get_summed_flux that first calculates
        all molecule intensities in parallel before summing.
        
        Parameters
        ----------
        wave_data : np.ndarray
            Input wavelength array. When _match_spectral_sampling is True, the model
            flux will be interpolated to match this wavelength grid pixel-by-pixel.
        visible_only : bool, default True
            Whether to include only visible molecules
        max_workers : int, optional
            Maximum number of parallel workers
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Combined wavelengths and summed flux arrays. When _match_spectral_sampling
            is enabled, wavelengths will match wave_data exactly.
        """
        section = PerformanceSection("MoleculeDict.get_summed_flux_parallel")
        section.start()
        
        if wave_data is None:
            section.end()
            return np.array([]), np.array([])
        
        molecules = list(self.get_visible_molecules() if visible_only else self.keys())
        if not molecules:
            section.end()
            return np.array([]), np.array([])
        
        # Check cache first - include match_spectral_sampling flag in cache key
        cache_key = self._get_flux_cache_key(wave_data, molecules)
        if self._match_spectral_sampling:
            cache_key = hash((cache_key, 'matched_sampling', wave_data.tobytes() if hasattr(wave_data, 'tobytes') else str(wave_data)))
        current_param_hash = self._compute_molecules_parameter_hash(molecules)
        
        if cache_key in self._summed_flux_cache:
            cached_wavelengths, cached_flux, cached_param_hash = self._summed_flux_cache[cache_key]
            if (cached_param_hash == current_param_hash and
                self._validate_summed_cache_entry(cached_wavelengths, cached_flux)):
                section.mark("cache_hit")
                section.end()
                return cached_wavelengths.copy(), cached_flux.copy()
        
        section.mark("parallel_intensity_calc")
        
        # Pre-calculate all intensities in parallel
        self.calculate_intensities_parallel(molecules, max_workers=max_workers)
        
        section.mark("sum_fluxes")
        
        # Check if we need to interpolate to input wavelengths
        use_interpolation = self._match_spectral_sampling
        
        # Parallelize get_flux calls since spectrum convolution is CPU-intensive
        # NumPy releases the GIL, so threads provide real parallelism here
        def get_molecule_flux(mol_name: str) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], float]:
            """Get flux for a single molecule (runs in thread)."""
            import time as _time
            _start = _time.perf_counter()
            
            if mol_name not in self:
                return (mol_name, None, None, 0.0)
            
            molecule = self[mol_name]
            molecule._wavelength_range = self._global_wavelength_range
            
            try:
                if use_interpolation:
                    # Get flux interpolated to input wavelength array
                    mol_wavelengths, mol_flux = molecule.get_flux(
                        wavelength_array=wave_data,
                        return_wavelengths=True, 
                        interpolate_to_input=True
                    )
                else:
                    mol_wavelengths, mol_flux = molecule.get_flux(return_wavelengths=True, interpolate_to_input=False)
                _elapsed = _time.perf_counter() - _start
                return (mol_name, mol_wavelengths, mol_flux, _elapsed)
            except Exception as e:
                print(f"Warning: Failed to get flux for molecule {mol_name}: {e}")
                return (mol_name, None, None, _time.perf_counter() - _start)
        
        # Run get_flux in parallel
        flux_results = {}
        flux_times = {}
        
        if max_workers is None:
            flux_workers = min(os.cpu_count() or 4, 6, len(molecules))
        else:
            flux_workers = max_workers
        
        with ThreadPoolExecutor(max_workers=flux_workers) as executor:
            futures = {executor.submit(get_molecule_flux, name): name for name in molecules}
            
            for future in as_completed(futures):
                mol_name, mol_wavelengths, mol_flux, elapsed = future.result()
                flux_times[mol_name] = elapsed
                if mol_wavelengths is not None and mol_flux is not None:
                    flux_results[mol_name] = (mol_wavelengths, mol_flux)
        
        # Print timing info
        #for mol_name, elapsed in sorted(flux_times.items(), key=lambda x: x[1]):
            #print(f"  [SUM] get_flux({mol_name}): {elapsed*1000:.1f}ms")
        
        #total_flux_time = sum(flux_times.values())
        #print(f"[SUM] Total get_flux time: {total_flux_time:.2f}s (parallel)")
        
        # Now combine the results (fast array operations)
        if use_interpolation:
            # Use input wavelength grid
            combined_wavelengths = wave_data.copy()
            combined_flux = np.zeros_like(wave_data, dtype=np.float64)
        else:
            combined_wavelengths = None
            combined_flux = None
        
        for mol_name in molecules:
            if mol_name not in flux_results:
                continue
            
            mol_wavelengths, mol_flux = flux_results[mol_name]
            
            if len(mol_wavelengths) > 0:
                if not np.all(np.isfinite(mol_flux)):
                    mol_flux = np.nan_to_num(mol_flux, nan=0.0, posinf=0.0, neginf=0.0)
                
                if use_interpolation:
                    # Direct summation on matched grid
                    combined_flux += mol_flux
                else:
                    if combined_wavelengths is None:
                        combined_wavelengths = mol_wavelengths.copy()
                        combined_flux = mol_flux.copy()
                    else:
                        if len(mol_wavelengths) != len(combined_wavelengths):
                            raise ValueError(f"Grid size mismatch for {mol_name}")
                        combined_flux += mol_flux
        
        if combined_wavelengths is None:
            section.end()
            return np.array([]), np.array([])
        
        # Cache result
        self._cache_summed_flux_result(cache_key, combined_wavelengths, combined_flux, current_param_hash)
        
        section.end()
        section.get_breakdown(print_output=True)
        
        return combined_wavelengths, combined_flux

    def process_molecules(self, operation: str, molecule_names: Optional[List[str]] = None, 
                         use_parallel: bool = None, max_workers: Optional[int] = None, 
                         **kwargs) -> Dict[str, Any]:
        """Unified method for processing molecules with different operations.
        
        Parameters
        ----------
        operation : str
            Type of operation: 'recalculate', 'intensity', 'batch_intensity'
        molecule_names : Optional[List[str]]
            Molecules to process (None for all)
        use_parallel : bool, optional
            Whether to use parallel processing (auto-detect if None)
        max_workers : Optional[int]
            Maximum number of workers for parallel processing
        **kwargs
            Additional parameters specific to the operation
            
        Returns
        -------
        Dict[str, Any]
            Results dictionary with success/failure statistics
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        valid_molecules = [name for name in molecule_names if name in self]
        if not valid_molecules:
            return {'success': 0, 'failed': 0, 'molecules': [], 'errors': []}
        
        # Auto-detect parallel processing
        if use_parallel is None:
            use_parallel = len(valid_molecules) >= 3
        
        # Apply parameter overrides if provided - use bulk update for efficiency
        if 'parameter_overrides' in kwargs and kwargs['parameter_overrides'] is not None:
            print("Applying parameter overrides to multiple molecules...")
            self._apply_bulk_parameter_overrides(valid_molecules, kwargs['parameter_overrides'])
        
        # Execute operation
        if use_parallel and len(valid_molecules) >= 2:
            return self._process_molecules_parallel(operation, valid_molecules, max_workers, **kwargs)
        else:
            return self._process_molecules_sequential(operation, valid_molecules, **kwargs)
    
    def _apply_bulk_parameter_overrides(self, molecule_names: List[str], overrides: Dict[str, Any]) -> None:
        """Apply parameter overrides to multiple molecules efficiently using bulk updates."""
        if not overrides:
            return
            
        # Apply overrides to all molecules in one pass using their bulk_update_parameters method
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                if hasattr(molecule, 'bulk_update_parameters'):
                    # Use the molecule's efficient bulk update method
                    molecule.bulk_update_parameters(overrides.copy(), skip_notification=True)
                else:
                    # Fallback to individual parameter setting (should not be needed)
                    self._apply_parameter_overrides_fallback(molecule, overrides)
    
    def _apply_parameter_overrides_fallback(self, molecule: 'Molecule', overrides: Dict[str, Any]) -> None:
        """Fallback method for applying parameter overrides (for compatibility)."""
        for param_name, value in overrides.items():
            if hasattr(molecule, param_name):
                setattr(molecule, param_name, value)
            elif hasattr(molecule, f'_{param_name}'):
                setattr(molecule, f'_{param_name}', value)
    
    def _process_molecules_sequential(self, operation: str, molecule_names: List[str], **kwargs) -> Dict[str, Any]:
        """Process molecules sequentially."""
        results = {'success': 0, 'failed': 0, 'molecules': molecule_names, 'errors': []}
        
        for mol_name in molecule_names:
            try:
                if self._execute_molecule_operation(mol_name, operation, **kwargs):
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Operation failed for {mol_name}")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error processing {mol_name}: {str(e)}")
        
        self._clear_all_caches()
        return results
    
    def _process_molecules_parallel(self, operation: str, molecule_names: List[str], 
                                   max_workers: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Process molecules in parallel."""
        if max_workers is None:
            max_workers = min(len(molecule_names), mp.cpu_count())
        
        results = {'success': 0, 'failed': 0, 'molecules': molecule_names, 'errors': []}
        
        def worker(mol_name):
            try:
                return self._execute_molecule_operation(mol_name, operation, **kwargs), mol_name, None
            except Exception as e:
                return False, mol_name, str(e)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker, mol_name): mol_name for mol_name in molecule_names}
            
            for future in as_completed(futures):
                try:
                    success, mol_name, error = future.result()
                    if success:
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                        results['errors'].append(f"{mol_name}: {error or 'Operation failed'}")
                except Exception as e:
                    mol_name = futures[future]
                    results['failed'] += 1
                    results['errors'].append(f"{mol_name}: {str(e)}")
        
        self._clear_all_caches()
        return results
    
    def _execute_molecule_operation(self, mol_name: str, operation: str, **kwargs) -> bool:
        """Execute a specific operation on a molecule."""
        if mol_name not in self:
            return False
            
        molecule = self[mol_name]
        
        if operation == 'recalculate':
            if hasattr(molecule, 'force_recalculate'):
                molecule.force_recalculate()
            else:
                molecule.clear_all_caches()
                if hasattr(molecule, 'calculate_intensity'):
                    molecule.calculate_intensity()
            return True
            
        elif operation == 'intensity':
            if hasattr(molecule, '_ensure_intensity_calculated'):
                molecule._ensure_intensity_calculated()
            elif hasattr(molecule, 'calculate_intensity'):
                molecule.calculate_intensity()
            return True
            
        elif operation == 'batch_intensity':
            parameter_combinations = kwargs.get('parameter_combinations', [])
            method = kwargs.get('method', self.global_intensity_calculation_method if hasattr(self, 'global_intensity_calculation_method') else "curve_growth")
            
            if not parameter_combinations:
                return False
                
            if hasattr(molecule, 'intensity') and hasattr(molecule.intensity, 'bulk_parameter_update_vectorized'):
                try:
                    molecule.intensity.bulk_parameter_update_vectorized(parameter_combinations, method=method)
                    return True
                except Exception:
                    pass
            
            # Fallback to individual calculations
            for params in parameter_combinations:
                if hasattr(molecule, 'intensity') and molecule.intensity:
                    molecule.intensity.calc_intensity(
                        t_kin=params.get('t_kin', 300),
                        n_mol=params.get('n_mol', 1e17),
                        dv=params.get('dv', 1.0),
                        method=method
                    )
            return True
            
        return False

    # Convenience methods that use the unified process_molecules method
    def bulk_recalculate(self, molecule_names: Optional[List[str]] = None,
                        parameter_overrides: Optional[dict] = None,
                        use_parallel: bool = None,
                        max_workers: Optional[int] = None) -> dict:
        """Recalculate multiple molecules."""
        return self.process_molecules(
            'recalculate', 
            molecule_names=molecule_names,
            use_parallel=use_parallel,
            max_workers=max_workers,
            parameter_overrides=parameter_overrides
        )
    
    def bulk_calculate_intensities(self, molecule_names: Optional[List[str]] = None,
                                  use_parallel: bool = None,
                                  max_workers: Optional[int] = None) -> dict:
        """Calculate intensities for multiple molecules."""
        return self.process_molecules(
            'intensity',
            molecule_names=molecule_names,
            use_parallel=use_parallel,
            max_workers=max_workers
        )
    
    def batch_intensity_calculation(self, parameter_combinations: List[Dict[str, float]], 
                                   molecule_names: Optional[List[str]] = None,
                                   method: str = "curve_growth") -> Dict[str, Any]:
        """Calculate intensities for multiple parameter combinations."""
        results = self.process_molecules(
            'batch_intensity',
            molecule_names=molecule_names,
            use_parallel=True,
            parameter_combinations=parameter_combinations,
            method=method
        )
        return results
    
    # ================================
    # Unified Molecule Loading
    # ================================
    def load_molecules(self, molecules_data: List[Dict[str, Any]], 
                       initial_molecule_parameters: Dict[str, Dict[str, Any]],
                       update_global_parameters: bool = True, 
                       strategy: str = "auto",
                       max_workers: Optional[int] = None, 
                       force_multiprocessing: bool = False,
                       defer_calculations: bool = True) -> Dict[str, Any]:
        """Unified molecule loading method with automatic strategy selection.
        
        Parameters
        ----------
        molecules_data : List[Dict[str, Any]]
            List of molecule data dictionaries
        initial_molecule_parameters : Dict[str, Dict[str, Any]]
            Initial parameters for each molecule type
        update_global_parameters : bool, default True
            Whether to update global parameters from first molecule
        strategy : str, default "auto"
            Loading strategy: "auto", "sequential", or "parallel"
        max_workers : Optional[int]
            Maximum number of workers for parallel loading
        force_multiprocessing : bool, default False
            Force use of multiprocessing
        defer_calculations : bool, default True
            If True, defer intensity calculations until needed (faster startup).
            If False, calculate intensities immediately after loading.
        """
        section = PerformanceSection("MoleculeDict.load_molecules")
        section.start()
        
        if not molecules_data:
            return {"success": 0, "failed": 0, "errors": [], "molecules": []}
        
        # debug print
        #print(f"Loading {len(molecules_data)} molecules...")
        
        # Filter valid molecules
        section.mark("filter_valid")
        valid_molecules_data = [
            mol_data for mol_data in molecules_data 
            if (mol_data.get("Molecule Name") or mol_data.get("name")) 
            and (mol_data.get("Molecule Name") or mol_data.get("name")) not in self
        ]
        
        if not valid_molecules_data:
            return {'success': 0, 'failed': 0, 'molecules': [], 'errors': ['No valid molecules to load']}
        
        # Determine strategy
        n_molecules = len(valid_molecules_data)
        if strategy == "auto":
            if force_multiprocessing and self._should_use_multiprocessing(valid_molecules_data):
                strategy = "parallel"
            elif n_molecules >= 8:
                strategy = "parallel"
            else:
                strategy = "sequential"

        if update_global_parameters:
            self._global_dist = float(valid_molecules_data[0].get("Dist", default_parms.DEFAULT_DISTANCE))
            self._global_stellar_rv = float(valid_molecules_data[0].get("StellarRV", default_parms.DEFAULT_STELLAR_RV))

        # Execute loading
        section.mark("execute_loading")
        #start_time = time.time()
        if strategy == "parallel":
            results = self._load_molecules_parallel(valid_molecules_data, initial_molecule_parameters, max_workers)
        else:
            results = self._load_molecules_sequential(valid_molecules_data, initial_molecule_parameters)
        section.mark("loading_complete")

        # Calculate intensities for loaded molecules (unless deferred for faster startup)
        section.mark("calculate_intensities")
        if results['success'] > 0 and not defer_calculations:
            print("Calculating intensities immediately (defer_calculations=False)...")
            intensity_results = self.bulk_calculate_intensities(results['molecules'])
            results['intensity_calculation'] = intensity_results
        elif results['success'] > 0:
            print(f"Deferring intensity calculations for {results['success']} molecules")
            results['intensity_calculation'] = {'deferred': True, 'molecules': results['molecules']}
        section.mark("intensities_complete")
        
        #elapsed_time = time.time() - start_time
        #print(f"Loading completed in {elapsed_time:.2f}s - Success: {results['success']}, Failed: {results['failed']}")
        
        section.end()
        section.get_breakdown(print_output=True)
        
        return results
    
    def _load_molecules_sequential(self, molecules_data: List[Dict[str, Any]], 
                                  initial_molecule_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Load molecules sequentially."""
        results = {'success': 0, 'failed': 0, 'molecules': [], 'errors': []}
        
        for mol_data in molecules_data:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            try:
                if self._load_single_molecule(mol_data, initial_molecule_parameters):
                    results['success'] += 1
                    results['molecules'].append(mol_name)
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to load {mol_name}")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error loading {mol_name}: {str(e)}")
        
        return results
    
    def _load_molecules_parallel(self, molecules_data: List[Dict[str, Any]], 
                                initial_molecule_parameters: Dict[str, Dict[str, Any]],
                                max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Load molecules in parallel using multiprocessing."""
        if max_workers is None:
            max_workers = min(len(molecules_data), mp.cpu_count())
        
        # Prepare worker arguments
        worker_args = [
            (mol_data, self._global_parms, initial_molecule_parameters)
            for mol_data in molecules_data
        ]
        
        results = {"success": 0, "failed": 0, "molecules": [], "errors": []}
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_data = {
                    executor.submit(self._create_molecule_worker, args): args[0]
                    for args in worker_args
                }
                
                for future in as_completed(future_to_data):
                    mol_data = future_to_data[future]
                    try:
                        success, result, mol_name = future.result()
                        
                        if success:
                            self[mol_name] = result
                            results["molecules"].append(mol_name)
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                            results["errors"].append(f"{mol_name}: {result}")
                            
                    except Exception as e:
                        mol_name = mol_data.get("Molecule Name", "Unknown")
                        results["failed"] += 1
                        results["errors"].append(f"{mol_name}: {str(e)}")
        
        except Exception as e:
            print(f"Parallel loading failed, falling back to sequential: {e}")
            return self._load_molecules_sequential(molecules_data, initial_molecule_parameters)
        
        return results

    def _load_single_molecule(self, mol_data: Dict[str, Any], 
                             initial_molecule_parameters: Dict[str, Any]) -> bool:
        """Load a single molecule with optimized parameter extraction."""
        try:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            if not mol_name or mol_name in self:
                return mol_name in self  # Return True if already exists, False if no name
            
            # Get initial parameters
            mol_initial_params = initial_molecule_parameters.get(
                mol_name, 
                initial_molecule_parameters.get('default_initial_params', {})
            )

            # Extract parameters efficiently
            use_user_save_data = "Molecule Name" in mol_data
            
            # Create molecule using module-level _safe_float for performance
            molecule = Molecule(
                user_save_data=mol_data if use_user_save_data else None,
                hitran_data=mol_data.get("hitran_data"),
                name=mol_name,
                filepath=mol_data.get("file") or mol_data.get("File Path"),
                displaylabel=mol_data.get("label") or mol_data.get("Molecule Label", mol_name),
                temp=_safe_float(mol_data, "Temp"),
                radius=_safe_float(mol_data, "Rad"),
                n_mol=_safe_float(mol_data, "N_Mol"),
                color=mol_data.get("Color"),
                is_visible=mol_data.get("Vis", True),
                wavelength_range=self._global_wavelength_range,
                distance=self._global_dist,
                intensity_calculation_method=self._global_intensity_calculation_method,
                fwhm=_safe_float(mol_data, "FWHM"),
                rv_shift=_safe_float(mol_data, "RV Shift"),
                broad=mol_data.get("Broad"),
                model_pixel_res=self._global_model_pixel_res,
                model_line_width=self._global_model_line_width,
                initial_molecule_parameters=mol_initial_params
            )
            
            self[mol_name] = molecule
            return True
            
        except Exception as e:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            print(f"Error loading molecule '{mol_name}': {e}")
            return False

    def bulk_update_parameters(self, parameter_dict: Dict[str, Any], 
                              molecule_names: Optional[List[str]] = None) -> None:
        """Update parameters for multiple molecules efficiently."""
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        affected_molecules = []
        for mol_name in molecule_names:
            if mol_name in self:
                molecule = self[mol_name]
                old_params = {param: getattr(molecule, param, None) for param in parameter_dict.keys()}
                molecule.bulk_update_parameters(parameter_dict, skip_notification=True)
                
                # Check if any parameters actually changed
                if any(old_params[param] != parameter_dict[param] for param in parameter_dict.keys() if param in old_params):
                    affected_molecules.append(mol_name)
        
        # Only invalidate cache for molecules that actually changed
        if affected_molecules:
            self._selective_cache_invalidation(affected_molecules)
        
        # debug print
        #print(f"Bulk updated parameters for {len(affected_molecules)} molecules (out of {len(molecule_names)} requested)")
    
    def _clear_all_caches(self) -> None:
        """Clear all caches."""
        self._summed_flux_cache.clear()
    
    def _selective_cache_invalidation(self, molecule_names: Optional[List[str]] = None) -> None:
        """Selectively invalidate cache entries that involve specific molecules."""
        if not molecule_names:
            self._clear_all_caches()
            return
        
        # Convert to set for faster lookup
        mol_set = set(molecule_names)
        
        # Find cache entries that need to be invalidated
        keys_to_remove = []
        for cache_key in list(self._summed_flux_cache.keys()):
            # For now, we'll use a conservative approach and remove all entries
            # In the future, we could track which molecules contributed to each cache entry
            keys_to_remove.append(cache_key)
        
        # Remove invalidated entries
        for key in keys_to_remove:
            if key in self._summed_flux_cache:
                del self._summed_flux_cache[key]
    
    def _on_molecule_parameter_changed(self, molecule_name: str, parameter_name: str, 
                                      old_value: Any, new_value: Any) -> None:
        """Handle molecule parameter changes with selective cache invalidation."""
        if old_value != new_value:
            # For flux-affecting parameters, clear all caches for now
            # In the future, we could be more selective
            if parameter_name in ['temp', 'radius', 'n_mol', 'rv_shift', 'fwhm', 'broad', 'distance', 'wavelength_range']:
                self._selective_cache_invalidation([molecule_name])
            else:
                # For other parameters, we might not need to clear all caches
                pass
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'summed_flux_cache_size': len(self._summed_flux_cache),
            'summed_flux_cache_entries': []
        }
        
        # Get details about each cache entry
        for cache_key, (wavelengths, flux, param_hash) in self._summed_flux_cache.items():
            entry_stats = {
                'cache_key': cache_key,
                'wavelength_points': len(wavelengths) if wavelengths is not None else 0,
                'flux_points': len(flux) if flux is not None else 0,
                'param_hash': param_hash,
                'memory_size_mb': (wavelengths.nbytes + flux.nbytes) / 1024 / 1024 if wavelengths is not None and flux is not None else 0
            }
            stats['summed_flux_cache_entries'].append(entry_stats)
        
        return stats
    
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """Validate cache integrity and return diagnostic information."""
        validation_results = {
            'valid_entries': 0,
            'invalid_entries': 0,
            'corrupted_entries': [],
            'total_memory_mb': 0
        }
        
        corrupted_keys = []
        for cache_key, (wavelengths, flux, param_hash) in self._summed_flux_cache.items():
            try:
                if self._validate_summed_cache_entry(wavelengths, flux):
                    validation_results['valid_entries'] += 1
                    if wavelengths is not None and flux is not None:
                        validation_results['total_memory_mb'] += (wavelengths.nbytes + flux.nbytes) / 1024 / 1024
                else:
                    validation_results['invalid_entries'] += 1
                    corrupted_keys.append(cache_key)
                    validation_results['corrupted_entries'].append({
                        'cache_key': cache_key,
                        'reason': 'Failed validation'
                    })
            except Exception as e:
                validation_results['invalid_entries'] += 1
                corrupted_keys.append(cache_key)
                validation_results['corrupted_entries'].append({
                    'cache_key': cache_key,
                    'reason': str(e)
                })
        
        # Clean up corrupted entries
        for key in corrupted_keys:
            if key in self._summed_flux_cache:
                del self._summed_flux_cache[key]
        
        return validation_results
    
    def refresh_summed_flux_cache(self, wave_data: np.ndarray = None, visible_only: bool = True) -> Dict[str, Any]:
        """Manually refresh the summed flux cache and return statistics."""
        # Clear existing cache
        old_cache_size = len(self._summed_flux_cache)
        self._clear_all_caches()
        
        # If wave_data is provided, pre-populate cache
        if wave_data is not None:
            molecules = list(self.get_visible_molecules() if visible_only else self.keys())
            if molecules:
                # This will populate the cache
                result_wavelengths, result_flux = self.get_summed_flux(wave_data, visible_only)
                
                return {
                    'cache_cleared': True,
                    'old_cache_size': old_cache_size,
                    'new_cache_size': len(self._summed_flux_cache),
                    'molecules_processed': len(molecules),
                    'result_points': len(result_wavelengths) if result_wavelengths is not None else 0
                }
        
        return {
            'cache_cleared': True,
            'old_cache_size': old_cache_size,
            'new_cache_size': 0,
            'molecules_processed': 0,
            'result_points': 0
        }
    
    @staticmethod
    def _create_molecule_worker(args):
        """Worker function for parallel molecule creation."""
        try:
            mol_data, global_params, init_params = args
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            
            if not mol_name:
                return False, "Missing molecule name", None
            
            from .Molecule import Molecule
            
            def safe_float(key, default=None):
                value = mol_data.get(key, default)
                try:
                    return float(value) if value is not None else default
                except (ValueError, TypeError):
                    return default
            
            molecule = Molecule(
                user_save_data=mol_data if "Molecule Name" in mol_data else None,
                hitran_data=mol_data.get("hitran_data"),
                name=mol_name,
                filepath=mol_data.get("file") or mol_data.get("File Path"),
                displaylabel=mol_data.get("label") or mol_data.get("Molecule Label", mol_name),
                temp=safe_float("Temp"),
                radius=safe_float("Rad"),
                n_mol=safe_float("N_Mol"),
                color=mol_data.get("Color"),
                is_visible=mol_data.get("Vis", True),
                fwhm=safe_float("FWHM"),
                rv_shift=safe_float("RV Shift"),
                broad=mol_data.get("Broad"),
                initial_molecule_parameters=init_params.get(mol_name, {}),
                **global_params
            )
            
            return True, molecule, mol_name
            
        except Exception as e:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name", "Unknown")
            return False, str(e), mol_name
    
    def _should_use_multiprocessing(self, molecules_data: List[Dict[str, Any]]) -> bool:
        """Determine if multiprocessing would be beneficial."""
        num_molecules = len(molecules_data)
        
        if num_molecules < 4:
            return False
            
        # Estimate workload based on file sizes
        total_estimated_lines = 0
        for mol_data in molecules_data:
            file_path = mol_data.get("hitran_data") or mol_data.get("File Path")
            if file_path and os.path.exists(file_path):
                try:
                    file_size = os.path.getsize(file_path)
                    total_estimated_lines += file_size // 80  # Rough estimate
                except:
                    total_estimated_lines += 25000  # Default estimate
        
        return total_estimated_lines > 100000 and num_molecules > 3
    
    # Global parameter management
    def add_global_parameter_change_callback(self, callback: Callable) -> None:
        """Add a callback for global parameter changes."""
        if callback not in self._global_parameter_change_callbacks:
            self._global_parameter_change_callbacks.append(callback)
    
    def remove_global_parameter_change_callback(self, callback: Callable) -> None:
        """Remove a callback for global parameter changes."""
        if callback in self._global_parameter_change_callbacks:
            self._global_parameter_change_callbacks.remove(callback)

    def _notify_global_parameter_change(self, parameter_name: str, old_value: Any, new_value: Any) -> None:
        """Notify callbacks of global parameter changes."""
        for callback in self._global_parameter_change_callbacks:
            try:
                callback(parameter_name, old_value, new_value)
            except Exception as e:
                print(f"Error in global parameter change callback: {e}")

    # Simplified global parameter properties
    @property
    def global_distance(self) -> float:
        return self._global_dist
    
    @global_distance.setter
    def global_distance(self, value: float) -> None:
        old_value = self._global_dist
        self._global_dist = value
        self.bulk_update_parameters({'distance': value})
        self._notify_global_parameter_change('distance', old_value, value)
    
    @property
    def global_wavelength_range(self) -> Tuple[float, float]:
        return self._global_wavelength_range
    
    @global_wavelength_range.setter
    def global_wavelength_range(self, value: Tuple[float, float]) -> None:
        old_value = self._global_wavelength_range
        if value != old_value:
            self._global_wavelength_range = value
            # Update all molecules to use new range
            for molecule in self.values():
                molecule._wavelength_range = value
                if hasattr(molecule, '_dirty_flags'):
                    molecule._dirty_flags['spectrum'] = True
            
            self.bulk_update_parameters({'wavelength_range': value})
            self._notify_global_parameter_change('wavelength_range', old_value, value)
    
    @property
    def global_stellar_rv(self) -> float:
        return self._global_stellar_rv
    
    @global_stellar_rv.setter
    def global_stellar_rv(self, value: float) -> None:
        old_value = self._global_stellar_rv
        self._global_stellar_rv = value
        self._notify_global_parameter_change('stellar_rv', old_value, value)

    @property
    def global_model_pixel_res(self) -> Optional[float]:
        return self._global_model_pixel_res
    
    @global_model_pixel_res.setter
    def global_model_pixel_res(self, value: Optional[float]) -> None:
        old_value = self._global_model_pixel_res
        #value = f'{value:.2e}'
        value = float(value)
        self._global_model_pixel_res = value
        self.bulk_update_parameters({'model_pixel_res': value})
        self._notify_global_parameter_change('model_pixel_res', old_value, value)

    @property
    def global_intensity_calculation_method(self) -> Optional[str]:
        return self._global_intensity_calculation_method
    
    @global_intensity_calculation_method.setter
    def global_intensity_calculation_method(self, value: Optional[str]) -> None:
        old_value = self._global_intensity_calculation_method
        try:
            value = str(value)
        except Exception as e:
            print(f"Error setting intensity_calculation_method: {e}\ndefaulting to 'curve_growth'")
            value = "curve_growth"
        self._global_intensity_calculation_method = value
        self.bulk_update_parameters({'intensity_calculation_method': value})
        self._notify_global_parameter_change('intensity_calculation_method', old_value, value)

    @property
    def match_spectral_sampling(self) -> bool:
        """Whether to interpolate model flux to match data wavelengths pixel-by-pixel."""
        return self._match_spectral_sampling
    
    @match_spectral_sampling.setter
    def match_spectral_sampling(self, value: bool) -> None:
        old_value = self._match_spectral_sampling
        self._match_spectral_sampling = bool(value)
        if old_value != value:
            # Clear summed flux cache when this changes
            self._summed_flux_cache.clear()
            self._notify_global_parameter_change('match_spectral_sampling', old_value, value)