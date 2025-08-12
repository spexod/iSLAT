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
        self._global_star_rv: float = default_parms.DEFAULT_STELLAR_RV
        self._global_fwhm: float = default_parms.DEFAULT_FWHM
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
        """Add multiple molecules to the dictionary."""
        molecules = molecules[0]
        for mol in molecules:
            if isinstance(mol, Molecule):
                self[mol.name] = mol
            else:
                raise TypeError("Expected a Molecule instance.")

    def load_molecules_data(self, molecules_data: List[Dict[str, Any]], 
                           initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                           save_file_data: Dict[str, Dict[str, Any]], 
                           wavelength_range: Tuple[float, float], 
                           intrinsic_line_width: float, 
                           model_pixel_res: float, 
                           model_line_width: float, 
                           distance: float, 
                           hitran_data: Dict[str, Any]) -> None:
        """Load multiple molecules data into the dictionary."""
        self.initial_molecule_parameters = initial_molecule_parameters
        self.save_file_data = save_file_data
        for mol_entry in molecules_data:
            self.add_molecule(
                mol_entry,
                intrinsic_line_width=intrinsic_line_width,
                wavelength_range=wavelength_range,
                model_pixel_res=model_pixel_res,
                model_line_width=model_line_width,
                distance=distance,
                hitran_data=hitran_data[mol_entry["name"]] if mol_entry["name"] in hitran_data else None
            )
    
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
        try:
            wave_data_hash = hash(wave_data.tobytes()) if hasattr(wave_data, 'tobytes') else hash(str(wave_data))
            visible_molecules = self.get_visible_molecules_fast() if visible_only else set(self.keys())
            visible_molecules = {str(name) for name in visible_molecules}
            cache_key = hash((wave_data_hash, frozenset(visible_molecules)))
        except (TypeError, ValueError):
            return self._calculate_summed_flux_uncached(wave_data, visible_only)
        
        if cache_key in self._summed_flux_cache:
            return self._summed_flux_cache[cache_key]
        
        summed_flux = self._calculate_summed_flux_uncached(wave_data, visible_only)
        
        self._summed_flux_cache[cache_key] = summed_flux
        if len(self._summed_flux_cache) > 50:
            oldest_keys = list(self._summed_flux_cache.keys())[:10]
            for key in oldest_keys:
                del self._summed_flux_cache[key]
        
        return summed_flux
    
    def _calculate_summed_flux_uncached(self, wave_data: np.ndarray, visible_only: bool = True) -> np.ndarray:
        summed_flux = np.zeros_like(wave_data)
        visible_molecules = self.get_visible_molecules_fast() if visible_only else set(self.keys())
        
        for mol_name in visible_molecules:
            if mol_name in self:
                molecule = self[mol_name]
                molecule.prepare_plot_data(wave_data)
                if hasattr(molecule, 'plot_flux'):
                    summed_flux += molecule.plot_flux
        
        return summed_flux
    
    def get_summed_flux_optimized(self, wave_data: np.ndarray, visible_only: bool = True) -> np.ndarray:
        """Memory-optimized summed flux calculation using hash keys."""
        
        # Use hash of wave_data bytes for cache key (more memory efficient)
        try:
            wave_hash = hash(wave_data.data.tobytes()) if hasattr(wave_data, 'data') else hash(wave_data.tobytes())
            visible_molecules = self.get_visible_molecules_fast() if visible_only else set(self.keys())
            
            # Ensure all molecules are strings (hashable)
            visible_molecules = {str(name) for name in visible_molecules}
            
            # Create a more compact cache key
            cache_key = hash((wave_hash, frozenset(visible_molecules)))
        except (TypeError, ValueError) as e:
            # If hashing fails, calculate without caching
            print(f"Warning: Cache key creation failed in optimized version: {e}")
            visible_molecules = self.get_visible_molecules_fast() if visible_only else set(self.keys())
            cache_key = None
        
        # Check cache
        if cache_key is not None and cache_key in self._summed_flux_cache:
            return self._summed_flux_cache[cache_key]
        
        # Pre-allocate result array with float32 to save memory
        summed_flux = np.zeros_like(wave_data, dtype=np.float32)
        
        # Vectorized summation where possible
        for mol_name in visible_molecules:
            if mol_name in self:
                molecule = self[mol_name]
                molecule.prepare_plot_data(wave_data)
                if hasattr(molecule, 'plot_flux'):
                    summed_flux += molecule.plot_flux.astype(np.float32)
        
        # Cache with size limit (only if cache_key is valid)
        if cache_key is not None:
            if len(self._summed_flux_cache) > 100:  # Limit cache size
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._summed_flux_cache))
                del self._summed_flux_cache[oldest_key]
            
            self._summed_flux_cache[cache_key] = summed_flux
            
        return summed_flux
    
    def get_visible_molecules_fast(self) -> set:
        """Get set of visible molecule names for fast operations."""
        # Lazy update of visible molecules set
        current_visible = {name for name, mol in self.items() if mol.is_visible}
        self._visible_molecules = current_visible
        return current_visible
    
    def bulk_set_visibility_fast(self, is_visible: bool, molecule_names: Optional[List[str]] = None) -> None:
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
    
    def get_ndarray_of_attributes(self, attribute_name: str) -> np.ndarray:
        """Get a numpy array of a specific attribute for all molecules."""
        return np.array([getattr(mol, attribute_name, None) for mol in self.values()])
    
    def get_ndarray_of_attributes_optimized(self, attribute_name: str) -> np.ndarray:
        """Get a numpy array of a specific attribute for all molecules - optimized version."""
        # Pre-allocate array if we know the size
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
    
    def bulk_set_temperature(self, temperature: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update temperature for multiple molecules."""
        self.bulk_update_parameter('temp', temperature, molecule_names)
    
    def bulk_set_radius(self, radius: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update radius for multiple molecules."""
        self.bulk_update_parameter('radius', radius, molecule_names)
    
    def bulk_set_column_density(self, n_mol: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update column density for multiple molecules."""
        self.bulk_update_parameter('n_mol', n_mol, molecule_names)
    
    def bulk_set_distance(self, distance: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update distance for multiple molecules."""
        self.bulk_update_parameter('distance', distance, molecule_names)
    
    def bulk_set_fwhm(self, fwhm: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update FWHM for multiple molecules."""
        self.bulk_update_parameter('fwhm', fwhm, molecule_names)
    
    def bulk_set_stellar_rv(self, stellar_rv: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update stellar RV for multiple molecules."""
        self.bulk_update_parameter('stellar_rv', stellar_rv, molecule_names)
    
    def bulk_set_intrinsic_line_width(self, width: float, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update intrinsic line width for multiple molecules."""
        self.bulk_update_parameter('intrinsic_line_width', width, molecule_names)
    
    def bulk_set_visibility(self, is_visible: bool, molecule_names: Optional[List[str]] = None) -> None:
        """Bulk update visibility for multiple molecules using optimized set operations."""
        print(f"molecule names for visibliy: {molecule_names}")
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        # Use set operations for efficiency
        molecule_set = set(molecule_names) & set(self.keys())
        
        updated_count = 0
        for mol_name in molecule_set:
            molecule = self[mol_name]
            if molecule.is_visible != is_visible:  # Only update if different
                molecule.is_visible = is_visible
                updated_count += 1
                
                # Update visibility tracking set
                if is_visible:
                    self._visible_molecules.add(mol_name)
                else:
                    self._visible_molecules.discard(mol_name)
        
        if updated_count > 0:
            self._clear_flux_caches()
        
        print(f"Updated visibility for {updated_count} molecules")
    
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
    
    def apply_parameter_template(self, template: Dict[str, Any], molecule_names: Optional[List[str]] = None) -> None:
        """
        Apply a parameter template to multiple molecules.
        
        Args:
            template: Dictionary of parameter names and values to apply
            molecule_names: List of molecule names to apply template to (None for all)
        """
        self.bulk_update_parameters(template, molecule_names)
        print(f"Applied parameter template to {len(molecule_names or self.keys())} molecules")
    
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
    
    def load_molecules_parallel(self, molecules_data: List[Dict[str, Any]], 
                               initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                               max_workers: Optional[int] = None) -> Dict[str, Any]:
        """
        Load multiple molecules in parallel using multiprocessing.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
            max_workers: Maximum number of worker processes (None for auto-detect)
        
        Returns:
            Dictionary with loading statistics and results
        """
        if not molecules_data:
            return {"success": 0, "failed": 0, "molecules": []}
        
        # Prepare global parameters
        global_params = {
            "wavelength_range": self._global_wavelength_range,
            "distance": self._global_dist,
            "fwhm": self._global_fwhm,
            "stellar_rv": self._global_star_rv,
            "intrinsic_line_width": self._global_intrinsic_line_width,
            "model_pixel_res": self._global_model_pixel_res,
            "model_line_width": self._global_model_line_width
        }
        
        # Prepare arguments for workers
        worker_args = [
            (mol_data, global_params, initial_molecule_parameters)
            for mol_data in molecules_data
        ]
        
        # Determine optimal number of workers
        if max_workers is None:
            max_workers = min(len(molecules_data), mp.cpu_count())
        
        print(f"Loading {len(molecules_data)} molecules using {max_workers} worker processes...")
        
        results = {
            "success": 0,
            "failed": 0,
            "molecules": [],
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # Use ProcessPoolExecutor for CPU-bound molecule creation
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_args = {
                    executor.submit(self._create_molecule_worker, args): args[0]
                    for args in worker_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_args):
                    mol_data = future_to_args[future]
                    try:
                        success, result, mol_name = future.result()
                        
                        if success:
                            # Add molecule to dictionary
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
            return self._load_molecules_sequential(molecules_data, initial_molecule_parameters)
        
        elapsed_time = time.time() - start_time
        print(f"Parallel loading completed in {elapsed_time:.2f}s")
        print(f"Successfully loaded: {results['success']}, Failed: {results['failed']}")
        
        # Clear and rebuild caches after loading
        self._clear_flux_caches()
        
        return results
    
    def _load_molecules_sequential(self, molecules_data: List[Dict[str, Any]], 
                                  initial_molecule_parameters: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fallback sequential molecule loading method.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
        
        Returns:
            Dictionary with loading statistics and results
        """
        print("Falling back to sequential molecule loading...")
        
        results = {
            "success": 0,
            "failed": 0,
            "molecules": [],
            "errors": []
        }
        
        for mol_data in molecules_data:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            if not mol_name:
                results["failed"] += 1
                results["errors"].append("Unknown: Missing molecule name")
                continue
            
            try:
                success, result, name = self._create_molecule_worker((
                    mol_data, 
                    {
                        "wavelength_range": self._global_wavelength_range,
                        "distance": self._global_dist,
                        "fwhm": self._global_fwhm,
                        "stellar_rv": self._global_star_rv,
                        "intrinsic_line_width": self._global_intrinsic_line_width,
                        "model_pixel_res": self._global_model_pixel_res,
                        "model_line_width": self._global_model_line_width
                    },
                    initial_molecule_parameters
                ))
                
                if success:
                    self[mol_name] = result
                    results["molecules"].append(mol_name)
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"{mol_name}: {result}")
                    
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(f"{mol_name}: {str(e)}")
        
        return results
    
    def bulk_recalculate_sequential(self, molecule_names: Optional[List[str]] = None) -> None:
        """
        Recalculate intensity and spectrum for multiple molecules sequentially (default method).
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        if not molecule_names:
            return
        
        print(f"Recalculating {len(molecule_names)} molecules sequentially...")
        start_time = time.time()
        success_count = 0
        
        for mol_name in molecule_names:
            try:
                if mol_name in self:
                    molecule = self[mol_name]
                    # Force invalidation and recalculation
                    molecule._intensity_valid = False
                    molecule._spectrum_valid = False
                    molecule._clear_flux_caches()
                    molecule._invalidate_parameter_hash()
                    
                    # Trigger recalculation by accessing properties
                    if hasattr(molecule, 'calculate_intensity'):
                        molecule.calculate_intensity()
                    
                    success_count += 1
                else:
                    print(f"Molecule '{mol_name}' not found")
            except Exception as e:
                print(f"Error recalculating '{mol_name}': {str(e)}")
        
        # Clear global caches
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Sequential recalculation completed in {elapsed_time:.2f}s")
        print(f"Successfully recalculated {success_count}/{len(molecule_names)} molecules")
    
    def bulk_recalculate_parallel(self, molecule_names: Optional[List[str]] = None, 
                                 max_workers: Optional[int] = None) -> None:
        """
        Recalculate intensity and spectrum for multiple molecules in parallel.
        This method is available but not used by default for better stability.
        
        Args:
            molecule_names: List of molecule names to recalculate (None for all)
            max_workers: Maximum number of worker threads (None for auto-detect)
        """
        if molecule_names is None:
            molecule_names = list(self.keys())
        
        if not molecule_names:
            return
        
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
                    
                    # Trigger recalculation by accessing properties
                    if hasattr(molecule, 'calculate_intensity'):
                        molecule.calculate_intensity()
                    
                    return True, mol_name
                else:
                    return False, f"Molecule '{mol_name}' not found"
            except Exception as e:
                return False, f"Error recalculating '{mol_name}': {str(e)}"
        
        start_time = time.time()
        success_count = 0
        
        # Use ThreadPoolExecutor for I/O-bound recalculation tasks
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {
                executor.submit(recalculate_molecule, mol_name): mol_name
                for mol_name in molecule_names
            }
            
            for future in as_completed(future_to_name):
                mol_name = future_to_name[future]
                try:
                    success, result = future.result()
                    if success:
                        success_count += 1
                    else:
                        print(f"Failed to recalculate {mol_name}: {result}")
                except Exception as e:
                    print(f"Error recalculating {mol_name}: {e}")
        
        # Clear global caches
        self._clear_flux_caches()
        
        elapsed_time = time.time() - start_time
        print(f"Parallel recalculation completed in {elapsed_time:.2f}s")
        print(f"Successfully recalculated {success_count}/{len(molecule_names)} molecules")
    
    def load_molecules_ultra_fast(self, molecules_data: List[Dict[str, Any]], 
                                 initial_molecule_parameters: Dict[str, Dict[str, Any]], 
                                 max_workers: Optional[int] = None, 
                                 force_multiprocessing: bool = False) -> Dict[str, Any]:
        """
        Load multiple molecules with sequential loading by default, multiprocessing only when forced.
        
        This method uses sequential loading by default for better compatibility and stability.
        Multiprocessing is only used when explicitly requested via force_multiprocessing=True.
        
        Args:
            molecules_data: List of molecule data dictionaries
            initial_molecule_parameters: Dictionary of initial parameters by molecule name
            max_workers: Maximum number of worker processes (None for auto-detect)
            force_multiprocessing: If True, forces multiprocessing even for small datasets
        
        Returns:
            Dictionary with loading statistics and results
        """
        if not molecules_data:
            return {"success": 0, "failed": 0, "errors": []}
        
        print(f"Starting molecule loading for {len(molecules_data)} molecules...")
        start_time = time.time()
        
        # Use multiprocessing only if explicitly forced AND conditions are met
        use_multiprocessing = force_multiprocessing and self._should_use_multiprocessing(molecules_data, max_workers)
        
        results = {
            "success": 0,
            "failed": 0,
            "errors": []
        }
        
        if use_multiprocessing:
            print(f"Using multiprocessing for {len(molecules_data)} molecules (forced)...")
            # Use the existing parallel loading method
            parallel_results = self.load_molecules_parallel(
                molecules_data, 
                initial_molecule_parameters, 
                max_workers
            )
            results.update(parallel_results)
        else:
            print(f"Using sequential loading for {len(molecules_data)} molecules...")
            # Use sequential loading (default and safer)
            for mol_data in molecules_data:
                success = self._load_single_molecule_ultra_fast(mol_data, initial_molecule_parameters)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to load {mol_data.get('Molecule Name', 'unknown')}")
        
        elapsed_time = time.time() - start_time
        print(f"Ultra-fast loading completed in {elapsed_time:.3f}s")
        print(f"Success: {results['success']}, Failed: {results['failed']}")
        
        return results
    
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
    
    def _load_single_molecule_ultra_fast(self, mol_data: Dict[str, Any], 
                                        initial_molecule_parameters: Dict[str, Dict[str, Any]]) -> bool:
        """
        Load a single molecule using ultra-fast optimized methods.
        """
        try:
            mol_name = mol_data.get("Molecule Name") or mol_data.get("name")
            
            if not mol_name:
                print("Error: Missing molecule name")
                return False
            
            # Create molecule with optimized loading
            molecule = Molecule(
                user_save_data=mol_data if "Molecule Name" in mol_data else None,
                hitran_data=mol_data.get("hitran_data") if "hitran_data" in mol_data else None,
                name=mol_name,
                filepath=mol_data.get("file") or mol_data.get("File Path"),
                displaylabel=mol_data.get("label") or mol_data.get("Molecule Label", mol_name),
                wavelength_range=self._global_wavelength_range,
                distance=self._global_dist,
                fwhm=self._global_fwhm,
                stellar_rv=self._global_star_rv,
                broad=self._global_intrinsic_line_width,
                model_pixel_res=self._global_model_pixel_res,
                model_line_width=self._global_model_line_width,
                temp=mol_data.get("Temp"),
                radius=mol_data.get("Rad"),
                n_mol=mol_data.get("N_Mol"),
                color=mol_data.get("Color"),
                is_visible=mol_data.get("Vis", True),
                initial_molecule_parameters=initial_molecule_parameters.get(mol_name, {})
            )
            
            # Add to dictionary
            self[mol_name] = molecule
            print(f"Successfully loaded molecule: {mol_name}")
            return True
            
        except Exception as e:
            print(f"Error loading molecule '{mol_name}': {e}")
            return False

    def create_memory_mapped_flux_storage(self, max_molecules: int = 1000, max_wavelengths: int = 10000):
        """Create memory-mapped arrays for flux storage to handle large datasets."""
        import tempfile
        
        # Create temporary file for memory mapping
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        
        # Memory-mapped array for flux storage
        self._flux_memmap = np.memmap(
            temp_file.name, 
            dtype=np.float32, 
            mode='w+', 
            shape=(max_molecules, max_wavelengths)
        )
        
        self._flux_molecule_index = {}  # Map molecule names to array indices
        self._flux_memmap_file = temp_file.name
        print(f"Created memory-mapped flux storage: {max_molecules}x{max_wavelengths}")
    
    def get_molecule_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about molecules in the dictionary."""
        if not self:
            return {"count": 0}
        
        # Update parameter arrays for efficient computation
        self.update_parameter_arrays()
        
        stats = {
            "count": len(self),
            "visible_count": len(self.get_visible_molecules_fast()),
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
    
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """Estimate memory usage of the molecule dictionary in MB."""
        total_size = 0
        flux_size = 0
        cache_size = 0
        
        # Estimate molecule objects size
        mol_count = len(self)
        estimated_mol_size = mol_count * 1024  # Rough estimate per molecule in bytes
        
        # Flux arrays size
        for flux_array in self.fluxes.values():
            if hasattr(flux_array, 'nbytes'):
                flux_size += flux_array.nbytes
        
        # Cache size
        for cache_array in self._summed_flux_cache.values():
            if hasattr(cache_array, 'nbytes'):
                cache_size += cache_array.nbytes
        
        # Parameter arrays size
        param_array_size = 0
        for param_array in self._parameter_arrays.values():
            if hasattr(param_array, 'nbytes'):
                param_array_size += param_array.nbytes
        
        total_size = estimated_mol_size + flux_size + cache_size + param_array_size
        
        return {
            "total_mb": total_size / (1024 * 1024),
            "molecules_mb": estimated_mol_size / (1024 * 1024),
            "fluxes_mb": flux_size / (1024 * 1024),
            "cache_mb": cache_size / (1024 * 1024),
            "parameter_arrays_mb": param_array_size / (1024 * 1024)
        }
    
    def optimize_memory(self) -> None:
        """Optimize memory usage by cleaning up caches and converting to more efficient dtypes."""
        # Clear old caches
        if len(self._summed_flux_cache) > 50:
            # Keep only the 25 most recent entries
            cache_items = list(self._summed_flux_cache.items())
            self._summed_flux_cache = dict(cache_items[-25:])
        
        # Convert flux arrays to float32 if they're float64
        for mol_name, flux_array in self.fluxes.items():
            if flux_array.dtype == np.float64:
                self.fluxes[mol_name] = flux_array.astype(np.float32)
        
        # Update parameter arrays to more efficient dtypes
        for param_name, param_array in self._parameter_arrays.items():
            if param_array.dtype == np.float64:
                self._parameter_arrays[param_name] = param_array.astype(np.float32)
        
        print("Memory optimization completed")
    
    def cleanup_memory_mapped_storage(self) -> None:
        """Clean up memory-mapped storage files."""
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
            self.bulk_set_distance(value)
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
            self.cleanup_memory_mapped_storage()
        except:
            pass

    # Performance testing and demonstration methods
    def benchmark_bulk_operations(self, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark different bulk operation methods for performance comparison."""
        import time
        
        if not self:
            print("No molecules to benchmark")
            return {}
        
        molecule_names = list(self.keys())
        results = {}
        
        # Test 1: Traditional bulk update vs vectorized
        print(f"Benchmarking bulk operations with {len(molecule_names)} molecules...")
        
        # Vectorized parameter update
        start_time = time.time()
        for _ in range(num_iterations):
            test_values = np.random.uniform(100, 1000, len(molecule_names))
            self.bulk_update_parameter_vectorized('temp', test_values, molecule_names)
        results['vectorized_update_time'] = time.time() - start_time
        
        # Traditional parameter update
        start_time = time.time()
        for _ in range(num_iterations):
            test_value = np.random.uniform(100, 1000)
            self.bulk_update_parameter('temp', test_value, molecule_names)
        results['traditional_update_time'] = time.time() - start_time
        
        # Test 2: Visibility operations
        start_time = time.time()
        for _ in range(num_iterations):
            self.bulk_set_visibility_fast(True, molecule_names[:len(molecule_names)//2])
            self.bulk_set_visibility_fast(False, molecule_names[len(molecule_names)//2:])
        results['fast_visibility_time'] = time.time() - start_time
        
        # Test 3: Parameter array operations
        start_time = time.time()
        for _ in range(num_iterations):
            self.update_parameter_arrays()
            temp_array = self.get_parameter_array('temp')
            scaled_temps = temp_array * 1.1
        results['array_operations_time'] = time.time() - start_time
        
        print("Benchmark Results:")
        for operation, time_taken in results.items():
            print(f"  {operation}: {time_taken:.3f} seconds")
        
        return results
    
    def demo_optimized_features(self) -> None:
        """Demonstrate the new optimized features of MoleculeDict."""
        if not self:
            print("No molecules available for demonstration")
            return
            
        print("=== MoleculeDict Optimized Features Demo ===\n")
        
        # 1. Memory usage analysis
        print("1. Memory Usage Analysis:")
        memory_stats = self.get_memory_usage_estimate()
        for key, value in memory_stats.items():
            print(f"   {key}: {value:.2f} MB")
        print()
        
        # 2. Parameter statistics
        print("2. Parameter Statistics:")
        stats = self.get_molecule_statistics()
        print(f"   Total molecules: {stats['count']}")
        print(f"   Visible molecules: {stats['visible_count']}")
        for param, param_stats in stats.get('parameter_stats', {}).items():
            print(f"   {param}: mean={param_stats['mean']:.2f}, std={param_stats['std']:.2f}")
        print()
        
        # 3. Vectorized operations demo
        print("3. Vectorized Operations Demo:")
        molecule_names = list(self.keys())[:5]  # Use first 5 molecules for demo
        
        # Scale temperatures by 1.1
        print("   Scaling temperatures by 1.1...")
        self.bulk_scale_parameter('temp', 1.1, molecule_names)
        
        # Apply log function to column densities
        print("   Applying log10 to column densities...")
        self.bulk_apply_function('n_mol', lambda x: np.log10(max(x, 1e-10)), molecule_names)
        print()
        
        # 4. Conditional updates demo
        print("4. Conditional Updates Demo:")
        high_temp_molecules = self.bulk_filter_molecules(lambda params: params['temp'] > 500)
        print(f"   Found {len(high_temp_molecules)} molecules with temp > 500K")
        
        if high_temp_molecules:
            print("   Setting radius to 1.0 for high-temperature molecules...")
            self.bulk_conditional_update(
                lambda params: params['temp'] > 500,
                {'radius': 1.0}
            )
        print()
        
        # 5. Fast visibility operations
        print("5. Fast Visibility Operations:")
        visible_before = len(self.get_visible_molecules_fast())
        print(f"   Visible molecules before: {visible_before}")
        
        # Hide half the molecules
        half_molecules = molecule_names[:len(molecule_names)//2]
        self.bulk_set_visibility_fast(False, half_molecules)
        
        visible_after = len(self.get_visible_molecules_fast())
        print(f"   Visible molecules after hiding {len(half_molecules)}: {visible_after}")
        
        # Restore visibility
        self.bulk_set_visibility_fast(True, half_molecules)
        print()
        
        # 6. Memory optimization
        print("6. Memory Optimization:")
        print("   Running memory optimization...")
        self.optimize_memory()
        
        new_memory_stats = self.get_memory_usage_estimate()
        print(f"   New total memory usage: {new_memory_stats['total_mb']:.2f} MB")
        print()
        
        print("=== Demo Complete ===")
    
    def create_performance_report(self) -> str:
        """Create a comprehensive performance report for the molecule dictionary."""
        report = []
        report.append("MoleculeDict Performance Report")
        report.append("=" * 40)
        
        # Basic statistics
        stats = self.get_molecule_statistics()
        report.append(f"Total Molecules: {stats['count']}")
        report.append(f"Visible Molecules: {stats['visible_count']}")
        report.append("")
        
        # Memory usage
        memory_stats = self.get_memory_usage_estimate()
        report.append("Memory Usage:")
        for key, value in memory_stats.items():
            report.append(f"  {key}: {value:.2f} MB")
        report.append("")
        
        # Cache statistics
        report.append("Cache Statistics:")
        report.append(f"  Summed flux cache entries: {len(self._summed_flux_cache)}")
        report.append(f"  Flux arrays stored: {len(self.fluxes)}")
        report.append("")
        
        # Parameter distribution
        report.append("Parameter Distributions:")
        for param, param_stats in stats.get('parameter_stats', {}).items():
            report.append(f"  {param}:")
            report.append(f"    Range: {param_stats['min']:.2e} - {param_stats['max']:.2e}")
            report.append(f"    Mean ± Std: {param_stats['mean']:.2e} ± {param_stats['std']:.2e}")
        
        return "\n".join(report)