# -*- coding: utf-8 -*-

"""
The class Intensity calculates the intensities

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
        frequencies : np.ndarray
            Line frequencies in Hz with shape (n_lines,)
        sqrt_ln2_inv : float
            Normalization constant

        Returns read access to the fields is granted through properties

- 01/06/2020: SB, initial version
- 11/01/2025 Johnny McCaskill, redesigned for performance and to enable overlapping line treatment

"""

import numpy as np
from typing import Optional, Union, Literal, TYPE_CHECKING, Any
import iSLAT.Constants as c

# Lazy imports for performance
_scipy_imported = False
_pd_imported = False

def _get_scipy():
    """Lazy import of scipy components"""
    global _scipy_imported
    if not _scipy_imported:
        global fixed_quad
        from scipy.integrate import fixed_quad
        _scipy_imported = True
    return fixed_quad

def _get_pandas():
    """Lazy import of pandas"""
    global _pd_imported, pd
    if not _pd_imported:
        try:
            import pandas as pd
            _pd_imported = True
        except ImportError:
            pd = None
            _pd_imported = True
    return pd

if TYPE_CHECKING:
    import pandas as pd
    from .MoleculeLineList import MoleculeLineList
else:
    # Lazy import for runtime
    def _get_molecule_line_list():
        from .MoleculeLineList import MoleculeLineList
        return MoleculeLineList

__all__ = ["Intensity"]

class Intensity:
    __slots__ = ('_molecule', '_intensity', '_tau', '_t_kin', '_n_mol', '_dv', '_cache_valid')
    
    def __init__(self, molecule_line_list: 'MoleculeLineList') -> None:
        """Initialize an intensity class which calculates the intensities for a given molecule and provided
        physical parameters.

        Parameters
        ----------
        molecule_line_list: MoleculeLineList
            Molecular line list data to calculate the intensity
        """
        if TYPE_CHECKING:
            self._molecule: "MoleculeLineList" = molecule_line_list
        else:
            # Runtime: accept any compatible object
            self._molecule = molecule_line_list
            
        self._intensity: Optional[np.ndarray] = None
        self._tau: Optional[np.ndarray] = None
        self._t_kin: Optional[float] = None
        self._n_mol: Optional[float] = None
        self._dv: Optional[float] = None
        self._cache_valid: bool = False

    @staticmethod
    def _bb(nu: np.ndarray, T: float) -> np.ndarray:
        """Blackbody function for one temperature and an array of frequencies. 
        Uses optimized approximations for accuracy and performance.

        Parameters
        ----------
        nu: np.ndarray
            Frequency in Hz to calculate the blackbody values
        T: float
            Temperature in K

        Returns
        -------
        np.ndarray:
            Blackbody intensity in erg/s/cm**2/sr/Hz
        """
        x = c.PLANCK_CONSTANT * nu / (c.BOLTZMANN_CONSTANT * T)
        
        # Use vectorized conditions for better performance
        bb_RJ = 2.0 * nu ** 2 * c.BOLTZMANN_CONSTANT * T / (c.SPEED_OF_LIGHT_CGS ** 2)
        bb_Wien = 2.0 * c.PLANCK_CONSTANT * nu ** 3 / c.SPEED_OF_LIGHT_CGS ** 2 * np.exp(-x)
        bb_Planck = 2. * c.PLANCK_CONSTANT * nu ** 3 / c.SPEED_OF_LIGHT_CGS ** 2 / (np.exp(np.clip(x, None, 20.0)) - 1.)
        
        # Use np.select for cleaner vectorized selection
        conditions = [x < 1.0e-5, x > 20.0]
        choices = [bb_RJ, bb_Wien]
        
        return np.select(conditions, choices, default=bb_Planck)

    # Pre-computed Gaussian quadrature points and weights for maximum efficiency
    _GAUSS_QUAD_X = None
    _GAUSS_QUAD_W = None
    _GAUSS_QUAD_INITIALIZED = False
    
    @classmethod
    def _initialize_gauss_quad(cls):
        """Initialize Gaussian quadrature points and weights once for all calculations."""
        if not cls._GAUSS_QUAD_INITIALIZED:
            try:
                from numpy.polynomial.legendre import leggauss
                x_quad, w_quad = leggauss(20)
                # Transform from [-1,1] to [-6,6] and pre-compute exp(-x^2)
                #cls._GAUSS_QUAD_X = np.exp(-(6.0 * x_quad) ** 2)  # Pre-compute exp(-x^2)
                cls._GAUSS_QUAD_X = 6.0 * x_quad  # Store actual quadrature points
                cls._GAUSS_QUAD_EXP = np.exp(-(cls._GAUSS_QUAD_X ** 2))
                cls._GAUSS_QUAD_W = 6.0 * w_quad
                cls._GAUSS_QUAD_INITIALIZED = True
            except ImportError:
                # Fallback to scipy method if numpy.polynomial not available
                fixed_quad = _get_scipy()
                # Use a simple tau value to initialize quadrature
                cls._GAUSS_QUAD_X = np.linspace(-6, 6, 20)  # Store actual x values
                cls._GAUSS_QUAD_W = np.ones(20) * (12.0 / 20.0)  # Simple uniform weighting
                cls._GAUSS_QUAD_INITIALIZED = True
    
    @classmethod 
    def _fint_multi(cls, center_tau: np.ndarray, dv_cond: np.ndarray,
                   bb_vals: np.ndarray, freq_ratio: np.ndarray, sqrt_ln2_inv: float, 
                   frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate spectral line intensities with proper treatment of overlapping lines.
        
        This method implements the curve-of-growth integral for molecular line intensities,
        handling both isolated and overlapping lines using correct radiative transfer physics.
        For overlapping lines, optical depths are summed before applying the curve-of-growth
        to avoid unphysical intensity enhancement and maintain conservation.
        
        Algorithm Overview:
        1. Identify isolated vs. overlapping lines using velocity separation criteria  
        2. Process isolated lines in batch using vectorized operations
        3. For overlapping lines: sum optical depths → apply curve-of-growth → distribute intensity
        
        Physical Basis:
        - Isolated lines: I = (physical_factors) x ∫[1 - exp(-τ)] dv
        - Overlapping lines: I_total = ∫[1 - exp(-Στ_i)] dv, then distribute proportionally
        
        Performance:
        - Memory complexity: O(n) instead of O(n²) for overlap detection
        - Time complexity: O(n log n) for sorting + O(n) for processing  
        - Vectorized operations for isolated lines (typically 85%+ of all lines)
        
        Parameters
        ----------
        center_tau : np.ndarray, shape (n_conditions, n_lines)
            Line center optical depths for all physical conditions and lines
        dv_cond : np.ndarray, shape (n_conditions,)
            Doppler broadening velocities (FWHM) in km/s for each condition
        bb_vals : np.ndarray, shape (n_conditions, n_lines)
            Blackbody source function values at line frequencies  
        freq_ratio : np.ndarray, shape (n_conditions, n_lines)
            Frequency ratios (v/c) for unit conversion
        sqrt_ln2_inv : float
            Normalization constant: 1/(2√ln(2)) ≈ 0.6005 for Gaussian integration
        frequencies : np.ndarray, shape (n_lines,)
            Line rest frequencies in Hz
            
        Returns
        -------
        np.ndarray, shape (n_conditions, n_lines)
            Calculated line intensities in CGS units (erg cm⁻² s⁻¹ Hz⁻¹)
            
        Notes
        -----
        - Uses Gaussian quadrature for numerical integration of curve-of-growth
        - Velocity overlap criterion: lines within 10 km/s are considered overlapping
        """
        # Initialize quadrature points and weights
        if not cls._GAUSS_QUAD_INITIALIZED:
            cls._initialize_gauss_quad()
        
        # Setup arrays and extract dimensions
        dv_cond = np.atleast_1d(dv_cond)
        intensity = np.zeros_like(center_tau)
        
        # Quadrature setup for curve-of-growth integration
        exp_neg_x_squared = cls._GAUSS_QUAD_EXP[np.newaxis, :]  # (1, n_quad)
        weights = cls._GAUSS_QUAD_W
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Identify isolated vs. overlapping lines
        # ═══════════════════════════════════════════════════════════════
        
        # Physical overlap criterion: lines within 10 km/s velocity separation. prevents spurious overlap detection for large datasets
        #cutoff_velocity = min(10.0, 3 * np.max(dv_cond))  # km/s
        #cutoff_velocity = np.max(dv_cond)  # km/s
        
        # Convert velocity cutoff to frequency tolerance
        freq_tolerance = np.max(dv_cond) * frequencies / c.SPEED_OF_LIGHT_KMS

        # Sort indices by frequency for efficient range queries
        sort_indices = np.argsort(frequencies)
        sorted_frequencies = frequencies[sort_indices]
        tol_sorted = freq_tolerance[sort_indices]

        # If every adjacent pair is separated by more than the stricter of the two tolerances,
        # then there are no overlaps anywhere; do the isolated batch and return.
        gaps = np.diff(sorted_frequencies)
        if gaps.size == 0 or np.all(gaps > np.maximum(tol_sorted[:-1], tol_sorted[1:])):
            # Pre-compute physical factors (same as in STEP 2)
            physical_factors = sqrt_ln2_inv * 1e5 * dv_cond[:, np.newaxis] * freq_ratio * bb_vals

            # Vectorized curve-of-growth for all lines at once
            integrand = 1.0 - np.exp(-center_tau[:, :, np.newaxis] * exp_neg_x_squared)  # (n_cond, n_lines, n_quad)
            fint_vals = integrand @ weights                                             # (n_cond, n_lines)
            return physical_factors * fint_vals
        
        # Lines are already sorted by frequency; expand contiguous windows while
        # adjacent pairs are within either neighbor's tolerance (max of the pair).
        isolated_lines = []
        blended_groups = {}

        i = 0
        n_sorted = sorted_frequencies.size
        while i < n_sorted:
            j = i + 1
            # Grow the group while neighbors mutually overlap
            while j < n_sorted:
                gap = sorted_frequencies[j] - sorted_frequencies[j - 1]
                if gap <= max(tol_sorted[j], tol_sorted[j - 1]):
                    j += 1
                else:
                    break

            # Map back to original indices
            group_orig_idx = sort_indices[i:j]
            if group_orig_idx.size == 1:
                isolated_lines.append(int(group_orig_idx[0]))
            else:
                key = tuple(np.sort(group_orig_idx))
                blended_groups[key] = group_orig_idx

            i = j
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Process isolated lines (optimized vectorization)
        # ═══════════════════════════════════════════════════════════════
        
        # Pre-compute physical factors for all lines (more efficient)
        physical_factors = sqrt_ln2_inv * 1e5 * dv_cond[:, np.newaxis] * freq_ratio * bb_vals
        
        if isolated_lines:
            isolated_indices = np.array(isolated_lines)
            
            # Process in batches to manage memory for very large line lists
            batch_size = 10000
            for batch_start in range(0, len(isolated_indices), batch_size):
                batch_end = min(batch_start + batch_size, len(isolated_indices))
                batch_indices = isolated_indices[batch_start:batch_end]
                
                tau_batch = center_tau[:, batch_indices]
                
                # Vectorized curve-of-growth: (n_cond, n_batch, n_quad)
                integrand = 1.0 - np.exp(-tau_batch[:, :, np.newaxis] * exp_neg_x_squared)
                fint_vals = integrand @ weights
                
                # Apply pre-computed physical factors
                intensity[:, batch_indices] = physical_factors[:, batch_indices] * fint_vals
        
        # ═══════════════════════════════════════════════════════════════  
        # STEP 3: Process overlapping lines
        # ═══════════════════════════════════════════════════════════════
        
        for overlapping_indices in blended_groups.values():
            overlapping_indices = np.asarray(overlapping_indices)  # Faster than np.array
            n_overlap = len(overlapping_indices)
            
            if n_overlap <= 1:
                continue
            
            # Extract optical depths for this blended group
            tau_group = center_tau[:, overlapping_indices]  # (n_cond, n_overlap)
            
            # Sum optical depths (vectorized)
            tau_total = np.sum(tau_group, axis=1)  # (n_cond,)
            
            # Apply curve-of-growth to total blended optical depth
            integrand_total = 1.0 - np.exp(-tau_total[:, np.newaxis] * exp_neg_x_squared)
            fint_total = integrand_total @ weights
            
            # VECTORIZED: Calculate all fractional contributions at once
            tau_fractions = tau_group / (tau_total[:, np.newaxis])  # (n_cond, n_overlap)
            #tau_fractions = np.clip(tau_fractions, 0.0, 1.0)
            
            # VECTORIZED: Calculate all line shares at once
            fint_lines = fint_total[:, np.newaxis] * tau_fractions  # (n_cond, n_overlap)
            
            # VECTORIZED: Apply physical factors to all overlapping lines at once
            intensity[:, overlapping_indices] = (
                physical_factors[:, overlapping_indices] * fint_lines
            )

        return intensity

    def _calc_intensity_core(self, t_kin_vals: np.ndarray, n_mol_vals: np.ndarray, 
                            dv_vals: np.ndarray, method: str = "curve_growth") -> tuple:
        """Core intensity calculation using unified vectorized operations.
        
        This method handles all intensity calculations efficiently using a single
        vectorized code path by converting scalar inputs to arrays.
        
        Parameters
        ----------
        t_kin_vals : np.ndarray
            Temperature values (K) - can be scalar or array
        n_mol_vals : np.ndarray  
            Column density values (cm^-2) - can be scalar or array
        dv_vals : np.ndarray
            Line width values (km/s) - can be scalar or array
        method : str
            Calculation method ("curve_growth" or "radex")
            
        Returns
        -------
        tuple
            (intensities, tau_values) as numpy arrays
        """
        m = self._molecule
        lines = m.lines_as_namedtuple
        partition = m.partition
        
        # Ensure inputs are numpy arrays with consistent shape
        t_kin_vals = np.asarray(t_kin_vals)
        n_mol_vals = np.asarray(n_mol_vals) 
        dv_vals = np.asarray(dv_vals)
        
        # Convert scalar inputs to 1-element arrays
        was_scalar = t_kin_vals.ndim == 0 and n_mol_vals.ndim == 0 and dv_vals.ndim == 0
        if was_scalar:
            t_kin_vals = np.atleast_1d(t_kin_vals)
            n_mol_vals = np.atleast_1d(n_mol_vals)
            dv_vals = np.atleast_1d(dv_vals)
        
        # Validate temperature bounds
        t_min, t_max = np.min(partition.t), np.max(partition.t)
        if np.any(t_kin_vals < t_min) or np.any(t_kin_vals > t_max):
            raise ValueError(f'Temperature values outside partition function range [{t_min}, {t_max}]')
        
        # Broadcast to common shape
        t_kin_vals, n_mol_vals, dv_vals = np.broadcast_arrays(t_kin_vals, n_mol_vals, dv_vals)
        output_shape = t_kin_vals.shape
        
        # Flatten for efficient processing
        t_kin_flat = t_kin_vals.ravel()
        n_mol_flat = n_mol_vals.ravel()
        dv_flat = dv_vals.ravel()
        
        # Vectorized partition function
        q_sum_vals: np.ndarray = np.interp(t_kin_flat, partition.t, partition.q)
        
        # Efficient broadcasting for line calculations - minimize memory usage
        inv_t_kin_flat = 1.0 / t_kin_flat
        exp_factor_low = np.exp(-np.outer(inv_t_kin_flat, lines.e_low))
        exp_factor_up = np.exp(-np.outer(inv_t_kin_flat, lines.e_up))
        
        # Calculate populations directly without storing intermediate arrays
        q_sum_inv: np.ndarray = 1.0 / q_sum_vals[:, np.newaxis]
        x_low = (exp_factor_low * lines.g_low[np.newaxis, :]) * q_sum_inv
        x_up = (exp_factor_up * lines.g_up[np.newaxis, :]) * q_sum_inv
        
        freq_factor = (c.SPEED_OF_LIGHT_CGS ** 3 / 
                      (8.0 * np.pi * lines.freq[np.newaxis, :] ** 3 * 
                       (1e5 * dv_flat[:, np.newaxis] * c.FGAUSS_PREFACTOR)))
        
        population_factor = (x_low * lines.g_up[np.newaxis, :] / lines.g_low[np.newaxis, :] - x_up)
        center_tau = (lines.a_stein[np.newaxis, :] * freq_factor * 
                     n_mol_flat[:, np.newaxis] * population_factor)

        # Group overlapping lines (within broadening parameter) 
        #line_groups = self._find_overlapping_line_groups(lines.freq, dv_flat)
        
        # Blackbody calculation - vectorized for all temperatures at once
        bb_vals = self._bb(lines.freq[np.newaxis, :], t_kin_flat[:, np.newaxis])
        
        # Blackbody and frequency ratio calculations
        freq_ratio = lines.freq[np.newaxis, :] / c.SPEED_OF_LIGHT_CGS
        
        # Calculate intensity with proper normalization factors from original code
        #intensity = np.zeros_like(center_tau)
        
        # Pre-compute constants for efficiency
        sqrt_ln2_inv = 1.0 / (2.0 * np.sqrt(np.log(2.0)))  # For curve_growth method
        
        if method == "radex":
            intensity = (c.FGAUSS_PREFACTOR * 1e5 * dv_flat[:, np.newaxis] * 
                                freq_ratio * bb_vals * (1.0 - np.exp(-center_tau)))
        elif method == "curve_growth":
            fint_val = self._fint_multi(center_tau, dv_flat, 
                                      bb_vals, freq_ratio, sqrt_ln2_inv, lines.freq)
            intensity = fint_val
        
        if not was_scalar:
            intensity = intensity.reshape(output_shape + (len(lines.freq),))
            center_tau = center_tau.reshape(output_shape + (len(lines.freq),))
        elif was_scalar:
            intensity = intensity.squeeze()
            center_tau = center_tau.squeeze()

        return intensity, center_tau

    def calc_intensity(self, t_kin: Optional[float] = None, n_mol: Optional[float] = None, 
                      dv: Optional[float] = None, method: Literal["curve_growth", "radex"] = "curve_growth") -> None:
        """Calculate the intensity for a given set of physical parameters. This implements Eq. A1 and A2 in
        Banzatti et al. 2012.

        The calculation method to obtain the intensity from the opacity can be switched between the curve-of-growth
        method used in Banzatti et al. 2012, which considers broadening for high values of tau and the simple expression
        used e.g. in RADEX (van der Tak et al. 2007) which saturates at tau ~ few. For low values (tau < 1), they
        yield the same values.

        Parameters
        ----------
        t_kin: float, optional
            Kinetic temperature in K
        n_mol: float, optional
            Column density in cm**-2
        dv: float, optional
            Intrinsic (turbulent) line width in km/s
        method: Literal["curve_growth", "radex"], default "curve_growth"
            Calculation method, either "curve_growth" for Eq. A1 or "radex" for less accurate approximation
        """
        # Check if we can use cached result
        if (self._cache_valid and 
            self._t_kin == t_kin and 
            self._n_mol == n_mol and 
            self._dv == dv and
            self._intensity is not None):
            return

        # Validate inputs
        if t_kin is None or n_mol is None or dv is None:
            raise ValueError("t_kin, n_mol, and dv must all be provided")

        # Use optimized core calculation
        intensity, tau = self._calc_intensity_core(
            np.asarray(t_kin), np.asarray(n_mol), np.asarray(dv), method
        )
        
        # Update instance variables and cache
        self._t_kin = t_kin
        self._n_mol = n_mol
        self._dv = dv
        self._tau = tau
        self._intensity = intensity
        self._cache_valid = True

    def calc_intensity_batch(self, t_kin_array: np.ndarray, n_mol_array: np.ndarray, 
                           dv_array: np.ndarray, method: Literal["curve_growth", "radex"] = "curve_growth") -> np.ndarray:
        """Calculate intensities for multiple parameter combinations using vectorized operations.
        
        This method is optimized for processing many parameter sets simultaneously using the
        unified core calculation engine for maximum efficiency.

        Parameters
        ----------
        t_kin_array: np.ndarray
            Array of kinetic temperatures in K
        n_mol_array: np.ndarray  
            Array of column densities in cm**-2
        dv_array: np.ndarray
            Array of intrinsic line widths in km/s
        method: Literal["curve_growth", "radex"], default "curve_growth"
            Calculation method

        Returns
        -------
        np.ndarray
            Array with intensities - shape depends on input array broadcasting
        """
        # Use the optimized core calculation
        intensity, _ = self._calc_intensity_core(
            np.asarray(t_kin_array), 
            np.asarray(n_mol_array), 
            np.asarray(dv_array), 
            method
        )
        
        return intensity

    def invalidate_cache(self) -> None:
        """Invalidate the calculation cache, forcing recalculation on next call."""
        self._cache_valid = False

    def bulk_parameter_update_vectorized(self, parameter_combinations: list, method: Literal["curve_growth", "radex"] = "curve_growth") -> np.ndarray:
        """Update multiple parameter combinations and calculate intensities in a vectorized manner.
        
        Parameters
        ----------
        parameter_combinations: list
            List of dictionaries, each containing 't_kin', 'n_mol', and 'dv' keys
        method: Literal["curve_growth", "radex"], default "curve_growth"
            Calculation method
            
        Returns
        -------
        np.ndarray
            2D array with shape (n_combinations, n_lines) containing intensities
        """
        if not parameter_combinations:
            return np.array([])
        
        # Extract parameter arrays efficiently
        t_kin_vals = np.array([combo['t_kin'] for combo in parameter_combinations])
        n_mol_vals = np.array([combo['n_mol'] for combo in parameter_combinations]) 
        dv_vals = np.array([combo['dv'] for combo in parameter_combinations])
        
        # Use optimized core calculation only
        intensity, _ = self._calc_intensity_core(t_kin_vals, n_mol_vals, dv_vals, method)
        return intensity

    def get_table_in_range(self, lam_min: float, lam_max: float) -> Any:
        """Get a table with the lines in the specified wavelength range.

        Parameters
        ----------
        lam_min: float
            Minimum wavelength in microns
        lam_max: float
            Maximum wavelength in microns

        Returns
        -------
        pd.DataFrame:
            Dataframe with the lines in the specified range
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")

        mask = (self.molecule.lines_as_namedtuple.lam >= lam_min) & (self.molecule.lines_as_namedtuple.lam <= lam_max)
        return self.get_table[mask]
    
    def get_lines_in_range_with_intensity(self, lam_min: float, lam_max: float):
        """
        Get MoleculeLine objects in the specified wavelength range with computed intensity and tau values.
        
        Parameters
        ----------
        lam_min : float
            Minimum wavelength in microns
        lam_max : float
            Maximum wavelength in microns
            
        Returns
        -------
        list
            List of tuples (MoleculeLine, intensity, tau) within the range
        """
        # Get lines in range from the molecule line list
        lines_in_range = self.molecule.get_lines_in_range(lam_min, lam_max)
        
        # If no intensity calculated yet, return empty list
        if self._intensity is None or self._tau is None:
            return []
        
        # Create list of tuples with line, intensity, and tau
        result = []
        lines_array = self.molecule.lines_as_namedtuple
        
        for line in lines_in_range:
            # Find the index of this line in the full array
            line_idx = None
            for i, (lam, lev_up, lev_low) in enumerate(zip(lines_array.lam, lines_array.lev_up, lines_array.lev_low)):
                if (line.lam == lam and line.lev_up == lev_up and line.lev_low == lev_low):
                    line_idx = i
                    break
            
            if line_idx is not None:
                intensity = self._intensity[line_idx]
                tau = self._tau[line_idx]
                result.append((line, intensity, tau))
        
        return result

    @property
    def tau(self) -> Optional[np.ndarray]:
        """np.ndarray: Opacities per line"""
        return self._tau

    @property
    def intensity(self) -> Optional[np.ndarray]:
        """np.ndarray: Calculated intensity per line in erg/s/cm**2/sr/Hz"""
        return self._intensity

    @property
    def molecule(self) -> 'MoleculeLineList':
        """MoleculeLineList: Molecular line list data used for calculation"""
        return self._molecule

    @property
    def t_kin(self) -> Optional[float]:
        """float: Kinetic temperature in K used for calculation"""
        return self._t_kin

    @property
    def n_mol(self) -> Optional[float]:
        """float: Molecular column density in cm**-2 used for calculation"""
        return self._n_mol

    @property
    def dv(self) -> Optional[float]:
        """float: Line width in km/s used for calculation"""
        return self._dv

    def __repr__(self) -> str:
        return f"Intensity(Mol-Name={self.molecule.name}, t_kin={self.t_kin} n_mol={self.n_mol} dv={self.dv}, " \
               f"tau={self.tau}, intensity={self.intensity})"

    @property
    def get_table(self) -> Any:
        """pd.DataFrame: Pandas dataframe with line data"""
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")

        # Use optimized approach with individual MoleculeLine objects for better performance
        if hasattr(self.molecule, 'lines') and self.molecule.lines:
            # Pre-calculate list comprehensions for better performance
            tau_list = self.tau.tolist() if self.tau is not None else [None] * len(self.molecule.lines)
            intens_list = self.intensity.tolist() if self.intensity is not None else [None] * len(self.molecule.lines)
            
            data_dict = {
                'lev_up': [line.lev_up for line in self.molecule.lines],
                'lev_low': [line.lev_low for line in self.molecule.lines],
                'lam': [line.lam for line in self.molecule.lines],
                'tau': tau_list,
                'intens': intens_list,
                'a_stein': [line.a_stein for line in self.molecule.lines],
                'e_up': [line.e_up for line in self.molecule.lines],
                'g_up': [line.g_up for line in self.molecule.lines]
            }
            return pd.DataFrame(data_dict)

        # Fallback to namedtuple approach
        lines = self.molecule.lines_as_namedtuple
        return pd.DataFrame({
            'lev_up': lines.lev_up,
            'lev_low': lines.lev_low,
            'lam': lines.lam,
            'tau': self.tau,
            'intens': self.intensity,
            'a_stein': lines.a_stein,
            'e_up': lines.e_up,
            'g_up': lines.g_up
        })

    def _repr_html_(self) -> Optional[str]:
        # noinspection PyProtectedMember
        pd = _get_pandas()
        return self.get_table._repr_html_() if pd is not None else None