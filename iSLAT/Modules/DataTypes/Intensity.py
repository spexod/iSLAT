# -*- coding: utf-8 -*-

"""
The class Intensity calculates the intensities

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version
- 07/21/2025 Johnny McCaskill, refactored for performance and clarity

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
    def _fint(cls, tau: np.ndarray) -> np.ndarray:
        """Vectorized calculation of the curve-of-growth integral.
        
        Computes the integral: (2/sqrt(pi)) * integral from -inf to +inf of (1 - exp(-tau*exp(-x^2))) dx
        for an array of tau values using Gaussian quadrature. This method uses optimized broadcasting
        to calculate the integral for all tau values simultaneously.
        
        Parameters
        ----------
        tau: np.ndarray
            Array with tau values (any shape)
            
        Returns
        -------
        np.ndarray
            Array with same shape as tau containing integral values
        """
        if not cls._GAUSS_QUAD_INITIALIZED:
            cls._initialize_gauss_quad()
        
        # Store original shape and flatten for vectorized computation
        original_shape = tau.shape
        tau_flat = tau.ravel()
        
        # Vectorized calculation of the curve-of-growth integral
        # The integral is: (2/sqrt(pi)) * integral from -inf to +inf of (1 - exp(-tau*exp(-x^2))) dx
        # We approximate with Gaussian quadrature from -6 to +6
        
        # Broadcasting: (n_tau_values, 1) * (1, n_quad_points) -> (n_tau_values, n_quad_points)
        x_quad = cls._GAUSS_QUAD_X[np.newaxis, :]  # Shape: (1, n_quad_points)
        tau_broadcast = tau_flat[:, np.newaxis]    # Shape: (n_tau_values, 1)
        
        # Compute exp(-x^2) for each quadrature point
        exp_neg_x_squared = np.exp(-(x_quad ** 2))
        
        # Compute the integrand: (1 - exp(-tau * exp(-x^2)))
        # The exp(-x^2) factor is NOT included here as it's part of the Gaussian quadrature weights
        #integrand = (1.0 - np.exp(-tau_broadcast * exp_neg_x_squared)) * exp_neg_x_squared
        #integrand = (1.0 - np.exp(-tau_broadcast * exp_neg_x_squared))
        integrand = (1 - np.exp(-tau_broadcast * exp_neg_x_squared))

        # Apply weights and integrate
        # The Gaussian quadrature already includes the exp(-x^2) weighting
        weights = cls._GAUSS_QUAD_W #* exp_neg_x_squared[0, :]  # Shape: (20,)
        integrand = 1.0 - np.exp(-tau_flat[:, None] * exp_neg_x_squared)  # (n_tau, n_quad)
        #integral_values = np.dot(integrand, weights)  # Shape: (n_tau_values,)
        
        integral_values = integrand @ weights  # no extra exp(-x^2) in weights
        
        # Apply the normalization factor (2/sqrt(pi))
        integral_values *= (2.0 / np.sqrt(np.pi))
        
        return integral_values.reshape(original_shape)

    @classmethod
    def _fint_multi(cls, tau0_neighbors: np.ndarray, delta_v: np.ndarray, dv_cond: np.ndarray) -> np.ndarray:
        """
        Multi-line curve-of-growth integral for a blended set of Gaussian profiles.

        Parameters
        ----------
        tau0_neighbors : (n_cond, n_neighbors) array
            Line-center optical depths for all neighbors, including the target line.
        delta_v : (n_neighbors,) array
            Velocity offsets (km/s) of each neighbor relative to the target line center.
        dv_cond : (n_cond,) array
            Intrinsic FWHM (km/s) for each condition.

        Returns
        -------
        F : (n_cond,) array
            The dimensionless curve-of-growth integral for the blend.
        """
        if not cls._GAUSS_QUAD_INITIALIZED:
            cls._initialize_gauss_quad()

        # Shapes
        tau0_neighbors = np.atleast_2d(tau0_neighbors)        # (n_cond, n_neighbors)
        delta_v = np.atleast_1d(delta_v)                      # (n_neighbors,)
        dv_cond = np.atleast_1d(dv_cond)                      # (n_cond,)
        original_shape = tau0_neighbors.shape
        n_cond, n_neighbors = original_shape

        sigma_v = dv_cond / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        x = cls._GAUSS_QUAD_X[np.newaxis, :]
        w = cls._GAUSS_QUAD_W[np.newaxis, :]
        
        v_diff = x[:, :, np.newaxis] - delta_v[np.newaxis, np.newaxis, :]
        sigma_v_sq = (sigma_v[:, np.newaxis, np.newaxis] ** 2)
        
        gaussian_profiles = np.exp(-0.5 * (v_diff ** 2) / sigma_v_sq)
        
        tau_total = (tau0_neighbors[:, np.newaxis, :] * gaussian_profiles).sum(axis=2)

        integrand = 1.0 - np.exp(-tau_total)
        
        F = (2.0 / np.sqrt(np.pi)) * (integrand * w).sum(axis=1)

        return F

    def _find_overlapping_line_groups(self, frequencies: np.ndarray, dv_vals: np.ndarray, c_light_ratio: float) -> list:
        """
        Find groups of lines that overlap within the broadening parameter.
        
        Parameters
        ----------
        frequencies : np.ndarray
            Line frequencies in Hz
        dv_vals : np.ndarray
            Broadening velocities in km/s for each condition
        c_light_ratio : float
            Speed of light ratio (c_cgs / 1e5)
            
        Returns
        -------
        list
            List of lists, each containing indices of overlapping lines
        """
        n_lines = len(frequencies)
        if n_lines <= 1:
            return [[i] for i in range(n_lines)]
        
        # Use maximum broadening to be conservative
        max_dv = np.max(dv_vals)  # km/s
        
        # Sort lines by frequency for efficient grouping
        sorted_indices = np.argsort(frequencies)
        sorted_freqs = frequencies[sorted_indices]
        
        groups = []
        used = set()
        
        for i, idx in enumerate(sorted_indices):
            if idx in used:
                continue
                
            # Start a new group with this line
            group = [idx]
            used.add(idx)
            freq_center = frequencies[idx]
            
            # Find all lines within broadening distance
            for j, idx2 in enumerate(sorted_indices[i+1:], i+1):
                if idx2 in used:
                    continue
                    
                freq2 = frequencies[idx2]
                # Calculate velocity separation
                delta_v = abs(freq_center - freq2) / freq_center * c_light_ratio
                
                if delta_v <= max_dv:  # Within broadening parameter
                    group.append(idx2)
                    used.add(idx2)
                else:
                    # Lines are sorted, so we can break here
                    break
            
            groups.append(group)
        
        return groups

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
        q_sum_vals = np.interp(t_kin_flat, partition.t, partition.q)
        
        # Efficient broadcasting for line calculations - minimize memory usage
        inv_t_kin_flat = 1.0 / t_kin_flat
        exp_factor_low = np.exp(-np.outer(inv_t_kin_flat, lines.e_low))
        exp_factor_up = np.exp(-np.outer(inv_t_kin_flat, lines.e_up))
        
        # Calculate populations directly without storing intermediate arrays
        q_sum_inv = 1.0 / q_sum_vals[:, np.newaxis]
        x_low = (exp_factor_low * lines.g_low[np.newaxis, :]) * q_sum_inv
        x_up = (exp_factor_up * lines.g_up[np.newaxis, :]) * q_sum_inv
        
        wavelength_microns = c.SPEED_OF_LIGHT_MICRONS / lines.freq[np.newaxis, :]
        tau_amp = ((np.log(2) / np.pi)**0.5 * lines.a_stein[np.newaxis, :] * 
                  n_mol_flat[:, np.newaxis] * (wavelength_microns * 1e-4)**3 / 
                  (4.0 * np.pi * dv_flat[:, np.newaxis] * 1e5))

        population_factor = (x_low * lines.g_up[np.newaxis, :] / lines.g_low[np.newaxis, :] - x_up)
        center_tau = tau_amp * population_factor
        # This grid will contain the tau values at the center of the line profile 

        n_lines = len(lines.freq)
        
        # Handle line opacity overlap
        c_light_ratio = c.SPEED_OF_LIGHT_CGS / 1e5  # km/s
        
        # Group overlapping lines (within broadening parameter)
        line_groups = self._find_overlapping_line_groups(lines.freq, dv_flat, c_light_ratio)
        
        # Blackbody calculation - vectorized for all temperatures at once
        bb_vals = self._bb(lines.freq[np.newaxis, :], t_kin_flat[:, np.newaxis])
        
        # Pre-compute common factors for efficiency
        dv_factor = 1e5 * dv_flat[:, np.newaxis]
        freq_ratio = lines.freq[np.newaxis, :] / c.SPEED_OF_LIGHT_CGS
        
        # Calculate intensity accounting for line opacity overlap
        intensity = np.zeros_like(center_tau)
        
        for group in line_groups:
            if len(group) == 1:
                # Single line - calculate normally
                j = group[0]
                if method == "radex":
                    exp_neg_tau = np.exp(-center_tau[:, j])
                    intensity[:, j] = (c.FGAUSS_PREFACTOR * dv_factor[:, 0] * freq_ratio[:, j] * 
                                     bb_vals[:, j] * (1.0 - exp_neg_tau))
                elif method == "curve_growth":
                    fint_val = self._fint(center_tau[:, j])
                    intensity[:, j] = (c.FGAUSS_PREFACTOR * dv_factor[:, 0] * freq_ratio[:, j] * 
                                     bb_vals[:, j] * fint_val)
            else:
                # Overlapping lines - combine opacities first, then calculate intensity
                combined_tau = np.sum(center_tau[:, group], axis=1)  # Sum opacities
                
                # Use representative values (frequency-weighted averages)
                group_freqs = lines.freq[group]
                group_weights = group_freqs / np.sum(group_freqs)
                
                avg_bb = np.sum(bb_vals[:, group] * group_weights[np.newaxis, :], axis=1)
                avg_dv_factor = dv_factor[:, 0]  # Same for all lines
                avg_freq_ratio = np.sum(freq_ratio[:, group] * group_weights[np.newaxis, :], axis=1)
                
                if method == "radex":
                    exp_neg_combined_tau = np.exp(-combined_tau)
                    combined_intensity = (c.FGAUSS_PREFACTOR * avg_dv_factor * avg_freq_ratio * 
                                        avg_bb * (1.0 - exp_neg_combined_tau))
                elif method == "curve_growth":
                    fint_combined = self._fint(combined_tau)
                    combined_intensity = (c.FGAUSS_PREFACTOR * avg_dv_factor * avg_freq_ratio * 
                                        avg_bb * fint_combined)
                
                # Distribute the combined intensity among group members weighted by their individual tau
                for j in group:
                    weight = center_tau[:, j] / np.maximum(combined_tau, 1e-10)
                    intensity[:, j] = combined_intensity * weight
        
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