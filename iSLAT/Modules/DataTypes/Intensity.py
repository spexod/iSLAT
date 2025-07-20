# -*- coding: utf-8 -*-

"""
The class Intensity calculates the intensities

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

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

    @staticmethod
    def _fint(tau: np.ndarray) -> np.ndarray:
        """Evaluates the integral in Eq. A1 of Banzatti et al. 2012 for an array of tau values.

        To calculate the integral

        int 1.0 - exp(-tau*exp(-x**2)) dx

        for all values of tau simultaneously, we cannot use an adaptive algorithm such as quad in scipy.integrate.
        Thus, the same approach as in the Fortran 90 code is used and the integral is evaluated using a 20-point
        Gaussian quadrature.

        Parameters
        ----------
        tau: np.ndarray
            Array with tau values

        Returns
        -------
        np.ndarray:
            Array with the integral values
        """
        fixed_quad = _get_scipy()
        tau_param = np.array(tau).repeat(20).reshape(-1, 20)
        i, _ = fixed_quad(lambda x: 1.0 - np.exp(-tau_param * np.exp(-x ** 2)), -6, 6, n=20)
        return i

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

        self._t_kin = t_kin
        self._n_mol = n_mol
        self._dv = dv

        m = self._molecule

        # Early validation to avoid expensive calculations
        if t_kin is None or n_mol is None or dv is None:
            raise ValueError("t_kin, n_mol, and dv must all be provided")

        # 1. calculate the partition function
        partition = m.partition
        if t_kin < np.min(partition.t) or t_kin > np.max(partition.t):
            raise ValueError('t_kin outside range of partition function')

        q_sum = np.interp(t_kin, partition.t, partition.q)

        # 2. line opacity - use optimized data access
        lines = m.lines_as_namedtuple
        
        # Use vectorized operations for better performance
        exp_e_low = np.exp(-lines.e_low / t_kin)
        exp_e_up = np.exp(-lines.e_up / t_kin)
        
        x_low = lines.g_low * exp_e_low / q_sum
        x_up = lines.g_up * exp_e_up / q_sum

        # Eq. A2 of Banzatti et al. 2012 - optimized calculation
        freq_factor = c.SPEED_OF_LIGHT_CGS ** 3 / (8.0 * np.pi * lines.freq ** 3 * 1e5 * dv * c.FGAUSS_PREFACTOR)
        population_factor = (x_low * lines.g_up / lines.g_low - x_up)
        tau = lines.a_stein * freq_factor * n_mol * population_factor
        
        # Check for problematic values only if debugging
        if __debug__:
            zero_g_low = np.sum(lines.g_low == 0)
            if zero_g_low > 0:
                print(f"Warning: {zero_g_low} lines have g_low = 0")

        # 3. line intensity - optimized calculation
        bb_vals = self._bb(lines.freq, t_kin)
        freq_ratio = lines.freq / c.SPEED_OF_LIGHT_CGS
        
        if method == "radex":
            intensity = (c.FGAUSS_PREFACTOR * 1e5 * dv * freq_ratio * bb_vals * 
                        (1.0 - np.exp(-tau)))
        elif method == "curve_growth":
            # Eq. A1 of Banzatti et al. 2012
            fint_vals = self._fint(tau)
            intensity = (1.0 / (2.0 * np.sqrt(np.log(2.0))) * 1e5 * dv * freq_ratio * 
                        bb_vals * fint_vals)
        else:
            raise ValueError("Intensity calculation method not known")

        self._tau = tau
        self._intensity = intensity
        self._cache_valid = True

    def invalidate_cache(self) -> None:
        """Invalidate the calculation cache, forcing recalculation on next call."""
        self._cache_valid = False

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
        return self.get_table._repr_html_() if pd is not None else None