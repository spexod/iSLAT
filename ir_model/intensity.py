# -*- coding: utf-8 -*-

"""
The class Intensity calculates the intensities

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

"""

import numpy as np
from scipy.integrate import fixed_quad

try:
    import pandas as pd
except ImportError:
    pd = None
    pass

from ir_model.moldata import MolData
from ir_model.constants import constants as c


__all__ = ["Intensity"]


class Intensity:

    def __init__(self, molecule):
        """Initialize an intensity class which calculates the intensities for a given molecule and provided
        physical parameters.

        Parameters
        ----------
        molecule: MolData
            Molecular data to calculate the intensity
        """

        self._molecule = molecule
        self._intensity = None
        self._tau = None
        self._t_kin = None
        self._n_mol = None
        self._dv = None

    @staticmethod
    def _bb(nu, T):
        """Blackbody function for one temperature and an array of frequencies. Uses the short and long wavelength
        approximations for accuracy.

        Parameters
        ----------
        nu: np.ndarray
            Frequency in Hz to calculate the blackbody values
        T: float
            Temperature in K

        Returns
        -------
        array:
            Blackbody intensity in erg/s/cm**2/sr/Hz
        """

        x = c.h * nu / (c.k * T)
        bb_RJ = np.where(x < 1.0e-5, 2.0 * nu ** 2 * c.k * T / (c.c ** 2), 0.0)
        bb_Wien = np.where(x > 20.0, 2.0 * c.h * nu ** 3 / c.c ** 2 * np.exp(-x), 0.0)
        bb_Planck = np.where((x >= 1.e-5) * (x <= 20.0),
                             2. * c.h * nu ** 3 / c.c ** 2 * 1. / (np.exp(np.where(x <= 20.0, x, 20.0)) - 1.), 0.0)

        return bb_RJ + bb_Wien + bb_Planck

    @staticmethod
    def _fint(tau):
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

        tau_param = np.array(tau).repeat(20).reshape(-1, 20)

        i, _ = fixed_quad(lambda x: 1.0 - np.exp(-tau_param * np.exp(-x ** 2)), -6, 6, n=20)

        return i

    def calc_intensity(self, t_kin=None, n_mol=None, dv=None, method="curve_growth"):
        """Calculate the intensity for a given set of physical parameters. This implements Eq. A1 and A2 in
        Banzatti et al. 2012.

        The calculation method to obtain the intensity from the opacity can be switched between the curve-of-growth
        method used in Banzatti et al. 2012, which considers broadening for high values of tau and the simple expression
        used e.g. in RADEX (van der Tak et al. 2007) which saturates at tau ~ few. For low values (tau < 1), they
        yield the same values.

        Parameters
        ----------
        t_kin: float
            Kinetic temperature in K
        n_mol: float
            Column density in cm**-2
        dv: float
            Intrinsic (turbulent) line width in km/s
        method: str
            Calculation method, either "curve_growth" for Eq. A1 or "radex" for less accurate approximation
        """

        self._t_kin = t_kin
        self._n_mol = n_mol
        self._dv = dv

        m = self._molecule

        # 1. calculate the partition function
        if t_kin < np.min(m.partition.t) or t_kin > np.max(m.partition.t):
            raise ValueError('t_kin outside range of partition function')

        q_sum = np.interp(t_kin, m.partition.t, m.partition.q)

        # 2. line opacity
        x_low = m.lines.g_low * np.exp(-m.lines.e_low / t_kin) / q_sum
        x_up = m.lines.g_up * np.exp(-m.lines.e_up / t_kin) / q_sum

        # Eq. A2 of Banzatti et al. 2012
        tau = m.lines.a_stein * c.c ** 3 / (8.0 * np.pi * m.lines.freq ** 3 * 1e5 * dv * c.fgauss) * n_mol \
            * (x_low * m.lines.g_up / m.lines.g_low - x_up)

        # 3. line intensity
        if method == "radex":
            intensity = c.fgauss * (1e5 * dv) * m.lines.freq / c.c * self._bb(m.lines.freq, t_kin) * \
                        (1.0 - np.exp(-tau))
        elif method == "curve_growth":
            # Eq. A1 of Banzatti et al. 2012
            intensity = 1.0 / (2.0 * np.sqrt(np.log(2.0))) * (1e5 * dv) * m.lines.freq / c.c * \
                        self._bb(m.lines.freq, t_kin) * self._fint(tau)
        else:
            raise ValueError("Intensity calculation method not known")

        self._tau = tau
        self._intensity = intensity

    @property
    def tau(self):
        """np.ndarray: Opacities per line"""
        return self._tau

    @property
    def intensity(self):
        """np.ndarray: Calculated intensity per line in erg/s/cm**2/sr/Hz"""
        return self._intensity

    @property
    def molecule(self):
        """MolData: Molecular data used for calculation"""
        return self._molecule

    @property
    def t_kin(self):
        """float: Kinetic temperature in K used for calculation"""
        return self._t_kin

    @property
    def n_mol(self):
        """float: Molecular column density in cm**-2 used for calculation"""
        return self._n_mol

    @property
    def dv(self):
        """float: Line width in km/s used for calculation"""
        return self._dv

    def __repr__(self):
        return f"Intensity(Mol-Name={self.molecule.name}, t_kin={self.t_kin} n_mol={self.n_mol} dv={self.dv}, " \
               f"tau={self.tau}, intensity={self.intensity})"

    @property
    def get_table(self):
        """pd.Dataframe: Pandas dataframe"""

        if pd is None:
            raise ImportError("Pandas required to create table")

        return pd.DataFrame({'lev_up': self.molecule.lines.lev_up,
                             'lev_low': self.molecule.lines.lev_low,
                             'lam': self.molecule.lines.lam,
                             'tau': self.tau,
                             'intens': self.intensity,
                             'a_stein': self.molecule.lines.a_stein,
                             'e_up': self.molecule.lines.e_up,
                             'g_up': self.molecule.lines.g_up})

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.get_table._repr_html_()
