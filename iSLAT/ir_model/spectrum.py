# -*- coding: utf-8 -*-

"""
This class Spectrum creates a spectrum from intensity instances

* The same algorithm as in the Fortran 90 code are used. This is explained in the appendix of Banzatti et al. 2012.
* After a Spectrum instance is created with the wavelength grid and resolution, different intensity components can be
  added. When they are added, only the area-scaled intensities per line are stored. The convolution with the resolution
  of the instrument is done at the end (a la lazy evaluation at the first call to retrieve the flux). This improves the
  performance very much when different components of the same molecule are added.
* The get the spectral convolution of the raw lines acceptably quick, only a range of about 15 sigma around the line
  center is evaluated. This is the same trick as has been used in the Fortran 90 code.
* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

"""

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None
    pass

from ir_model.constants import constants as c
from ir_model.intensity import Intensity


class Spectrum:

    def __init__(self, lam_min=None, lam_max=None, dlambda=None, R=None, distance=None):
        """Initialize a spectrum class and prepare it to add intensity components

        Parameters
        ----------
        lam_min: float
            Lower border of the spectrum in micron
        lam_max: float
            Upper border of the spectrum in micron
        dlambda: float
            Resolution of the spectrum to calculate in micron
        R: float
            Spectral resolution of the instrument in R = lambda/delta_lambda
        distance: float
            Distance to the disk in pc
        """

        # assure valid lambda grid range
        if lam_min >= lam_max:
            raise ValueError('lam_min must be < lam_max')

        # store parameters
        self._lam_min = lam_min
        self._lam_max = lam_max
        self._dlambda = dlambda
        self._R = R
        self._distance = distance

        # create wavelength grid
        self._lamgrid = np.linspace(lam_min, lam_max, int(1 + (lam_max - lam_min) / dlambda))

        # flux array
        self._flux = None

        # list with area scaled intensities and wavelengths to add to the spectrum
        self._I_list = np.array([])
        self._lam_list = np.array([])

        # list with the different intensity components building up the spectrum
        self._components = []

    def add_intensity(self, intensity, dA):
        """Adds an intensity component to the spectrum

        Parameters
        ----------
        intensity: Intensity
            Intensity structure to add to the spectrum
        dA: float
            Area of the component in au**2
        """

        # 0. make sure to invalidate the flux field, so that the intensity list and returned spectrum are consistent
        self._flux = None

        # 1. get intensity and wavelength of the lines
        I_all = intensity.intensity
        lam_all = intensity.molecule.lines.lam

        # 2. select only lines within the selected wavelength range
        select_border = 100 * self._lam_max / self._R
        lines_selected = np.where(np.logical_and(lam_all > self._lam_min - select_border,
                                                 lam_all < self._lam_max + select_border))[0]

        # 3. scale for area in au**2
        I_scaled = dA * I_all[lines_selected]

        # 4. append to list
        self._I_list = np.hstack((self._I_list, I_scaled))
        self._lam_list = np.hstack((self._lam_list, lam_all[lines_selected]))

        # 5. append to components
        self._components.append({'name': intensity.molecule.name, 'fname': intensity.molecule.fname,
                                't_kin': intensity.t_kin, 'n_mol': intensity.n_mol, 'dv': intensity.dv,
                                 'area': dA})

    def _convol_flux(self):
        """Internal procedure to carry out the convolution, should never be called directly

        Returns
        -------
        np.ndarray:
            List with fluxes, convolved to the spectra resolution
        """

        # 1. summarize intensities at the (exactly) same wavelength, this improves performance, as only
        #    one convolution kernel needs to be evaluated per line of a molecule (independent of intensity components)
        lam, index_wavelength = np.unique(self._lam_list, return_inverse=True)

        intens = np.zeros(lam.shape[0])
        np.add.at(intens, index_wavelength, self._I_list)

        # 2. calculate width and normalization of convolution kernel
        fwhm = lam / self._R
        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        # 3. calculation mask (=range around the lines, where the kernel needs to be evaluated)
        #    * index_lam contains the points of the wavelength grid that should be calculated
        #    * index_line contains the index of the line
        max_sigma = np.max(sigma)
        kernel_range = np.arange(-15 * max_sigma / self._dlambda, 15 * max_sigma / self._dlambda, dtype=np.int64)
        lam_grid_position = (self._lamgrid.shape[0] * (lam - self._lam_min) /
                             (self._lam_max - self._lam_min)).astype(np.int64)

        index_lam = (kernel_range.reshape(1, -1) + lam_grid_position[:, np.newaxis]).reshape((-1,))
        index_line = np.array(np.arange(lam.shape[0])).repeat(kernel_range.shape[0])

        # filter out only allowed index range (>=0 and < number of points of wavelength grid)
        allowed_index = np.logical_and(index_lam >= 0, index_lam < self._lamgrid.shape[0])
        index_lam = index_lam[allowed_index]
        index_line = index_line[allowed_index]

        # 4. calculate kernel, Eq. A3 in Banzatti et al. 2012
        kernel = norm[index_line] * intens[index_line] * \
            np.exp(-(self._lamgrid[index_lam] - lam[index_line]) ** 2 / (2.0 * sigma[index_line] ** 2))

        # 5. add up spectrum
        flux = np.zeros_like(self._lamgrid)
        np.add.at(flux, index_lam, kernel)

        # 6. scale for distance and correct units for the area
        #    note that area scaling is already performed in add_intensity
        return flux * (c.aucm / c.pccm) ** 2 * (1.0 / self._distance ** 2)

    @property
    def flux(self):
        """np.ndarray: Fluxdensity in erg/s/cm^2/micron"""
        if self._flux is None:
            self._flux = self._convol_flux()
        return self._flux

    @property
    def flux_jy(self):
        """np.ndarray: Fluxdensity in Jy/micron"""
        if self._flux is None:
            self._flux = self._convol_flux()
        flux_jy = self._flux * (1e19 / c.c) * self._lamgrid ** 2
        return flux_jy

    @property
    def lamgrid(self):
        """np.ndarray: Wavelength grid in micron"""
        return self._lamgrid

    @property
    def components(self):
        """list of dict: Intensity components added to the spectrum"""
        return self._components

    @property
    def get_table(self):
        """pd.Dataframe: Pandas dataframe"""

        if pd is None:
            raise ImportError("Pandas required to create table")

        return pd.DataFrame({'lam': self.lamgrid,
                             'flux': self.flux,
                             'flux_jy': self.flux_jy})

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.get_table._repr_html_()
