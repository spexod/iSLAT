# -*- coding: utf-8 -*-

"""
The class Chi2Spectrum calculates the chi2 statistics from a spectrum
* There are two way of adding flux measurments to a Chi2Spectrum instance
  1.) Load a text file with the flux measurements:

        >>> chi2_evaluate = Chi2Spectrum()
        >>> chi2_evaluate.load_file("input_chi2.dat")

      with an input_file containing the lam_min, lam_max, flux and flux_error as columns

  2.) Through adding individual flux measurements to the Chi2Spectrum instance:

        >>> chi2_evaluate = Chi2Spectrum()
        >>> chi2_evaluate.add_measurement(FluxMeasurement(lam_min=4.60, lam_max=4.62, flux=2e-16, flux_error=1e-17))
        >>> chi2_evaluate.add_measurement(FluxMeasurement(lam_min=4.62, lam_max=4.66, flux=2e-16, flux_error=2e-17))
        >>> chi2_evaluate.add_measurement(FluxMeasurement(lam_min=4.66, lam_max=4.68, flux=4e-17, flux_error=2e-18))
        >>> chi2_evaluate.add_measurement(FluxMeasurement(lam_min=4.86, lam_max=4.94, flux=4e-17, flux_error=1e-18))

* Only read access to the fields is granted through properties

- 01/06/2020: SB, initial version

"""

from collections import namedtuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None
    pass


__all__ = ["Chi2Spectrum", "FluxMeasurement"]

# Can be used to manually add a flux measurement
FluxMeasurement = namedtuple("FluxMeasurement", ["lam_min", "lam_max", "flux", "flux_error"])


class Chi2Spectrum:
    _Chi2Comparison = namedtuple("Chi2Comparison", ["lam_min", "lam_max", "flux", "flux_error", "flux_model", "chi2"])

    def __init__(self):
        """Initialization of the Chi2Spectrum class
        """

        # Spectrum object
        self._spectrum = None

        # List of FluxMeasurements objects
        self._measurements = []

        # List of Chi2Comparison objects after the statistics has been calculated
        self._chi2 = []

        # Sum of all Chi2 values
        self._chi2_total = 0

    def add_measurement(self, measurement):
        """Adds one individual flux measurement"

        Parameters
        ----------
        measurement: FluxMeasurement
            Measurment to add
        """

        self._measurements.append(measurement)

    def load_file(self, fname):
        """Reads a file with a list of flux measurments

        Parameters
        ----------
        fname: str
            Path/filename of the input file

        Notes
        -----

        An example file with 4 measurements look like this:

        # lammin    lammax    flux           flux_error
        # [micron]  [micron]  [erg/cm**2/s]  [erg/cm**2/s]
        4.60      4.62      2e-16          1e-17
        4.62      4.66      2e-16          2e-17
        4.66      4.68      4e-17          2e-18
        4.86      4.94      4e-17          1e-18

        The file is format free (spaces to separate columns)
        """

        #lam_min, lam_max, flux, flux_error = np.loadtxt(fname, unpack=True, usecols = (3,4,5,6))
        print(f"Loading flux measurements from {fname}")
        measur = pd.read_csv(fname, sep=',', usecols=['xmin','xmax','Flux_islat','Err_islat'])
        lam_min = measur['xmin']
        lam_max = measur['xmax']
        flux = measur['Flux_islat']
        flux_error = measur['Err_islat']

        for d in zip(lam_min, lam_max, flux, flux_error):
            self.add_measurement(FluxMeasurement(*d))

    def evaluate_spectrum(self, spectrum, flux_unit="ergscm2"):
        """Evaluates the Chi2 values for a spectrum

        Parameters
        ----------
        spectrum: Spectrum
            Input Spectrum instance from which the chi2 value should be calculated
        flux_unit:
            Which units of the flux should be used for comparison? Either "ergscm2" for erg/s/cm**2 or "jy" for Jansky
        """

        # 1. get wavelength grid and select flux array
        lam = spectrum.lamgrid
        if flux_unit == "ergscm2":
            flux = spectrum.flux
        elif flux_unit == "jy":
            flux = spectrum.flux_jy
        else:
            raise ValueError("Flux units not known")

        # 2. loop over all measurements and calculate the chi**2 values
        self._chi2 = []
        self._chi2_total = 0

        for d in self._measurements:

            # integrate the model flux of the range of the measurement range
            integral_range = np.where(np.logical_and(lam > d.lam_min, lam < d.lam_max))
            flux_model = np.trapz(flux[integral_range], x=lam[integral_range])

            # calculate the chi2 statistics
            chi2 = (d.flux - flux_model) ** 2 / d.flux_error ** 2

            # add one chi2 comparison to the list
            self._chi2.append(self._Chi2Comparison(*d, flux_model, chi2))

            # add to the total value
            self._chi2_total += chi2

    @property
    def measurements(self):
        """list of FluxMeasurement: Flux measurements"""
        return self._measurements

    @property
    def chi2(self):
        """list of _Chi2Comparison: Chi2 values for each flux measurements"""
        return self._chi2

    @property
    def chi2_total(self):
        """float: Total chi2 value"""
        return self._chi2_total

    @property
    def get_table(self):
        """pd.Dataframe: Pandas dataframe"""

        if pd is None:
            raise ImportError("Pandas required to create table")

        return pd.DataFrame({'lam_min': [c.lam_min for c in self._chi2],
                             'lam_max': [c.lam_max for c in self._chi2],
                             'flux': [c.flux for c in self._chi2],
                             'flux_error': [c.flux_error for c in self._chi2],
                             'flux_model': [c.flux_model for c in self._chi2],
                             'chi2': [c.chi2 for c in self._chi2]})

    def _repr_html_(self):
        # noinspection PyProtectedMember
        return self.get_table._repr_html_()