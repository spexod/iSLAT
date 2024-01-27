# -*- coding: utf-8 -*-

"""
This file provides the necessary constants in CGS.

* For benchmark puroposes, the same values as in the  old Fortran 90 code have been used.
* In order to make the constants immutable stored them in a namedtuple.

- 01/06/2020: SB, initial version

"""

import numpy as np
from collections import namedtuple


__all__ = ["constants"]

_k = 1.3806504e-16  # Boltzman constant
_c = 2.99792458e10  # speed of light
_h = 6.62606896e-27  # Planck
_fgauss = np.sqrt(np.pi)/(2.0*np.sqrt(np.log(2.0)))  # prefactor for opacity and intensity
_aucm = 1.496e13  # au in cm
_pccm = 3.086e18  # pc in cm

_Constants = namedtuple('Constants', ['k', 'c', 'h', 'fgauss', 'aucm', 'pccm'])

constants = _Constants(_k, _c, _h, _fgauss, _aucm, _pccm)
