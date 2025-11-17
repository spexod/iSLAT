# iSLAT v5

**This is version v5.00.01 - beta release**\

**Key updates from v5.00.00:**
- Further improvements in speed and computational efficiency.
- The IR model now accounts for Line opacity overlap/mutually saturation. 
- - Enabled by a rewrite of the IR model used by the code.
- - Importantly, this feature does not slow the code and in some cases is faster. 
- New full spectrum output and display features
- - Output full spectrum function under the spectrum functions tab, save the full spectral range as a PDF.
- - Display full spectrum function under the spectrum functions tab, open a new window to view the full spectrum at the same time as the regular set of plots.
- - Toggle full spectrum function button at the of the window, toggles an interactive and live view of the full spectra in the main window.
- Display thermal broading for each molecule

*Important note: In version 5.00.01 of the beta release of iSLAT version five, most of the new functions are optimized for JWST data. We are working to improve the use for other data sets as well, and many of the new features might still work with other data sets.*

** **RECOMMENDED**: ** If updating from version 4: clone the new iSLAT in a new folder on your machine, and copy over any save files you may have made with version 4 into the respective folders (e.g. SAVES/ or LINESAVES/ or LINELISTS/). 

**Acknowledgements:** McCaskill et al., in prep. (version 5), [Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract) (version 4), [Johnson et al. 2024](https://doi.org/10.5281/zenodo.12167854) (version 4), 
[Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract) (if you use the "MIRI_general" line list), 
[LMFIT](https://lmfit.github.io/lmfit-py/index.html) (if you use any of the fitting functions in iSLAT), plus other packages as appropriate.

**Past Updates**\

**This is version v5.00 - beta release**\
*We are in the process of releasing a new version of iSLAT, version 5; here you can start testing it before its official release (you may need to update the code from here more frequently until we officially release the new version). Please refer to the readme file in version 4 for a general description of the tool and its functions. Please let us know of any bugs you may find!*

**Main updates from version 4:** 
- Substantial speed improvement from global refactoring of the code and data architecture
- Improved management of molecular models (now including the option to have individual RV shift and broadening)
- New function to measure line fluxes on a sample rather than on single spectrum ("Fit Saved Lines to sample")
- Several improvements to GUI organization and interactivity