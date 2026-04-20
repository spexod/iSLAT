# iSLAT v5.02.04

**iSLAT** (the **i**nteractive **S**pectral-**L**ine **A**nalysis **T**ool) is a python package that provides an interactive interface for the visualization, exploration, and analysis of molecular spectra. Synthetic spectra are made using a simple slab model written by Simon Bruderer and originally described in [Banzatti et al. 2012](https://ui.adsabs.harvard.edu/abs/2012ApJ...745...90B/abstract); the code uses molecular data from [HITRAN](https://hitran.org/) ([Gordon et al. 2022](https://www.sciencedirect.com/science/article/pii/S0022407321004416), [Gordon et al. 2026](https://www.sciencedirect.com/science/article/pii/S0022407326000014?via%3Dihub)). iSLAT has been developed for and tested on spectra at infrared wavelengths as observed at different resolving powers (R = 700-90,000) with: JWST-MIRI, Spitzer-IRS, VLT-CRIRES, IRTF-ISHELL. Examples of these spectra are included for users to practice with the tool functionalities across a range of resolving powers. iSLAT has been built as a flexible tool that should work with one-dimensional molecular spectra observed with other instruments too, provided that some requirements are met (see below). The first public version of iSLAT (version 4) is presented and described in [Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract) and is still accessible in the branch version4-retired. This is the new version of iSLAT (version 5), which includes significant improvements and updates as described in the Wiki. 

Please see the new [wiki page](https://github.com/spexod/iSLAT/wiki) for detailed information on installation, examples, and GUI functions and parameters.

## **RECOMMENDED**
If updating from version 4: clone the new iSLAT in a new folder on your machine, and copy over any save files you may have made with version 4 into the respective folders (e.g. SAVES/ or LINESAVES/ or LINELISTS/). 

## Acknowledgements
- McCaskill et al., 2026 (version 5, code) 
- [Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract) (version 4, paper)
- [Johnson et al. 2024](https://doi.org/10.5281/zenodo.12167854) (version 4, code)
- [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract) (if you use the "MIRI_general" or other line lists provided in LINELISTS/)
- [LMFIT](https://lmfit.github.io/lmfit-py/index.html) (if you use any of the fitting functions in iSLAT), plus other packages as appropriate.


**Main updates from version 4:** 
- Substantial speed improvement from global refactoring of the code and data architecture
- Improved management of molecular models (now including the option to have individual RV shift and broadening)
- New function to measure line fluxes on a sample rather than on single spectrum ("Fit Saved Lines to sample")
- Several improvements to GUI organization and interactivity
- Added a large number of example Juypter Notebooks showing some of the ways that you can use iSLAT functions without the GUI
- Updated to the latest [HITRAN](https://hitran.org/) release (2024) [Gordon et al. 2026](https://www.sciencedirect.com/science/article/pii/S0022407326000014?via%3Dihub)
- The slab model now accounts for line opacity overlap and mutual saturation, which is particularly important in the case of water ortho-para lines that overlap (see Figure 5 in [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract)).
- New functionality to display and save the full spectrum to obtain something similar to Figures 1-4 in [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract) (warning: for now this is optimized for JWST MIRI spectra only!)
    - "Output full spectrum" in the spectrum parameters menu saves the full spectral range as a single-page PDF file that can be directly included in a paper.
    - "Display full spectrum" in the spectrum parameters menu opens a new interactive window to view the full spectrum at the same time as the set of plots in the GUI, to display, for instance, on a second large screen.
    - "Toggle full spectrum" in the main GUI toggles an interactive view of the full spectrum in the main window, by replacing the default zoomed-in plots.
- The thermal broadening is calculated and displayed for reference for each molecule at the temperature set by the user.
- The molar mass of each molecule is now saved in their respective Hitran Files.
