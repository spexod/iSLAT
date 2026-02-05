# Changelog

## v5.02.00 - February 05, 2026

### New Features

#### Keyboard Shortcuts
- Added `Ctrl+S` and `Ctrl+L` shortcuts to save and load spectrum parameters from the top bar menu
- Added `Ctrl+Shift+F` shortcut for Output Full Spectrum
- Added `F` key to toggle full spectrum view
- Added `Ctrl+F` (or `Cmd+F` on Mac) to open the full spectrum in a new window
- Added up/down arrow keys to cycle through molecules in the molecule list

#### Full Spectrum View
- Added the ability to span select on the full spectrum plot, which triggers a switch back to the 3 plot grid
- Made atomic and saved lines toggles work correctly in full spectrum plot mode

#### User Settings
- Added `default_spectra_file_to_load` setting to allow loading a default spectrum file on startup
- Added `save_fit_plot_grid_directly_to_PDF` user setting
- Added setting to update save file names in the user molecules CSV
- Added new default setting for line list file in configuration files
- Added global setting for intensity calculation method (`curve_growth`, `curve_growth_no_overlap`, or `radex`)

#### Batch Fitting
- Implemented multi-processing in batch fitting service using ProcessPoolExecutor for improved performance
- Created new data processing service files for batch fitting, deblending, and line saving
- Reworked TopBar file to use new helper service classes

#### UI Improvements
- Added tooltip to the load spectrum button indicating Ctrl/Cmd click functionality
- Ctrl+clicking the load spectrum button now also loads the spectrum's parameters if they exist
- Added scroll bar to fit lines sample window
- Added minimum size for subplot cells in plot grid window
- When fitting a sample with no input line list, the selection dialog now opens automatically

### Performance Improvements
- Created new PerformanceLogger module to track execution time of key functions
- Integrated PerformanceLogger into iSlat, MoleculeDict, Molecule, Intensity, and MoleculeLineList classes
- Made GUI loading asynchronous on startup
- Added parallelization to spectrum calculations with improved performance logging
- Added caching for HITRAN data files to reduce load times
- Reworked HITRAN caching system
- Optimized the `_get_pandas_dataframe` method of the MoleculeLineList class
- Made optimization changes to the population diagram and line inspection plots
- Population diagram now renders on startup
- Fixed issue where all molecules were being rendered even when not visible

### Bug Fixes
- Fixed minor bug with setting output line measurements
- Fixed batch fitting to output correct columns
- Fixed setting span extent on main plot
- Corrected datatype handling in iSLATFileHandling.py
- Fixed minor bug when exiting the line list selection dialog
- Fixed thermal broadening calculation to properly access molar mass from MoleculeLineList
- Fixed save figure button in plot grid window
- Fixed file path for the example data folder to use the correct variable name

### Data & Configuration Updates
- Updated HITRAN catalog year from 2020 to 2024 in multiple files
- Updated default configuration and molecule list files to remove "2020" from HITRAN data file paths
- Updated default user settings file
- Reformatted output of fit saved lines CSV file to move info fields to after xmin/xmax and before flux/err fields

### Code Quality
- Changed usage of `np.trapz` to `np.trapezoid` for numerical integration (numpy deprecation fix)
- Removed debug print statements from InteractionHandler and other modules
- Cleaned up leftover code
- Updated gitignore to ignore files in the example data folder that are not the provided data files
- Updated the requirements file to specify the minimum version for several packages
- Added a version field to the user settings file, planning to implement version checking and add the file to the gitignore

---

## v5.01.00 - Beta Release

### Key Updates from v5.00.00
- Further improvements in speed and computational efficiency
- The slab model now accounts for line opacity overlap and mutual saturation, which is particularly important in the case of water ortho-para lines that overlap (see Figure 5 in [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract))
- New functionality to display and save the full spectrum to obtain something similar to Figures 1-4 in [Banzatti et al. 2025](https://ui.adsabs.harvard.edu/abs/2025AJ....169..165B/abstract) (warning: for now this is optimized for JWST MIRI spectra only!)
    - "Output full spectrum" in the spectrum parameters menu saves the full spectral range as a single-page PDF file that can be directly included in a paper
    - "Display full spectrum" in the spectrum parameters menu opens a new interactive window to view the full spectrum at the same time as the set of plots in the GUI, to display, for instance, on a second large screen
    - "Toggle full spectrum" in the main GUI toggles an interactive view of the full spectrum in the main window, by replacing the default zoomed-in plots
- The thermal broadening is calculated and displayed for reference for each molecule at the temperature set by the user
- The molar mass of each molecule is now saved in their respective HITRAN files

---

## v5.00.00 - Beta Release

### Main Updates from Version 4
- Substantial speed improvement from global refactoring of the code and data architecture
- Improved management of molecular models (now including the option to have individual RV shift and broadening)
- New function to measure line fluxes on a sample rather than on single spectrum ("Fit Saved Lines to sample")
- Several improvements to GUI organization and interactivity