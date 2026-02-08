# Changelog

## v5.02.02 - February 08, 2026

### iSLAT Package
- Added new project toml file to define the package
- As of commit [10eec...](https://github.com/John-M87/iSLATTests/commit/10eec425a25fd7973497b52d9acfad376b0c355c) (Made iSLAT installable as a package) you will need to do one of the follow to run this branch:
    - Run ``pip install -e .`` from the project root (i.e. the directory where the `pyproject.toml` is located)
    - Run ``pip install -e git://github.com/John-M87/iSLATTests.git@{ tag name }#egg={ desired egg name }`` (Untested)
    - In the future you will be able to pip install as usual
- Updated some files (such as iSLAT-launch.py) to remove code that is no longer needed
- Added a new `iSLAT` console command when installed as a package
    - When you type `iSLAT` into the command line it will auto run a start script and launch the GUI
    - Could be used with a bat file if desired

### New Features

#### GUI Improvements
- Added a new plus button next to the load spectrum button
    - Lets you add multiple spectra to be used as a sample
    - When a sample is loaded the Fit Saved Lines to Sample function automatically uses it
    - When a sample is loaded you are also able to use the left and right arrow keys to dynamically move between the loaded spectra

#### Example Notebooks
- Edited all Notebooks to use a regular import for iSLAT instead of the sys path appending method

### Bug Fixes
- Updated HITRANdata file path in the FileHandling module

### Code Quality
- Bumped version to `v5.02.02`

---

## v5.02.01 - February 07, 2026

### New Features

#### User Settings System
- Added a versioning system for user settings (`user_settings_version` field in `DefaultUserSettings.json`)
- User settings are now automatically migrated when the default version is bumped: new fields are added, removed fields are pruned, and key order matches the defaults file
- If `UserSettings.json` does not exist, it is now created from `DefaultUserSettings.json` with `first_startup` set to `true`
- The `first_startup` and `reload_default_files` flags are now correctly persisted back to `false` after the first launch
- Added `save_user_settings()` function for writing settings back to disk while handling theme dict-to-string conversion and stripping runtime-only keys

#### Full Spectrum Plot
- The full spectrum plot now updates automatically when a new spectrum file is loaded
- Wavelength range is recalculated and the figure is recreated when loading data that spans a different range
- The Tk canvas is properly rebuilt when the underlying matplotlib figure changes

#### Plotting Module Refactor
- Added new modular plot classes: `BasePlot`, `FullSpectrumPlot`, `LineInspectionPlot`, `PopulationDiagramPlot`, `MainPlotGrid`
- Refactored `PlotRenderer` and `MainPlot` to use the new plot classes for improved modularity
- Plot classes now clear previous axes with `fig.clf()` so regeneration doesn't stack on top
- Line inspection plot now includes information about lines in the current selection, displayed in the data field

#### GUI Improvements
- Added a toggle for the visibility of all molecules at once by pressing the "On" label in the control panel
- Changed the "Toggle Summed Flux" button label to "Toggle Total Model"
- Repositioned the X buttons for removing input and output files
- Updated tooltips for several control panel items
- First draft of model subtraction from data
- The toggle total model button now toggles the visibility of the total model line in the main plot, and the total model is now included in the full spectrum plot

#### Full Spectrum Mode
- Reworked full spectrum mode to work with instruments other than MIRI (MIRI output remains the same)
- First draft of updated pixel sampling using [spectres](https://github.com/ACCarnall/spectres) adaptation

#### Example Notebooks
- Reorganized example notebooks into `DataTypes`, `Interactive Widgets`, `Plotting`, and `Fitting` folders
- Added `DataTypes_Overview.ipynb` — overview of the data type classes
- Added `Molecule_Examples.ipynb` — examples for the Molecule class
- Added `MoleculeDict_Examples.ipynb` — creating, managing, and visualizing molecular data
- Added `MoleculeLineList_Examples.ipynb` — working with HITRAN line list data
- Added `Plotting_Examples.ipynb` — static plotting examples
- Added `Animated_Parameter_Evolution.ipynb` — animated parameter sweep plots
- Added 10 interactive widget notebooks: Bulk Parameter Explorer, Intensity Parameter Explorer, Full Spectrum Plot, Line Inspection, Molecule Parameters, Low-Level Spectrum Viewer, Molecule Selector Dropdown, Side-by-Side Comparison, Spectrum Builder, and Visibility Toggles

### Bug Fixes
- Fixed `reload_default_files` defaulting to `True` in `check_HITRAN()`, which caused default files to reload on every launch
- Fixed molecule color access when setting molecule from a new load
- Fixed incorrect caching behavior in FileHandling `__init__.py`
- Removed unused import of `defaultdict` from `Molecule.py`

### Code Quality
- Improved type hints in `Molecule` and `MoleculeDict` classes
- Updated `bulk_update_parameters` in `MoleculeDict` to accept multiple kinds of input for specifying which molecules to update
- Updated `safe_float` method in `MoleculeDict` with improved validation
- Added `UserSettings.json` to `.gitignore` and removed it from Git tracking
- Bumped version to `v5.02.01`

---

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