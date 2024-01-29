# iSLAT
**iSLAT** (the **i**nteractive **S**pectral-**L**ine **A**nalysis 
**T**ool) is a python package that provides an interactive interface
for the visualization, exploration, and analysis of molecular spectra.
Synthetic models are made using a simple slab model written by Simon 
Bruderer and originally described in [Banzatti et al. 2012](https://ui.adsabs.harvard.edu/abs/2012ApJ...745...90B/abstract);
the code uses molecular data from [HITRAN](https://hitran.org/).
iSLAT has been developed and currently tested on spectra at infrared wavelengths 
as observed at different resolving powers (R = 700-90,000) with: 
JWST-MIRI, Spitzer-IRS, IRTF-ISHELL. iSLAT has been built as a flexible
tool that should work with one-dimensional molecular spectra observed 
with other instruments too, provided some requirements are met (see below).

## Installation and updates

### Download and run the latest version from GitHub:

    git clone https://github.com/spexod/iSLAT
    cd iSLAT
    pip install -r requirements.txt
    cd iSLAT
    python iSLAT.py

Remember to update the repository from GitHub when a new version is available.

## Input and outputs
iSLAT requires a flat, continuum-subtracted spectrum as input data 
file in csv format, with a wavelength array in μm ("wave") and flux 
array in Jy ("flux").
The outputs are txt or csv files that save model parameters, spectra,
or line lists as defined by users (see more below).

## Parameters definitions and units
- Model parameters: temperature is in K, radius in au, column density
in cm<sup>-2</sup> 
- Other parameters: 
  - plot start/range and min/max wavelength (which the spectral range
  for computing the model) are in μm
  - distance is in pc
  - stellar RV is in km/s, and will shift the observed spectrum accordingly
  - line FWHM is in km/s, and is used in convolution of the model to
  match the observed line widths
  - line broadening is in km/s and is the FWHM of the intrinsic line
  broadening due to thermal motion or turbulence
  - line separation is in μm, and is used to identify isolated lines
  in the current water model

## Quick reference for main functions
- Save parameters: save in an output file the current model parameters 
(T, R, N) for each molecule; the output file will have the same name 
as the input observed spectrum plus "-save.txt" and will be stored in
the folder SAVES
- Load parameters: loads previously saved model parameters from output
file created with "Save parameters"
- Export models: export a specific or all model spectra in an output 
csv file
- Selecting a line: by dragging a region in the top spectrum plot, the
spectral region gets visualized in the bottom left plot with the 
addition of the individual transitions that dominate the emission;
the strongest of these transitions is highlighted in orange and its
properties are reported in the text box at the bottom left of the GUI
- Save line: save strongest line currently selected into an output 
csv file that will include all the line parameters; this function
needs the selection/definition of an output file from the folder 
LINESAVES under "Saved lines file"
- Fit line: fit selected line with a Gaussian function using LMFIT;
full fit results are reported in the terminal, while a selection in
the text box
- Show saved lines: marks lines saved into the output file selected 
under "Saved lines file"
- Show atomic lines: marks and labels atomic lines from the list 
saved in the folder ATOMLINES
- Find single lines: identifies and marks isolated water lines using
the parameter "Line separ." as their spectral separation
- Add molecule: adds a molecule to the list of available molecules
at the top left of the GUI; the new molecule must already be downloaded
from HITRAN and stored in a .par file in the folder "HITRANdata"
- Clear molecules: removes any additional molecules and leaves the 
default ones only

## Data samples
iSLAT's release includes some continuum-subtracted spectra that
users can use to get familiar with the different functionalities 
and applications across a wide range of resolving powers:
- one M-band spectrum from iSHELL (FZTau) from [Banzatti et al. 2023a](https://ui.adsabs.harvard.edu/abs/2023AJ....165...72B/abstract)
- two spectra from MIRI (CITau and FZTau) from [Banzatti et al. 2023b](https://ui.adsabs.harvard.edu/abs/2023ApJ...957L..22B/abstract)
and [Pontoppidan et al. 2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231117020P/abstract), respectively

## Known issues
If you use iSLAT on an Apple-silicon Mac, there is a known issue with 
tkinter where the GUI often does not take a new input or click; the
solution is to "wake up" the GUI by slightly resizing it windows in 
any way. Sometimes even just moving the GUI window around for a 
moment will do the trick.

## Acknowledging iSLAT
If you use iSLAT for your research, we would appreciate its citation 
in any publications and oral/poster presentations. The main reference
to cite is: Jellison et al. 2024
