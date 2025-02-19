# iSLAT
**Current version: v4.04.02**\
*If you are running a previous version, make sure to update to the latest one (see below for instructions)!*\
**Please cite:** [Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract) (paper) and [Johnson et al. 2024](https://doi.org/10.5281/zenodo.12167854) (code)

[![DOI](https://zenodo.org/badge/731677457.svg)](https://zenodo.org/doi/10.5281/zenodo.12167853)

*If you are running iSLAT on an Apple silicon processor, please read "Caveats" below*

**iSLAT** (the **i**nteractive **S**pectral-**L**ine **A**nalysis 
**T**ool) is a python package that provides an interactive interface
for the visualization, exploration, and analysis of molecular spectra.
Synthetic spectra are made using a simple slab model written by Simon 
Bruderer and originally described in [Banzatti et al. 2012](https://ui.adsabs.harvard.edu/abs/2012ApJ...745...90B/abstract);
the code uses molecular data from [HITRAN](https://hitran.org/).
iSLAT has been developed and currently tested on spectra at infrared wavelengths 
as observed at different resolving powers (R = 700-90,000) with: 
JWST-MIRI, Spitzer-IRS, VLT-CRIRES, IRTF-ISHELL. Examples of these spectra are
included for users to practice with the tool functionalities across 
a range of resolving powers. iSLAT has been built as a flexible
tool that should work with one-dimensional molecular spectra observed 
with other instruments too, provided some requirements are met (see below).
iSLAT is presented and described in [Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract). A screenshot of iSLAT's GUI is provided here:

![iSLAT's GUI v3.10](./iSLAT_v403.jpg)

*We built iSLAT to make the analysis of infrared molecular spectra
an accessible, enjoyable, and rewarding experience for all, including
students and junior researchers who are starting in the field today. We 
believe iSLAT will be useful both to students and to professional researchers,
and we will continue to support it and expand it with care for all users.
If you find a bug, please be patient and let us know; we want to fix
it as much as you do! And if you have ideas to improve the tool, 
we'd love to hear.*

Questions? Feedback? Contributions? Submit an issue, a pull request,
or email us at spexodisks@gmail.com

## Installation and updates

### Download and run the latest version from GitHub:

Make sure you have [git](https://github.com/git-guides/install-git) and [pip](https://pip.pypa.io/en/stable/installation/) installed, 
then run these terminal commands sequentially from within the
folder where you wish your local copy of iSLAT to be:

    git clone https://github.com/spexod/iSLAT
    cd iSLAT
    pip install -r requirements.txt

To launch iSLAT, simply type:

    cd iSLAT
    python iSLAT.py

### Update to the latest version from GitHub:

Remember to update the repository from GitHub from time to time (check at the top of the page to see if you're running an older version);
from your local iSLAT folder, type on terminal:

    git pull https://github.com/spexod/iSLAT

## Input and outputs
iSLAT requires a continuum-subtracted spectrum as input data 
file in csv format, with a wavelength array in μm (assumed to be in a
column called "wave") and flux array in Jy (assumed to be called "flux"). 
The step of estimating and subtracting the continuum can be done e.g. 
with this tool: [ctool](https://github.com/pontoppi/ctool).
iSLAT's outputs are txt or csv files that save model parameters, spectra,
or line lists as defined by users (see more below). These output files 
are stored in the folders /SAVES, /LINESAVES, and /MODELS. The folder
/LINELISTS instead includes some curated line lists provided to users 
(see [Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract)).

## HITRAN data
At first launch by the user, iSLAT will download from [HITRAN](https://hitran.org/) the data
for a default list of molecules: H<sub>2</sub>, H<sub>2</sub>O, CO, 
CO<sub>2</sub>, CH<sub>4</sub>, HCN, NH<sub>3</sub>, OH, 
C<sub>2</sub>H<sub>2</sub>, C<sub>2</sub>H<sub>4</sub>, 
C<sub>2</sub>H<sub>6</sub>, C<sub>4</sub>H<sub>2</sub>, 
HCN, HC<sub>3</sub>N and some of their isotopologues. 
These datafiles are stored into the folder "HITRANdata" and are available to 
load and use in iSLAT (see below). Any additional molecule or isotopologue can be downloaded from HITRAN using the "HITRAN query" function in iSLAT. 
The current HITRAN data release as of 2024 is described in 
[Gordon et al. 2022](https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract).

## Parameters definitions and units
(Definitions and instructions are also available by hovering over GUI elements.)

**General note for all parameters in the GUI:** *every time you update any 
value, hit "return/enter" on your keyboard to apply the new value to the model.
This has to be done for each parameter individually, e.g. when submitting 
a new temperature for a given molecule, or updating the distance for a given 
target. A confirmation will appear in the message box in iSLAT.*

- **Molecular model parameters**: temperature is in K, radius in au (this is the 
radius of the equivalent emitting area, not the orbital radius of the 
emission), column density in cm<sup>-2</sup>. You can control which molecules are
ready for use in the GUI with the buttons: "Default Molecules" and "Add Molecule"
(see more below on these functions), and the "Del." button to remove them. You can
also select each molecule color by using the "Color" button.
- Other parameters: 
  - **Plot start/range** (just for the current plot) and 
  **min/max wavelength** (the spectral range
  for computing the slab models) are in μm
  - **Distance** is in pc
  - **Stellar RV** is Heliocentric and in km/s, and will shift the observed 
  spectrum (not the model)
  - line **FWHM** is in km/s, and is used for convolution of the model to
  match the observed line widths (which can be either set by the instrument
  resolving power, or by the gas kinematics if the resolving power is high 
  enough to resolve lines)
  - line **broadening** is in km/s and it is the FWHM of the intrinsic line
  broadening due to thermal motion or turbulence
  - **line separation** is in μm, and is used to identify isolated lines
  in the model for the molecule selected in the drop-down menu

## Quick reference for main functions
(Definitions and instructions are also available by hovering over GUI elements.)

### General functions (top of the GUI)
- **HITRAN query**: opens up a window where you can select and dowload any
molecules or isotopologues available in HITRAN; you need to run this first
if you wish to use any molecule that is not part of the default list above.
- **Default molecules**: load the default list of molecules for the GUI; this
list includes: H<sub>2</sub>O, OH, HCN, CO<sub>2</sub>, C<sub>2</sub>H<sub>2</sub>, CO.
- **Add molecule**: loads a molecule into the list of available molecules
at the top left of the GUI; the new molecule must already be downloaded
from HITRAN and stored in a .par file in the folder "HITRANdata"; the user 
can also load multiple times the same molecule, as long as different labels 
are assigned (for instance to simultaneously plot different temperature
components of a same molecule).
- **Save parameters**: save in an output file the current model parameters 
(T, R, N) for each molecule, plus the distance, RV, FWHM, and broadening values
set by users for any input spectrum; the output file will have the same name 
as the input observed spectrum plus "-molsave.csv" and will be stored in
the folder iSLAT/SAVES.
- **Load parameters**: loads previously saved model parameters from output
file created with "Save Parameters" from the folder iSLAT/SAVES; it will
use the input spectrum file name to identify the correct save file, so if
you change the input spectrum name make sure you update the save filename too.
- **Export models**: export specific or all model spectra in an output 
csv file in the folder iSLAT/MODELS.

### Spectral analysis functions
- **Selecting a line**: by dragging a region in the top spectrum plot, the
spectral region gets visualized in the bottom left plot with the 
addition of the individual transitions that dominate the emission;
the strongest of these transitions is highlighted in orange and its
properties are reported in the text box at the bottom left of the GUI.
Starting with version 4.04, other lines can be selected interactively in this
plot and the population diagram by clicking on them; this new feature allows
users to explore and save different lines within the same selected range.
- **Save line**: save the currently selected line into an output 
csv file that will include all its line parameters; this function
needs the definition of an output file in the box "Output Line Measurements"
that will be saved in the LINESAVES folder.
- **Fit line**: fit selected line with a Gaussian function using [LMFIT](https://lmfit.github.io/lmfit-py/index.html);
full fit results are reported on terminal, and a selection in
the GUI text box.
- **Show saved lines**: marks in the plot all lines saved into the 
file selected in "Input Line List" (this list can be one of those
provided with iSLAT, or one you make using the "Save Line" function).
- **Fit saved lines**: this is the same as "Fit line" but it fits in
one shot all lines defined in the "Input Line List" and will save
their measurements in the LINESAVES folder in the file defined in the
"Output Line Measurements" box. The output will also include rotation
diagram values for each line, which you can use to make a rotation diagram
like the one in the GUI, but with measured lines instead of a model.
- **Show atomic lines**: marks and labels atomic lines from the list 
provided in the folder LINELISTS
- **Find single lines**: identifies and marks isolated lines using
the value in "Line separ." as their spectral separation.
- **Molecule drop-down menu**: select which molecule is considered for
the zoomed-in plot, the rotation diagram, the "Find single
lines" function, and the "Save line" function.

## Data examples
iSLAT's release includes some continuum-subtracted spectra of 
protoplanetary disks that users can use to get familiar with iSLAT 
and its applications across a range of resolving powers:
- one M-band spectrum from iSHELL with R~60,000 (FZ Tau) from [Banzatti et al. 2023a](https://ui.adsabs.harvard.edu/abs/2023AJ....165...72B/abstract)
- two spectra from MIRI with R~2000-3000 (CI Tau and FZ Tau) from [Banzatti et al. 2023b](https://ui.adsabs.harvard.edu/abs/2023ApJ...957L..22B/abstract)
and [Pontoppidan et al. 2024](https://ui.adsabs.harvard.edu/abs/2023arXiv231117020P/abstract), respectively
- one spectrum from Spitzer-IRS with R~700 (CI Tau) from [Banzatti et al. 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...903..124B/abstract)
, reduced as described in [Pontoppidan et al. 2010](https://ui.adsabs.harvard.edu/abs/2010ApJ...720..887P/abstract)

The original spectra (not continuum-subtracted) are or will be available on
[spexodisks.com](www.spexodisks.com). Users who wish to use any of these spectra 
in their research work should use the original data and cite the
original publication papers.

## Caveats and known issues
The simple slab model used in iSLAT includes line opacity and saturation
with a curve-of-growth method, but does not account for mutual
saturation of lines that overlap within the thermal/turbulent line
broadening (see details in the appendix of [Banzatti et al. 2012](https://ui.adsabs.harvard.edu/abs/2012ApJ...745...90B/abstract)).
This implies that some emission lines in iSLAT will look stronger 
than they should be, specifically where there are clusters of transitions
that are partially or fully optically thick. Users that wish to model
those lines correctly can use other available slab model codes: 
[spectools-ir](https://github.com/csalyk/spectools_ir/) or [iris](https://github.com/munozcar/IRIS).

If you use iSLAT on an Apple-silicon Mac (M1, M2, etc.), there is a known issue with 
tkinter where the GUI often seems to freeze and does not take a new 
input or click; the solution is to click and drag very slightly on any
element you're working on (whether you're clicking a button or 
updating a parameter value), and the GUI will take your input. 
Hopefully this will be fixed in a new
release of tkinter (this is not an issue of iSLAT).

## Acknowledgements
It is hard to remember to acknowledge and cite all the tools 
we use in research, and often it is impossible to offer more than 
personal gratitude to the many people who have contributed to the
tools and instruments we use. Science is a community endeavor that 
always builds "on the shoulders of giants", but these giants are
by now a multitude and are most often unknown. They simply did
good work and shared it, and left a heritage that became useful 
to others.

iSLAT's story is similar: it is the outcome of experience, passion,
and complementary contributions of multiple people spanning more 
than 14 years, even if we focus on just the four main authors.
We offer our gratitude for all the tools we have used to build iSLAT. 
If you find iSLAT useful for your research, and if you can remember
doing it, please add its citation in your publications and oral/poster 
presentations. It's just a way to say "thank you". 
The main reference to cite is: 
[Jellison et al. 2024](https://ui.adsabs.harvard.edu/abs/2024arXiv240204060J/abstract)
