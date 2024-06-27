iSLAT_version = 'v4.03.04'
print(' ')
print('Loading iSLAT '+ iSLAT_version +': Please Wait ...')

# Import necessary modules
import numpy as np
import pandas as pd
import warnings
import matplotlib
#matplotlib.use('Agg')
matplotlib.use("TKAgg")
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, SpanSelector, TextBox, CheckButtons
from matplotlib.artist import Artist
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import sys
import os
from astropy.io import ascii, fits
from astropy.table import vstack, Table
from astropy import stats
import lmfit
from lmfit.models import GaussianModel
from ir_model import *
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk  # For ttk.Style
from tkinter import colorchooser
import inspect
#from PyQt5.QtWidgets import QApplication, QMainWindow
from datetime import datetime as dt
import csv
import time
import threading
from astroquery import hitran
import numpy
from astropy import units as un
from scipy import constants as con
import datetime
import certifi
import ssl
import urllib
context = ssl.create_default_context(cafile=certifi.where())

from COMPONENTS.chart_window import MoleculeSelector
from COMPONENTS.Hitran_data import get_Hitran_data
from COMPONENTS.partition_function_writer import write_partition_function
from COMPONENTS.line_data_writer import write_line_data


# create HITRAN folder, only needed for first start
HITRAN_folder = "HITRANdata"
os.makedirs(HITRAN_folder, exist_ok=True)


if __name__ == "__main__":

    mols = ["H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N"]
    basem = ["H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N"]
    isot = [1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1]

    min_wave = 0.3  # micron
    max_wave = 1000  # micron

    min_vu = 1 / (min_wave / 1E6) / 100.
    max_vu = 1 / (max_wave / 1E6) / 100.

    print(' ')
    print ('Checking for HITRAN files: ...')

    for mol, bm, iso in zip(mols, basem, isot):
        save_folder = 'HITRANdata'
        file_path = os.path.join(save_folder, "data_Hitran_2020_{:}.par".format(mol))

        if os.path.exists(file_path):
            print("File already exists for mol: {:}. Skipping.".format(mol))
            continue

        print("Downloading data for mol: {:}".format(mol))
        Htbl, qdata, M, G = get_Hitran_data(bm, iso, min_vu, max_vu)

        with open(file_path, 'w') as fh:
            fh.write("# HITRAN 2020 {:}; id:{:}; iso:{:};gid:{:}\n".format(mol, M, iso, G))
            fh.write("# Downloaded from the Hitran website\n")
            fh.write("# {:s}\n".format(str(datetime.date.today())))
            fh = write_partition_function(fh, qdata)
            fh = write_line_data(fh, Htbl)

        print("Data for Mol: {:} downloaded and saved.".format(mol))


# Define the default molecules and their file path; the folder must be in the same path as iSLAT
molecules_data = [
    ("H2O", "HITRANdata/data_Hitran_2020_H2O.par", "H$_2$O"),
    ("OH", "HITRANdata/data_Hitran_2020_OH.par", "OH"),
    ("HCN", "HITRANdata/data_Hitran_2020_HCN.par", "HCN"),
    ("C2H2", "HITRANdata/data_Hitran_2020_C2H2.par", "C$_2$H$_2$"),
    ("CO2", "HITRANdata/data_Hitran_2020_CO2.par", "CO$_2$"),
    ("CO", "HITRANdata/data_Hitran_2020_CO.par", "CO")
    # Add more molecules here if needed
]

default_data = [
    ("H2O", "HITRANdata/data_Hitran_2020_H2O.par", "H$_2$O"),
    ("OH", "HITRANdata/data_Hitran_2020_OH.par", "OH"),
    ("HCN", "HITRANdata/data_Hitran_2020_HCN.par", "HCN"),
    ("C2H2", "HITRANdata/data_Hitran_2020_C2H2.par", "C$_2$H$_2$"),
    ("CO2", "HITRANdata/data_Hitran_2020_CO2.par", "CO$_2$"),
    ("CO", "HITRANdata/data_Hitran_2020_CO.par", "CO")
] 



molecules_data_default = molecules_data.copy()

deleted_molecules = []

# Create necessary folders, if it doesn't exist (typically at first launch of iSLAT)
save_folder = "SAVES"
os.makedirs(save_folder, exist_ok=True)
output_dir = "MODELS"
os.makedirs(output_dir, exist_ok=True)
linesave_folder = "LINESAVES"
os.makedirs(linesave_folder, exist_ok=True)


# read more molecules if saved by the user in a previous iSLAT session
def read_from_csv():
    global file_name
    filename = os.path.join(save_folder, f"{file_name}-molsave.csv")
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row
                return [tuple(row[:3]) for row in reader] 
        except FileNotFoundError:
            pass
    return molecules_data

def read_default_csv():
    global file_name
    filename = os.path.join(save_folder, f"default.csv")
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row
                return [tuple(row[:3]) for row in reader] 
        except FileNotFoundError:
            pass
    return molecules_data

# read more molecules if saved by the user in a previous iSLAT session
def read_from_user_csv():
    global file_name
    filename = os.path.join(save_folder, f"molecules_list.csv")
    
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row
                return [tuple(row[:3]) for row in reader] 
        except FileNotFoundError:
            pass
    return molecules_data




# Set default initial parameters for a new molecule
default_initial_params = {
    "scale_exponent": 17,
    "scale_number": 1,
    "t_kin": 600,
    "radius_init": 0.5
}

# Define the initial parameters for default molecules
initial_parameters = {
    "H2O": {
        "scale_exponent": 18,
        "scale_number": 1,
        "t_kin": 850,
        "radius_init": 0.5
    },
    "OH": {
        "scale_exponent": 16,
        "scale_number": 1,
        "t_kin": 2000,
        "radius_init": 0.3
    },
    "HCN": {
        "scale_exponent": 16,
        "scale_number": 1,
        "t_kin": 850,
        "radius_init": 0.5
    },
    "C2H2": {
        "scale_exponent": 17,
        "scale_number": 1,
        "t_kin": 600,
        "radius_init": 0.1
    },
    "CO2": {
        "scale_exponent": 17,
        "scale_number": 1,
        "t_kin": 300,
        "radius_init": 0.5
    },
    "CO": {
        "scale_exponent": 18,
        "scale_number": 1,
        "t_kin": 1200,
        "radius_init": 0.4
    }
}

    

# Set-up default input parameters for model generation
min_lamb = 4.5
max_lamb = 28.
dist = 160.0
star_rv = 0.0
fwhm = 130. # FWHM of the observed lines or instrument
pix_per_fwhm = 10 # number of pixels per fwhm element

intrinsic_line_width = 1.0
cc = 2.99792458e5  # speed of light in km/s
model_line_width = cc / fwhm
model_pixel_res = (np.mean([min_lamb, max_lamb]) / cc * fwhm) / pix_per_fwhm

# Constants used in generating the rotation diagram
au = 1.496e11 # 1AU in m
pc = 3.08567758128e18 # From parsec to cm
ccum = 2.99792458e14 # speed of light in um/s
hh = 6.62606896e-27 # erg s

# Dictionary to store the initial values for each chemical
initial_values = {}

# define other defaults needed below
spanmol = "h2o"

line_threshold = 0.03   # this is the percent value (where 0.01 = 1%) of the strongest line in the plot;
                        # lines below this this limit are ignored in the plot and in the single line selection



"""Creating all the functions used for the tool"""

# """
# find_nearest() is a simple function that finds a value in a given array that matches closest to a value input.
# The function then returns the index of the value in the array.
# """
# def find_nearest(array,value):
#     idx = (np.abs(array-value)).argmin()
#     return idx

    
"""
Save() is connected to the "Save Line" button of the tool.
This function appends information of the the strongest line (as determined by intensity) in the spanned area graph to a csv file. 
The name of the csv file is set with the "svd_line_file" variable in the second code block above. 
For the parameters of the line that is saved, refer to the "line2save" variable in onselect().
When starting the tool up, the "headers" variable is set to False. After apending a line to the csv for the first time, the "headers" variable is changed to False.
"""
def Save():
    global line2save
    global headers
    global selectedline
    global linesavepath

    # This section is necessary for refreshing the text feed area to the left of the tool
    data_field.delete('1.0', "end")

    try:
        linesavepath
    except NameError:
        data_field.delete ('1.0', "end")
        data_field.insert ('1.0', 'Line save file is not defined!')
    else:
        if selectedline == True:  # "selectedline" variable is determined by whether or not an area was selected in the top graph or not

            line2save.to_csv (linesavepath, mode='a', index=False, header=False)

            data_field.insert ('1.0', 'Line Saved!')
            fig.canvas.draw_idle ()
        else:
            data_field.insert ('1.0', 'No Line Selected!')
            fig.canvas.draw_idle ()
            return

    canvas.draw()


"""
fit_onselect() is connected to the "Fit Line" button of the tool.
This function fits the line selected in the top graph using LMFIT"""
def fit_onselect():
    global selectedline

    print (' ')
    print ('Fitting line with LMFIT ...')

    if selectedline == True:  # "selectedline" variable is determined by whether or not an area was selected in the top graph or not

        # using one less pixel on each side here, because of how data_region_x is defined: to include 1 more pixel on each side
        gauss_fit, gauss_fwhm, gauss_area, x_fit = fit_line(data_region_x[1],data_region_x[-2])

        dely = gauss_fit.eval_uncertainty (sigma=3)
        ax2.fill_between (x_fit, gauss_fit.best_fit - dely, gauss_fit.best_fit + dely, color="#ABABAB",
                          label=r'3-$\sigma$ uncertainty band')
        ax2.plot (x_fit, gauss_fit.best_fit, label='Gauss. fit', color='lime', ls='--')

        data_field.insert(tk.END, ('\n ' + '\nGaussian fit results: ' + '\nCentroid (μm) = ' + str(
            np.round(gauss_fit.params['center'].value, decimals=5)) + ' +/- ' + str(
            np.round(gauss_fit.params['center'].stderr, decimals=5)) + '\nFWHM (km/s) = ' + str(
            np.round(gauss_fwhm[0], decimals=1)) + ' +/- ' + str(
            np.round(gauss_fwhm[1], decimals=1)) + '\nArea (erg/s/cm2) = ' + f'{gauss_area[0]:.{3}e}' +
            ' +/- ' + f'{gauss_area[1]:.{3}e}'))

        fig.canvas.draw_idle()
    else:
        data_field.delete ('1.0', "end")
        data_field.insert('1.0', 'No Line Selected!')
        fig.canvas.draw_idle()
        return
    canvas.draw()

"""
fit_line() uses LMFIT to fit a line and provide best-fit parameters.
"""
def fit_line(xmin, xmax):

    fit_range = np.where (np.logical_and (wave_data >= xmin, wave_data <= xmax))  # define spectral range for the fit
    x_fit = wave_data[fit_range[::-1]]  # reverse the wavelength array to use it in the fit
    model = GaussianModel ()  # use gaussian model from LMFIT
    params = model.guess (flux_data[fit_range[::-1]], x=x_fit)  # get initial guess for parameters
    # the fit in the next line uses the data error array as weights, as described in the LMFIT docs and this discussion: https://groups.google.com/g/lmfit-py/c/SmO19HbXGcc/m/xa3tsPJcBgAJ
    gauss_fit = model.fit (flux_data[fit_range[::-1]], params, x=x_fit, weights=1/err_data[fit_range[::-1]], nan_policy='omit')  # make the fit, ignoring nans
    print (gauss_fit.fit_report ())  # print full fit report

    gauss_fwhm = gauss_fit.params['fwhm'].value / gauss_fit.params['center'].value * cc # get FWHM in km/s
    # these if statements are made to avoid problems when the fit does not converge and stderr are returned as NoneType
    if gauss_fit.params['fwhm'].stderr is not None:
        gauss_fwhm_err = gauss_fit.params['fwhm'].stderr / gauss_fit.params['center'].value * cc # get FWHM error
    else:
        gauss_fwhm_err = np.nan

    sigma_freq = ccum / (gauss_fit.params['center'].value ** 2) * gauss_fit.params[
        'sigma'].value  # sigma from wavelength to frequency
    if gauss_fit.params['sigma'].stderr is not None:
        sigma_freq_err = ccum / (gauss_fit.params['center'].value ** 2) * gauss_fit.params['sigma'].stderr # error on sigma
    else:
        sigma_freq_err = np.nan

    gauss_area = gauss_fit.params['height'].value * sigma_freq * np.sqrt (2 * np.pi) * (1.e-23)  # to get line flux in erg/s/cm2
    if gauss_fit.params['height'].stderr is not None:
        gauss_area_err = (gauss_area * np.sqrt(
                        (gauss_fit.params['height'].stderr/gauss_fit.params['height'].value)**2 +
                        (sigma_freq_err/sigma_freq)**2) ) # get area error
    else:
        gauss_area_err = np.nan

    return gauss_fit, [gauss_fwhm,gauss_fwhm_err], [gauss_area,gauss_area_err], x_fit


"""
update() is main function of this tool. It is called any time and of the sliders or text imput are changed. 
This function is where the models are rebuilt with new parameters and the graphs are recreated.
"""
def update(*val):
    global skip # See reference in reset()
                # If skip is False, then update() does not run. This cuts down needless processing
    if skip == True:
        return

    ax1.clear() # Reference: matplotlib.widgets
    ax2.clear() # These functions clear the three plots of the tool
    ax3.clear() # They are rebuilt in the update() function
        
    global xp1, xp2, span, model_line_select, data_line_select, fig_height, fig_bottom_height, n_mol, selectedline, spanmol, sum_line, int_pars, molecules_data, total_fluxes, dist, fwhm, max_lamd, min_lamd
    
    
    span.set_visible(False) # Clears the blue area created by the span selector (range selector in the top graph of the tool)
    selectedline = False # See reference in save()

    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    # Make empty lines for the second plot
    data_line_select, = ax2.plot([],[],color=foreground,linewidth=2)
    data_line, = ax1.plot([], [], color=foreground, linewidth=1)
    data_line.set_label('Data')

    ax1.set_prop_cycle (color=['dodgerblue', 'darkorange', 'orangered', 'limegreen', 'darkviolet', 'magenta',
                               'hotpink', 'cyan', 'gold', 'turquoise', 'chocolate', 'royalblue', 'sienna', 'lime', 'blue'])

    # Make empty lines for the top graph for each molecule
    for mol_name, mol_filepath, mol_label in molecules_data:
        molecule_name_lower = mol_name.lower()
        
        if molecule_name_lower in deleted_molecules:
            continue
        
        exec(f"{molecule_name_lower}_line, = ax1.plot([], [], alpha=1, linewidth=2, ls='--')", globals())
        exec(f"{molecule_name_lower}_line.set_label('{mol_label}')", globals())
        
        try:
            color_var = globals().get(f"{molecule_name_lower}_color")
            
            if color_var:
                # Set the color of the molecule line
                exec(f"{molecule_name_lower}_line.set_color('{color_var}')", globals())
                exec(f"global {molecule_name_lower}_line_color; {molecule_name_lower}_line_color = '{color_var}'")
        except NameError:
            print('not changed!')
            return
            
    #sum_line, = ax1.plot([], [], color='gray', linewidth=1)
    #sum_line.set_label('Sum')
    ax1.legend()

    # h2o, oh, hcn, and c2h2 are variables that are set to True or false depending if the molecule is currently selected in the tool
    # If True, then that molecule's model is rebuilt with any new conditions (as set by the sliders or text input) that may have called the update() function
    # See h2o_select()
    for mol_name, mol_filepath, mol_label in molecules_data:
        molecule_name_lower = mol_name.lower()

        # Intensity calculation
        exec(f"{molecule_name_lower}_intensity.calc_intensity(t_{molecule_name_lower}, n_mol_{molecule_name_lower}, dv=intrinsic_line_width)", globals())

        # Spectrum creation
        exec(f"{molecule_name_lower}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

        # Adding intensity to the spectrum
        exec(f"{molecule_name_lower}_spectrum.add_intensity({molecule_name_lower}_intensity, {molecule_name_lower}_radius ** 2 * np.pi)", globals())

        # Fluxes and lambdas
        exec(f"fluxes_{molecule_name_lower} = {molecule_name_lower}_spectrum.flux_jy; lambdas_{molecule_name_lower} = {molecule_name_lower}_spectrum.lamgrid", globals())
        
        linevar = eval(f"{molecule_name_lower}_line")
        linecolor = linevar.get_color()
        exec(f"global {molecule_name_lower}_line_color; {mol_name.lower()}_line_color = '{linecolor}'")
                       

    
    # Redefining plot parameters that were deleted by the clear functions at the begining of this function
    ax1.set_ylabel('Flux density (Jy)')
    ax1.set_xlabel('Wavelength (μm)')
    ax2.set_ylabel('Flux density (Jy)')
    ax2.set_xlabel('Wavelength (μm)')
    ax3.set_label('Population Diagram')
    plt.rcParams['font.size'] = 10

    # xp1 and xp2 define the range of spectrum shown in the top graph
    # See Prev(), Next(), and on_xlims_change()
    # Originally defined in code block below
    ax1.set_xlim(xmin=xp1, xmax=xp2)

    #Scaling the y-axis based on tallest peak of data in the range of xp1 and xp2
    range_flux_cnts = input_spectrum_data[(input_spectrum_data['wave'] > xp1) & (input_spectrum_data['wave'] < xp2)]
    range_flux_cnts.index = range(len(range_flux_cnts.index))
    fig_height = np.nanmax(range_flux_cnts.flux)
    fig_bottom_height = np.nanmin(range_flux_cnts.flux)
    ax1.set_ylim(ymin=fig_bottom_height, ymax=fig_height+(fig_height/8))

    # Initialize total fluxes list
    total_fluxes = []
    # Calculate total fluxes based on visibility conditions
    for i in range(len(lambdas_h2o)):
        flux_sum = 0
        for mol_name, mol_filepath, mol_label in molecules_data:
            

            mol_name_lower = mol_name.lower()
            visibility_flag = f"{mol_name_lower}_vis"
            fluxes_molecule = f"fluxes_{mol_name_lower}"

            if visibility_flag in globals() and globals()[visibility_flag]:
                flux_sum += globals()[fluxes_molecule][i]

        total_fluxes.append(flux_sum)

    if mode == True:
            ax1.fill_between(lambdas_h2o, total_fluxes, color='gray', alpha=1)
    
    if mode == False:
            ax1.fill_between(lambdas_h2o, total_fluxes, color='lightgray', alpha=1)

    # populating the empty lines created earlier in the function
    for mol_name, mol_filepath, mol_label in molecules_data:
        molecule_name_lower = mol_name.lower()

        # Dynamically set the data for each molecule's line using exec and globals()
        exec(f"{molecule_name_lower}_line.set_data(lambdas_{molecule_name_lower}, fluxes_{molecule_name_lower})", globals())
    data_line.set_data(wave_data, flux_data)
    
    
    #This is the opacity value that will be used for all shades (if you want to change the opacity just change this value)
    alpha_set = .2

    
    for mol_name, mol_filepath, mol_label in molecules_data:
        molecule_name_lower = mol_name.lower()
        vis_status_var = globals()[f"{molecule_name_lower}_vis"]
        line_var = globals()[f"{molecule_name_lower}_line"]
        #lambdas_var = globals()[f"lambdas_{molecule_name_lower}"]
        #fluxes_var = globals()[f"fluxes_{molecule_name_lower}"]

        if vis_status_var:
            line_var.set_visible(True)
        else:
            line_var.set_visible(False)
            label = line_var.get_label()
            if label:
                ax1.legend().get_legend_handler_map().pop(label, None)
                line_var.set_label("_nolegend_")
            
    ax1.legend()
        
    # Creating an array that contains the data of every line in the water model and reseting the index of this array
    # Reference: "ir_model" > "intensity.py"
    # int_pars is used to call up information on water lines like when using the spanning feature of the tool
    int_pars = eval(f"{spanmol}_intensity.get_table")
    int_pars.index = range(len(int_pars.index))

    # Storing the callback for on_xlims_change()
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    
    # Storing the callback for the span selector
    span = SpanSelector(
        ax1,
        onselect,
        "horizontal",
        useblit=False,
        props=dict(alpha=0.5, facecolor="lime"),
        interactive=True,
        drag_from_anywhere=True
    )

    # Rebuilding the population diagram graph
    # See pop_diagram()
    pop_diagram()
    plt.draw(), canvas.draw()
    
    

"""
onselect() is the function for the span selector functionality in the top graph of the tool.
Here, the user selects a range in the top graph and the range of the data and water model spectrum are rebuilt in the zoom graph (bottom left graph) 
along with the water lines in that range.
The strongest line is determined and its info is printed in the text feed on the left. 
The lines are also highlighted in the population diagram graph in this function.
"""

def onselect(xmin, xmax):
    
    global wave_data, flux_data, line2save, selectedline, spanmol, model_indmin, model_indmax, data_region_x, model_line_select



    xdif = xmax - xmin
    if xdif > 0:
        # Clearing the bottom two graphs
        # Reference: matplotlib.pyplot
        ax3.clear()
        ax2.clear()

        int_pars = eval(f"{spanmol}_intensity.get_table")
        int_pars.index = range(len(int_pars.index))

        #print(spanmol)
        # Clearing the text feed box.
        data_field.delete('1.0', "end")

        # Repopulating the population diagram graph with all the lines of the water molecule (gray dots)
        pop_diagram()

        # Resetting the labels of graphs after they were deleted by the clear function above
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Flux density (Jy)')
        
        linevar = eval(f"{spanmol}_line")
        linecolor = linevar.get_color()
        #'royalblue'
        # Make empty lines for the zoom plot
        model_line_select, = ax2.plot([], [], color=linecolor, linewidth=3, ls='--')
        data_line_select, = ax2.plot([],[],color=foreground,linewidth=1)

        # Getting all the water lines for the selected range
        int_pars_line = int_pars[(int_pars['lam']>xmin) & (int_pars['lam']<xmax)]
        int_pars_line.index = range(len(int_pars_line.index))

        # Parsing out the columns of the lines in int_pars_line to be used later
        lamb_cnts = int_pars_line['lam']
        intensities = int_pars_line['intens']
        einstein = int_pars_line['a_stein']
        e_up = int_pars_line['e_up']
        up_lev = int_pars_line['lev_up']
        low_lev = int_pars_line['lev_low']
        g_up = int_pars_line['g_up']
        tau = int_pars_line['tau']

        # Creating zero variables to be used later
        max_value = intensities[0]
        max_index = 0

        # Checking to see if there are any lines in the range selected
        # If there aren't then this function does not continue
        # If there are, then the strongest intensity of the lines in the range selected is identified along with its index
        if len(intensities) >= 1:
            selectedline = True
            for i in range(len(intensities)):
                if intensities[i] > max_value:
                    max_value = intensities[i]
                    max_index = i
        else:
            return

        # Defining the other parameters of the line with the strongest intensity as found previously
        max_lamb_cnts = lamb_cnts[max_index]
        max_up_lev = up_lev[max_index]
        max_low_lev = low_lev[max_index]
        max_intensity = intensities[max_index]
        max_einstein = einstein[max_index]
        max_e_up = e_up[max_index]
        max_g_up = g_up[max_index]
        max_tau = tau[max_index]

        # Finding the index of the minimum and maximimum flux for both the data and model to be used in scaling the zoom graph (section below)
        model_indmin, model_indmax = np.searchsorted(lambdas_h2o, (xmin, xmax))
        data_indmin, data_indmax = np.searchsorted(wave_data, (xmin, xmax))
        # this below is to avoid taking too few pixels for the plot, useful in case of e.g. MIRI spectra
        data_indmin = data_indmin - 1
        data_indmax = data_indmax + 1
        model_indmax = min(len(lambdas_h2o) - 1, model_indmax)
        data_indmax = min(len(wave_data) - 1, data_indmax)

        # Dynamically set the x variable
        print (' ')
        print('Molecule selected: ')
        print(spanmol)
        model_region_x_str = f"lambdas_{spanmol}[model_indmin:model_indmax]"
        model_region_x = eval(model_region_x_str)

        # Scaling the zoom graph
        # First, it's determined if the max intensity of the model is bigger than that of the max intensity of the data or vice versa
        # Then, the max for the y-axis is determined by the max intensity of either the model or data, whichever is bigger
        # The minimum for the y-axis of the zoom graph is set to zero here

        #model_region_y = fluxes_h2o[model_indmin:model_indmax]
        # Dynamically set the variable
        model_region_y_str = f"fluxes_{spanmol}[model_indmin:model_indmax]"
        model_region_y = eval(model_region_y_str)
        data_region_x = wave_data[data_indmin:data_indmax]
        data_region_y = flux_data[data_indmin:data_indmax]
        max_data_y = np.nanmax(data_region_y)
        max_model_y = np.nanmax(model_region_y)
        if (max_model_y) >= (max_data_y):
            max_y = max_model_y
        else:
            max_y = max_data_y
        ax2.set_ylim(0, max_y)
        print (' ')
        print('Data range:')
        print(data_region_x[0],data_region_x[-1])
        #print(xmin, xmax)

        # Calling the flux function to calculate the flux for the data in the range selected
        # Also printing the flux in the notebook for easy copying
        # See flux_integral
        line_flux = flux_integral(wave_data, flux_data, xmin, xmax)

        data_field.delete('1.0', "end")
        data_field.insert ('1.0', (
                    'Strongest line:' + '\nUpper level = ' + str (max_up_lev) + '\nLower level = ' + str (
                max_low_lev) + '\nWavelength (μm) = ' + str (max_lamb_cnts) + '\nEinstein-A coeff. (1/s) = ' + str (
                max_einstein) + '\nUpper level energy (K) = ' + str (f'{max_e_up:.{0}f}') +'\nOpacity = '+ str(
                f'{max_tau:.{3}f}')+ '\nFlux in sel. range (erg/s/cm2) = ' + str (f'{line_flux:.{3}e}')))


        # Creating a pandas dataframe for all the info of the strongest line in the selected range
        # This dataframe is used in the Save() function to save the strongest line in a csv file
        line2save = {'species': [spanmol.upper()], 'lev_up': [max_up_lev], 'lev_low': [max_low_lev], 'lam': [max_lamb_cnts], 'tau': [max_tau],
                     'intens': [max_intensity], 'a_stein': [max_einstein], 'e_up': [max_e_up], 'g_up': [max_g_up],
                     'xmin': [f'{xmin:.{4}f}'], 'xmax': [f'{xmax:.{4}f}']
                     }
        line2save = pd.DataFrame(line2save)


        # This section prints vertical lines on the zoom graph at the wavelengths for each line in the model
        # The strongest line is colored differently than the other lines
        # The height of the lines represent the ratio of their intensities to the strongest line's intensity 
        # e.g. the strongest line is the tallest, a line that has 50% the int of the strongest line will be half as tall as that line
        if len(model_region_x) >= 1:
            k=0
            model_line_select.set_data(model_region_x, model_region_y), globals()
            data_line_select.set_data(data_region_x, data_region_y)
            ax2.set_xlim(model_region_x[0], model_region_x[-1])
            print(' ')
            print('Other strong lines in selected range (wavelength, upper and lower levels, E_up, A-coeff, opacity):')
            for j in range(len(lamb_cnts)):
                if j == max_index:
                    k = j
                if j != max_index:
                    lineheight = (intensities[j]/max_intensity)*max_y
                    if intensities[j] > max_intensity/50:
                        ax2.vlines(lamb_cnts[j], 0, lineheight, linestyles='dashed',color='green')
                        ax2.text(lamb_cnts[j], lineheight, (str(f'{e_up[j]:.{0}f}')+', '+str(f'{einstein[j]:.{3}f}')), color = 'green', fontsize = 'small')
                        print(str(f'{lamb_cnts[j]:.{5}f}'), up_lev[j], low_lev[j], str(f'{e_up[j]:.{0}f}'), einstein[j], str(f'{tau[j]:.{3}f}'))
                        area = eval(f"np.pi*({spanmol}_radius*au*1e2)**2") # In cm^2
                        Dist = dist*pc
                        beam_s = area/Dist**2
                        F = intensities[j]*beam_s
                        freq = ccum/lamb_cnts[j]
                        rd_yax = np.log(4*np.pi*F/(einstein[j]*hh*freq*g_up[j]))
                        ax3.scatter(e_up[j], rd_yax, s=30, color='green', edgecolors='black')
            lineheight = (intensities[k]/max_model_y)*max_model_y
            ax2.vlines(lamb_cnts[k], 0, lineheight, linestyles='dashed',color='orange')
            ax2.text(lamb_cnts[k], max_y, (str(f'{e_up[k]:.{0}f}')+', '+str(f'{einstein[k]:.{3}f}')), color = 'orange', fontsize = 'small')
            area = eval(f"np.pi*({spanmol}_radius*au*1e2)**2") # In cm^2
            Dist = dist*pc
            beam_s = area/Dist**2
            F = intensities[k]*beam_s
            freq = ccum/lamb_cnts[k]
            rd_yax = np.log(4*np.pi*F/(einstein[k]*hh*freq*g_up[k]))
            ax3.scatter(e_up[k], rd_yax, s=30, color='orange', edgecolors='black')
            fig.canvas.flush_events()
    else:
        pop_diagram()
        ax2.clear()


# """
# group() is a function used in the single_finder() function.
# This function groups model lines together based on a wavelength separation threshold (thr) set when the function is called.
# (a) is the array of lines being grouped
# """
# def group(a,thr):
#     x = np.sort(a)
#     diff = x[1:]-x[:-1]
#     gps = np.concatenate([[0],np.cumsum(diff>=thr)])
#     return [x[gps==i] for i in range(gps[-1]+1)]

"""
single_finder() is connected to the "Find Singles" button.
This function is a filter that finds molecular lines in the model that are isolated then prints vertical lines in the top graph where these lines are located
e.g. they are either a set distance away from other strong lines, or the intensity of the lines near the line are negligible.
"""
specsep = .01  # This is the default value for the separation to determine if line is single


def single_finder():
    update ()
    global fig_height
    global fig_bottom_height
    counter = 0
    specsep = float (specsep_entry.get ())

    # Resetting the text feed box
    data_field.delete ('1.0', "end")

    # Getting all the water lines in the range of xp1 and xp2
    int_pars_line = int_pars[(int_pars['lam'] > xp1) & (int_pars['lam'] < xp2)]
    int_pars_line.index = range (len (int_pars_line.index))

    # Parsing the wavelengths and intensities of the lines in int_pars_line
    lamb_cnts = int_pars_line['lam']
    intensities = int_pars_line['intens']

    # Determining an max threshold for lines we may want to consider
    # This threshold is based on the max line intensity found in the range of xp1 and xp2
    # This threshold will be used to filter out weak lines regardless of them being single
    max_intens = 0
    for i in range (len (intensities)):
        if intensities[i] > max_intens:
            max_intens = intensities[i]
    max_threshold = max_intens * line_threshold  # This will be used to filter out lines with intensities below a percentage of the max intensity

    # This is the main function. First, it will only consider lines with intensities above "max_threshold."
    # Of those lines, it will inspect all lines within "specsep" (user defined) below and above their wavelength.
    # If any lines within this range have an intensity above "threshold" (a percentage of the intensity for the line of interest),
    # then the line of interest is determined to be non-single. Otherwise, it's determined to be single.
    for j in int_pars_line.index:
        include = True  # Boolean for determining line is single or not.
        j_lam = lamb_cnts[int_pars_line.index[j]]  # Wavelength of line of interest
        sub_xmin = j_lam - specsep
        sub_xmax = j_lam + specsep
        j_intens = intensities[int_pars_line.index[j]]  # Intensity of line of interest
        loc_threshold = j_intens * 0.1  # Creating a threshold for determining locally if line of interest is single
        if j_intens >= max_threshold:  # Filter out weak lines
            chk_range = int_pars[(int_pars['lam'] > sub_xmin) & (int_pars['lam'] < sub_xmax)]
            chk_range.index = range (len (chk_range.index))
            range_intens = chk_range[
                'intens']  # Intensities of lines +/- "specsep" wavelength away from line of interest
            for k in chk_range.index:
                k_intens = range_intens[chk_range.index[k]]
                if k_intens >= loc_threshold:  # Filter determining if line of interest is not single
                    if k_intens != j_intens:  # Making sure we are not excluding line of interest by accidently considering its own intensity for the filter
                        include = False  # If both filters above are true, then the line of interest is not single
            if include == True:  # If both filters above are false for all lines in range of "specsep", then line of interest is single
                ax1.vlines (lamb_cnts[int_pars_line.index[j]], fig_bottom_height, fig_height, linestyles='dashed',
                            color='blue')
                counter = counter + 1

    # Storing the callback for on_xlims_change()
    ax1.callbacks.connect ('xlim_changed', on_xlims_change)

    # Print the number of isolated lines that the function found in the region of xp1 and xp2
    if counter == 0:
        data_field.insert ('1.0', 'No single lines found in the current wavelength range.')
    if counter > 0:
        data_field.insert ('1.0',
                           'There are ' + str (counter) + ' single lines found in the current wavelength range.')
    canvas.draw ()


"""
print_saved_lines() prints, as vertical dashed lines, on the top graph the locations of all lines saved to the current csv connected to the Save() function.
This csv can be changed in the user adjustable variables code block, but the change won't take into effect until the user regenerates the tool.
"""
def print_saved_lines():
    global linelistpath

    try:
        linelistpath

    except NameError:
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Input line list is not defined!')

    else:
        update()
        ax1.callbacks.connect('xlim_changed', on_xlims_change)

        svd_lns=pd.read_csv(linelistpath, sep=',')
        svd_lamb = np.array(svd_lns['lam'])
        if 'xmin' in svd_lns:
            x_min = np.array(svd_lns['xmin'])
            x_max = np.array(svd_lns['xmax'])
        for i in range(len(svd_lamb)):
            ax1.vlines(svd_lamb[i], -2, 10, linestyles='dashed',color='red')
            if 'xmin' in svd_lns:
                ax1.vlines (x_min[i], -2, 10, color='coral', alpha=0.5)
                ax1.vlines (x_max[i], -2, 10, color='coral', alpha=0.5)
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Saved lines retrieved from file.')
        canvas.draw()


"""
fit_saved_lines() will fit all saved lines in one click, and save them to output"""
def fit_saved_lines():

    try:
        linelistpath
    except NameError:
        data_field.delete ('1.0', "end")
        data_field.insert ('1.0', 'Input line list is not defined!')
    else:
        try:
            linesavepath
        except NameError:
            data_field.delete ('1.0', "end")
            data_field.insert ('1.0', 'Output file is not defined!')
        else:
            svd_lns = pd.read_csv(linelistpath, sep=',')

            x_min = np.array(svd_lns['xmin'])
            x_max = np.array(svd_lns['xmax'])
            restwl = np.array(svd_lns['lam'])
            # define new columns to be filled with values line by line in the loop below
            #svd_lns["Flux_data"], svd_lns["Flux_fit"], svd_lns["Flux_err"] = [np.empty_like(x_min, dtype=None),np.empty_like(x_min, dtype=None),np.empty_like(x_min, dtype=None)]
            #svd_lns["Centr_fit"], svd_lns["Centr_err"], svd_lns["Doppler"], svd_lns["FWHM_fit"], svd_lns["FWHM_err"] = [np.empty_like(x_min),np.empty_like(x_min),np.empty_like(x_min),np.empty_like(x_min),np.empty_like(x_min)]

            for i in range(len(x_min)):
                ax1.vlines(x_min[i], -2, 10, color='lime', alpha=0.5)
                ax1.vlines(x_max[i], -2, 10, color='lime', alpha=0.5)
                gauss_fit, gauss_fwhm, gauss_area, x_fit = fit_line(x_min[i],x_max[i])

                dely = gauss_fit.eval_uncertainty (sigma=3)
                ax1.fill_between (x_fit, gauss_fit.best_fit - dely, gauss_fit.best_fit + dely, color="#ABABAB",
                                  label=r'3-$\sigma$ uncertainty band')
                ax1.plot (x_fit, gauss_fit.best_fit, label='Gauss. fit', color='lime', ls='--')
                flux_nofit = flux_integral (wave_data, flux_data, x_min[i], x_max[i])

                sig_det_lim = 1
                # these reformatting below is for reducing the number of decimals and then get back to a float
                svd_lns.loc[i,"Flux_data"] = np.float64(f'{flux_nofit:.{3}e}')
                svd_lns.loc[i,"Flux_fit"] = np.float64(f'{gauss_area[0]:.{3}e}')
                svd_lns.loc[i,"Flux_err"] = np.float64(f'{gauss_area[1]:.{3}e}')
                # store fit results only if fit is good and line is detected; for now we're using a condition on line detection, as the goodness of fit is not very informative in MIRI spectra, it seems..
                if gauss_area[0] > sig_det_lim*gauss_area[1]:
                    svd_lns.loc[i,"FWHM_fit"] = np.round(gauss_fwhm[0], decimals=1)
                    svd_lns.loc[i,"FWHM_err"] = np.round(gauss_fwhm[1], decimals=1)
                    svd_lns.loc[i,"Centr_fit"] = np.round (gauss_fit.params['center'].value, decimals=5)
                    svd_lns.loc[i,"Centr_err"] = np.round (gauss_fit.params['center'].stderr, decimals=5)
                    svd_lns.loc[i,"Doppler"] = np.round ((gauss_fit.params['center'].value - restwl[i]) / restwl[i] * cc, decimals=1)

                else:
                    svd_lns.loc[i,"Flux_fit"] = svd_lns.loc[i,"Flux_data"] # safety measure, LMFIT gives much larger fluxes to undetected lines sometimes..
                    svd_lns.loc[i,"FWHM_fit"] = np.nan
                    svd_lns.loc[i,"FWHM_err"] = np.nan
                    svd_lns.loc[i,"Centr_fit"] = np.nan
                    svd_lns.loc[i,"Centr_err"] = np.nan
                    svd_lns.loc[i, "Doppler"] = np.nan

                svd_lns.loc[i,"Red-chisq"] = np.round (gauss_fit.redchi, decimals=2)

            # add rotation diagram values
            freq = ccum / svd_lns['lam']
            svd_lns['RD_y'] = np.round(np.log(4 * np.pi * svd_lns["Flux_fit"] / (svd_lns['a_stein'] * hh * freq * svd_lns['g_up'])), decimals=3)

            # save output file with measurements as csv file
            svd_lns.to_csv(linesavepath, header=True, index=False)

            data_field.delete ('1.0', "end")
            data_field.insert ('1.0', 'Input lines fitted and saved.')
            canvas.draw ()


def print_atomic_lines():
    update()
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    
    svd_lns = pd.read_csv("LINELISTS/Atomic_lines.csv", sep=',')
    svd_lamb = np.array(svd_lns['wave'])
    svd_species = svd_lns['species']
    svd_lineID = np.array(svd_lns['line'])

    for i in range(len(svd_lamb)):
        ax1.vlines(svd_lamb[i], -2, 2, linestyles='dashed', color='tomato')
        
        # Adjust the y-coordinate to place labels within the borders
        label_y = ax1.get_ylim()[1] - 0.21 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        
        # Adjust the x-coordinate to place labels just to the right of the line
        label_x = svd_lamb[i] + 0.006 * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
        
        ax1.text(label_x, label_y, svd_species[i]+' '+svd_lineID[i], fontsize=8, rotation=90, va='bottom', ha='left', color='tomato')
    
    data_field.insert('1.0', 'Atomic lines retrieved from file.')

    canvas.draw()




        


"""
on_xlims_change() saves the current xp1 and xp2 for use in other functions.
This Function is necessary to allow the user to use matplotlib's interactive graph scrolling feature without 
breaking the functionality of other features of this tool (e.g. Next() or Prev())
"""
def on_xlims_change(event_ax):
    global xp1
    global xp2
    xp1, xp2 = event_ax.get_xlim()

"""
pop_diagram() is the function for populating the population diagram with the lines of the water model 
in the entire range of the model
"""
def pop_diagram():
    ax3.clear()
    global spanmol
    ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
    ax3.set_xlabel(r'$E_{u}$ (K)')

    # Getting all the water lines in the range of min_lamb, max_lamb as set by the user in the adjustable variables code block
    int_pars = eval(f"{spanmol}_intensity.get_table")
    int_pars.index = range(len(int_pars.index))

    # Parsing the components of the lines in int_pars
    wl = int_pars['lam']
    intens_mod = int_pars['intens']
    Astein_mod = int_pars['a_stein']
    gu = int_pars['g_up']
    eu = int_pars['e_up']

    # Calculating the y-axis for the population diagram for each line in int_pars
    area = eval(f"np.pi*({spanmol}_radius*au*1e2)**2") # In cm^2    
    Dist = dist*pc
    beam_s = area/Dist**2
    F = intens_mod*beam_s
    freq = ccum/wl
    rd_yax = np.log(4*np.pi*F/(Astein_mod*hh*freq*gu))
    threshold = np.nanmax(F)/100

    ax3.set_ylim(np.nanmin(rd_yax[F > threshold]),np.nanmax(rd_yax)+0.5)
    ax3.set_xlim(np.nanmin(eu)-50,np.nanmax(eu[F > threshold]))

    # Populating the population diagram graph with the lines
    line6 = ax3.scatter(eu, rd_yax, s=0.5, color='#838B8B')
    #plt.show()

"""
submit_col() is connected to the text input box for adjusting the 
column density of the currently selected molecule
"""
def submit_col(event, text):
    
    #global text_box
    #global text_box_data

    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Submitting Density...')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 
  
    
    val = float(event)
    exec(f"n_mol_{text} = {val}", globals())
    
    # Intensity calculation
    exec(f"{text}_intensity.calc_intensity(t_{text}, n_mol_{text}, dv=intrinsic_line_width)", globals())

    # Spectrum creation
    exec(f"{text}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

    # Adding intensity to the spectrum
    exec(f"{text}_spectrum.add_intensity({text}_intensity, {text}_radius ** 2 * np.pi)", globals())

    # Fluxes and lambdas
    exec(f"fluxes_{text} = {text}_spectrum.flux_jy; lambdas_{text} = {text}_spectrum.lamgrid", globals())

    # Dynamically set the data for each molecule's line using exec and globals()
    exec(f"{text}_line.set_data(lambdas_{text}, fluxes_{text})", globals())
    
    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Density Updated!')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 

    #plt.pause(3)

    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    plt.draw(), canvas.draw()  
    
    # Initialize total fluxes list
    total_fluxes = []
    # Calculate total fluxes based on visibility conditions
    for i in range(len(lambdas_h2o)):
        flux_sum = 0
        for mol_name, mol_filepath, mol_label in molecules_data:
            

            mol_name_lower = mol_name.lower()
            visibility_flag = f"{mol_name_lower}_vis"
            fluxes_molecule = f"fluxes_{mol_name_lower}"

            if visibility_flag in globals() and globals()[visibility_flag]:
                flux_sum += globals()[fluxes_molecule][i]

        total_fluxes.append(flux_sum)
    
    sum_line.set_data(lambdas_h2o, total_fluxes)
    update()
    pop_diagram()
    canvas.draw()
    
def submit_temp(event, text):
    #print(event)
    #global text_box
    #global text_box_data
    global xp1
    global xp2
    global span
    global model_line_select
    global data_line_select
    global fig_height
    global fig_bottom_height
    global n_mol
    global selectedline
    global int_pars
    global molecules_data
    global total_fluxes
    
    
    
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Submitting Temperature...')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 



    
    val = float(event)
    exec(f"t_{text} = {val}", globals())
    
    # Intensity calculation
    exec(f"{text}_intensity.calc_intensity(t_{text}, n_mol_{text}, dv=intrinsic_line_width)", globals())

    # Spectrum creation
    exec(f"{text}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

    # Adding intensity to the spectrum
    exec(f"{text}_spectrum.add_intensity({text}_intensity, {text}_radius ** 2 * np.pi)", globals())

    # Fluxes and lambdas
    exec(f"fluxes_{text} = {text}_spectrum.flux_jy; lambdas_{text} = {text}_spectrum.lamgrid", globals())

    # Dynamically set the data for each molecule's line using exec and globals()
    exec(f"{text}_line.set_data(lambdas_{text}, fluxes_{text})", globals())
    
    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Temperature Updated!')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 

    #plt.pause(3)

    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    plt.draw(), canvas.draw()
    
    #sum_line, = ax1.plot([], [], color='gray', linewidth=1)
    #sum_line.set_label('Sum')
    #ax1.legend()
    
    # Initialize total fluxes list
    total_fluxes = []
    # Calculate total fluxes based on visibility conditions
    for i in range(len(lambdas_h2o)):
        flux_sum = 0
        for mol_name, mol_filepath, mol_label in molecules_data:
            

            mol_name_lower = mol_name.lower()
            visibility_flag = f"{mol_name_lower}_vis"
            fluxes_molecule = f"fluxes_{mol_name_lower}"

            if visibility_flag in globals() and globals()[visibility_flag]:
                flux_sum += globals()[fluxes_molecule][i]

        total_fluxes.append(flux_sum)
    
    sum_line.set_data(lambdas_h2o, total_fluxes)
    #exec(sum_line.set_data(lambdas_h2o, total_fluxes), globals())
    update()
    pop_diagram()
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 
    
def submit_rad(event, text):
    
    #global text_box
    #global text_box_data

    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Submitting Radius...')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 
    
    val = float(event)
    exec(f"{text}_radius = {val}", globals())
    
    # Intensity calculation
    exec(f"{text}_intensity.calc_intensity(t_{text}, n_mol_{text}, dv=intrinsic_line_width)", globals())

    # Spectrum creation
    exec(f"{text}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

    # Adding intensity to the spectrum
    exec(f"{text}_spectrum.add_intensity({text}_intensity, {text}_radius ** 2 * np.pi)", globals())

    # Fluxes and lambdas
    exec(f"fluxes_{text} = {text}_spectrum.flux_jy; lambdas_{text} = {text}_spectrum.lamgrid", globals())

    # Dynamically set the data for each molecule's line using exec and globals()
    exec(f"{text}_line.set_data(lambdas_{text}, fluxes_{text})", globals())
    
    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Radius updated!')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 


    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    plt.draw(), canvas.draw()
    
    # Initialize total fluxes list
    total_fluxes = []
    # Calculate total fluxes based on visibility conditions
    for i in range(len(lambdas_h2o)):
        flux_sum = 0
        for mol_name, mol_filepath, mol_label in molecules_data:
            

            mol_name_lower = mol_name.lower()
            visibility_flag = f"{mol_name_lower}_vis"
            fluxes_molecule = f"fluxes_{mol_name_lower}"

            if visibility_flag in globals() and globals()[visibility_flag]:
                flux_sum += globals()[fluxes_molecule][i]

        total_fluxes.append(flux_sum)
    
    sum_line.set_data(lambdas_h2o, total_fluxes)
    update()
    pop_diagram()
    canvas.draw()


"""
flux_integral() calculates the flux of the data line in the selected region of the top graph.
This function is used in onselect().
"""
def flux_integral(lam, flux, lam_min, lam_max):
    # calculate flux integral
    integral_range = np.where(np.logical_and(lam > lam_min, lam < lam_max))
    line_flux_meas = np.trapz(flux[integral_range[::-1]], x=ccum/lam[integral_range[::-1]])
    line_flux_meas = -line_flux_meas*1e-23 # to get (erg s-1 cm-2); it's using frequency array, so need the - in front of it
    return line_flux_meas


"""
model_visible() is connected to the "Visible" button in the tool.
This function turn on/off the visibility of the line for the currently selected model
"""

# Function to update visibility when a button is clicked
def model_visible(event):
    # Update visibility based on the button that was clicked
    if globals()[f"{event}_vis"] == True:
        globals()[f"{event}_vis"] = False
       
        if event == 'h2o':
            ax1.fill_between(lambdas_h2o, fluxes_h2o, 0, facecolor="red", color='red', alpha=0)
        # Add similar blocks for other molecules
    else:
        globals()[f"{event}_vis"] = True
        if event == 'h2o':
            ax1.fill_between(lambdas_h2o, fluxes_h2o, 0, facecolor="red", color='red', alpha=0.2)


    update()  # Call the update function to refresh the plot
    canvas.draw()
#def LATInit():
skip = False


selectedline = False


root = tk.Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

def selectfileinit():
    global file_path
    global file_name
    global wave_data, flux_data, err_data, wave_original
    global input_spectrum_data
    global filename_box_data
    global mode
    global xp1, rng, xp2

    spectra_directory = os.path.abspath("EXAMPLE-data")
    filetypes = [('CSV Files', '*.csv')]
    # Ask the user to select a file
    infiles = filedialog.askopenfilename(multiple=True, title='Choose Spectrum Data File', filetypes=filetypes, initialdir=spectra_directory)

    if infiles:
        for file_path in infiles:
            # Process each selected file
            print(' ')
            print("Selected file:", file_path)
            file_name = os.path.basename(file_path)
            # code to process each file
            input_spectrum_data = pd.read_csv(filepath_or_buffer=file_path, sep=',')
            wave_data = np.array(input_spectrum_data['wave'])
            wave_original = np.array(input_spectrum_data['wave'])
            flux_data = np.array(input_spectrum_data['flux'])
            if 'err' in input_spectrum_data:
                err_data = np.array(input_spectrum_data['err'])
            else:
                err_data = np.empty_like(flux_data) + np.nanmax(flux_data)/100 # assumed, if not present

                # Set initial values of xp1 and rng
            fig_max_limit = np.nanmax(wave_data)
            fig_min_limit = np.nanmin(wave_data)
            xp1 = np.around(fig_min_limit + (fig_max_limit - fig_min_limit)/2, decimals=2)
            rng = np.around((fig_max_limit - fig_min_limit)/10, decimals=2)
            xp2 = xp1 + rng

            # now = dt.now()
            # dateandtime = now.strftime("%d-%m-%Y-%H-%M-%S")
            # print(dateandtime)
            # svd_line_file = f'savedlines-{dateandtime}.csv'
            
        # Ask the user to select the mode (light or dark)
        mode_dialog = tk.messagebox.askquestion("Select Mode", "Would you like to start iSLAT in Dark Mode?")
        
        if mode_dialog == 'yes':
            mode = True  # Dark mode
        else:
            mode = False  # Light mode
    else:
        print("No files selected.")


selectfileinit()

print(' ')
print ('Loading molecule files: ...')

molecules_data = read_from_user_csv()
# Loop through each molecule and set up the necessary objects and variables
for mol_name, mol_filepath, mol_label in molecules_data:
    # Import line lists from the ir_model folder
    mol_data = MolData(mol_name, mol_filepath)

    # Get the initial parameters for the current molecule, use default if not defined
    params = initial_parameters.get(mol_name, default_initial_params)
    scale_exponent = params["scale_exponent"]
    scale_number = params["scale_number"]
    t_kin = params["t_kin"]
    radius_init = params["radius_init"]

    # Calculate and set n_mol_init for the current molecule
    n_mol_init = float(scale_number * (10 ** scale_exponent))

    # Use exec() to create the variables with specific variable names for each molecule
    exec(f"mol_{mol_name.lower()} = MolData('{mol_name}', '{mol_filepath}')", globals())
    exec(f"scale_exponent_{mol_name.lower()} = {scale_exponent}", globals())
    exec(f"scale_number_{mol_name.lower()} = {scale_number}", globals())
    exec(f"n_mol_{mol_name.lower()}_init = {n_mol_init}", globals())
    exec(f"t_kin_{mol_name.lower()} = {t_kin}", globals())
    exec(f"{mol_name.lower()}_radius_init = {radius_init}", globals())

    # Print the results (you can modify this part as needed)
    print(f"Molecule Initialized: {mol_name}")
    #print(f"scale_exponent_{mol_name.lower()} = {scale_exponent}")
    #print(f"scale_number_{mol_name.lower()} = {scale_number}")
    #print(f"n_mol_{mol_name.lower()}_init = {n_mol_init}")
    #print(f"t_kin_{mol_name.lower()} = {t_kin}")
    #print(f"{mol_name.lower()}_radius_init = {radius_init}")
    #print()  # Empty line for spacing


    # Store the initial values in the dictionary
    initial_values[mol_name.lower()] = {
        "scale_exponent": scale_exponent,
        "scale_number": scale_number,
        "t_kin": t_kin,
        "radius_init": radius_init,
        "n_mol_init": n_mol_init
    }
    
# Initialize visibility booleans for each molecule
molecule_names = [mol_name.lower() for mol_name, _ , _ in molecules_data]
for mol_name in molecule_names:
    if mol_name == 'h2o':
        globals()[f"{mol_name}_vis"] = True
    else:
        globals()[f"{mol_name}_vis"] = False

for mol_name, mol_filepath, mol_label in molecules_data:
    molecule_name_lower = mol_name.lower()

    # Column density
    exec(f"global n_mol_{molecule_name_lower}; n_mol_{molecule_name_lower} = n_mol_{molecule_name_lower}_init")

    # Temperature
    exec(f"global t_{molecule_name_lower}; t_{molecule_name_lower} = t_kin_{molecule_name_lower}")

    # Radius
    exec(f"global {molecule_name_lower}_radius; {molecule_name_lower}_radius = {molecule_name_lower}_radius_init")

if mode:
    global background
    global foreground
    
    #plt.style.use('dark_background')
    # Set Matplotlib rcParams for dark background
    matplotlib.rcParams['figure.facecolor'] = 'black'
    matplotlib.rcParams['axes.facecolor'] = 'black'
    matplotlib.rcParams['axes.edgecolor'] = 'white'
    matplotlib.rcParams['xtick.color'] = 'white'
    matplotlib.rcParams['ytick.color'] = 'white'
    matplotlib.rcParams['text.color'] = 'white'
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    background = 'black'
    foreground = 'white'
    #self.toolbar.setStyleSheet("background-color:Gray;")
else:
    
    background = 'white'
    foreground = 'black'


def update_xp1_rng():
    global xp1, rng, xp2 
    # Get the values from the Tkinter Entry widgets and convert them to floats
    xp1 = float(xp1_entry.get())
    rng = float(rng_entry.get())
    xp2 = xp1 + rng
    ax1.set_xlim(xmin=xp1, xmax=xp2)
    print("Updated values: xp1 =", xp1, ", rng =", rng)
    update()
    canvas.draw()
    
def update_initvals():
    global min_lamb, max_lamb, dist, fwhm, star_rv, model_line_width, model_pixel_res, intrinsic_line_width, wave_data, pix_per_fwhm
    # Get the values from the Tkinter Entry widgets and convert them to floats
    min_lamb = float(min_lamb_entry.get())
    max_lamb = float(max_lamb_entry.get())
    dist = float(dist_entry.get())
    fwhm = float(fwhm_entry.get())
    if fwhm >= 70:
        pix_per_fwhm = 10
    if fwhm < 70:
        pix_per_fwhm = 20 # increase model pixel sampling in case of higher resolution spectra, usually in the M band
    intrinsic_line_width = float(intrinsic_line_width_entry.get())
    model_line_width = cc / fwhm
    model_pixel_res = (np.mean([min_lamb, max_lamb]) / cc * fwhm) / pix_per_fwhm
    # this below needs to be updated to act on the wave array in the data
    wave_data = wave_original - (wave_original / cc * float(star_rv_entry.get()))
    loadsavedmessage()
    update()
    canvas.draw()

    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Parameter updated!')
    # time.sleep(2)


# # Functing to limit the user input from the previous prompts to the range of the data you're inspecting
# # The tool with stop and the associated dialogs will be printed in the serial line
# if xp2 > fig_max_limit:
#     sys.exit("Your wavelength range extends past the model, please start with a new range.")
# if xp1 < fig_min_limit:
#     sys.exit("Your wavelength range extends past the model, please start with a new range.")

# Set the headers for the saved lines csv to start at True
headers = True

for mol_name, mol_filepath, mol_label in molecules_data:
    molecule_name_lower = mol_name.lower()

    # Intensity calculation
    exec(f"{molecule_name_lower}_intensity = Intensity(mol_{molecule_name_lower})")
    exec(f"{molecule_name_lower}_intensity.calc_intensity(t_kin_{molecule_name_lower}, n_mol_{molecule_name_lower}, dv=intrinsic_line_width)")

    # Spectrum creation
    exec(f"{molecule_name_lower}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)")

    # Adding intensity to the spectrum
    exec(f"{molecule_name_lower}_spectrum.add_intensity({molecule_name_lower}_intensity, {molecule_name_lower}_radius ** 2 * np.pi)")

    # Fluxes and lambdas
    exec(f"fluxes_{molecule_name_lower} = {molecule_name_lower}_spectrum.flux_jy; lambdas_{molecule_name_lower} = {molecule_name_lower}_spectrum.lamgrid")


#Setting up the line identifier tool
int_pars = h2o_intensity.get_table
int_pars.index = range(len(int_pars.index))

#Creating the graph
fig = plt.figure(figsize=(15, 8.5))
#fig = plt.figure()
gs = GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[1, 1.5])
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
# Create a text box for streaming useful information in the third section
#text_box = fig.add_subplot(gs[1, 0])  # This is the third section
#text_box_data = TextBox(text_box, label='', color = background, hovercolor= background)
ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
ax3.set_xlabel(r'$E_{u}$')
ax3.set_label('Population Diagram')
ax2.set_xlabel('Wavelength (μm)')
ax1.set_ylabel('Flux density (Jy)')
ax2.set_ylabel('Flux density (Jy)')
ax1.set_xlim(xmin=xp1, xmax=xp2)
plt.rcParams['font.size'] = 10
data_line, = ax1.plot(wave_data, flux_data, color=foreground, linewidth=1)

for mol_name, mol_filepath, mol_label in molecules_data:
        molecule_name_lower = mol_name.lower()
        
        if molecule_name_lower == 'h2o':
            exec(f"{molecule_name_lower}_line, = ax1.plot({molecule_name_lower}_spectrum.lamgrid, fluxes_{molecule_name_lower}, alpha=0.8, linewidth=1)", globals())
        else:
            exec(f"{molecule_name_lower}_line, = ax1.plot([], [], alpha=0.8, linewidth=1)", globals())
        exec(f"{molecule_name_lower}_line.set_label('{mol_label}')", globals())
data_line.set_label('Data')
sum_line, = ax1.plot([], [], color='purple', linewidth=1)
sum_line.set_label('Sum')
ax1.legend()


ax2.set_frame_on(False)
ax3.set_frame_on(False)


#make empty lines for the second plot
data_line_select, = ax2.plot([],[],color=foreground,linewidth=1)


#Scaling the y-axis based on tallest peak of data
range_flux_cnts = input_spectrum_data[(input_spectrum_data['wave'] > xp1) & (input_spectrum_data['wave'] < xp2)]
range_flux_cnts.index = range(len(range_flux_cnts.index))
fig_height = np.nanmax(range_flux_cnts.flux)
fig_bottom_height = np.min(range_flux_cnts.flux)
ax1.set_ylim(ymin=fig_bottom_height, ymax=fig_height+(fig_height/8))

# adjust the plots to make room for the widgets
fig.subplots_adjust(left=0.06, right=0.97, top = 0.97, bottom=0.09)

# Populating the population diagram graph
pop_diagram()



num_rows = 9

# Calculate the height and width of each row
row_height = 0.035
row_width = 0.19

# Calculate the total height of all rows
total_height = row_height * num_rows

# Calculate the starting y-position for the first row within the control_border
start_y = 0.52 + (0.45 - total_height) / 2  # Center vertically
   
# Define the column labels
column_labels = ['Molecule', 'Temp.', 'Radius', 'Col. Dens', 'On', 'Del.', 'Color']

# Create a dictionary to store the visibility buttons
vis_buttons_dict = {}

# Create a tkinter window
window = tk.Tk()
window.title("iSLAT "+ iSLAT_version)


# MENU of tkinter GUI frame parts and definitions:
# title_frame : menu at the top of the GUI (Add molecule, etc)
# molecule_frame (contained by outer_frame) : top right frame for molecules and their parameters (temp, rad, coldens)
# files_frame : file input and output
# plotparams_frame : plot parameters (plot start, range, etc)
# functions_frame : buttons to activate functions (Save line, Fit line, etc)
# text_frame : bottom right empty frame for text messages

# create buttons for top of GUI
nb_of_columns = 10 # to be replaced by the relevant number
title_frame = tk.Frame(window, bg ="gray")
title_frame.grid(row=0, column=0, columnspan=nb_of_columns, sticky='ew')

# Create a frame to hold the canvasscroll and both scrollbars
outer_frame = tk.Frame(window)
outer_frame.grid(row=title_frame.grid_info()['row'] + title_frame.grid_info()['rowspan'], column=0, rowspan=10, columnspan=5, sticky="nsew")

# Create a canvasscroll widget
canvasscroll = tk.Canvas(outer_frame)
canvasscroll.grid(row=0, column=0, sticky="nsew")

# Create vertical and horizontal scrollbar widgets and associate them with the canvasscroll
vscrollbar = tk.Scrollbar(outer_frame, orient="vertical", command=canvasscroll.yview)
vscrollbar.grid(row=0, column=1, sticky="ns")
hscrollbar = tk.Scrollbar(outer_frame, orient="horizontal", command=canvasscroll.xview)
hscrollbar.grid(row=1, column=0, sticky="ew")

# Configure the canvasscroll to use the scrollbars
canvasscroll.configure(yscrollcommand=vscrollbar.set, xscrollcommand=hscrollbar.set)

# Allow resizing of the canvasscroll and outer_frame
outer_frame.grid_rowconfigure(0, weight=1)
outer_frame.grid_columnconfigure(0, weight=1)

# Create the frame that will contain your actual content
molecule_frame = tk.Frame(canvasscroll, borderwidth=2)  # , relief="groove")
canvasscroll.create_window((0, 0), window=molecule_frame, anchor="nw")

# Configure the canvasscroll scroll region
def on_frame_configure(event):
    canvasscroll.configure(scrollregion=canvasscroll.bbox("all"))

molecule_frame.bind("<Configure>", on_frame_configure)


# Create the frame with the specified properties
#molecule_frame = tk.Frame(window, borderwidth=2, relief="groove")
#molecule_frame.grid(row=1, column=0, rowspan=10, columnspan=5, sticky="nsew")

# Create labels for columns
for col, label in enumerate(column_labels):
    label_widget = tk.Label(molecule_frame, text=label)
    label_widget.grid(row=0, column=col)

def choose_color(widget):
    # Get the row number from the widget's grid info
    row = widget.grid_info()["row"]
    
    # Get the molecule name from the entry widget in the same row
    mol_name = molecule_frame.grid_slaves(row=row, column=0)[0].get().lower()
    
    # Ask user to choose a color
    color = colorchooser.askcolor(title="Choose a color")
    if color[1]:  # Check if a color was selected
        # Set the color of the molecule line
        globals()[f"{mol_name.lower()}_color"] = color[1]
        exec(f"{mol_name.lower()}_line.set_color('{color[1]}')", globals())
        # Get the color button from the grid_slaves list
        color_button = molecule_frame.grid_slaves(row=row, column=6)[0]

        # Set the background color of the color button
        color_button.configure(bg=color[1])
        update()

    
def delete_row(widget):
    global molecules_data, nextrow
    
    data_field.delete('1.0', "end")
    
    row = widget.grid_info()["row"]
    mol_name = molecule_frame.grid_slaves(row=row, column=0)[0].get().lower()
    
    if mol_name == "h2o":
        data_field.delete ('1.0', "end")
        data_field.insert ('1.0', f'You can not delete {mol_name.upper()}!')
        return

    # Destroy all widgets in the row
    for w in molecule_frame.grid_slaves(row=row):
        w.destroy()
        
    exec(f"{mol_name.lower()}_line.remove()", globals())

    # Remove the molecule from molecules_data
    molecules_data = [molecule for molecule in molecules_data if molecule[0].lower() != mol_name]
    
    write_user_csv(molecules_data)
    
    # Move all rows below this row up by one
    for r in range(row + 1, nextrow):
        for col in range(7):  # Adjust the range if you have more columns
            widget_list = molecule_frame.grid_slaves(row=r, column=col)
            for widget in widget_list:
                widget.grid(row=r-1, column=col)

    nextrow -= 1
    
    spanoptionsvar = [m[0] for m in molecules_data]
    spandropd['values'] = spanoptionsvar
    if spanoptionsvar:
        spandropd.set(spanoptionsvar[0])
    update()

    data_field.delete ('1.0', "end")
    data_field.insert ('1.0', f'{mol_name.upper()} deleted!')

def update_csv():
    filename = os.path.join(save_folder, f"{file_name}-molsave.csv")
    csv_file = filename
    try:
        # Read existing data from CSV
        with open(csv_file, 'r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)

        # Identify the row to delete
        for i, row in enumerate(rows):
            if row[0].lower() == mol_name:
                del rows[i]
                break

        # Write updated data back to CSV
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        print(f"{csv_file} updated.")
    except Exception as e:
        print(f"Error updating {csv_file}: {e}")
      

# Loop to create rows of input fields and buttons for each chemical
nextrow = 1  # Start with row 1
for row, (mol_name, mol_filepath, mol_label) in enumerate(molecules_data):
    #global nextrow
    y_row = start_y + row_height * (num_rows - row - 1)
    row = row + 1
    # Get the initial values for the current chemical from the dictionary
    params = initial_values[mol_name.lower()]
    scale_exponent = params["scale_exponent"]
    scale_number = params["scale_number"]
    t_kin = params["t_kin"]
    radius_init = params["radius_init"]
    n_mol_init = params["n_mol_init"]

    # Row label
    exec(f"{mol_name.lower()}_rowl_field = tk.Entry(molecule_frame, width=6)")
    eval(f"{mol_name.lower()}_rowl_field").grid(row=row, column=0)
    eval(f"{mol_name.lower()}_rowl_field").insert(0, f"{mol_name}")

    # Temperature input field
    exec(f"{mol_name.lower()}_temp_field = tk.Entry(molecule_frame, width=4)")
    eval(f"{mol_name.lower()}_temp_field").grid(row=row, column=1)
    eval(f"{mol_name.lower()}_temp_field").insert(0, f"{t_kin}")
    eval(f"{mol_name.lower()}_temp_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
    

    # Radius input field
    exec(f"{mol_name.lower()}_rad_field = tk.Entry(molecule_frame, width=4)")
    eval(f"{mol_name.lower()}_rad_field").grid(row=row, column=2)
    eval(f"{mol_name.lower()}_rad_field").insert(0, f"{radius_init}")
    eval(f"{mol_name.lower()}_rad_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))

    
    # Column Density input field
    exec(f"{mol_name.lower()}_dens_field = tk.Entry(molecule_frame, width=6)")
    eval(f"{mol_name.lower()}_dens_field").grid(row=row, column=3)
    eval(f"{mol_name.lower()}_dens_field").insert(0, f"{n_mol_init:.{1}e}")
    eval(f"{mol_name.lower()}_dens_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
    
    # Visibility Checkbutton
    if mol_name.lower() == 'h2o':
        exec(f"{mol_name.lower()}_vis_status = tk.BooleanVar()")
        exec(f"{mol_name.lower()}_vis_status.set(True)")  # Set the initial state
        exec(f"{mol_name.lower()}_vis_checkbutton = tk.Checkbutton(molecule_frame, text='', variable={mol_name.lower()}_vis_status, onvalue=True, offvalue=False, command=lambda mn=mol_name.lower(): model_visible(mn))")
        exec(f"{mol_name.lower()}_vis_checkbutton.select()")
    else:
        globals()[f"{mol_name.lower()}_vis_status"] = tk.BooleanVar()
        globals()[f"{mol_name.lower()}_vis_checkbutton"] = tk.Checkbutton(molecule_frame, text='', variable=eval(f"{mol_name.lower()}_vis_status"), command=lambda mn=mol_name.lower(): model_visible(mn))
        globals()[f"{mol_name.lower()}_vis_status"].set(False)  # Set the initial state

    eval(f"{mol_name.lower()}_vis_checkbutton").grid(row=row, column=4)

    # Delete button
    del_button = tk.Button(molecule_frame, text="X", command=lambda widget=eval(f"{mol_name.lower()}_rowl_field"): delete_row(widget))
    del_button.grid(row=row, column=5)
    
    color_button = tk.Button(molecule_frame, text=" ", command=lambda widget=eval(f"{mol_name.lower()}_rowl_field"): choose_color(widget))
    color_button.grid(row=row, column=6)
    

    nextrow = row + 1




variable_names = ['t_h2o', 'h2o_radius', 'n_mol_h2o', 't_oh', 'oh_radius', 'n_mol_oh', 't_hcn', 'hcn_radius', 'n_mol_hcn', 't_c2h2', 'c2h2_radius', 'n_mol_c2h2']


def loadsavedmessage():
    global data_field
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Parameter updated')
    fig.canvas.draw_idle()

def saveparams_button_clicked():
    write_to_csv(molecules_data, True)


def load_variables_from_file(file_name):
    #global text_box_data
    #global text_box
    global molecules_data, nextrow
    # Display a confirmation dialog
    confirmed = tk.messagebox.askquestion("Confirmation", "Sure you want to load parameters? Make sure to save any unsaved changes!")
    if confirmed == "no":  # Check if user clicked "no"
        return
    if not os.path.exists(os.path.join(save_folder, f"{file_name}-molsave.csv")):
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'No save for data file found.')
        return
    
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Loading saved parameters, this may take a moment...')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events()
    
    #del_molecule_data()
    
    molecules_data = read_from_csv()
    
   # Read molecules_list.csv
    molecules_list = []
    new_molecules = []
    missing_molecules = []  # Initialize the missing_molecules array
    try:
        with open(os.path.join(save_folder, "molecules_list.csv"), 'r') as list_file:
            reader = csv.reader(list_file)
            next(reader)  # Skip header
            for row in reader:
                molecules_list.append(tuple(row[:3]))  # Taking the first three columns of each row
    except Exception as e:
        print("Error reading molecules_list.csv:", e)

    # Create a set of molecule names from molecules_data
    molecules_data_names = set(item[0] for item in molecules_data)
    
    # Create a set of molecule names from molecules_data
    molecules_list_names = set(item[0] for item in molecules_list)
    
    # Append missing molecules from molecules_list.csv
    for mol_name, mol_path, mol_label in molecules_data:
        if mol_name not in molecules_list_names:
            new_molecules.append((mol_name, mol_path, mol_label))

    # Append missing molecules from molecules_list.csv
    for mol_name, mol_path, mol_label in molecules_list:
        if mol_name not in molecules_data_names:
            missing_molecules.append((mol_name, mol_path, mol_label))  # Add to missing_molecules
            molecules_data.append((mol_name, mol_path, mol_label))
    
    print(f"new molecules:{new_molecules}")
    
    # Create labels for columns
    for col, label in enumerate(column_labels):
        label_widget = tk.Label(molecule_frame, text=label)
        label_widget.grid(row=0, column=col)

    # Loop to create rows of input fields and buttons for each chemical
    #nextrow = 1  # Start with row 1
    for row, (mol_name, mol_filepath, mol_label) in enumerate(new_molecules):
        #global nextrow
        y_row = start_y + row_height * (num_rows - row - 1)
        row = nextrow
        # Get the initial values for the current chemical from the dictionary
        params = initial_parameters.get(mol_name, default_initial_params)
        scale_exponent = params["scale_exponent"]
        scale_number = params["scale_number"]
        t_kin = params["t_kin"]
        radius_init = params["radius_init"]
        
        # Calculate and set n_mol_init for the current molecule
        n_mol_init = float(scale_number * (10 ** scale_exponent))
        
        # Import line lists from the ir_model folder
        mol_data = MolData(mol_name, mol_filepath)
        
        # Use exec() to create the variables with specific variable names for each molecule
        exec(f"mol_{mol_name.lower()} = MolData('{mol_name}', '{mol_filepath}')", globals())
               
        # Row label
        exec(f"{mol_name.lower()}_rowl_field = tk.Entry(molecule_frame, width=6)", globals())
        eval(f"{mol_name.lower()}_rowl_field").grid(row=row, column=0)
        eval(f"{mol_name.lower()}_rowl_field").insert(0, f"{mol_name}")
        #molecule_elements[mol_name.lower()] = {'rowl': mol_name.lower() + '_rowl_field'}

        # Temperature input field
        globals()[f"{mol_name.lower()}_temp_field"] = tk.Entry(molecule_frame, width=4)

        eval(f"{mol_name.lower()}_temp_field").grid(row=row, column=1)
        eval(f"{mol_name.lower()}_temp_field").insert(0, f"{t_kin}")
        #globals() [f"{mol_name.lower()}_submit_temp_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), te = globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(te.get(), mn))
        #eval(f"{mol_name.lower()}_submit_temp_button").grid(row=row + 1, column=2)
        #molecule_elements[mol_name.lower()] = {'temp': mol_name.lower() + '_temp_field'}
        eval(f"{mol_name.lower()}_temp_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
        exec(f"t_{mol_name.lower()} = {t_kin}", globals())
        
        # Radius input field
        globals()[f"{mol_name.lower()}_rad_field"] = tk.Entry(molecule_frame, width=4)
        eval(f"{mol_name.lower()}_rad_field").grid(row=row, column=2)
        eval(f"{mol_name.lower()}_rad_field").insert(0, f"{radius_init}")
        #globals() [f"{mol_name.lower()}_submit_rad_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), re = globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(re.get(), mn))
        #eval(f"{mol_name.lower()}_submit_rad_button").grid(row=row + 1, column=4)
        #molecule_elements[mol_name.lower()]['rad'] = mol_name.lower() + '_rad_field'
        eval(f"{mol_name.lower()}_rad_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))
        exec(f"{mol_name.lower()}_radius = {radius_init}", globals())
        
        # Column Density input field
        globals()[f"{mol_name.lower()}_dens_field"] = tk.Entry(molecule_frame, width=6)
        eval(f"{mol_name.lower()}_dens_field").grid(row=row, column=3)
        eval(f"{mol_name.lower()}_dens_field").insert(0, f"{n_mol_init:.{1}e}")
        #globals() [f"{mol_name.lower()}_submit_col_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
        #eval(f"{mol_name.lower()}_submit_col_button").grid(row=row + 1, column=6)
        #molecule_elements[mol_name.lower()]['dens'] = mol_name.lower() + '_dens_field'
        eval(f"{mol_name.lower()}_dens_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
        exec(f"n_mol_{mol_name.lower()} = {n_mol_init}", globals())
        
        # Visibility Checkbutton
        if mol_name.lower() == 'h2o':
            exec(f"{mol_name.lower()}_vis_status = tk.BooleanVar()")
            exec(f"{mol_name.lower()}_vis_status.set(True)")  # Set the initial state
            exec(f"{mol_name.lower()}_vis_checkbutton = tk.Checkbutton(molecule_frame, text='', variable={mol_name.lower()}_vis_status, onvalue=True, offvalue=False, command=lambda mn=mol_name.lower(): model_visible(mn))")
            exec(f"{mol_name.lower()}_vis_checkbutton.select()")
        else:
            globals()[f"{mol_name.lower()}_vis_status"] = tk.BooleanVar()
            globals()[f"{mol_name.lower()}_vis_checkbutton"] = tk.Checkbutton(molecule_frame, text='', variable=eval(f"{mol_name.lower()}_vis_status"), command=lambda mn=mol_name.lower(): model_visible(mn))
            globals()[f"{mol_name.lower()}_vis_status"].set(False)  # Set the initial state
            
        globals()[f"{mol_name}_vis"] = False
        eval(f"{mol_name.lower()}_vis_checkbutton").grid(row=row, column=4)

        # Delete button
        del_button = tk.Button(molecule_frame, text="X", command=lambda widget=eval(f"{mol_name.lower()}_rowl_field"): delete_row(widget))
        del_button.grid(row=row, column=5)

        color_button = tk.Button(molecule_frame, text=" ", command=lambda widget=eval(f"{mol_name.lower()}_rowl_field"): choose_color(widget))
        color_button.grid(row=row, column=6)
        
        exec(f"{mol_name.lower()}_line, = ax1.plot([], [], alpha=0.8, linewidth=1)", globals())
        exec(f"{mol_name.lower()}_line.set_label('{mol_name}')", globals())
        
        # Intensity calculation
        exec(f"{mol_name.lower()}_intensity = Intensity(mol_{mol_name.lower()})", globals())
        exec(f"{mol_name.lower()}_intensity.calc_intensity(t_{mol_name.lower()}, n_mol_{mol_name.lower()}, dv=intrinsic_line_width)", globals())
        #print(f"{mol_name.lower()}_intensity")
        # Add the variables to the globals dictionary
        globals()[f"{mol_name.lower()}_intensity"] = eval(f"{mol_name.lower()}_intensity")

        # Spectrum creation
        exec(f"{mol_name.lower()}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

        # Adding intensity to the spectrum
        exec(f"{mol_name.lower()}_spectrum.add_intensity({mol_name.lower()}_intensity, {mol_name.lower()}_radius ** 2 * np.pi)", globals())

        # Fluxes and lambdas
        exec(f"fluxes_{mol_name.lower()} = {mol_name.lower()}_spectrum.flux_jy; lambdas_{mol_name.lower()} = {mol_name.lower()}_spectrum.lamgrid", globals())

        #delete_button = tk.Button(molecule_frame, text="Delete", command=lambda r=row, mn=mol_name: delete_row(r, mn))
        #delete_button.grid(row=row, column=5)

        nextrow = nextrow + 1
    
    filename = os.path.join(save_folder, f"{file_name}-molsave.csv")
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)  # Read the header row
                rows = list(reader)  # Read all rows into a list
                for i, row in enumerate(rows):
                    mol_name, mol_filepath, mol_label, temp, rad, n_mol, color, vis, dist, stellarrv, fwhm, ilw = row
                    # Update global variables or GUI fields with the loaded values
                    exec(f"global t_{mol_name.lower()}; t_{mol_name.lower()} = {temp}")
                    exec(f"global {mol_name.lower()}_radius; {mol_name.lower()}_radius = {rad}")
                    exec(f"global n_mol_{mol_name.lower()}; n_mol_{mol_name.lower()} = {n_mol}")
                    exec(f"global {mol_name.lower()}_line_color; {mol_name.lower()}_line_color = '{color}'")
                    exec(f"global {mol_name.lower()}_vis; {mol_name.lower()}_vis = {vis}")
                    exec(f"global dist; dist = {dist}")
                    exec(f"global star_rv; star_rv = {stellarrv}")
                    exec(f"global fwhm; fwhm = {fwhm}")
                    exec(f"global intrinsic_line_width; intrinsic_line_width = {ilw}")

                    # Update GUI fields
                    eval(f"{mol_name.lower()}_temp_field").delete(0, "end")
                    eval(f"{mol_name.lower()}_temp_field").insert(0, temp)

                    eval(f"{mol_name.lower()}_rad_field").delete(0, "end")
                    eval(f"{mol_name.lower()}_rad_field").insert(0, rad)

                    eval(f"{mol_name.lower()}_dens_field").delete(0, "end")
                    eval(f"{mol_name.lower()}_dens_field").insert(0, f"{float(n_mol):.{1}e}")

                    dist_entry.delete(0, "end")
                    dist_entry.insert(0, f"{dist}")

                    star_rv_entry.delete(0, "end")
                    star_rv_entry.insert(0, f"{star_rv}")

                    fwhm_entry.delete(0, "end")
                    fwhm_entry.insert(0, f"{fwhm}")

                    intrinsic_line_width_entry.delete(0, "end")
                    intrinsic_line_width_entry.insert(0, f"{intrinsic_line_width}")

                    # Call update_initvals() only on the last iteration
                    if i == len(rows) - 1:
                        update_initvals()

            #update()
            spanoptionsvar = [m[0] for m in molecules_data]
            spandropd['values'] = spanoptionsvar
            if spanoptionsvar:
                spandropd.set(spanoptionsvar[0])
                
            print("Variables loaded from CSV file.")
        except Exception as e:
            print("Error loading variables from CSV:", e)


    for row, (mol_name, _, _) in enumerate(molecules_data, start=1):

            linecolor = eval(f"{mol_name.lower()}_line_color")
            exec(f"{mol_name.lower()}_line.set_color('{linecolor}')", globals())
            # Get the molecule name in lower case
            mol_name_lower = mol_name.lower()

            # Get the line object
            line_var = globals().get(f"{mol_name_lower}_line")

            # Check if the line object exists and has a color attribute
            if line_var and hasattr(line_var, 'get_color'):
                # Get the color of the line
                line_color = line_var.get_color()
                globals()[f"{mol_name.lower()}_color"] = line_color

                # Get the color button from the grid_slaves list
                color_button = molecule_frame.grid_slaves(row=row, column=6)[0]
                # Set the background color of the color button
                color_button.configure(bg=line_color)
                
            if eval(f"{mol_name.lower()}_vis"):
                exec(f"{mol_name.lower()}_vis_checkbutton.select()")
                
                
    
    else:
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Saved parameters file not found.')
    update()
    write_user_csv(molecules_data)
    data_field.delete ('1.0', "end")
    data_field.insert ('1.0', 'Saved parameters loaded from file.')


def load_defaults_from_file():
    #global text_box_data
    #global text_box
    global molecules_data, nextrow
    
    confirmed = tk.messagebox.askquestion("Confirmation", "Sure you want to load the default molecules? This will erase all current parameters (save first if you wish to).")
    if confirmed == "no":  # Check if user clicked "no"
        return
    
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Loading default molecules, this may take a moment...')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events()
    
    del_molecule_data()
    
    molecules_data = read_default_csv()
    #print(molecules_data)
    # Create labels for columns
    for col, label in enumerate(column_labels):
        label_widget = tk.Label(molecule_frame, text=label)
        label_widget.grid(row=0, column=col)

    # Loop to create rows of input fields and buttons for each chemical
    nextrow = 1  # Start with row 1
    for row, (mol_name, mol_filepath, mol_label) in enumerate(molecules_data):
        #global nextrow
        y_row = start_y + row_height * (num_rows - row - 1)
        row = row + 1
        # Get the initial values for the current chemical from the dictionary
        params = initial_parameters.get(mol_name, default_initial_params)
        scale_exponent = params["scale_exponent"]
        scale_number = params["scale_number"]
        t_kin = params["t_kin"]
        radius_init = params["radius_init"]
        
        # Calculate and set n_mol_init for the current molecule
        n_mol_init = float(scale_number * (10 ** scale_exponent))
        
        # Import line lists from the ir_model folder
        mol_data = MolData(mol_name, mol_filepath)
        
        # Use exec() to create the variables with specific variable names for each molecule
        exec(f"mol_{mol_name.lower()} = MolData('{mol_name}', '{mol_filepath}')", globals())
               
        # Row label
        exec(f"{mol_name.lower()}_rowl_field = tk.Entry(molecule_frame, width=6)", globals())
        eval(f"{mol_name.lower()}_rowl_field").grid(row=row, column=0)
        eval(f"{mol_name.lower()}_rowl_field").insert(0, f"{mol_name}")
        #molecule_elements[mol_name.lower()] = {'rowl': mol_name.lower() + '_rowl_field'}

        # Temperature input field
        globals()[f"{mol_name.lower()}_temp_field"] = tk.Entry(molecule_frame, width=4)

        eval(f"{mol_name.lower()}_temp_field").grid(row=row, column=1)
        eval(f"{mol_name.lower()}_temp_field").insert(0, f"{t_kin}")
        #globals() [f"{mol_name.lower()}_submit_temp_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), te = globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(te.get(), mn))
        #eval(f"{mol_name.lower()}_submit_temp_button").grid(row=row + 1, column=2)
        #molecule_elements[mol_name.lower()] = {'temp': mol_name.lower() + '_temp_field'}
        eval(f"{mol_name.lower()}_temp_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
        exec(f"t_{mol_name.lower()} = {t_kin}", globals())
        
        # Radius input field
        globals()[f"{mol_name.lower()}_rad_field"] = tk.Entry(molecule_frame, width=4)
        eval(f"{mol_name.lower()}_rad_field").grid(row=row, column=2)
        eval(f"{mol_name.lower()}_rad_field").insert(0, f"{radius_init}")
        #globals() [f"{mol_name.lower()}_submit_rad_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), re = globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(re.get(), mn))
        #eval(f"{mol_name.lower()}_submit_rad_button").grid(row=row + 1, column=4)
        #molecule_elements[mol_name.lower()]['rad'] = mol_name.lower() + '_rad_field'
        eval(f"{mol_name.lower()}_rad_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))
        exec(f"{mol_name.lower()}_radius = {radius_init}", globals())
        
        # Column Density input field
        globals()[f"{mol_name.lower()}_dens_field"] = tk.Entry(molecule_frame, width=6)
        eval(f"{mol_name.lower()}_dens_field").grid(row=row, column=3)
        eval(f"{mol_name.lower()}_dens_field").insert(0, f"{n_mol_init:.{1}e}")
        #globals() [f"{mol_name.lower()}_submit_col_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
        #eval(f"{mol_name.lower()}_submit_col_button").grid(row=row + 1, column=6)
        #molecule_elements[mol_name.lower()]['dens'] = mol_name.lower() + '_dens_field'
        eval(f"{mol_name.lower()}_dens_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
        exec(f"n_mol_{mol_name.lower()} = {n_mol_init}", globals())
        
        # Visibility Checkbutton
        if mol_name.lower() == 'h2o':
            exec(f"{mol_name.lower()}_vis_status = tk.BooleanVar()")
            exec(f"{mol_name.lower()}_vis_status.set(True)")  # Set the initial state
            exec(f"{mol_name.lower()}_vis_checkbutton = tk.Checkbutton(molecule_frame, text='', variable={mol_name.lower()}_vis_status, onvalue=True, offvalue=False, command=lambda mn=mol_name.lower(): model_visible(mn))")
            exec(f"{mol_name.lower()}_vis_checkbutton.select()")
        else:
            globals()[f"{mol_name.lower()}_vis_status"] = tk.BooleanVar()
            globals()[f"{mol_name.lower()}_vis_checkbutton"] = tk.Checkbutton(molecule_frame, text='', variable=eval(f"{mol_name.lower()}_vis_status"), command=lambda mn=mol_name.lower(): model_visible(mn))
            globals()[f"{mol_name.lower()}_vis_status"].set(False)  # Set the initial state
            
        globals()[f"{mol_name}_vis"] = False
        eval(f"{mol_name.lower()}_vis_checkbutton").grid(row=row, column=4)

        # Delete button
        del_button = tk.Button(molecule_frame, text="X", command=lambda widget=eval(f"{mol_name.lower()}_rowl_field"): delete_row(widget))
        del_button.grid(row=row, column=5)

        color_button = tk.Button(molecule_frame, text=" ", command=lambda widget=eval(f"{mol_name.lower()}_rowl_field"): choose_color(widget))
        color_button.grid(row=row, column=6)

        exec(f"{mol_name.lower()}_line, = ax1.plot([], [], alpha=0.8, linewidth=1)", globals())
        exec(f"{mol_name.lower()}_line.set_label('{mol_name}')", globals())
        
        # Intensity calculation
        exec(f"{mol_name.lower()}_intensity = Intensity(mol_{mol_name.lower()})", globals())
        exec(f"{mol_name.lower()}_intensity.calc_intensity(t_{mol_name.lower()}, n_mol_{mol_name.lower()}, dv=intrinsic_line_width)", globals())
        #print(f"{mol_name.lower()}_intensity")
        # Add the variables to the globals dictionary
        globals()[f"{mol_name.lower()}_intensity"] = eval(f"{mol_name.lower()}_intensity")

        # Spectrum creation
        exec(f"{mol_name.lower()}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

        # Adding intensity to the spectrum
        exec(f"{mol_name.lower()}_spectrum.add_intensity({mol_name.lower()}_intensity, {mol_name.lower()}_radius ** 2 * np.pi)", globals())

        # Fluxes and lambdas
        exec(f"fluxes_{mol_name.lower()} = {mol_name.lower()}_spectrum.flux_jy; lambdas_{mol_name.lower()} = {mol_name.lower()}_spectrum.lamgrid", globals())

        #delete_button = tk.Button(molecule_frame, text="Delete", command=lambda r=row, mn=mol_name: delete_row(r, mn))
        #delete_button.grid(row=row, column=5)

        nextrow = row + 1
    
    write_user_csv(molecules_data)
    spanoptionsvar = [m[0] for m in molecules_data]
    spandropd['values'] = spanoptionsvar
    if spanoptionsvar:
        spandropd.set(spanoptionsvar[0])
    

    filename = os.path.join(save_folder, f"default.csv")
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)  # Read the header row
                for row in reader:
                    mol_name, mol_filepath, mol_label, temp, rad, n_mol, vis = row
                    # Update global variables or GUI fields with the loaded values
                    exec(f"global t_{mol_name.lower()}; t_{mol_name.lower()} = {temp}")
                    exec(f"global {mol_name.lower()}_radius; {mol_name.lower()}_radius = {rad}")
                    exec(f"global n_mol_{mol_name.lower()}; n_mol_{mol_name.lower()} = {n_mol}")
                    exec(f"global {mol_name.lower()}_vis; {mol_name.lower()}_vis = {vis}")
                    

                    # Update GUI fields
                    eval(f"{mol_name.lower()}_temp_field").delete(0, "end")
                    eval(f"{mol_name.lower()}_temp_field").insert(0, temp)

                    eval(f"{mol_name.lower()}_rad_field").delete(0, "end")
                    eval(f"{mol_name.lower()}_rad_field").insert(0, rad)

                    eval(f"{mol_name.lower()}_dens_field").delete(0, "end")
                    eval(f"{mol_name.lower()}_dens_field").insert(0, f"{float(n_mol):.{1}e}")
                    
                    update_initvals()

            #update()
            data_field.delete('1.0', "end")
            data_field.insert('1.0', 'Defaults loaded!')
        except Exception as e:
            print("Error loading defaults:", e)

    for row, (mol_name, _, _) in enumerate(molecules_data, start=1):

            # Get the molecule name in lower case
            mol_name_lower = mol_name.lower()

            # Get the line object
            line_var = globals().get(f"{mol_name_lower}_line")

            # Check if the line object exists and has a color attribute
            if line_var and hasattr(line_var, 'get_color'):
                # Get the color of the line
                line_color = line_var.get_color()
                globals()[f"{mol_name.lower()}_color"] = line_color

                # Get the color button from the grid_slaves list
                color_button = molecule_frame.grid_slaves(row=row, column=6)[0]

                # Set the background color of the color button
                color_button.configure(bg=line_color)
                
            if eval(f"{mol_name.lower()}_vis"):
                exec(f"{mol_name.lower()}_vis_checkbutton.select()")
       
                
                
def write_default_csv(data):
    csv_filename = os.path.join(save_folder, f"default.csv")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Molecule Name', 'File Path', 'Molecule Label', 'Temp', 'Rad', 'N_Mol', 'Vis']
            writer.writerow(header)

            for mol_name, mol_filepath, mol_label in data:
                row = [mol_name, mol_filepath, mol_label]
                # Append the variables for the current molecule to the row
                row.append(globals().get(f"t_{mol_name.lower()}", ''))
                row.append(globals().get(f"{mol_name.lower()}_radius", ''))
                row.append(globals().get(f"n_mol_{mol_name.lower()}", ''))
                row.append(globals().get(f"{mol_name.lower()}_vis", ''))
                writer.writerow(row)
    except Exception as e:
        print("Error:", e)

write_default_csv(default_data)
    
def loadparams_button_clicked():
    load_variables_from_file(file_name)
    #plt.draw(), canvas.draw()
    #fig.canvas.flush_events() 

files_frame = tk.Frame(window, borderwidth=2, relief="groove")
files_frame.grid(row=outer_frame.grid_info()['row'] + outer_frame.grid_info()['rowspan'], column=0, rowspan=7, columnspan=5, sticky="nsew")

# Function to open spectrum data file from the GUI using Open File for "Spectrum data file"
def selectfile():
    global file_path
    global file_name
    global wave_data, flux_data, err_data, wave_original
    global input_spectrum_data
    global filename_box_data
    global xp1, rng, xp2, xp1_entry, rng_entry

    filetypes = [('CSV Files', '*.csv')]
    spectra_directory = os.path.abspath ("EXAMPLE-data")
    infiles = filedialog.askopenfilename(multiple=True, title='Choose Spectrum Data File', filetypes=filetypes, initialdir=spectra_directory)

    if infiles:
        for file_path in infiles:
            # Process each selected file
            print("Selected file:", file_path)
            file_name = os.path.basename(file_path)
            file_name_label.config(text=str(file_name))
            #filename_box_data.set_val(file_name)
            # Add your code to process each file
            #THIS IS THE OLD FILE SYSTEM (THIS WILL BE USED UNTIL THE NEW FILE SYSTEM IS DEVELOPED) USE THIS!!!!!
            input_spectrum_data=pd.read_csv(filepath_or_buffer=(file_path), sep=',')
            wave_data = np.array(input_spectrum_data['wave'])
            wave_original = np.array(input_spectrum_data['wave'])
            flux_data = np.array(input_spectrum_data['flux'])
            if 'err' in input_spectrum_data:
                err_data = np.array(input_spectrum_data['err'])
            else:
                err_data = np.empty_like(flux_data) + np.nanmax(flux_data)/100 # assumed, if not present

            # Set initial values of xp1 and rng
            fig_max_limit = np.nanmax(wave_data)
            fig_min_limit = np.nanmin(wave_data)
            xp1 = fig_min_limit + (fig_max_limit - fig_min_limit)/2
            rng = (fig_max_limit - fig_min_limit)/10
            xp2 = xp1 + rng
            xp1_entry.delete(0, "end")
            xp1_entry.insert(0, np.around(xp1, decimals=2))
            rng_entry.delete(0, "end")
            rng_entry.insert(0, np.around(rng, decimals=2))

            #now = dt.now()
            #dateandtime = now.strftime("%d-%m-%Y-%H-%M-%S")
            #print(dateandtime)
            #svd_line_file = f'savedlines-{dateandtime}.csv'

            update()

            data_field.delete('1.0', "end")
            data_field.insert('1.0', 'New spectrum loaded!')
    else:
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'No file selected.')


def selectlinefile():
    global linelistfile
    global linelistpath

    # Create the folder if it doesn't exist
    linelist_folder = "LINELISTS"

    # Set the initial directory to the created folder
    initial_directory = os.path.abspath(linelist_folder)

    filetypes = [('CSV Files', '*.csv')]
    infile = filedialog.askopenfilename(
        title='Choose Line List File',
        filetypes=filetypes,
        defaultextension=".csv",
        initialdir=initial_directory  # Set the initial directory
    )

    if infile:
        linelistpath = infile
        linelistfile = os.path.basename(linelistpath)
        # Update the label with the selected/created file
        linefile_name_label.config(text=str(linelistfile))

        #headers = "lev_up,lev_low,lam, tau,intens,a_stein,e_up,g_up,xmin,xmax"

        # Check if the file already exists
        if os.path.exists(infile):
            # File already exists, so check if the headers match
            with open(infile, 'r') as existing_file:
                first_line = existing_file.readline().strip()
            #if first_line == headers:
            #    # Headers match, no need to write them
            #    pass
            #else:
            #    print("File selected is not a line save file")
        else:
            # File doesn't exist
            print("File selected does not exist")
        
        
        
def savelinefile():
    global linesavefile
    global linesavepath


    # Set the initial directory to the created folder
    initial_directory = os.path.abspath(linesave_folder)

    filetypes = [('CSV Files', '*.csv')]
    infile = filedialog.asksaveasfilename(
        title='Choose or Define a File',
        filetypes=filetypes,
        defaultextension=".csv",
        initialdir=initial_directory  # Set the initial directory
    )

    if infile:
        linesavepath = infile
        linesavefile = os.path.basename(linesavepath)
        # Update the label with the selected/created file
        savelinefile_name_label.config(text=str(linesavefile))

        headers = "species,lev_up,lev_low,lam,tau,intens,a_stein,e_up,g_up,xmin,xmax"

        # Check if the file already exists
        if os.path.exists(infile):
            # File already exists, so check if the headers match
            with open(infile, 'r') as existing_file:
                first_line = existing_file.readline().strip()
            if first_line == headers:
                # Headers match, no need to write them
                pass
            elif not first_line:
                # First line is empty, so write the headers
                with open(infile, 'a') as file:
                    file.write(headers + '\n')
            else:
                print("File selected is not a line save file")
        else:
            # File doesn't exist, create a new one and write headers
            with open(infile, 'w') as file:
                file.write(headers + '\n')

# Configure columns to expand and fill the width
for i in range(5):
    files_frame.columnconfigure(i, weight=1)

# Create a frame to hold the box outline
box_frame = tk.Frame(files_frame)
box_frame.grid(row=1, column=0, columnspan=5, sticky='nsew')

specfile_label = tk.Label(files_frame, text='Spectrum Data File:')
specfile_label.grid(row=0, column=0, columnspan=5, sticky='nsew')  # Center-align using grid

linefile_label = tk.Label(files_frame, text='Input Line List:')
linefile_label.grid(row=2, column=0, columnspan=5, sticky='nsew')  # Center-align using grid

linefile_label = tk.Label(files_frame, text='Output Line Measurements:')
linefile_label.grid(row=4, column=0, columnspan=5, sticky='nsew')  # Center-align using grid

# Create a frame to hold the box outline
linebox_frame = tk.Frame(files_frame)
linebox_frame.grid(row=3, column=0, columnspan=5, sticky='nsew')

# Create a label widget inside the frame to create the box outline
linebox_label = tk.Label(linebox_frame, text='', relief='solid', borderwidth=1, height=2)  # Adjust the height value as needed
linebox_label.pack(side="top", fill="both", expand=True)

# Create a label inside the box_frame and center-align it
linefile_name_label = tk.Label(linebox_label, text='')
linefile_name_label.grid(row=0, column=0, sticky='nsew')  # Center-align using grid

# Create a frame to hold the box outline
savelinebox_frame = tk.Frame(files_frame)
savelinebox_frame.grid(row=5, column=0, columnspan=5, sticky='nsew', pady=(0,10))

# Create a label widget inside the frame to create the box outline
linesavebox_label = tk.Label(savelinebox_frame, text='', relief='solid', borderwidth=1, height=2)  # Adjust the height value as needed
linesavebox_label.pack(side="top", fill="both", expand=True)

# Create a label inside the box_frame and center-align it
savelinefile_name_label = tk.Label(linesavebox_label, text='')
savelinefile_name_label.grid(row=0, column=0, sticky='nsew')  # Center-align using grid

# Create a label widget inside the frame to create the box outline
box_label = tk.Label(box_frame, text='', relief='solid', borderwidth=1, height=2)  # Adjust the height value as needed
box_label.pack(fill=tk.BOTH, expand=True)

# Create a label inside the box_frame and center-align it
file_name_label = tk.Label(box_label, text=str(file_name))
file_name_label.grid(row=0, column=0, sticky='nsew')  # Center-align using grid

file_button = tk.Button(files_frame, text='Open File', command=selectfile)
file_button.grid(row=1, column=5)

linefile_button = tk.Button(files_frame, text='Open File', command=selectlinefile)
linefile_button.grid(row=3, column=5)

linesave_button = tk.Button(files_frame, text='Define File', command=savelinefile)
linesave_button.grid(row=5, column=5, pady=(0,10))

# Add some space below files_frame
#tk.Label(files_frame, text="").grid(row=4, column=0)

plotparams_frame = tk.Frame(window, borderwidth=2, relief="groove")
plotparams_frame.grid(row=files_frame.grid_info()['row'] + files_frame.grid_info()['rowspan'], column=0, rowspan=6, columnspan=5, sticky="nsew")
    

# Create and place the xp1 text box in row 12, column 0
xp1_label = tk.Label(plotparams_frame, text="Plot start:")
xp1_label.grid(row=0, column=0)
xp1_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
xp1_entry.insert(0, str(xp1))
xp1_entry.grid(row=0, column=1)
xp1_entry.bind("<Return>", lambda event: update_xp1_rng())

# Create and place the rng text box in row 12, column 2
rng_label = tk.Label(plotparams_frame, text="Plot range:")
rng_label.grid(row=0, column=2)
rng_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
rng_entry.insert(0, str(rng))
rng_entry.grid(row=0, column=3)
rng_entry.bind("<Return>", lambda event: update_xp1_rng())

# Create and place the min_lamb text box in row 2, column 0
min_lamb_label = tk.Label(plotparams_frame, text="Min. Wave:")
min_lamb_label.grid(row=1, column=0)
min_lamb_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
min_lamb_entry.insert(0, str(min_lamb))
min_lamb_entry.grid(row=1, column=1)
min_lamb_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the max_lamb text box in row 2, column 2
max_lamb_label = tk.Label(plotparams_frame, text="Max. Wave:")
max_lamb_label.grid(row=1, column=2)
max_lamb_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
max_lamb_entry.insert(0, str(max_lamb))
max_lamb_entry.grid(row=1, column=3)
max_lamb_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the dist text box in row 3, column 0
dist_label = tk.Label(plotparams_frame, text="Distance:")
dist_label.grid(row=2, column=0)
dist_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
dist_entry.insert(0, str(dist))
dist_entry.grid(row=2, column=1)
dist_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the RV text box in row 3, column 0
dist_label = tk.Label(plotparams_frame, text="Stellar RV:")
dist_label.grid(row=2, column=2)
star_rv_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
star_rv_entry.insert(0, str(star_rv))
star_rv_entry.grid(row=2, column=3)
star_rv_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the fwhm text box in row 3, column 2
fwhm_label = tk.Label(plotparams_frame, text="FWHM:")
fwhm_label.grid(row=3, column=0)
fwhm_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
fwhm_entry.insert(0, str(fwhm))
fwhm_entry.grid(row=3, column=1)
fwhm_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the fwhm text box in row 3, column 2
intrinsic_line_width_label = tk.Label(plotparams_frame, text="Broadening:")
intrinsic_line_width_label.grid(row=3, column=2)
intrinsic_line_width_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
intrinsic_line_width_entry.insert(0, str(intrinsic_line_width))
intrinsic_line_width_entry.grid(row=3, column=3)
intrinsic_line_width_entry.bind("<Return>", lambda event: update_initvals())


def on_span_select(selected_item):
    global spanmol
    global model_line_select
    global model_indmin
    global model_indmax 
    
    # Suppress the divide by zero warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spanmol = selected_item

        pop_diagram()
        update()

        data_field.insert('1.0', 'Molecule selected: ' + spanmol)

        try:
            model_region_x_str = f"lambdas_{spanmol}[model_indmin:model_indmax]"
            model_region_x = eval(model_region_x_str)

            # Dynamically set the variable
            model_region_y_str = f"fluxes_{spanmol}[model_indmin:model_indmax]"
            model_region_y = eval(model_region_y_str)

            # Suppress the divide by zero warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)


            model_line_select.set_data(model_region_x, model_region_y)
            plt.draw()
            canvas.draw()
            fig.canvas.flush_events()
        except NameError:
            #print("model_indmin or model_indmax is not defined.")
            pass






# Create and place the xp1 text box in row 12, column 0
specsep_label = tk.Label(plotparams_frame, text="Line Separ.:")
specsep_label.grid(row=4, column=2, pady=(0,45))
specsep_entry = tk.Entry(plotparams_frame, bg='lightgray', width=8)
specsep_entry.insert(0, str(specsep))
specsep_entry.grid(row=4, column=3, pady=(0,45))


# Add some space below plotparams_frame
#tk.Label(plotparams_frame, text="").grid(row=4, column=0)

def generate_all_csv():

    for molecule in molecules_data:
        mol_name = molecule[0]
        mol_name_lower = mol_name.lower ()

        fluxes = globals ().get (f'fluxes_{mol_name_lower}', np.array ([]))
        lambdas = globals ().get (f'lambdas_{mol_name_lower}', np.array ([]))

        if fluxes.size == 0 or lambdas.size == 0 or len (fluxes) != len (lambdas):
            continue

        data = list (zip (lambdas, fluxes))

        os.makedirs (output_dir, exist_ok=True)

        csv_file_path = os.path.join (output_dir, f"{mol_name}_spec_output.csv")

        with open (csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer (csv_file)
            csv_writer.writerow (["wave", "flux"])
            for row in data:
                csv_writer.writerow (row)

    # Get the fluxes and lambdas for the selected molecule
    fluxes = globals ().get ('total_fluxes', [])
    lambdas = globals ().get ('lambdas_h2o', np.array ([]))

    if len (fluxes) == 0 or lambdas.size == 0 or len (fluxes) != len (lambdas):
        return

    # Combine fluxes and lambdas into rows
    data = list (zip (lambdas, fluxes))

    # Create a directory if it doesn't exist
    os.makedirs (output_dir, exist_ok=True)

    # Specify the full path for the CSV file
    csv_file_path = os.path.join (output_dir, "SUM_spec_output.csv")

    # Create a CSV file with the selected data in the "MODELS" directory
    with open (csv_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer (csv_file)
        csv_writer.writerow (["wave", "flux"])
        for row in data:
            csv_writer.writerow (row)

    data_field.delete ('1.0', "end")
    data_field.insert ('1.0', f'All models exported!')


def generate_csv(mol_name):

    if mol_name == "SUM":

        # Get the fluxes and lambdas for the selected molecule
        fluxes = globals ().get ('total_fluxes', [])
        lambdas = globals ().get ('lambdas_h2o', np.array ([]))

        if len (fluxes) == 0 or lambdas.size == 0 or len (fluxes) != len (lambdas):
            return

        # Combine fluxes and lambdas into rows
        data = list (zip (lambdas, fluxes))

        # Create a directory if it doesn't exist
        os.makedirs (output_dir, exist_ok=True)

        # Specify the full path for the CSV file
        csv_file_path = os.path.join (output_dir, "SUM_spec_output.csv")

        # Create a CSV file with the selected data in the "MODELS" directory
        with open (csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer (csv_file)
            csv_writer.writerow (["wave", "flux"])
            for row in data:
                csv_writer.writerow (row)

        data_field.delete ('1.0', "end")
        data_field.insert ('1.0', 'SUM model exported!')

    if mol_name == "ALL":
        generate_all_csv ()

    else:
        # Find the tuple for the selected molecule
        molecule = next ((m for m in molecules_data if m[0] == mol_name), None)
        if molecule is None:
            return

        # Extract the lowercase version of the molecule name
        mol_name_lower = mol_name.lower ()

        # Get the fluxes and lambdas for the selected molecule
        fluxes = globals ().get (f'fluxes_{mol_name_lower}', np.array ([]))
        lambdas = globals ().get (f'lambdas_{mol_name_lower}', np.array ([]))
        line_prop = eval (f"{spanmol}_intensity.get_table")
        line_prop.to_csv (output_dir + '/' + f"{mol_name}_line_params.csv", index=False)

        if fluxes.size == 0 or lambdas.size == 0 or len (fluxes) != len (lambdas):
            return

        # Combine fluxes and lambdas into rows
        data = list (zip (lambdas, fluxes))
        # Create a directory if it doesn't exist
        os.makedirs (output_dir, exist_ok=True)

        # Specify the full path for the CSV file
        csv_file_path = os.path.join (output_dir, f"{mol_name}_spec_output.csv")

        # Create a CSV file with the selected data in the "MODELS" directory
        with open (csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer (csv_file)
            csv_writer.writerow (["wave", "flux"])
            for row in data:
                csv_writer.writerow (row)

        data_field.delete ('1.0', "end")
        data_field.insert ('1.0', f'{mol_name} model exported!')


def export_spectrum():
    # Create a new window for exporting the spectrum
    export_window = tk.Toplevel(root)
    export_window.title("Export Spectrum")

    # Create a label in the new window
    label = tk.Label(export_window, text="Select a molecule:")
    label.grid(row=0, column=0)

    # Create a dropdown menu in the new window
    options = [molecule[0] for molecule in molecules_data] + ["SUM"] + ["ALL"]
    dropdown_var = tk.StringVar()
    dropdown = ttk.Combobox(export_window, textvariable=dropdown_var, values=options)
    dropdown.set(options[0])
    dropdown.grid(row=1, column=0)

    # Create a button in the new window
    button = tk.Button(export_window, text="Generate CSV", command=lambda: generate_csv(dropdown_var.get()))
    button.grid(row=1, column=1)

# Create a dropdown menu in the new window
spanselectlab = tk.Label(plotparams_frame, text="Molecule:")
spanselectlab.grid(row=4, column=0, pady=(0,45))
spanoptionsvar = [m[0] for m in molecules_data] # + ["SUM"]
spandropdowntext = tk.StringVar()
spandropd = ttk.Combobox(plotparams_frame, textvariable=spandropdowntext, values=spanoptionsvar, width=6)
spandropd.set(spanoptionsvar[0])
spandropd.grid(row=4, column=1, pady=(0,45))

spandropd.bind("<<ComboboxSelected>>", lambda event: on_span_select((spanoptionsvar[spandropd.current()]).lower()))

# Create a button for submitting changed molecule for ax2
#spanbutton = tk.Button(plotparams_frame, text="Change Mol", command=lambda: on_span_select((spanoptionsvar[spandropd.current()]).lower()))
#spanbutton.grid(row=4, column=2)


spanmol = (spanoptionsvar[spandropd.current()]).lower()   


def toggle_fullscreen():
    state = window.attributes('-fullscreen')
    window.attributes('-fullscreen', not state)

# # Create a button to toggle fullscreen
# fullscreen_button = tk.Button(title_frame, text="Toggle Fullscreen", bg='blue', activebackground='darkblue', command=toggle_fullscreen, width=14, height=1)
# fullscreen_button.grid(row = 0, column = 0)

def import_molecule():
    MoleculeSelector(root, data_field)
    #data_field.delete ('1.0', "end")
    #data_field.insert ('1.0', 'New molecule downloaded from HITRAN.')

# Create a Tkinter button to import additional hitran molecules
import_button = tk.Button (title_frame, text="HITRAN query", bg='lightgray', activebackground='gray', command=import_molecule)
import_button.grid (row=0, column=0)

addmol_button = tk.Button(title_frame, text='Default Molecules', bg='lightgray', activebackground='gray', command=lambda: load_defaults_from_file(), width=12, height=1)
addmol_button.grid(row=0, column=1)

# Create the 'Add Mol.' button
addmol_button = tk.Button(title_frame, text='Add Molecule', bg='lightgray', activebackground='gray', command=lambda: add_molecule_data(), width=12, height=1)
addmol_button.grid(row=0, column=2)

# Create the 'Save Changes' button
saveparams_button = tk.Button(title_frame, text='Save Parameters', bg='lightgray', activebackground='gray', command=lambda: saveparams_button_clicked(), width=12, height=1)
saveparams_button.grid(row=0, column=3)

# Create the 'Load Save' button
loadparams_button = tk.Button(title_frame, text='Load Parameters', bg='lightgray', activebackground='gray', command=lambda: loadparams_button_clicked(), width=12, height=1)
loadparams_button.grid(row=0, column=4)

export_button = tk.Button(title_frame, text='Export Models', bg='lightgray', command=export_spectrum, width=12, height=1)
export_button.grid(row=0, column=5)

def toggle_legend():
    if ax1.legend_ is None:
        ax1.legend ()
    else:
        ax1.legend_.remove ()
    canvas.draw ()

# Create a Tkinter button to toggle the legend
toggle_button = tk.Button (title_frame, text="Toggle Legend", bg='lightgray', activebackground='gray', command=toggle_legend, width=12)
toggle_button.grid (row=0, column=6)


# Create and place the buttons for other functions
functions_frame = tk.Frame(window, borderwidth=2, relief="groove")
functions_frame.grid(row=plotparams_frame.grid_info()['row'] + plotparams_frame.grid_info()['rowspan'], column=0, rowspan=5, columnspan=5, sticky='nsew')

save_button = tk.Button(functions_frame, text="Save Line", bg='lightgray', activebackground='gray', command=Save, width=13, height=1)
save_button.grid(row=0, column=0)

fit_button = tk.Button(functions_frame, text="Fit Line", bg='lightgray', activebackground='gray', command=fit_onselect, width=13, height=1)
fit_button.grid(row=0, column=1)

savedline_button = tk.Button(functions_frame, text="Show Saved Lines", bg='lightgray', activebackground='gray', command=print_saved_lines, width=13, height=1)
savedline_button.grid(row=1, column=0)

fitsavedline_button = tk.Button(functions_frame, text="Fit Saved Lines", bg='lightgray', activebackground='gray', command=fit_saved_lines, width=13, height=1)
fitsavedline_button.grid(row=1, column=1)

autofind_button = tk.Button(functions_frame, text="Find Single Lines", bg='lightgray', activebackground='gray', command=single_finder, width=13, height=1)
autofind_button.grid(row=2, column=0)

# Create the 'Atomic lines' button
atomlines_button = tk.Button(functions_frame, text='Show Atomic Lines', bg='lightgray', activebackground='gray', command=lambda: print_atomic_lines(), width=13, height=1)
atomlines_button.grid(row=2, column=1)





"""
# Create a label (thin horizontal line) to fill the rest of row 0 with gray
for col in range(2, 10):  # Adjust the column range as needed
    label = tk.Label(window, bg='gray', borderwidth=0, highlightthickness=0)
    label.grid(row=0, column=col, sticky='ew', columnspan=999)
"""


# Create a button to update xp1 and rng
#loadparams_box = plt.axes([0.085, .975, 0.07, 0.02])
#loadparams_button = Button(loadparams_box, 'Load Save', color = background, hovercolor = 'lightgray')
#loadparams_button.on_clicked(loadparams_button_clicked) #loadparams_button_clicked)

# Dictionary to store the references to text boxes for each molecule
molecule_text_boxes = {}

def set_file_permissions(filename, mode):
    print (' ')
    print ('Molecule paths file: ...')

    try:
        os.chmod(filename, mode)
        print(f"Permissions set for {filename}")
    except Exception as e:
        print(f"File not found, permissions will be set when molecules are saved")

# Your script code here...

# After you write the data to the CSV file, call the function to set the permissions
csv_perm_path = os.path.join(save_folder, f"{file_name}-molsave.csv")
set_file_permissions(csv_perm_path, 0o666)  # Here, 0o666 sets read and write permissions for all users.

def write_to_csv(data, confirmation=False):
    if confirmation:
        # Display a confirmation dialog
        confirmed = tk.messagebox.askquestion("Confirmation", "Sure you want to save? This will overwrite any previous save for this data file.")
        if confirmed == "no":  # Check if user clicked "no"
            return

    csv_filename = os.path.join(save_folder, f"{file_name}-molsave.csv")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Molecule Name', 'File Path', 'Molecule Label', 'Temp', 'Rad', 'N_Mol', 'Color', 'Vis', 'Dist', 'StellarRV', 'FWHM', 'Broad']
            writer.writerow(header)

            for mol_name, mol_filepath, mol_label in data:
                row = [mol_name, mol_filepath, mol_label]
                linevar = eval(f"{mol_name.lower()}_line")
                linecolor = linevar.get_color()
                # Append the variables for the current molecule to the row
                row.append(globals().get(f"t_{mol_name.lower()}", ''))
                row.append(globals().get(f"{mol_name.lower()}_radius", ''))
                row.append(globals().get(f"n_mol_{mol_name.lower()}", ''))
                row.append(linecolor)
                row.append(globals().get(f"{mol_name.lower()}_vis", ''))
                row.append(dist)
                row.append(star_rv_entry.get())
                row.append(fwhm)
                row.append(intrinsic_line_width)
                
                writer.writerow(row)

        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Molecule parameters saved into file.')
        fig.canvas.draw_idle()
    except Exception as e:
        print("Error:", e)

def write_user_csv(data):

    csv_filename = os.path.join(save_folder, f"molecules_list.csv")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Molecule Name', 'File Path', 'Molecule Label', 'Temp', 'Rad', 'N_Mol', 'Color', 'Vis', 'Dist', 'StellarRV', 'FWHM', 'Broad']
            writer.writerow(header)

            for mol_name, mol_filepath, mol_label in data:
                row = [mol_name, mol_filepath, mol_label]
                linevar = eval(f"{mol_name.lower()}_line")
                linecolor = linevar.get_color()
                # Append the variables for the current molecule to the row
                row.append(globals().get(f"t_{mol_name.lower()}", ''))
                row.append(globals().get(f"{mol_name.lower()}_radius", ''))
                row.append(globals().get(f"n_mol_{mol_name.lower()}", ''))
                row.append(linecolor)
                row.append(globals().get(f"{mol_name.lower()}_vis", ''))
                row.append(dist)
                row.append(star_rv_entry.get())
                row.append(fwhm)
                row.append(intrinsic_line_width)
                
                writer.writerow(row)

        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Molecule parameters saved into file.')
        fig.canvas.draw_idle()
    except Exception as e:
        print("Error:", e)


def add_molecule_data():
    global mol_file_path
    global mol_file_name
    global wave_data
    global flux_data
    global filename_box_data
    global nextrow 
    global vis_button
    global vis_status
    #global text_box
    #global text_box_data
    global text_boxes
    global molecule_elements
    global deleted_molecules
    global molecules_data
    global files_frame
    global spanoptionsvar
    global spandropd
    
    molecule_elements = {} 
    
    if nextrow == 16:
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Maximum Molecules Reached!')
        #fig.canvas.draw_idle()
        #update()
        return
    
    # Define the filetypes to accept, in this case, only .par files
    molfiletypes = [('PAR Files', '*.par')]
    hitran_directory = os.path.abspath ("HITRANdata")

    # Ask the user to select a data file
    inmolfiles = filedialog.askopenfilename(multiple=True, title='Choose HITRAN Molecule Data File', filetypes=molfiletypes, initialdir=hitran_directory)

    if inmolfiles:
        for mol_file_path in inmolfiles:
            # Process each selected file
            mol_file_name = os.path.basename(file_path)

            # Ask the user to enter the molecule name
            molecule_name = simpledialog.askstring("Assign label", "Enter a label for this model (LaTeX and case sensitive):", parent=window)
            molecule_label = molecule_name

            # remove unaccepted characters
            #molecule_name = molecule_name.replace("-","_")
            molecule_name = molecule_name.translate({ord(i): None for i in '_$^{}'})
            molecule_name = molecule_name.translate({ord(i): "_" for i in ' -'})

            # Check if the molecule_name starts with a number
            if molecule_name[0].isdigit():
                # Add a "m_" to the beginning of the molecule name because python cannot take strings starting with a number
                molecule_name = 'm_' + molecule_name

            molecule_name = molecule_name.upper()
            
            if molecule_name:

                data_field.delete('1.0', "end")
                data_field.insert('1.0', 'Importing Molecule...')
                plt.draw(), canvas.draw()
                fig.canvas.flush_events() 

                
                # Specify the common directory to start the relative path from
                common_directory = "HITRANdata"
                
                script_directory = os.path.dirname(os.path.realpath(sys.argv[0] if hasattr(sys, 'frozen') else sys.executable))
                
                # Add the molecule name and file path to the molecules_data list
                relative_path = os.path.relpath(mol_file_path, start=script_directory)
                if common_directory in relative_path:
                    relative_path = os.path.join(common_directory, relative_path.split(common_directory, 1)[1].lstrip(os.path.sep)).replace('\\', '/')
                molecules_data.append((molecule_name, relative_path, molecule_label))
                
                #make sure molecule is no longer in deleted array
                if molecule_name.lower() in deleted_molecules:
                    deleted_molecules.remove(molecule_name.lower())
                
                # Import line lists from the ir_model folder
                mol_data = MolData(mol_name, mol_filepath)

                # Get the initial parameters for the current molecule, use default if not defined
                params = initial_parameters.get(mol_name, default_initial_params)
                scale_exponent = params["scale_exponent"]
                scale_number = params["scale_number"]
                t_kin = params["t_kin"]
                radius_init = params["radius_init"]

                # Calculate and set n_mol_init for the current molecule
                n_mol_init = float(scale_number * (10 ** scale_exponent))

                # Use exec() to create the variables with specific variable names for each molecule
                exec(f"mol_{molecule_name.lower()} = MolData('{molecule_name}', '{mol_file_path}')", globals())
                exec(f"scale_exponent_{molecule_name.lower()} = {scale_exponent}", globals())
                exec(f"scale_number_{molecule_name.lower()} = {scale_number}", globals())
                exec(f"n_mol_{molecule_name.lower()}_init = {n_mol_init}", globals())
                exec(f"t_kin_{molecule_name.lower()} = {t_kin}", globals())
                exec(f"{molecule_name.lower()}_radius_init = {radius_init}", globals())
                
                # Print the results (you can modify this part as needed)
                print(f"Molecule Added: {molecule_name}")
                #print(f"scale_exponent_{molecule_name.lower()} = {scale_exponent}")
                #print(f"scale_number_{molecule_name.lower()} = {scale_number}")
                #print(f"n_mol_{molecule_name.lower()}_init = {n_mol_init}")
                #print(f"t_kin_{molecule_name.lower()} = {t_kin}")
                #print(f"{molecule_name.lower()}_radius_init = {radius_init}")
                print()  # Empty line for spacing
                
                # Store the initial values in the dictionary
                initial_values[molecule_name.lower()] = {
                    "scale_exponent": scale_exponent,
                    "scale_number": scale_number,
                    "t_kin": t_kin,
                    "radius_init": radius_init,
                    "n_mol_init": n_mol_init
                }
                
                # Create a new row of text boxes for the current molecule
                row = nextrow
                y_row = start_y + row_height * (num_rows - row - 1)
                

                # Row label
                exec(f"{molecule_name.lower()}_rowl_field = tk.Entry(molecule_frame, width=6)", globals())
                eval(f"{molecule_name.lower()}_rowl_field").grid(row=row, column=0)
                eval(f"{molecule_name.lower()}_rowl_field").insert(0, f"{molecule_name}")
                molecule_elements[molecule_name.lower()] = {'rowl': molecule_name.lower() + '_rowl_field'}
                
                # Temperature input field
                globals()[f"{molecule_name.lower()}_temp_field"] = tk.Entry(molecule_frame, width=4)

                eval(f"{molecule_name.lower()}_temp_field").grid(row=row, column=1)
                eval(f"{molecule_name.lower()}_temp_field").insert(0, f"{t_kin}")
                #globals() [f"{molecule_name.lower()}_submit_temp_button"] = tk.Button(window, text="Submit", command=lambda mn=molecule_name.lower(), te = globals()[f"{molecule_name.lower()}_temp_field"]: submit_temp(te.get(), mn))
                #eval(f"{molecule_name.lower()}_submit_temp_button").grid(row=row + 1, column=2)
                molecule_elements[molecule_name.lower()] = {'temp': molecule_name.lower() + '_temp_field'}
                eval(f"{molecule_name.lower()}_temp_field").bind("<Return>", lambda event, mn=molecule_name.lower(), ce=globals()[f"{molecule_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
                
                # Radius input field
                globals()[f"{molecule_name.lower()}_rad_field"] = tk.Entry(molecule_frame, width=4)
                eval(f"{molecule_name.lower()}_rad_field").grid(row=row, column=2)
                eval(f"{molecule_name.lower()}_rad_field").insert(0, f"{radius_init}")
                #globals() [f"{molecule_name.lower()}_submit_rad_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), re = globals()[f"{molecule_name.lower()}_rad_field"]: submit_rad(re.get(), mn))
                #eval(f"{molecule_name.lower()}_submit_rad_button").grid(row=row + 1, column=4)
                molecule_elements[molecule_name.lower()]['rad'] = molecule_name.lower() + '_rad_field'
                eval(f"{molecule_name.lower()}_rad_field").bind("<Return>", lambda event, mn=molecule_name.lower(), ce=globals()[f"{molecule_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))
                
                # Column Density input field
                globals()[f"{molecule_name.lower()}_dens_field"] = tk.Entry(molecule_frame, width=6)
                eval(f"{molecule_name.lower()}_dens_field").grid(row=row, column=3)
                eval(f"{molecule_name.lower()}_dens_field").insert(0, f"{n_mol_init:.{1}e}")
                #globals() [f"{molecule_name.lower()}_submit_col_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{molecule_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
                #eval(f"{molecule_name.lower()}_submit_col_button").grid(row=row + 1, column=6)
                molecule_elements[molecule_name.lower()]['dens'] = molecule_name.lower() + '_dens_field'
                eval(f"{molecule_name.lower()}_dens_field").bind("<Return>", lambda event, mn=molecule_name.lower(), ce=globals()[f"{molecule_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
                
                # Visibility Button
                globals()[f"{molecule_name.lower()}_vis_status"] = tk.BooleanVar()
                globals()[f"{molecule_name.lower()}_vis_checkbutton"] = tk.Checkbutton(molecule_frame, text='', variable=eval(f"{molecule_name.lower()}_vis_status"), command=lambda mn=molecule_name.lower(): model_visible(mn))
                globals()[f"{molecule_name.lower()}_vis_status"].set(False)  # Set the initial state
                eval(f"{molecule_name.lower()}_vis_checkbutton").grid(row=row, column=4)
                globals()[f"{molecule_name.lower()}_vis"] = False
                # Add the variable to the globals dictionary
                # Add the text boxes to the molecule_text_boxes dictionary
                #molecule_text_boxes[molecule_name.lower()] = text_boxes
                
                #print(f"{mol_name.lower()}_rowl_field")
                
                del_button = tk.Button(molecule_frame, text="X", command=lambda widget=eval(f"{molecule_name.lower()}_rowl_field"): delete_row(widget))
                del_button.grid(row=row, column=5)
                
                color_button = tk.Button(molecule_frame, text=" ", command=lambda widget=eval(f"{molecule_name.lower()}_rowl_field"): choose_color(widget))
                color_button.grid(row=row, column=6)
                
                # Increment nextrow
                nextrow += 1
                
                exec(f"{molecule_name.lower()}_line, = ax1.plot([], [], alpha=0.8, linewidth=1)", globals())
                exec(f"{molecule_name.lower()}_line.set_label('{molecule_label}')", globals())
                
                line_var = globals().get(f"{molecule_name.lower()}_line")
                linecolor = line_var.get_color()
                # Get the color button from the grid_slaves list
                colorbutton = molecule_frame.grid_slaves(row=row, column=6)[0]

                # Set the background color of the color button
                colorbutton.configure(bg=linecolor)

                # Column density
                exec(f"global n_mol_{molecule_name.lower()}; n_mol_{molecule_name.lower()} = n_mol_{molecule_name.lower()}_init")

                # Temperature
                exec(f"global t_{molecule_name.lower()}; t_{molecule_name.lower()} = t_kin_{molecule_name.lower()}")

                # Radius
                exec(f"global {molecule_name.lower()}_radius; {molecule_name.lower()}_radius = {molecule_name.lower()}_radius_init")
                
                # Intensity calculation
                exec(f"{molecule_name.lower()}_intensity = Intensity(mol_{molecule_name.lower()})", globals())
                exec(f"{molecule_name.lower()}_intensity.calc_intensity(t_{molecule_name.lower()}, n_mol_{molecule_name.lower()}, dv=intrinsic_line_width)", globals())
                #print(f"{molecule_name.lower()}_intensity")
                # Add the variables to the globals dictionary
                globals()[f"{molecule_name.lower()}_intensity"] = eval(f"{molecule_name.lower()}_intensity")
                
                # Spectrum creation
                exec(f"{molecule_name.lower()}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

                # Adding intensity to the spectrum
                exec(f"{molecule_name.lower()}_spectrum.add_intensity({molecule_name.lower()}_intensity, {molecule_name.lower()}_radius ** 2 * np.pi)", globals())

                # Fluxes and lambdas
                exec(f"fluxes_{molecule_name.lower()} = {molecule_name.lower()}_spectrum.flux_jy; lambdas_{molecule_name.lower()} = {molecule_name.lower()}_spectrum.lamgrid", globals())

                # Dynamically set the data for each molecule's line using exec and globals()
                #exec(f"{molecule_name.lower()}_line.set_data(lambdas_{molecule_name.lower()}, fluxes_{molecule_name.lower()})", globals())
                
                # Save the molecules_data to the CSV file
                write_user_csv(molecules_data)
                
                update()
                

                # Clearing the text feed box.
                data_field.delete('1.0', "end")
                data_field.insert('1.0', 'Molecule Imported')
                canvas.draw(), plt.draw()
                #fig.canvas.flush_events() 
                
                #plt.pause(2)
                # Sleep for 3 seconds
                #time.sleep(3)
                canvas.draw()
                
                # Clearing the text feed box.
                data_field.delete('1.0', "end")
                
                canvas.draw()
                
                spanoptionsvar = [m[0] for m in molecules_data]
                spandropd['values'] = spanoptionsvar
                if spanoptionsvar:
                    spandropd.set(spanoptionsvar[0])
                
                
            else:
                print("Molecule label not provided.")
    else:
        print("No files selected.")
        
def del_molecule_data():
    
    global molecules_data, nextrow
    
    default_list = []
    try:
        with open(os.path.join(save_folder, "default.csv"), 'r') as list_file:
            reader = csv.reader(list_file)
            next(reader)  # Skip header
            for row in reader:
                default_list.append(tuple(row[:3]))  # Taking the first three columns of each row
    except Exception as e:
        print("Error reading molecules_list.csv:", e)

    # Create a set of molecule names from molecules_data
    default_data_names = set(item[0] for item in default_data)
    indexsub=0
    for row, (mol_name, _, _) in enumerate(molecules_data, start = 1):
        if mol_name not in default_data_names:
            adjrow = row - indexsub
            # Destroy all widgets in the row
            for w in molecule_frame.grid_slaves(row=adjrow):
                w.destroy()

            exec(f"{mol_name.lower()}_line.remove()", globals())

            # Remove the molecule from molecules_data
            molecules_data = [molecule for molecule in molecules_data if molecule[0].lower() != mol_name]


            # Move all rows below this row up by one
            for r in range(adjrow + 1, nextrow):
                for col in range(7):  # Adjust the range if you have more columns
                    widget_list = molecule_frame.grid_slaves(row=r, column=col)
                    for widget in widget_list:
                        widget.grid(row=r-1, column=col)
                        
            indexsub += 1

    
    write_user_csv(molecules_data)
    nextrow = 7
    update()

    data_field.delete ('1.0', "end")
    data_field.insert ('1.0', f'{mol_name.upper()} deleted!')



# Function to download extra data from HITRAN, to be finished..
def down_molecule_data(val):
    url = "https://hitran.org/lbl/"
    browsers = ["chrome", "edge", "firefox", "safari"]

    for browser_name in browsers:
        try:
            # Attempt to open the URL using webbrowser
            webbrowser.get(browser_name).open(url)
            break  # Stop trying if the browser opens the URL successfully
        except webbrowser.Error:
            try:
                # Fallback to using 'os' to execute the browser's command directly
                os.system(f"{browser_name} {url}")
                break  # Stop trying if the command succeeds
            except OSError:
                continue  # Try the next browser if the current one fails       
        
# sum button
#sum_box = plt.axes([0.16, .975, 0.07, 0.02])
#sum_button = Button(sum_box, 'Show Sum', color = background, hovercolor = background)
#sum_button.on_clicked(total_flux)    


# Create a frame for the Text widget
text_frame = tk.Frame(window)
text_frame.grid(row=functions_frame.grid_info()['row'] + functions_frame.grid_info()['rowspan'], column=0, columnspan=5, sticky='nsew')

# Create a Text widget within the frame
data_field = tk.Text(text_frame, wrap="word", height=13, width=24)
data_field.pack(fill="both", expand=True)




# Define the span selecting function of the tool
span = SpanSelector(
    ax1,
    onselect,
    "horizontal",
    useblit=False,
    props=dict(alpha=0.5, facecolor="lime"),
    interactive=True,
    drag_from_anywhere=True
)

# Storing the callback for on_xlims_change()
ax1.callbacks.connect('xlim_changed', on_xlims_change)
# Set the window manager to display the figure in a separate window
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
#ax1.figure.canvas.draw_idle()





#plt.show()



# Create a FigureCanvasTkAgg widget to embed the figure in the tkinter window
canvas = FigureCanvasTkAgg(fig, master=window)
canvas_widget = canvas.get_tk_widget()

# Place the canvas widget in column 9, row 1
canvas_widget.grid(row=1, column=5, rowspan=100, sticky='nsew')

# Allow column 9 and row 1 to expandc
window.grid_columnconfigure(5, weight=1)
window.grid_rowconfigure(100, weight=1)


# Create a frame for the toolbar inside the title_frame
toolbar_frame = tk.Frame(title_frame)
toolbar_frame.grid(row=0, column=9, columnspan=2, sticky="nsew")  # Place the frame in row 0, column 9
# Create a toolbar and update it
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

title_frame.grid_columnconfigure(9, weight=1)

plt.interactive(False)

update()

for row, (mol_name, _, _) in enumerate(molecules_data, start=1):
    # Get the molecule name in lower case
    mol_name_lower = mol_name.lower()
    
    # Get the line object
    line_var = globals().get(f"{mol_name_lower}_line")
    # Check if the line object exists and has a color attribute
    if line_var and hasattr(line_var, 'get_color'):
        # Get the color of the line
        line_color = line_var.get_color()
        
        # Get the color button from the grid_slaves list
        color_button = molecule_frame.grid_slaves(row=row, column=6)[0]
        
        # Set the background color of the color button
        color_button.configure(bg=line_color)
    else:
        print('Line object or color attribute not found for:', mol_name)
        
        
#save_default_to_file(file_name)
window.mainloop()