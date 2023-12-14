#!/usr/bin/env python
# coding: utf-8

# <div style="text-align:center;">
#   <h1><u>iSLAT: Interactive Spectral-Line Analysis Tool</u></h1>
# </div>
# 
# <img src="https://i.imgur.com/AQLsXrt.png" alt="Alt text" width="300">
# 
# #### Version Changelog
# Version - 3.05.00<br>
# Last Revision - 11/20/2023<br>
# ***
# - [Will be added in official 3.05.00 release]
# 
# ***

# ### Execute All At Once For Ease Of Use 

# In[ ]:


#%matplotlib qt

print('Loading iSLAT V3.05.00: Please Wait...')

# Import necessary modules
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
matplotlib.use("TKAgg")
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, SpanSelector, TextBox
from matplotlib.artist import Artist
import sys
from astropy.io import ascii, fits
from astropy.table import vstack, Table
from astropy import stats
from ir_model import *
import tkinter as tk
from tkinter import filedialog, simpledialog
import os
import inspect
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow
from datetime import datetime
from matplotlib.widgets import TextBox, CheckButtons  # Import CheckButtons
import csv
import time
import threading
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from tkinter import ttk  # For ttk.Style
    
#("NH3", "2020HITRANdata/data_Hitran_2020_NH3.par")

# Define the molecules and their corresponding file paths
molecules_data = [
    ("H2O", "2020HITRANdata/data_Hitran_2020_H2O.par"),
    ("OH", "2020HITRANdata/data_Hitran_2020_OH.par"),
    ("HCN", "2020HITRANdata/data_Hitran_2020_HCN.par"),
    ("C2H2", "2020HITRANdata/data_Hitran_2020_C2H2.par"),
    ("CO2", "2020HITRANdata/data_Hitran_2020_CO2.par"),
    ("CO", "2020HITRANdata/data_Hitran_2020_CO.par")
    # Add more molecules here if needed
]

deleted_molecules = []

def read_from_csv():
    if os.path.exists('molecules_data.csv'):
        try:
            with open('molecules_data.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip the header row
                return [tuple(row) for row in reader]
        except FileNotFoundError:
            pass
    return molecules_data

# Define the molecules and their corresponding file paths
molecules_data = read_from_csv()

# Set default initial parameters for a chemical
default_initial_params = {
    "scale_exponent": 17,
    "scale_number": 1,
    "t_kin": 600,
    "radius_init": 0.5
}

# Define the initial parameters for each molecule
initial_parameters = {
    "H2O": {
        "scale_exponent": 18,
        "scale_number": 1,
        "t_kin": 800,
        "radius_init": 0.60
    },
    "OH": {
        "scale_exponent": 17,
        "scale_number": 1,
        "t_kin": 3000,
        "radius_init": 0.1
    },
    "HCN": {
        "scale_exponent": 17,
        "scale_number": 1,
        "t_kin": 800,
        "radius_init": 0.3
    },
    "C2H2": {
        "scale_exponent": 17,
        "scale_number": 1,
        "t_kin": 700,
        "radius_init": 0.3
    }
}

import tkinter as tk

def get_variable_values():
    global min_lamb, max_lamb, dist, fwhm, intrinsic_line_width
    min_lamb = float(entry_widgets["Minimum_Wavelength_(µm)"].get())
    max_lamb = float(entry_widgets["Maximum_Wavelength_(µm)"].get())
    dist = float(entry_widgets["Distance_(parsec)"].get())
    fwhm = float(entry_widgets["Resolving_Power_FWHM_(km/s)"].get())
    intrinsic_line_width = float(entry_widgets["Intrinsic_Line_Width_(km/s)"].get())
    root.destroy()
       
    
#root = tk.Tk()
#root.title("Variable Definition")

# Set the window as a top-level window
#root.attributes("-topmost", True)

# Default values
default_values = {
    "Minimum_Wavelength_(µm)": 4.9,
    "Maximum_Wavelength_(µm)": 28.0,
    "Distance_(parsec)": 160.0,
    "Resolving_Power_FWHM_(km/s)": 130.0,
    "Intrinsic_Line_Width_(km/s)": 1.0
}

# Entry Widgets
#entry_widgets = {}
#for var_name, default_value in default_values.items():
#    tk.Label(root, text=f"{var_name.capitalize().replace('_', ' ')}:").pack()
#    entry = tk.Entry(root)
#    entry.insert(0, str(default_value))
#    entry.pack()
#    entry_widgets[var_name] = entry

# Button to submit values
#submit_button = tk.Button(root, text="Submit", command=get_variable_values)
#submit_button.pack()

#root.mainloop()

# Set-up constants to be inputs for model generation
min_lamb = 4.9
max_lamb = 28.
dist = 160.0
fwhm = 130. # FWHM of the observed lines or instrument

intrinsic_line_width = 1.0
cc = 2.99792458e5  # speed of light in km/s
model_line_width = cc / fwhm
model_pixel_res = (np.mean([min_lamb, max_lamb]) / cc * fwhm) / 5


# Dictionary to store the initial values for each chemical
initial_values = {}


# Loop through each molecule and set up the necessary objects and variables

for mol_name, mol_filepath in molecules_data:
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


# Constants used in generating the RV diagram
au = 1.496e11 # 1AU in m
pc = 3.08567758128e18 # From parsec to cm
ccum = 2.99792458e14 # speed of light in um s^-1
hh = 6.62606896e-27 # erg s

"""Creating all the functions used for the tool"""

"""
find_nearest() is a simple function that finds a value in a given array that matches closest to a value input. 
The function then returns the index of the value in the array. 
"""
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


"""
Next() and Prev() are the functions connected to the arrow buttons in the tool's UI. 
They jump the x-axis in their respective directions by an amount saved as the rng variable. 
This variable is an input from the user when starting up the tool.
"""
def Next(val):
    global rng
    global xp1
    global xp2
    xp1 = xp1+rng
    xp2 = xp2+rng
    update()
    
def Prev(val):
    global rng
    global xp1
    global xp2
    xp1 = xp1-rng
    xp2 = xp2-rng
    update()
    
"""
Save() is connected to the "Save Line" button of the tool.
This function apends information of the the strongest line (as determined by intensity) in the spanned area graph to a csv file. 
The name of the csv file is set with the "svd_line_file" variable in the second code block above. 
For the parameters of the line that is saved, refer to the "line2save" variable in onselect().
When starting the tool up, the "headers" variable is set to False. After apending a line to the csv for the first time, the "headers" variable is changed to False.
"""
def Save():
    global line2save
    global headers
    global saveline
    global linesavepath

    # This section is necessary for refreshing the text feed area to the left of the tool
    data_field.delete('1.0', "end")
    
    if saveline == True: # "saveline" variable is determined by whether or not an area was selected in the top graph or not 
         
        line2save.to_csv(linesavepath, mode='a', index=False, header=False)

        data_field.insert('1.0', 'Line Saved!')
        fig.canvas.draw_idle()
    else:
        data_field.insert('1.0', 'No Line Selected!')
        fig.canvas.draw_idle()
        return
    canvas.draw()


"""
reset() is connected to the "Reset" button of the tool.
Clicking this button will retrurn the temperature, column density, and radius values of the currently selected molecule to their initial values.
These initial values are located in the adjustable variables code block above and can be changed prior to starting up the tool.
"""
def reset(event):
    global h2o
    global oh
    global hcn
    global c2h2
    global t_kin_h2o
    global t_kin_oh
    global t_kin_hcn
    global t_kin_c2h2
    global n_mol_h2o
    global n_mol_oh
    global n_mol_hcn
    global n_mol_c2h2
    global h2o_radius
    global oh_radius
    global hcn_radius
    global c2h2_radius
    global h2o_radius_init
    global oh_radius_init
    global hcn_radius_init
    global c2h2_radius_init
    global skip
    if h2o == True:
        skip = True # This variable is defined in the main code block below
                    # It's essentially a workaround of a matplotlib limitation that prevents the model being build each time any of the parameters are reset/changed and only when the last (after skip is set to False)
                    # This prevents the tool from being very slow. See its use in update()
        temp_slider.set_val(t_kin_h2o)
        temp_slider.initial = t_kin_h2o
        rad_slider.initial = h2o_radius_init
        rad_slider.set_val(h2o_radius_init)
        text_box_col.initial = n_mol_h2o_init
        skip = False
        text_box_col.set_val(n_mol_h2o_init)
    if oh == True:
        skip = True
        temp_slider.set_val(t_kin_oh)
        temp_slider.initial = t_kin_oh
        rad_slider.initial= oh_radius_init
        rad_slider.set_val(oh_radius_init)
        text_box_col.initial = n_mol_oh_init
        skip = False
        text_box_col.set_val(n_mol_oh_init)
    if hcn == True:
        skip = True
        temp_slider.set_val(t_kin_hcn)
        temp_slider.initial = t_kin_hcn
        rad_slider.initial = hcn_radius_init
        rad_slider.set_val(hcn_radius_init)
        text_box_col.initial = n_mol_hcn_init
        skip = False
        text_box_col.set_val(n_mol_hcn_init)
    if c2h2 == True:
        skip = True
        temp_slider.set_val(t_kin_c2h2)
        temp_slider.initial = t_kin_c2h2
        rad_slider.initial = c2h2_radius_init
        rad_slider.set_val(c2h2_radius_init)
        text_box_col.initial = n_mol_c2h2_init
        skip = False
        text_box_col.set_val(n_mol_c2h2_init)
    
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
    
    
    
    
    global xp1
    global xp2
    global span
    global fluxes_h2o
    global lambdas_h2o
    global fluxes_oh
    global lambdas_oh
    global fluxes_hcn
    global lambdas_hcn
    global fluxes_c2h2
    global lambdas_c2h2
    global h2o_line_select
    global data_line_select
    global fig_height
    global fig_bottom_height
    global h2o_radius
    global n_mol
    global saveline
    global h2o
    global oh
    global hcn
    global c2h2
    global h2o_vis
    global oh_vis
    global hcn_vis
    global c2h2_vis
    global temp_slider
    global h2o_line
    global oh_line
    global hcn_line
    global c2h2_line
    global sum_line
    global t_h2o
    global t_oh
    global t_hcn
    global t_c2h2
    global n_mol_h2o
    global n_mol_oh
    global n_mol_hcn
    global n_mol_c2h2
    global h2o_intensity
    global int_pars
    global molecules_data
    global total_fluxes
    global dist
    global fwhm
    global max_lamd
    global min_lamd
    
    span.set_visible(False) # Clears the blue area created by the span selector (range selector in the top graph of the tool)
    saveline = False # See reference in save()

    # Clearing the text feed box.
    data_field.delete('1.0', "end")
    # Make empty lines for the second plot
    h2o_line_select, = ax2.plot([], [], color='royalblue',linewidth=3, ls='--')
    data_line_select, = ax2.plot([],[],color=foreground,linewidth=2)
    data_line, = ax1.plot([], [], color=foreground, linewidth=1)
    data_line.set_label('Data')
    
    # Make empty lines for the top graph for each molecule
    for mol_name, mol_filepath in molecules_data:
        molecule_name_lower = mol_name.lower()
        
        if molecule_name_lower in deleted_molecules:
            continue
        
        exec(f"{molecule_name_lower}_line, = ax1.plot([], [], alpha=1, linewidth=2, ls='--')", globals())
        exec(f"{molecule_name_lower}_line.set_label('{mol_name}')", globals())
    #sum_line, = ax1.plot([], [], color='gray', linewidth=1)
    #sum_line.set_label('Sum')
    ax1.legend()

    # h2o, oh, hcn, and c2h2 are variables that are set to True or false depending if the molecule is currently selected in the tool
    # If True, then that molecule's model is rebuilt with any new conditions (as set by the sliders or text input) that may have called the update() function
    # See h2o_select()
    for mol_name, mol_filepath in molecules_data:
        molecule_name_lower = mol_name.lower()

        # Intensity calculation
        exec(f"{molecule_name_lower}_intensity.calc_intensity(t_{molecule_name_lower}, n_mol_{molecule_name_lower}, dv=intrinsic_line_width)", globals())

        # Spectrum creation
        exec(f"{molecule_name_lower}_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist)", globals())

        # Adding intensity to the spectrum
        exec(f"{molecule_name_lower}_spectrum.add_intensity({molecule_name_lower}_intensity, {molecule_name_lower}_radius ** 2 * np.pi)", globals())

        # Fluxes and lambdas
        exec(f"fluxes_{molecule_name_lower} = {molecule_name_lower}_spectrum.flux_jy; lambdas_{molecule_name_lower} = {molecule_name_lower}_spectrum.lamgrid", globals())

    
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
    range_flux_cnts = observation_data[(observation_data['wave'] > xp1) & (observation_data['wave'] < xp2)]
    range_flux_cnts.index = range(len(range_flux_cnts.index))
    fig_height = np.max(range_flux_cnts.flux)
    fig_bottom_height = np.min(range_flux_cnts.flux)
    ax1.set_ylim(ymin=fig_bottom_height, ymax=fig_height+(fig_height/8))

    # Initialize total fluxes list
    total_fluxes = []
    # Calculate total fluxes based on visibility conditions
    for i in range(len(lambdas_h2o)):
        flux_sum = 0
        for mol_name, mol_filepath in molecules_data:
            

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
    for mol_name, mol_filepath in molecules_data:
        molecule_name_lower = mol_name.lower()

        # Dynamically set the data for each molecule's line using exec and globals()
        exec(f"{molecule_name_lower}_line.set_data(lambdas_{molecule_name_lower}, fluxes_{molecule_name_lower})", globals())
    data_line.set_data(wave_cnts, flux_cnts)
    
    
    #This is the opacity value that will be used for all shades (if you want to change the opacity just change this value)
    alpha_set = .2
    
    #ensures that the data line is always shaded
    
    
    """
    ax1.fill_between(wave_cnts, flux_cnts, 0,
    facecolor="black", # The fill color
    color='black',       # The outline color
    alpha=alpha_set)
    
    """

    # Determining the visibility of each molecule's line in the top graph and the shades
    # See model_visible()
    
    for mol_name, mol_filepath in molecules_data:
        molecule_name_lower = mol_name.lower()
        vis_status_var = globals()[f"{molecule_name_lower}_vis"]
        line_var = globals()[f"{molecule_name_lower}_line"]
        #lambdas_var = globals()[f"lambdas_{molecule_name_lower}"]
        #fluxes_var = globals()[f"fluxes_{molecule_name_lower}"]

        if vis_status_var:
            line_var.set_visible(True)
        else:
            line_var.set_visible(False)
        
    # Creating an array that contains the data of every line in the water model and reseting the index of this array
    # Reference: "ir_model" > "intensity.py"
    # int_pars is used to call up information on water lines like when using the spanning feature of the tool
    int_pars = h2o_intensity.get_table
    int_pars.index = range(len(int_pars.index))

    # Storing the callback for on_xlims_change()
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    
    # Storing the callback for the span selector
    span = SpanSelector(
    ax1,
    onselect,
    "horizontal",
    useblit=False,
    props=dict(alpha=0.5, facecolor="tab:blue"),
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
    
    global wave_cnts
    global flux_cnts
    global lambdas_h2o
    global fluxes_h2o
    global line2save
    global saveline
    
    xdif = xmax - xmin
    if xdif > 0:
        # Clearing the bottom two graphs
        # Reference: matplotlib.pyplot
        ax3.clear()
        ax2.clear()


        global wave_cnts
        global flux_cnts
        global lambdas_h2o
        global fluxes_h2o
        global line2save
        global saveline


        # Clearing the text feed box.
        data_field.delete('1.0', "end")

        # Repopulating the population diagram graph with all the lines of the water molecule (gray dots)
        pop_diagram()

        # Resetting the labels of graphs after they were deleted by the clear function above
        ax2.set_xlabel('Wavelength (μm)')
        ax2.set_ylabel('Flux density (Jy)')

        # Make empty lines for the zoom plot
        h2o_line_select, = ax2.plot([], [], color='red',linewidth=1)
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
            saveline = True
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

        # Defining the wavelengths and fluxes for the data spectrum
        wave_cnts = np.array(observation_data['wave'])
        flux_cnts = np.array(observation_data['flux'])

        # Calling the flux function to calculate the flux for the data in the range selected
        # Also printing the flux in the notebook for easy copying
        # See flux_integral
        f = flux_integral(wave_cnts, flux_cnts, xmin, xmax)
        print(f)

        data_field.delete('1.0', "end")
        data_field.insert('1.0', ('Strongest line:'+'\nup_lev = '+str(max_up_lev)+'\nlow_lev = '+str(max_low_lev)+'\nWavelength = '+str(max_lamb_cnts)+'\nIntensity = '+str(f'{max_intensity:.{3}f}')+'\nA_Einstein = '+str(max_einstein)+'\nE_up = '+str(f'{max_e_up:.{4}f}')+'\ng_up = '+str(max_g_up)+'\nflux = '+str(f)))
        # Creating a pandas dataframe for all the info of the strongest line in the selected range
        # This dataframe is used in the Save() function to save the strongest line in a csv file
        line2save = {'lev_up':[max_up_lev],'lev_low':[max_low_lev],'lam':[max_lamb_cnts],'tau':[max_tau],'intens':[max_intensity],'a_stein':[max_einstein],'e_up':[max_e_up],'g_up':[max_g_up],'xmin':[f'{xmin:.{4}f}'],'xmax':[f'{xmax:.{4}f}'], 'flux':[f]}
        line2save = pd.DataFrame(line2save)

        # Finding the index of the minimum and maximimum flux for both the data and model to be used in scaling the zoom graph (section below)
        model_indmin, model_indmax = np.searchsorted(lambdas_h2o, (xmin, xmax))
        data_indmin, data_indmax = np.searchsorted(wave_cnts, (xmin-.3, xmax+.3))
        model_indmax = min(len(lambdas_h2o) - 1, model_indmax)
        data_indmax = min(len(wave_cnts) - 1, data_indmax)

        # Scaling the zoom graph
        # First, it's determined if the max intensity of the model is bigger than that of the max intensity of the data or vice versa
        # Then, the max for the y-axis is determined by the max intensity of either the model or data, whichever is bigger
        # The minimum for the y-axis of the zoom graph is set to zero here
        model_region_x = lambdas_h2o[model_indmin:model_indmax]
        model_region_y = fluxes_h2o[model_indmin:model_indmax]
        data_region_x = wave_cnts[data_indmin:data_indmax]
        data_region_x = np.array(data_region_x)
        data_region_y = flux_cnts[data_indmin:data_indmax]
        data_region_y = np.array(data_region_y)
        max_model_y = 0
        max_data_y = 0
        min_data_y = 0
        for i in range(len(model_region_y)):
            if model_region_y[i] > max_model_y:
                max_model_y = model_region_y[i]
        for i in range(len(data_region_y)):
            if data_region_y[i] > max_data_y:
                max_data_y = data_region_y[i]
        for i in range(len(data_region_y)):
            if data_region_y[i] < min_data_y:
                min_data_y = data_region_y[i]
        if (max_model_y) >= (max_data_y):
            max_y = max_model_y
        else:
            max_y = max_data_y
        ax2.set_ylim(0, max_y)

        # This section prints vertical lines on the zoom graph at the wavelengths for each line in the model
        # The strongest line is colored differently than the other lines
        # The height of the lines represent the ratio of their intensities to the strongest line's intensity 
        # e.g. the strongest line is the tallest, a line that has 50% the int of the strongest line will be half as tall as that line
        if len(model_region_x) >= 1:
            k=0
            h2o_line_select.set_data(model_region_x, model_region_y)
            data_line_select.set_data(data_region_x, data_region_y)
            ax2.set_xlim(model_region_x[0], model_region_x[-1])

            for j in range(len(lamb_cnts)):
                if j == max_index:
                    k = j
                if j != max_index:
                    lineheight = (intensities[j]/max_intensity)*max_y
                    ax2.vlines(lamb_cnts[j], 0, lineheight, linestyles='dashed',color='green')
                    ax2.text(lamb_cnts[j], lineheight, (str(f'{intensities[j]:.{3}f}')+','+str(f'{e_up[j]:.{0}f}')+','+str(f'{einstein[j]:.{3}f}')), color = 'green', fontsize = 'small')
                    area = np.pi*(h2o_radius*au*1e2)**2 # In cm^2
                    Dist = dist*pc
                    beam_s = area/Dist**2
                    F = intensities[j]*beam_s
                    freq = ccum/lamb_cnts[j]
                    rd_yax = np.log(4*np.pi*F/(einstein[j]*hh*freq*g_up[j]))
                    ax3.scatter(e_up[j], rd_yax, s=30, color='green', edgecolors='black')
            lineheight = (intensities[k]/max_model_y)*max_model_y
            ax2.vlines(lamb_cnts[k], 0, lineheight, linestyles='dashed',color='orange')
            ax2.text(lamb_cnts[k], max_y, (str(f'{intensities[k]:.{3}f}')+','+str(f'{e_up[k]:.{0}f}')+','+str(f'{einstein[k]:.{3}f}')), color = 'orange', fontsize = 'small')
            area = np.pi*(h2o_radius*au*1e2)**2 # In cm^2
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
    #plt.show()

"""
group() is a function used in the single_finder() function. 
This function groups model lines together based on a threshold (thr) set when the function is called.
(a) is the array of lines being grouped
"""
def group(a,thr):
    x = np.sort(a)
    diff = x[1:]-x[:-1]
    gps = np.concatenate([[0],np.cumsum(diff>=thr)])
    return [x[gps==i] for i in range(gps[-1]+1)]

"""
single_finder() is connected to the "Find Singles" button.
This function is a filter that finds water lines in the model that are isolated then prints vertical lines in the top graph where these lines are located
e.g. they are either a set distance away from other lines, or the intensity of the lines near the line are negligible.
"""

specsep = .004
    
def single_finder():
    update()
    global fig_height
    global fig_bottom_height
    
    
    specsep = float(specsep_entry.get())
    
    # Resetting the text feed box
    data_field.delete('1.0', "end")
    # Getting all the water lines in the range of xp1 and xp2
    int_pars_line = int_pars[(int_pars['lam']>xp1) & (int_pars['lam']<xp2)]
    int_pars_line.index = range(len(int_pars_line.index))

    # Parsing the wavelengths and intensities of the lines in int_pars_line
    lamb_cnts = int_pars_line['lam']
    intensities = int_pars_line['intens']

    # Calling the group() function
    # See group()
    # The threshold for the grouping (in wavelength units) can be set here
    lambgroups = group(lamb_cnts, specsep)

    counter = 0
    for z in range(len(lambgroups)):

        # If there are no other lines around the isolated line (the group it's in is only made up of the one line)
        # Then print where that line is in the top graph
        if len(lambgroups[z]) == 1:
            idx = find_nearest(lamb_cnts,lambgroups[z]);
            if intensities[idx] >= 0.5:
                ax1.vlines(lamb_cnts[idx], fig_bottom_height, fig_height, linestyles='dashed',color='blue')
                counter = counter +1

        # For the groups that have more than one line (they are closer to eachother than the threshold set when group() was called)        
        if len(lambgroups[z]) > 1:
            overlap = False
            multigroup = lambgroups[z]
            intensgroup = []
            max_value = 0
            max_index = 0
            int_pars_line_group = int_pars[(int_pars['lam']>=multigroup[0]) & (int_pars['lam']<=multigroup[len(multigroup)-1])]
            int_pars_line_group.index = range(len(int_pars_line_group.index))
            lamb_cnts_group = int_pars_line_group['lam']
            intensities_group = int_pars_line_group['intens']

            # Find the max intensity in the lines
            for i in range(len(intensities_group)):
                if intensities_group[i] > max_value:
                    max_value = intensities_group[i]
                    max_index = i
            
            # Set the cutoff threshold for the other lines in determining if the max line is isolated 
            max_cutoff = max_value/10

            # Filter out potentially isolated lines that aren't very strong (user adjusts what they determine to be "strong")
            if max_value >= 0.5:

                # Determine if the line is truly isolated based on the intensities of the closeby lines
                for i in range(len(intensities_group)):
                    if (intensities_group[i] > max_cutoff) & (intensities_group[i] != max_value):
                        overlap = True
                if overlap == False:
                    ax1.vlines(lamb_cnts_group[max_index], fig_bottom_height, fig_height, linestyles='dashed',color='blue')
                    counter = counter +1

    # Storing the callback for on_xlims_change()
    ax1.callbacks.connect('xlim_changed', on_xlims_change)

    # Print the number of isolated lines that the function found in the region of xp1 and xp2
    if counter == 0:
        data_field.insert('1.0', 'No single lines found in the current wavelength range.')
    if counter > 0:
        data_field.insert('1.0', 'There are '+ str(counter)+' single lines \nfound in the current \nwavelength range.')
    canvas.draw()
    #plt.show()

"""
print_saved_lines() prints, as vertical dashed lines, on the top graph the locations of all lines saved to the current csv connected to the Save() function.
This csv can be changed in the user adjustable variables code block, but the change won't take into effect until the user regenerates the tool.
"""
def print_saved_lines():
    global linesavepath
    
    update()
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    
    svd_lns=pd.read_csv(linesavepath, sep=',')
    svd_lamb = np.array(svd_lns['lam'])
    for i in range(len(svd_lamb)):
        ax1.vlines(svd_lamb[i], -2, 2, linestyles='dashed',color='green')
    canvas.draw()
    
def print_atomic_lines():
    update()
    ax1.callbacks.connect('xlim_changed', on_xlims_change)
    
    svd_lns = pd.read_csv("ATOMLINES/Extra_line_list.csv", sep=',')
    svd_lamb = np.array(svd_lns['wave'])
    svd_species = svd_lns['species']

    for i in range(len(svd_lamb)):
        ax1.vlines(svd_lamb[i], -2, 2, linestyles='dashed', color='magenta')
        
        # Adjust the y-coordinate to place labels within the borders
        label_y = ax1.get_ylim()[1] - 0.17 * (ax1.get_ylim()[1] - ax1.get_ylim()[0])
        
        # Adjust the x-coordinate to place labels just to the right of the line
        label_x = svd_lamb[i] + 0.006 * (ax1.get_xlim()[1] - ax1.get_xlim()[0])
        
        ax1.text(label_x, label_y, svd_species[i], rotation=90, va='bottom', ha='left', color='magenta')

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
    global h2o_radius
    ax3.set_ylabel(r'ln(4πF/(hν$A_{u}$$g_{u}$))')
    ax3.set_xlabel(r'$E_{u}$')

    # Getting all the water lines in the range of min_lamb, max_lamb as set by the user in the adjustable variables code block
    int_pars = h2o_intensity.get_table
    int_pars.index = range(len(int_pars.index))

    # Parsing the components of the lines in int_pars
    wl = int_pars['lam']
    intens_mod = int_pars['intens']
    Astein_mod = int_pars['a_stein']
    gu = int_pars['g_up']
    eu = int_pars['e_up']

    # Calculating the y-axis for the population diagram for each line in int_pars
    area = np.pi*(h2o_radius*au*1e2)**2 # In cm^2
    Dist = dist*pc
    beam_s = area/Dist**2
    F = intens_mod*beam_s
    freq = ccum/wl
    rd_yax = np.log(4*np.pi*F/(Astein_mod*hh*freq*gu))

    ax3.set_ylim(-12,1)
    ax3.set_xlim(0,11000)

    # Populating the population diagram graph with the lines
    line6 = ax3.scatter(eu, rd_yax, s=0.5, color='#838B8B')
    #plt.show()

"""
submit_col() is connected to the text input box for adjusting the 
column density of the currently selected molecule
"""
def submit_col(event, text):
    
    global text_box
    global text_box_data

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
    data_field.insert('1.0', 'Density Submitted!')
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
        for mol_name, mol_filepath in molecules_data:
            

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
    global text_box
    global text_box_data
    global xp1
    global xp2
    global span
    global fluxes_h2o
    global lambdas_h2o
    global fluxes_oh
    global lambdas_oh
    global fluxes_hcn
    global lambdas_hcn
    global fluxes_c2h2
    global lambdas_c2h2
    global h2o_line_select
    global data_line_select
    global fig_height
    global fig_bottom_height
    global h2o_radius
    global n_mol
    global saveline
    global h2o
    global oh
    global hcn
    global c2h2
    global h2o_vis
    global oh_vis
    global hcn_vis
    global c2h2_vis
    global temp_slider
    global h2o_line
    global oh_line
    global hcn_line
    global c2h2_line
    global sum_line
    global t_h2o
    global t_oh
    global t_hcn
    global t_c2h2
    global n_mol_h2o
    global n_mol_oh
    global n_mol_hcn
    global n_mol_c2h2
    global h2o_intensity
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
    data_field.insert('1.0', 'Temperature Submitted!')
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
        for mol_name, mol_filepath in molecules_data:
            

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
    
    global text_box
    global text_box_data

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
    data_field.insert('1.0', 'Radius Submitted!')
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
        for mol_name, mol_filepath in molecules_data:
            

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
    cc = 2.99792458e8*1e6 # speed of light in um/s, to convert wavelengths into frequencies
    # calculate flux integral
    integral_range = np.where(np.logical_and(lam > lam_min, lam < lam_max))
    line_flux_meas = np.trapz(flux[integral_range[::-1]], x=cc/lam[integral_range[::-1]])
    line_flux_meas = -line_flux_meas*1e-23 # to get (erg s-1 cm-2); the minus sign is because we're getting a negative value, probably from the frequency array
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


"""
global ax1
global ax2
global ax3
global h2o_intensity
global h2o_radius
"""


h2o = True
oh = False
hcn = False
c2h2 = False
saveline = False

# Initialize visibility booleans for each molecule
molecule_names = [mol_name.lower() for mol_name, _ in molecules_data]
for mol_name in molecule_names:
    if mol_name == 'h2o':
        globals()[f"{mol_name}_vis"] = True
    else:
        globals()[f"{mol_name}_vis"] = False

for mol_name, mol_filepath in molecules_data:
    molecule_name_lower = mol_name.lower()

    # Column density
    exec(f"global n_mol_{molecule_name_lower}; n_mol_{molecule_name_lower} = n_mol_{molecule_name_lower}_init")

    # Temperature
    exec(f"global t_{molecule_name_lower}; t_{molecule_name_lower} = t_kin_{molecule_name_lower}")

    # Radius
    exec(f"global {molecule_name_lower}_radius; {molecule_name_lower}_radius = {molecule_name_lower}_radius_init")


root = tk.Tk()
root.withdraw()
root.call('wm', 'attributes', '.', '-topmost', True)

def selectfileinit():
    global file_path
    global file_name
    global wave_cnts
    global flux_cnts
    global observation_data
    global filename_box_data
    global svd_line_file
    global mode
    
    
    filetypes = [('CSV Files', '*.csv')]
    # Ask the user to select a file
    infiles = filedialog.askopenfilename(multiple=True, title='Choose Spectra Data File', filetypes=filetypes)

    if infiles:
        for file_path in infiles:
            # Process each selected file
            print("Selected file:", file_path)
            file_name = os.path.basename(file_path)
            # code to process each file
            observation_data = pd.read_csv(filepath_or_buffer=file_path, sep=',')
            wave_cnts = np.array(observation_data['wave'])
            flux_cnts = np.array(observation_data['flux'])
            
            now = datetime.now()
            dateandtime = now.strftime("%d-%m-%Y-%H-%M-%S")
            print(dateandtime)
            svd_line_file = f'savedlines-{dateandtime}.csv'
            
        # Ask the user to select the mode (light or dark)
        mode_dialog = tk.messagebox.askquestion("Select Mode", "Would you like to start iSLAT in Dark Mode?")
        
        if mode_dialog == 'yes':
            mode = True  # Dark mode
        else:
            mode = False  # Light mode
    else:
        print("No files selected.")

        
selectfileinit()
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

#wave_cnts = np.array(observation_data['wave'])
#flux_cnts = np.array(observation_data['flux'])

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
    global min_lamb, max_lamb, dist, fwhm, model_line_width, model_pixel_res, intrinsic_line_width
    # Get the values from the Tkinter Entry widgets and convert them to floats
    min_lamb = float(min_lamb_entry.get())
    max_lamb = float(max_lamb_entry.get())
    dist = float(dist_entry.get())
    fwhm = float(fwhm_entry.get())
    intrinsic_line_width = float(intrinsic_line_width_entry.get())
    model_line_width = cc / fwhm
    model_pixel_res = (np.mean([min_lamb, max_lamb]) / cc * fwhm) / 5
    print("Updated init vals")
    update()
    canvas.draw()

# Set initial values of xp1 and rng
xp1 = 13
rng = 1
xp2 = xp1 + rng
fig_max_limit = np.max(wave_cnts)
fig_min_limit = np.min(wave_cnts)


# Functing to limit the user input from the previous prompts to the range of the data you're inspecting
# The tool with stop and the associated dialogs will be printed in the serial line
if xp2 > fig_max_limit:
    sys.exit("Your wavelength range extends past the model, please start with a new range.")
if xp1 < fig_min_limit:
    sys.exit("Your wavelength range extends past the model, please start with a new range.")

# Set the headers for the saved lines csv to start at True
headers = True

for mol_name, mol_filepath in molecules_data:
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
data_line, = ax1.plot(wave_cnts, flux_cnts, color=foreground, linewidth=1)

for mol_name, mol_filepath in molecules_data:
        molecule_name_lower = mol_name.lower()
        
        if molecule_name_lower == 'h2o':
            exec(f"{molecule_name_lower}_line, = ax1.plot({molecule_name_lower}_spectrum.lamgrid, fluxes_{molecule_name_lower}, alpha=0.8, linewidth=1)", globals())
        else:
            exec(f"{molecule_name_lower}_line, = ax1.plot([], [], alpha=0.8, linewidth=1)", globals())
        exec(f"{molecule_name_lower}_line.set_label('{mol_name}')", globals())
data_line.set_label('Data')
sum_line, = ax1.plot([], [], color='purple', linewidth=1)
sum_line.set_label('Sum')
ax1.legend()
if oh_vis == False:
    oh_line.set_visible(False)
if hcn_vis == False:
    hcn_line.set_visible(False)
if c2h2_vis == False:
    c2h2_line.set_visible(False)

ax2.set_frame_on(False)
ax3.set_frame_on(False)

#start the program with h2o and the data line shaded

"""
ax1.fill_between(wave_cnts, flux_cnts, 0,
facecolor="black", # The fill color
color='black',       # The outline color
alpha=0.2)
"""

#ax1.fill_between(lambdas_h2o, fluxes_h2o, 0,
#facecolor="red", # The fill color
#color='red',       # The outline color
#alpha=0.2)

#make empty lines for the second plot
h2o_line_select, = ax2.plot([], [], color='red',linewidth=1)
data_line_select, = ax2.plot([],[],color=foreground,linewidth=1)

# Set the tool to start on fullscreen each time it's generated
#manager = plt.get_current_fig_manager()
#manager.full_screen_toggle()

#Scaling the y-axis based on tallest peak of data
range_flux_cnts = observation_data[(observation_data['wave'] > xp1) & (observation_data['wave'] < xp2)]
range_flux_cnts.index = range(len(range_flux_cnts.index))
fig_height = np.max(range_flux_cnts.flux)
fig_bottom_height = np.min(range_flux_cnts.flux)
ax1.set_ylim(ymin=fig_bottom_height, ymax=fig_height+(fig_height/8))

# adjust the plots to make room for the widgets
fig.subplots_adjust(left=0.06, right=0.97, top = 0.97, bottom=0.09)

# Populating the population diagram graph
pop_diagram()


#Create a border around the spectrum graph controls
#control_border = plt.axes([0.01, 0.52, 0.22, 0.45])
#control_border_params = Button(control_border, label='', color = background, hovercolor= background)

"""
# Make a button to select the H2O molecule to adjust
axh2o = fig.add_axes([0.035, 0.9, 0.035, 0.04]);
h2o_button = Button(axh2o, 'H2O', color='green', hovercolor='green')
h2o_button.on_clicked(h2o_select)

# Make a button to select the OH molecule to adjust
axoh = fig.add_axes([0.08, 0.9, 0.035, 0.04]);
oh_button = Button(axoh, 'OH', color= background, hovercolor= background)
oh_button.on_clicked(oh_select)

# Make a button to select the HCN molecule to adjust
axhcn = fig.add_axes([0.125, 0.9, 0.035, 0.04]);
hcn_button = Button(axhcn, 'HCN', color=background, hovercolor=background)
hcn_button.on_clicked(hcn_select)

# Make a button to select the C2H2 molecule to adjust
axc2h2 = fig.add_axes([0.17, 0.9, 0.035, 0.04]);
c2h2_button = Button(axc2h2, 'C2H2', color=background, hovercolor=background)
c2h2_button.on_clicked(c2h2_select)



# Make a button to make the currently selected model visible in the graph
axvisible = fig.add_axes([0.035, 0.85, 0.1, 0.04]);
vis_button = Button(axvisible, 'Visible', color='green', hovercolor='green')
vis_button.on_clicked(model_visible)

# Make a horizontal slider to control the temperature for the currently selected model.
axtemp = fig.add_axes([0.065, 0.64, 0.13, 0.02]);
temp_slider = TextBox(axtemp, 'Temprature', color = background, initial=temp_box, textalignment='left', hovercolor=background)
temp_slider.on_submit(submit_temp)

# Make a horizontal slider to control the temperature for the currently selected model.
axrad = fig.add_axes([0.065, 0.67, 0.13, 0.02]);
rad_slider = TextBox(axrad, 'Radius', color = background, initial=rad_box, textalignment='left', hovercolor=background)
rad_slider.on_submit(submit_rad)

# Make a text input to adjust the column density of the currently selected model.
axcol = plt.axes([0.11, 0.6, 0.08, 0.03])
text_box_col = TextBox(axcol, 'Col. Density: ', color = background, initial=n_mol, textalignment='left', hovercolor=background)
text_box_col.on_submit(submit_col)

"""

import tkinter as tk


num_rows = 9

# Calculate the height and width of each row
row_height = 0.035
row_width = 0.19

# Calculate the total height of all rows
total_height = row_height * num_rows

# Calculate the starting y-position for the first row within the control_border
start_y = 0.52 + (0.45 - total_height) / 2  # Center vertically
   
# Define the column labels
column_labels = ['Molecule', 'Temp.', 'Radius', 'Col. Dens', 'On']

# Create a dictionary to store the visibility buttons
vis_buttons_dict = {}

# Create a tkinter window
window = tk.Tk()
window.title("iSLAT V3.05.00")

# Define your data (molecules_data, initial_values, column_labels, etc.)

# Create the frame with the specified properties
param_frame = tk.Frame(window, borderwidth=2, relief="groove")
param_frame.grid(row=1, column=0, rowspan=10, columnspan=5, sticky="nsew")

# Create labels for columns
for col, label in enumerate(column_labels):
    label_widget = tk.Label(param_frame, text=label)
    label_widget.grid(row=0, column=col)

# Loop to create rows of input fields and buttons for each chemical
for row, (mol_name, mol_filepath) in enumerate(molecules_data):
    global nextrow
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
    exec(f"{mol_name.lower()}_rowl_field = tk.Entry(param_frame, width=6)")
    eval(f"{mol_name.lower()}_rowl_field").grid(row=row + 1, column=0)
    eval(f"{mol_name.lower()}_rowl_field").insert(0, f"{mol_name}")

    # Temperature input field
    exec(f"{mol_name.lower()}_temp_field = tk.Entry(param_frame, width=4)")
    eval(f"{mol_name.lower()}_temp_field").grid(row=row + 1, column=1)
    eval(f"{mol_name.lower()}_temp_field").insert(0, f"{t_kin}")
    globals() [f"{mol_name.lower()}_submit_temp_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
    #eval(f"{mol_name.lower()}_submit_temp_button").grid(row=row + 1, column=2)
    eval(f"{mol_name.lower()}_temp_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
    
    # Radius input field
    exec(f"{mol_name.lower()}_rad_field = tk.Entry(param_frame, width=4)")
    eval(f"{mol_name.lower()}_rad_field").grid(row=row + 1, column=2)
    eval(f"{mol_name.lower()}_rad_field").insert(0, f"{radius_init}")
    globals() [f"{mol_name.lower()}_submit_rad_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))
    #eval(f"{mol_name.lower()}_submit_rad_button").grid(row=row + 1, column=4)
    eval(f"{mol_name.lower()}_rad_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))
    
    # Column Density input field
    exec(f"{mol_name.lower()}_dens_field = tk.Entry(param_frame, width=6)")
    eval(f"{mol_name.lower()}_dens_field").grid(row=row + 1, column=3)
    eval(f"{mol_name.lower()}_dens_field").insert(0, f"{n_mol_init}")
    globals() [f"{mol_name.lower()}_submit_col_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
    #eval(f"{mol_name.lower()}_submit_col_button").grid(row=row + 1, column=6)
    eval(f"{mol_name.lower()}_dens_field").bind("<Return>", lambda event, mn=mol_name.lower(), ce=globals()[f"{mol_name.lower()}_dens_field"]: submit_col(ce.get(), mn))


    # Visibility Checkbutton
    if mol_name.lower() == 'h2o':
        exec(f"{mol_name.lower()}_vis_status = tk.BooleanVar()")
        exec(f"{mol_name.lower()}_vis_status.set(True)")  # Set the initial state
        exec(f"{mol_name.lower()}_vis_checkbutton = tk.Checkbutton(param_frame, text='', variable={mol_name.lower()}_vis_status, onvalue=True, offvalue=False, command=lambda mn=mol_name.lower(): model_visible(mn))")
        exec(f"{mol_name.lower()}_vis_checkbutton.select()")
    else:
        globals()[f"{mol_name.lower()}_vis_status"] = tk.BooleanVar()
        globals()[f"{mol_name.lower()}_vis_checkbutton"] = tk.Checkbutton(param_frame, text='', variable=eval(f"{mol_name.lower()}_vis_status"), command=lambda mn=mol_name.lower(): model_visible(mn))
        globals()[f"{mol_name.lower()}_vis_status"].set(False)  # Set the initial state
        #globals()[f"{mol_name.lower()}_vis"] = eval(f"{mol_name.lower()}_vis_status")

    eval(f"{mol_name.lower()}_vis_checkbutton").grid(row=row + 1, column=4)
    nextrow = row + 1

# Add a slider if needed
# slider = tk.Scale(window, from_=0, to=10, orient=tk.HORIZONTAL, label="Radius")
# slider.grid(row=row + 2, column=0, columnspan=8)

#window.mainloop()


# Make a slider to adjust the radius of the currently slected model.

"""
axrad = fig.add_axes([0.065, 0.67, 0.13, 0.02]);
rad_slider = Slider(
    ax=axrad,
    label="Radius",
    valmin=0.01,
    valmax=2,
    valinit=h2o_radius,
    color = 'green'
)
"""


# Register the update function with each slider
#temp_slider.on_changed(update)
#rad_slider.on_changed(update)
"""
# Create a separate Matplotlib figure for the text box
text_box_fig, text_box = plt.subplots(figsize=(6, 4))
text_box.axis('off')  # Turn off axes for the text box
text_box_data = TextBox(text_box, label='', color='white', hovercolor='white')  # Adjust colors as needed

# Create a FigureCanvasTkAgg widget to embed the text box figure in the tkinter window
text_canvas = FigureCanvasTkAgg(text_box_fig, master=window)
text_canvas_widget = text_canvas.get_tk_widget()

# Place the canvas widget in row 11, column 0, spanning 20 rows
text_canvas_widget.grid(row=11, column=0, rowspan=20, columnspan=8)
"""


# Create a button to reset the sliders to initial values.
#resetax = fig.add_axes([0.07, 0.53, 0.1, 0.04]);
#reset_button = Button(resetax, 'Add Molecule',color=background, hovercolor=background)
#reset_button.on_clicked(reset)

#addmol_box = fig.add_axes([0.07, 0.53, 0.1, 0.04]);
#addmol_button = Button(addmol_box, 'Add Molecule', color = background, hovercolor = 'lightgray')
#addmol_button.on_clicked(add_molecule_data) #loadparams_button_clicked)

#Create a button to go to next wavelength range
#axnext = fig.add_axes([0.18, 0.53, 0.03, 0.04]);
#next_button = Button(axnext, r'$\rightarrow$',color=background, hovercolor='lightgray')
#next_button.on_clicked(Next)

#Create a button to go to previous wavelength range
#axprev = fig.add_axes([0.03, 0.53, 0.03, 0.04]);
#prev_button = Button(axprev, r'$\leftarrow$',color=background, hovercolor='lightgray')
#prev_button.on_clicked(Prev)



# Create a TextBox to print the name of the current data file being observed
#filename_box = plt.axes([0.01, 0.47, 0.22, 0.04])
#filename_box_data = TextBox(filename_box, initial=(str(file_name)), label='', color = 'gray', hovercolor='gray')

variable_names = ['t_h2o', 'h2o_radius', 'n_mol_h2o', 't_oh', 'oh_radius', 'n_mol_oh', 't_hcn', 'hcn_radius', 'n_mol_hcn', 't_c2h2', 'c2h2_radius', 'n_mol_c2h2']


def save_variables_to_file(file_name, variable_names, *variables):
    filename = f"{file_name}-save.txt"
    with open(filename, 'w') as file:
        for mol_name, mol_filepath in molecules_data:
            file.write(f"t_{mol_name.lower()}: {globals()['t_' + mol_name.lower()]}\n")
            file.write(f"{mol_name.lower()}_radius: {globals()[mol_name.lower() + '_radius']}\n")
            file.write(f"n_mol_{mol_name.lower()}: {globals()['n_mol_' + mol_name.lower()]}\n")

    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Changes Saved!')
    fig.canvas.draw_idle()
#    filename = f"{file_name}-save.txt"
#    with open(filename, 'w') as file:
#        for index, var in enumerate(variables, start=1):
#            file.write(f"Variable {index}: {var}\n")

def loadsavedmessage(var):
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Save Loaded!')
    fig.canvas.draw_idle()

def saveparams_button_clicked():
    save_variables_to_file(file_name, variable_names, t_h2o, h2o_radius, n_mol_h2o, t_oh, oh_radius, n_mol_oh, t_hcn, hcn_radius, n_mol_hcn, t_c2h2, c2h2_radius, n_mol_c2h2)

def load_variables_from_file(file_name):
    global text_box_data
    global text_box    
    global fluxes_h2o
    global lambdas_h2o
    global fluxes_oh
    global lambdas_oh
    global fluxes_hcn
    global lambdas_hcn
    global fluxes_c2h2
    global lambdas_c2h2
    
    
    filename = f"{file_name}-save.txt"
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        variable_name, variable_value = line.strip().split(': ')
        exec(f"global {variable_name}\n{variable_name} = {variable_value}")
    
    for mol_name, mol_filepath in molecules_data:
        eval(f"{mol_name.lower()}_temp_field").delete(0, "end")
        eval(f"{mol_name.lower()}_temp_field").insert(0, eval(f"t_{mol_name.lower()}"))
        
        eval(f"{mol_name.lower()}_rad_field").delete(0, "end")
        eval(f"{mol_name.lower()}_rad_field").insert(0, eval(f"{mol_name.lower()}_radius"))
        
        eval(f"{mol_name.lower()}_dens_field").delete(0, "end")
        eval(f"{mol_name.lower()}_dens_field").insert(0, eval(f"n_mol_{mol_name.lower()}"))
        
        #exec(f"{mol_name.lower()}_temp_field.insert('1.0', t_{mol_name.lower()})")
    
    fill_h2o = ax1.fill_between(lambdas_h2o, fluxes_h2o, 0,
            facecolor="red", # The fill color
            color='red',       # The outline color
            alpha=0.2)
    fill_c2h2 = ax1.fill_between(lambdas_c2h2, fluxes_c2h2, 0,
            facecolor="blue", # The fill color
            color='blue',       # The outline color
            alpha=0)
    fill_oh = ax1.fill_between(lambdas_oh, fluxes_oh, 0,
            facecolor="orange", # The fill color
            color='orange',       # The outline color
            alpha=0)
    fill_hcn = ax1.fill_between(lambdas_hcn, fluxes_hcn, 0,
            facecolor="green", # The fill color
            color='green',       # The outline color
            alpha=0)
    
    h2o_intensity.calc_intensity(t_h2o, n_mol_h2o, dv=intrinsic_line_width)
    h2o_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist) 
    h2o_spectrum.add_intensity(h2o_intensity, h2o_radius ** 2 * np.pi)
    fluxes_h2o = h2o_spectrum.flux_jy
    lambdas_h2o = h2o_spectrum.lamgrid

    oh_intensity.calc_intensity(t_oh,n_mol_oh, dv=intrinsic_line_width)
    oh_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist) 
    oh_spectrum.add_intensity(oh_intensity, oh_radius ** 2 * np.pi)
    fluxes_oh = oh_spectrum.flux_jy
    lambdas_oh = oh_spectrum.lamgrid

    hcn_intensity.calc_intensity(t_hcn,n_mol_hcn, dv=intrinsic_line_width)
    hcn_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist) 
    hcn_spectrum.add_intensity(hcn_intensity, hcn_radius ** 2 * np.pi)
    fluxes_hcn = hcn_spectrum.flux_jy
    lambdas_hcn = hcn_spectrum.lamgrid

    c2h2_intensity.calc_intensity(t_c2h2,n_mol_c2h2, dv=intrinsic_line_width)
    c2h2_spectrum = Spectrum(lam_min=min_lamb, lam_max=max_lamb, dlambda=model_pixel_res, R=model_line_width, distance=dist) 
    c2h2_spectrum.add_intensity(c2h2_intensity, c2h2_radius ** 2 * np.pi)
    fluxes_c2h2 = c2h2_spectrum.flux_jy
    lambdas_c2h2 = c2h2_spectrum.lamgrid
        
    oh_line.set_data(lambdas_oh, fluxes_oh)
    hcn_line.set_data(lambdas_hcn, fluxes_hcn)
    c2h2_line.set_data(lambdas_c2h2, fluxes_c2h2)
    h2o_line.set_data(lambdas_h2o, fluxes_h2o)
    data_line.set_data(wave_cnts, flux_cnts)
    
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Save Loaded!')
    fig.canvas.draw_idle()
    time.sleep(3)  # Pause for 3 seconds
    update()
    
def loadparams_button_clicked():
    load_variables_from_file(file_name)

param2_frame = tk.Frame(window, borderwidth=2, relief="groove")
param2_frame.grid(row=16, column=0, rowspan=5, columnspan=5, sticky="nsew")
    

# Create and place the xp1 text box in row 12, column 0
xp1_label = tk.Label(param2_frame, text="Plot start:")
xp1_label.grid(row=0, column=0)
xp1_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
xp1_entry.insert(0, str(xp1))
xp1_entry.grid(row=0, column=1)
xp1_entry.bind("<Return>", lambda event: update_xp1_rng())

# Create and place the rng text box in row 12, column 2
rng_label = tk.Label(param2_frame, text="Plot range:")
rng_label.grid(row=0, column=2)
rng_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
rng_entry.insert(0, str(rng))
rng_entry.grid(row=0, column=3)
rng_entry.bind("<Return>", lambda event: update_xp1_rng())

# Create and place the xp1 text box in row 12, column 0
specsep_label = tk.Label(param2_frame, text="Line Separ.:")
specsep_label.grid(row=1, column=0)
specsep_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
specsep_entry.insert(0, str(specsep))
specsep_entry.grid(row=1, column=1)

# Create and place the min_lamb text box in row 2, column 0
min_lamb_label = tk.Label(param2_frame, text="Min. Wave:")
min_lamb_label.grid(row=1, column=2)
min_lamb_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
min_lamb_entry.insert(0, str(min_lamb))
min_lamb_entry.grid(row=1, column=3)
min_lamb_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the max_lamb text box in row 2, column 2
max_lamb_label = tk.Label(param2_frame, text="Max. Wave:")
max_lamb_label.grid(row=2, column=0)
max_lamb_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
max_lamb_entry.insert(0, str(max_lamb))
max_lamb_entry.grid(row=2, column=1)
max_lamb_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the dist text box in row 3, column 0
dist_label = tk.Label(param2_frame, text="Distance:")
dist_label.grid(row=2, column=2)
dist_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
dist_entry.insert(0, str(dist))
dist_entry.grid(row=2, column=3)
dist_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the fwhm text box in row 3, column 2
fwhm_label = tk.Label(param2_frame, text="FWHM:")
fwhm_label.grid(row=3, column=0)
fwhm_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
fwhm_entry.insert(0, str(fwhm))
fwhm_entry.grid(row=3, column=1)
fwhm_entry.bind("<Return>", lambda event: update_initvals())

# Create and place the fwhm text box in row 3, column 2
intrinsic_line_width_label = tk.Label(param2_frame, text="Line width:")
intrinsic_line_width_label.grid(row=3, column=2)
intrinsic_line_width_entry = tk.Entry(param2_frame, bg='lightgray', width=8)
intrinsic_line_width_entry.insert(0, str(intrinsic_line_width))
intrinsic_line_width_entry.grid(row=3, column=3)
intrinsic_line_width_entry.bind("<Return>", lambda event: update_initvals())

# Add some space below param2_frame
tk.Label(param2_frame, text="").grid(row=4, column=0)

def generate_all_csv():
    for molecule in molecules_data:
        mol_name = molecule[0]
        mol_name_lower = mol_name.lower()

        fluxes = globals().get(f'fluxes_{mol_name_lower}', np.array([]))
        lambdas = globals().get(f'lambdas_{mol_name_lower}', np.array([]))

        if fluxes.size == 0 or lambdas.size == 0 or len(fluxes) != len(lambdas):
            continue

        data = list(zip(lambdas, fluxes))

        output_dir = "OUTPUT"
        os.makedirs(output_dir, exist_ok=True)

        csv_file_path = os.path.join(output_dir, f"{mol_name}_spec_output.csv")

        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["wave", "flux"])
            for row in data:
                csv_writer.writerow(row)
    
    # Get the fluxes and lambdas for the selected molecule
    fluxes = globals().get('total_fluxes', [])
    lambdas = globals().get('lambdas_h2o', np.array([]))

    if len(fluxes) == 0 or lambdas.size == 0 or len(fluxes) != len(lambdas):
        return

    # Combine fluxes and lambdas into rows
    data = list(zip(lambdas, fluxes))

    # Create a directory if it doesn't exist
    output_dir = "OUTPUT"
    os.makedirs(output_dir, exist_ok=True)

    # Specify the full path for the CSV file
    csv_file_path = os.path.join(output_dir, "SUM_spec_output.csv")

    # Create a CSV file with the selected data in the "OUTPUT" directory
    with open(csv_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["wave", "flux"])
        for row in data:
            csv_writer.writerow(row)

    data_field.delete('1.0', "end")
    data_field.insert('1.0', f'All lines exported!')

def generate_csv(mol_name):
    if mol_name == "SUM":
        
        # Get the fluxes and lambdas for the selected molecule
        fluxes = globals().get('total_fluxes', [])
        lambdas = globals().get('lambdas_h2o', np.array([]))

        if len(fluxes) == 0 or lambdas.size == 0 or len(fluxes) != len(lambdas):
            return

        # Combine fluxes and lambdas into rows
        data = list(zip(lambdas, fluxes))

        # Create a directory if it doesn't exist
        output_dir = "OUTPUT"
        os.makedirs(output_dir, exist_ok=True)

        # Specify the full path for the CSV file
        csv_file_path = os.path.join(output_dir, "SUM_spec_output.csv")

        # Create a CSV file with the selected data in the "OUTPUT" directory
        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["wave", "flux"])
            for row in data:
                csv_writer.writerow(row)
                
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'SUM line exported!')
        
    if mol_name == "ALL":
        generate_all_csv()
        
    else:
        # Find the tuple for the selected molecule
        molecule = next((m for m in molecules_data if m[0] == mol_name), None)
        if molecule is None:
            return

        # Extract the lowercase version of the molecule name
        mol_name_lower = mol_name.lower()

        # Get the fluxes and lambdas for the selected molecule
        fluxes = globals().get(f'fluxes_{mol_name_lower}', np.array([]))
        lambdas = globals().get(f'lambdas_{mol_name_lower}', np.array([]))

        if fluxes.size == 0 or lambdas.size == 0 or len(fluxes) != len(lambdas):
            return

        # Combine fluxes and lambdas into rows
        data = list(zip(lambdas, fluxes))
        # Create a directory if it doesn't exist
        output_dir = "OUTPUT"
        os.makedirs(output_dir, exist_ok=True)

        # Specify the full path for the CSV file
        csv_file_path = os.path.join(output_dir, f"{mol_name}_spec_output.csv")

        # Create a CSV file with the selected data in the "OUTPUT" directory
        with open(csv_file_path, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["wave", "flux"])
            for row in data:
                csv_writer.writerow(row)
                        
        data_field.delete('1.0', "end")
        data_field.insert('1.0', f'{mol_name} line exported!')
        
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


nb_of_columns = 10 # to be replaced by the relevant number
titleframe = tk.Frame(window, bg ="gray")
titleframe.grid(row=0, column=0, columnspan=nb_of_columns, sticky='ew')

def toggle_fullscreen():
    state = window.attributes('-fullscreen')
    window.attributes('-fullscreen', not state)

# Create a button to toggle fullscreen
fullscreen_button = tk.Button(titleframe, text="Toggle Fullscreen", bg='blue', activebackground='darkblue', command=toggle_fullscreen, width=12, height=1)
fullscreen_button.grid(row = 0, column = 0)

# Create the 'Save Changes' button
saveparams_button = tk.Button(titleframe, text='Save Parameters', bg='lightgray', activebackground='gray', command=lambda: saveparams_button_clicked(), width=12, height=1)
saveparams_button.grid(row=0, column=1)

# Create the 'Load Save' button
loadparams_button = tk.Button(titleframe, text='Load Parameters', bg='lightgray', activebackground='gray', command=lambda: loadparams_button_clicked(), width=12, height=1)
loadparams_button.grid(row=0, column=2)


export_button = tk.Button(titleframe, text='Export Models', bg='lightgray', command=export_spectrum, width=12, height=1)
export_button.grid(row=0, column=3)

buttonframe = tk.Frame(window)
buttonframe.grid(row=21, column=0, rowspan=1, columnspan=5, sticky='nsew')

# Create the 'Add Mol.' button
addmol_button = tk.Button(buttonframe, text='Add Molecule', bg='lightgray', activebackground='gray', command=lambda: add_molecule_data(), width=13, height=1)
addmol_button.grid(row=2, column=0)

# Create the 'Clear Mol.' button
clearmol_button = tk.Button(buttonframe, text='Clear Molecules', bg='lightgray', activebackground='gray', command=lambda: del_molecule_data(), width=13, height=1)
clearmol_button.grid(row=2, column=1)

# Create the 'Clear Mol.' button
atomlines_button = tk.Button(buttonframe, text='Show Atom. Lines', bg='lightgray', activebackground='gray', command=lambda: print_atomic_lines(), width=13, height=1)
atomlines_button.grid(row=1, column=1)

#delmol_box = fig.add_axes([0.16, .975, 0.07, 0.02]);
#delmol_button = Button(delmol_box, 'Clear Molecules', color = background, hovercolor = 'lightgray')
#delmol_button.on_clicked(del_molecule_data) #loadparams_button_clicked)



# Create and place the buttons in the specified rows and columns
save_button = tk.Button(buttonframe, text="Save Line", bg='lightgray', activebackground='gray', command=Save, width=13, height=1)
save_button.grid(row=0, column=0)

autofind_button = tk.Button(buttonframe, text="Find Single Lines", bg='lightgray', activebackground='gray', command=single_finder, width=13, height=1)
autofind_button.grid(row=0, column=1)

savedline_button = tk.Button(buttonframe, text="Show Saved Lines", bg='lightgray', activebackground='gray', command=print_saved_lines, width=13, height=1)
savedline_button.grid(row=1, column=0)


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
    try:
        os.chmod(filename, mode)
        print(f"Permissions set for {filename}")
    except Exception as e:
        print(f"Error setting permissions for {filename}: {e}")

# Your script code here...

# After you write the data to the CSV file, call the function to set the permissions
set_file_permissions("molecules_data.csv", 0o666)  # Here, 0o666 sets read and write permissions for all users.

def write_to_csv(data):
    with open('molecules_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Molecule Name', 'File Path'])
        writer.writerows(data)

def add_molecule_data():
    global mol_file_path
    global mol_file_name
    global wave_cnts
    global flux_cnts
    global filename_box_data
    global nextrow 
    global vis_button
    global vis_status
    global text_box
    global text_box_data
    global text_boxes
    global molecule_elements
    import csv
    global deleted_molecules
    global molecules_data
    global files_frame
    
    molecule_elements = {} 
    
    if nextrow == 16:
        data_field.delete('1.0', "end")
        data_field.insert('1.0', 'Maximum Molecules Reached!')
        #fig.canvas.draw_idle()
        #update()
        return
    
    # Define the filetypes to accept, in this case, only .par files
    molfiletypes = [('PAR Files', '*.par')]
    
    # Ask the user to select a data file
    inmolfiles = filedialog.askopenfilename(multiple=True, title='Choose HITRAN Molecule Data File', filetypes=molfiletypes)

    if inmolfiles:
        for mol_file_path in inmolfiles:
            # Process each selected file
            mol_file_name = os.path.basename(file_path)

            # Ask the user to enter the molecule name
            molecule_name = simpledialog.askstring("Molecule Name", "Enter the molecule name (not case sensitive):", parent=window)
            
            # Check if the molecule_name starts with a number
            if molecule_name[0].isdigit():
                # Add 'm' to the beginning of the molecule name
                molecule_name = 'm' + molecule_name
            
            molecule_name = molecule_name.upper()
            
            if molecule_name:

                data_field.delete('1.0', "end")
                data_field.insert('1.0', 'Importing Molecule...')
                plt.draw(), canvas.draw()
                fig.canvas.flush_events() 

                
                # Add the molecule name and file path to the molecules_data list
                molecules_data.append((molecule_name, mol_file_path))
                
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
                exec(f"{molecule_name.lower()}_rowl_field = tk.Entry(param_frame, width=6)", globals())
                eval(f"{molecule_name.lower()}_rowl_field").grid(row=row+1, column=0)
                eval(f"{molecule_name.lower()}_rowl_field").insert(0, f"{molecule_name}")
                molecule_elements[molecule_name.lower()] = {'rowl': molecule_name.lower() + '_rowl_field'}
                
                # Temperature input field
                globals()[f"{molecule_name.lower()}_temp_field"] = tk.Entry(param_frame, width=4)

                eval(f"{molecule_name.lower()}_temp_field").grid(row=row + 1, column=1)
                eval(f"{molecule_name.lower()}_temp_field").insert(0, f"{t_kin}")
                #globals() [f"{molecule_name.lower()}_submit_temp_button"] = tk.Button(window, text="Submit", command=lambda mn=molecule_name.lower(), te = globals()[f"{molecule_name.lower()}_temp_field"]: submit_temp(te.get(), mn))
                #eval(f"{molecule_name.lower()}_submit_temp_button").grid(row=row + 1, column=2)
                molecule_elements[molecule_name.lower()] = {'temp': molecule_name.lower() + '_temp_field'}
                eval(f"{molecule_name.lower()}_temp_field").bind("<Return>", lambda event, mn=molecule_name.lower(), ce=globals()[f"{molecule_name.lower()}_temp_field"]: submit_temp(ce.get(), mn))
                
                # Radius input field
                globals()[f"{molecule_name.lower()}_rad_field"] = tk.Entry(param_frame, width=4)
                eval(f"{molecule_name.lower()}_rad_field").grid(row=row + 1, column=2)
                eval(f"{molecule_name.lower()}_rad_field").insert(0, f"{radius_init}")
                #globals() [f"{molecule_name.lower()}_submit_rad_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), re = globals()[f"{molecule_name.lower()}_rad_field"]: submit_rad(re.get(), mn))
                #eval(f"{molecule_name.lower()}_submit_rad_button").grid(row=row + 1, column=4)
                molecule_elements[molecule_name.lower()]['rad'] = molecule_name.lower() + '_rad_field'
                eval(f"{molecule_name.lower()}_rad_field").bind("<Return>", lambda event, mn=molecule_name.lower(), ce=globals()[f"{molecule_name.lower()}_rad_field"]: submit_rad(ce.get(), mn))
                
                # Column Density input field
                globals()[f"{molecule_name.lower()}_dens_field"] = tk.Entry(param_frame, width=6)
                eval(f"{molecule_name.lower()}_dens_field").grid(row=row + 1, column=3)
                eval(f"{molecule_name.lower()}_dens_field").insert(0, f"{n_mol_init}")
                #globals() [f"{molecule_name.lower()}_submit_col_button"] = tk.Button(window, text="Submit", command=lambda mn=mol_name.lower(), ce = globals()[f"{molecule_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
                #eval(f"{molecule_name.lower()}_submit_col_button").grid(row=row + 1, column=6)
                molecule_elements[molecule_name.lower()]['dens'] = molecule_name.lower() + '_dens_field'
                eval(f"{molecule_name.lower()}_dens_field").bind("<Return>", lambda event, mn=molecule_name.lower(), ce=globals()[f"{molecule_name.lower()}_dens_field"]: submit_col(ce.get(), mn))
                
                # Visibility Button
                globals()[f"{molecule_name.lower()}_vis_status"] = tk.BooleanVar()
                globals()[f"{molecule_name.lower()}_vis_checkbutton"] = tk.Checkbutton(param_frame, text='', variable=eval(f"{molecule_name.lower()}_vis_status"), command=lambda mn=molecule_name.lower(): model_visible(mn))
                globals()[f"{molecule_name.lower()}_vis_status"].set(False)  # Set the initial state
                eval(f"{molecule_name.lower()}_vis_checkbutton").grid(row=row + 1, column=4)
                globals()[f"{molecule_name.lower()}_vis"] = False
                # Add the variable to the globals dictionary
                # Add the text boxes to the molecule_text_boxes dictionary
                #molecule_text_boxes[molecule_name.lower()] = text_boxes

                # Increment nextrow
                nextrow += 1
                
                exec(f"{molecule_name.lower()}_line, = ax1.plot([], [], alpha=0.8, linewidth=1)", globals())
                exec(f"{molecule_name.lower()}_line.set_label('{molecule_name}')", globals())
                
                # Column density
                exec(f"global n_mol_{molecule_name.lower()}; n_mol_{molecule_name.lower()} = n_mol_{molecule_name.lower()}_init")

                # Temperature
                exec(f"global t_{molecule_name.lower()}; t_{molecule_name.lower()} = t_kin_{molecule_name.lower()}")

                # Radius
                exec(f"global {molecule_name.lower()}_radius; {molecule_name.lower()}_radius = {molecule_name.lower()}_radius_init")
                
                # Intensity calculation
                exec(f"{molecule_name.lower()}_intensity = Intensity(mol_{molecule_name.lower()})", globals())
                exec(f"{molecule_name.lower()}_intensity.calc_intensity(t_{molecule_name.lower()}, n_mol_{molecule_name.lower()}, dv=intrinsic_line_width)", globals())
                print(f"{molecule_name.lower()}_intensity")
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
                print(molecules_data)
                write_to_csv(molecules_data)
                
                update()
                
                print('test')
                
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
                 
                
                
            else:
                print("Molecule name not provided.")
    else:
        print("No files selected.")
        
def del_molecule_data():
    global molecules_data
    global temp_field
    global text_boxes
    global nextrow
    global deleted_molecules
    global text_box
    global text_box_data
    global molecule_elements
    
    data_field.delete('1.0', "end")
    data_field.insert('1.0', 'Clearing Molecules...')
    plt.draw(), canvas.draw()
    fig.canvas.flush_events() 
    
    deleted_molecules = []
    # Define the molecules and their corresponding file paths
    molecules_data = []
    molecules_data = [
        ("H2O", "2020HITRANdata/data_Hitran_2020_H2O.par"),
        ("OH", "2020HITRANdata/data_Hitran_2020_OH.par"),
        ("HCN", "2020HITRANdata/data_Hitran_2020_HCN.par"),
        ("C2H2", "2020HITRANdata/data_Hitran_2020_C2H2.par"),
        ("CO2", "2020HITRANdata/data_Hitran_2020_CO2.par"),
        ("CO", "2020HITRANdata/data_Hitran_2020_CO.par")
        # Add more molecules here if needed
    ]
    print(molecules_data)
    # New array to store the molecules found in the CSV but not in molecules_data
    new_molecules_data = []

    if os.path.exists('molecules_data.csv'):
        try:
            # Assuming the chemical names are in the first column of the CSV file
            with open('molecules_data.csv', 'r') as csvfile:
                reader = csv.reader(csvfile)
                # Skip the first header row
                next(reader)
                csv_chemicals = [row[0].strip().lower() for row in reader]

            for chem in csv_chemicals:
                if chem not in [mol[0].lower() for mol in molecules_data]:
                    # You can define the file path for each missing molecule here
                    # For now, we'll set it to an empty string
                    file_path = ""
                    new_molecules_data.append((chem.upper(), file_path))

            if len(new_molecules_data) > 0:
                print("Molecules found in the CSV but not in molecules_data:", new_molecules_data)
                
                for data in new_molecules_data:
                    nextrow = nextrow - 1
                    chem = data[0]
                    temp_field_var = f"{chem.lower()}_temp_field"
                    rad_field_var = f"{chem.lower()}_rad_field"
                    dens_field_var = f"{chem.lower()}_dens_field"
                    vis_button_var = f"vis_button_{chem.lower()}"
                    label_var = f"{chem.lower()}_rowl_field"
                    line_var = f"{chem.lower()}_line"
                    
                    
                    eval(f"{chem.lower()}_temp_field").destroy()
                    eval(f"{chem.lower()}_vis_checkbutton").destroy()
                    eval(f"{chem.lower()}_rowl_field").destroy()
                    eval(f"{chem.lower()}_rad_field").destroy()
                    eval(f"{chem.lower()}_dens_field").destroy()
                    
                # Clearing the text feed box.
                data_field.delete('1.0', "end")
                data_field.insert('1.0', 'Molecules Cleared!')
                plt.draw(), canvas.draw()
                fig.canvas.flush_events() 

                #plt.pause(3)

                # Clearing the text feed box.
                data_field.delete('1.0', "end")
                print(molecules_data)
                plt.draw(), canvas.draw()
            else:
                print("No new molecules found in the CSV.")
        except OSError as e:
            print(f"Error reading the CSV file: {e}")
    else:
        print("CSV file not found.")

    if os.path.exists('molecules_data.csv'):
        try:
            os.remove('molecules_data.csv')
            print('molecules_data.csv deleted.')
        except OSError as e:
            print(f"Error deleting molecules_data.csv: {e}")
    else:
        print('No molecule Save Found.')  

#addmol_box = fig.add_axes([0.07, 0.53, 0.1, 0.04]);
#addmol_button = Button(addmol_box, 'Add Molecule', color = background, hovercolor = 'lightgray')
#addmol_button.on_clicked(add_molecule_data) #loadparams_button_clicked)

#delmol_box = fig.add_axes([0.16, .975, 0.07, 0.02]);
#delmol_button = Button(delmol_box, 'Clear Molecules', color = background, hovercolor = 'lightgray')
#delmol_button.on_clicked(del_molecule_data) #loadparams_button_clicked)

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


#downmol_box = fig.add_axes([0.235, .975, 0.07, 0.02]);
#downmol_button = Button(downmol_box, 'Install Molecules', color = background, hovercolor = 'lightgray')
#downmol_button.on_clicked(down_molecule_data) #loadparams_button_clicked)



#delmol_box = fig.add_axes([0.235, .975, 0.07, 0.02]);
#delmol_button = Button(delmol_box, 'Delete Test', color = background, hovercolor = 'lightgray')
#delmol_button.on_clicked(delete_test) #loadparams_button_clicked)

#def fileselectupdate(val):
#    selectfile(1)
#    update()
    

def selectfile():
    global file_path
    global file_name
    global wave_cnts
    global flux_cnts
    global observation_data
    global filename_box_data
    
    filetypes = [('CSV Files', '*.csv')]
    infiles = filedialog.askopenfilename(multiple=True, title='Choose Spectra Data File', filetypes=filetypes)

    if infiles:
        for file_path in infiles:
            # Process each selected file
            print("Selected file:", file_path)
            file_name = os.path.basename(file_path)
            file_name_label.config(text=str(file_name))
            #filename_box_data.set_val(file_name)
            # Add your code to process each file
            #THIS IS THE OLD FILIE SYSTEM (THIS WILL BE USED UNTIL THE NEW FILE SYSTEM IS DEVELOPED) USE THIS!!!!!
            observation_data=pd.read_csv(filepath_or_buffer=(file_path), sep=',')
            wave_cnts = np.array(observation_data['wave'])
            flux_cnts = np.array(observation_data['flux'])
            now = datetime.now()
            dateandtime = now.strftime("%d-%m-%Y-%H-%M-%S")
            print(dateandtime)
            svd_line_file = f'savedlines-{dateandtime}.csv'
    else:
        print("No files selected.")
    update()
    
def selectlinefile():
    global linesavefile
    global linesavepath
    
    filetypes = [('CSV Files', '*.csv')]
    infile = filedialog.asksaveasfilename(title='Choose or Create a Spectra Data File', filetypes=filetypes, defaultextension=".csv")

    if infile:
        linesavepath = infile
        linesavefile = os.path.basename(linesavepath)
        # Update the label with the selected/created file
        linefile_name_label.config(text=str(linesavefile))

        headers = "lev_up,lev_low,lam, tau,intens,a_stein,e_up,g_up,xmin,xmax,flux"

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
        
        
        
        
        
        
        
# sum button
#sum_box = plt.axes([0.16, .975, 0.07, 0.02])
#sum_button = Button(sum_box, 'Show Sum', color = background, hovercolor = background)
#sum_button.on_clicked(total_flux)    

files_frame = tk.Frame(window, borderwidth=2, relief="groove")
files_frame.grid(row=11, column=0, rowspan=5, columnspan=5, sticky="nsew")

# Configure columns to expand and fill the width
for i in range(5):
    files_frame.columnconfigure(i, weight=1)

# Create a frame to hold the box outline
box_frame = tk.Frame(files_frame)
box_frame.grid(row=1, column=0, columnspan=5, sticky='nsew')

specfile_label = tk.Label(files_frame, text='Spectrum Data File:')
specfile_label.grid(row=0, column=0, columnspan=5, sticky='nsew')  # Center-align using grid

linefile_label = tk.Label(files_frame, text='Saved Lines File:')
linefile_label.grid(row=2, column=0, columnspan=5, sticky='nsew')  # Center-align using grid

# Create a frame to hold the box outline
linebox_frame = tk.Frame(files_frame)
linebox_frame.grid(row=3, column=0, columnspan=5, sticky='nsew')

# Create a label widget inside the frame to create the box outline
linebox_label = tk.Label(linebox_frame, text='', relief='solid', borderwidth=1, height=2)  # Adjust the height value as needed
linebox_label.pack(side="top", fill="both", expand=True)

# Create a label inside the box_frame and center-align it
linefile_name_label = tk.Label(linebox_label, text='')
linefile_name_label.grid(row=0, column=0, sticky='nsew')  # Center-align using grid

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

# Add some space below linebox_frame
tk.Label(files_frame, text="").grid(row=4, column=0)


# Create a frame for the Text widget
text_frame = tk.Frame(window)
text_frame.grid(row=26, column=0, columnspan=5, sticky='nsew')

# Create a Text widget within the frame
data_field = tk.Text(text_frame, wrap="word", height=10, width=24)
data_field.pack(fill="both", expand=True)




# Define the span selecting function of the tool
span = SpanSelector(
    ax1,
    onselect,
    "horizontal",
    useblit=False,
    props=dict(alpha=0.5, facecolor="tab:blue"),
    interactive=True,
    drag_from_anywhere=True
)

# Storing the callback for on_xlims_change()
ax1.callbacks.connect('xlim_changed', on_xlims_change)
fig.canvas.manager.set_window_title('iSLAT-V2.00.00') #Interactive Spectral-Line Analysis Tool
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

# Allow column 9 and row 1 to expand
window.grid_columnconfigure(5, weight=1)
#window.grid_rowconfigure(1, weight=1)


# Create a frame for the toolbar inside the titleframe
toolbar_frame = tk.Frame(titleframe)
toolbar_frame.grid(row=0, column=9, columnspan=2, sticky="nsew")  # Place the frame in row 0, column 9

# Create a toolbar and update it
toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
toolbar.update()

titleframe.grid_columnconfigure(9, weight=1)

plt.interactive(False)
update()
window.mainloop()    
#LATInit()


# In[ ]:





# In[ ]:




