import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path 
from tkinter import filedialog
# import tkinter as tk

from iSLAT.Modules.FileHandling import *


def output_full_spectrum(islat_ref):

    version = 'v8.3'
    Wfct = 4

    step = 2.3
    xlim1 = np.arange(4.9,26,step)
    fct = 130
    offslabl = 0.003
    ymaxfct = 0.2
    plotname='compact'
    figs = (12, 16.)

    plot_renderer = islat_ref.GUI.get_plot_renderer()

    spectrum_path = Path(islat_ref.loaded_spectrum_file)

    svd_lns = pd.read_csv(atomic_lines_file_name, sep=',')
    svd_lamb = np.array (svd_lns['wave'])
    svd_species = svd_lns['species']
    svd_lineID = np.array (svd_lns['line'])

    def plot_atomic(xr,ymin,ymax):
        for i in range(len(svd_lamb)):
            if svd_lamb[i] > xr[0] and svd_lamb[i] < xr[1]:
                plt.vlines(svd_lamb[i], ymin,ymax, linestyles='dotted', color='grey', linewidth=0.7)
                # Adjust the y-coordinate to place labels within the borders
                label_y = ymax
                # Adjust the x-coordinate to place labels just to the right of the line
                label_x = svd_lamb[i] + offslabl

                plt.text(label_x, label_y, svd_species[i] + ' ' + svd_lineID[i]+' ', fontsize=6, rotation=90, va='top',
                        ha='left', color='grey')


    # read input spectrum
    cc = 2.99792458e5  # speed of light in km/s

    RV = 15. # this should be taken from the GUI
    spectrum = pd.read_csv(spectrum_path, sep=',')
    rv = islat_ref.molecules_dict._global_parms.get("stellar_rv", RV)

    waveCI = spectrum['wave']
    waveCI = waveCI - (waveCI / cc * rv)
    fluxCI = spectrum['flux']
    offsetCI = 0.


    fig = plt.figure(figsize=figs)

    mol_dict = islat_ref.molecules_dict
    mol_labels = []
    mol_colors = []

    for key, mol in mol_dict.items():
        if mol._is_visible == "True":
            print(f"adding {key} to list")
            mol_labels.append(mol.displaylabel)
            mol_colors.append(mol.color)

    lw = 1.

    subplots = {}

    for n, xlim in enumerate(xlim1):
        # add a new subplot iteratively
        xr = [xlim1[n], xlim1[n]+step]
        subplots[n] = plt.subplot(len (xlim1), 1, n + 1)
        maxv = np.nanmax(fluxCI[(waveCI > xr[0]-0.02) & (waveCI < xr[1])])
        ymax = maxv + maxv*ymaxfct
        #ymax = 0.12
        ymin = -0.005
        plt.xlim(xr)
        plt.ylim([ymin,ymax])
        plt.ylabel("Flux dens. (Jy)")
        plot_atomic(xr,ymin,ymax)
   
        summed_wavelengths, summed_flux = islat_ref.molecules_dict.get_summed_flux(islat_ref.wave_data_original, visible_only=True)

        plot_renderer.render_main_spectrum_output(
            subplot = subplots[n],
            wave_data = waveCI, 
            flux_data = fluxCI + offsetCI,
            molecules = islat_ref.molecules_dict,
            summed_wavelengths = summed_wavelengths,
            summed_flux = summed_flux
            )
        plt.draw()
        
        #plot_waterPAR(fct, xr)
        if n == 0:
            plt.legend()
    
            plt.legend(
                mol_labels,
                labelcolor = mol_colors,
                loc = 'upper center',
                ncols = 9,
                handletextpad = 0.2,
                bbox_to_anchor = (0.5,1.4),
                handlelength = 0,
                fontsize = 10,
                prop = {'weight':'bold'},
            )
        if n == len (xlim1) - 1:
            plt.xlabel("Wavelength (μm)")
    default_name = spectrum_path.stem + "_full_output.pdf"

    save_path = filedialog.asksaveasfilename(
    title="Save Spectrum Output",
    defaultextension=".pdf",          
    initialfile=default_name,
    initialdir=absolute_data_files_path,
    filetypes=[("PDF files", "*.pdf")]
    )

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format = 'pdf')

        islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")
        print('done!')