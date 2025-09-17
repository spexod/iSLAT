import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Tuple, Callable, Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from lmfit.model import ModelResult

class FitLinesPlotGrid:
    def __init__(self, #files : List[str], 
                 #molecules_dict: 'MoleculeDict', 
                 fit_data: Dict[str, Any] = None,
                 rows: int = None, cols: int = None, 
                 #figsize: Tuple[int, int] = (15, 15),
                 **kwargs
                 ):
        #self.files = files
        #self.molecules_dict = molecules_dict
        self.fit_csv_dict: Dict[str, Any]
        self.fit_data_tuple_list: List[Tuple[ModelResult, np.ndarray, np.ndarray]]
        self.fit_csv_dict, self.fit_data_tuple_list = fit_data
        #self.gauss_fit, self.fitted_wave, self.fitted_flux = fit_data
        
        # Set rows and cols by the length of fit data if not specified
        if rows is None or cols is None:
            n_plots = len(self.fit_csv_dict)
            if rows is None and cols is None:
                cols = int(np.ceil(np.sqrt(n_plots)))
                rows = int(np.ceil(n_plots / cols))
            elif rows is None:
                rows = int(np.ceil(n_plots / cols))
            elif cols is None:
                cols = int(np.ceil(n_plots / rows))
        self.rows = rows
        self.cols = cols

        self.spectrum_name = kwargs.get('spectrum_name', 'Spectrum')
        self.figsize = kwargs.get('figsize', (5 * self.cols, 5 * self.rows))
        self.fig: Figure
        self.axs: np.ndarray[Axes]
        self.fig, self.axs = plt.subplots(rows, cols, figsize=self.figsize, layout='constrained')
        #self.fig.tight_layout(pad=2.0)
        self.plt_extra_range = kwargs.get('plt_extra_range', 0.015)  # extra range to plot for each line
        self.wave_data = kwargs.get('wave_data', None)
        self.flux_data = kwargs.get('flux_data', None)
        self.err_data = kwargs.get('err_data', None)
        self.fit_line_uncertainty = kwargs.get('fit_line_uncertainty', 3.0)

    def generate_plot(self):
        gauss_fits, fitted_waves, fitted_fluxes = self.fit_data_tuple_list
        for idx, (gauss_fit, fitted_wave, fitted_flux) in enumerate(zip(gauss_fits, fitted_waves, fitted_fluxes)):
            if idx >= self.rows * self.cols:
                break
            row = idx // self.cols
            col = idx % self.cols
            ax: Axes = self.axs[row, col]

            xmin = self.fit_csv_dict[idx]['xmin']
            xmax = self.fit_csv_dict[idx]['xmax']

            fit_mask = (self.wave_data >= xmin - self.plt_extra_range) & (self.wave_data <= xmax + self.plt_extra_range)
            spectrum_wave = self.wave_data[fit_mask]
            spectrum_flux = self.flux_data[fit_mask]
            spectrum_err = self.err_data[fit_mask]

            #fit_wave = fitted_wave[fit_mask]
            #fit_flux = fitted_flux[fit_mask]

            # Plot the spectrum
            ax.plot(spectrum_wave, spectrum_flux, color='black', linewidth=1, zorder=5)
            ax.errorbar(spectrum_wave, spectrum_flux, yerr=spectrum_err, fmt='-', color='black')
            
            # plot the fit result
            if fitted_wave is None or fitted_flux is None:
                ax.set_title(f"Line {idx+1}: Fit Error")
                #ax.text(0, 0, 'Fit Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                continue

            # get color based on fit det
            if self.fit_csv_dict[idx]['Fit_det'] == True:
                line_color = 'lime'
            else:
                line_color = 'red'
            try:
                ax.plot(fitted_wave, fitted_flux, color=line_color, linewidth=2, zorder=10, linestyle='--')[0]#, label=f'Gauss Fit {i}')[0]
                dely = gauss_fit.eval_uncertainty(sigma = self.fit_line_uncertainty)
                ax.fill_between(fitted_wave, fitted_flux - dely, fitted_flux + dely,
                                        color=line_color, alpha=0.3)#, label=r'3-$\sigma$ uncertainty band')

                # plot the xmin and xmax for each line
                #ax.vlines([lam_min, lam_max], -2, 10, colors='lime', alpha=0.5)
            
                ax.set_title(f"Line {idx+1}: {self.fit_csv_dict[idx]['species']}_{self.fit_csv_dict[idx]['lam']:.2f}")
                # set y lim to 10% above and below the observed flux in the fit range
                y_min = np.min(spectrum_flux) - 0.1 * np.abs(np.min(spectrum_flux))
                y_max = np.max(spectrum_flux) + 0.1 * np.abs(np.max(spectrum_flux))
                ax.set_ylim(y_min, y_max)
                #ax.set_xlabel("Wavelength")
                ax.set_ylabel("Flux (Jy)")
            except Exception as e:
                ax.set_title(f"Line {idx+1}: Plot Error")
                #ax.text(0, 0, 'Plot Error', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                print(f"Error plotting line {idx+1}: {e}")       

            self.axs[row, col] = ax

        #plt.show(block=False)
    
    def plot(self):
        self.generate_plot()
        plt.show(block=False)