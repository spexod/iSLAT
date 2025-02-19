from scipy.optimize import fmin
import numpy as np
from ir_model import Spectrum

class ModelFitting:
    def __init__(self, data_loader, config, data_field):
        self.data_loader = data_loader
        self.config = config
        self.data_field = data_field

    def eval_function(self, t_kin, n_mol, radius):
        intensity_h2o = self.data_loader.intensity_h2o
        chi2_h2o = self.data_loader.chi2_h2o
        intensity_h2o.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=self.config.intrins_line_broad)
        
        test_spectrum = Spectrum(
            lam_min=self.config.model_lam_min, lam_max=self.config.model_lam_max, 
            dlambda=self.config.model_pixel_res, R=self.config.model_line_width, 
            distance=self.config.dist)
        test_spectrum.add_intensity(intensity_h2o, radius**2 * np.pi)

        chi2_h2o.evaluate_spectrum(test_spectrum)
    
        print(f"For t_kin = {t_kin:.2f}, n_mol = {n_mol:.2e}, radius = {radius:.3f} chi2 = {chi2_h2o.chi2_total:.3e}")

        return chi2_h2o.chi2_total

    def fit_model(self, start_t, start_n_mol, start_r):
        func = lambda p: self.eval_function(p[0], 10**p[1], p[2])
        x_0 = [start_t, np.log10(start_n_mol), start_r]
        result = fmin(func, x_0)
        return result
    
    def message(self):
        # Update the main GUI data_field
        self.data_field.delete('1.0', "end")
        self.data_field.insert('1.0', "Fitting slab...")