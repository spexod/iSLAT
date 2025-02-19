import pandas as pd
from astropy.io import ascii
from ir_model import Chi2Spectrum, MolData, Intensity

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.chi2_h2o = Chi2Spectrum()
        self.mol_h2o = None
        self.intensity_h2o = None
        
    def load_data(self):
        self.chi2_h2o.load_file(self.config.input_file)
        self.mol_h2o = MolData(f"{self.config.molecule_name}", f"{self.config.molecule_path}")
        self.intensity_h2o = Intensity(self.mol_h2o)

