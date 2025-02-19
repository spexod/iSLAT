import os

class Config:
    def __init__(self, target, save_folder, mol, molpath, dist, fwhm, min_lamb, max_lamb, pix_per_fwhm, intrinsic_line_width, cc):
        self.target = target
        self.input_file = os.path.join(save_folder, target)
        self.dist = dist
        self.fwhm = fwhm
        self.npix = pix_per_fwhm
        self.model_lam_min = min_lamb
        self.model_lam_max = max_lamb
        self.intrins_line_broad = intrinsic_line_width
        self.rings = 1
        self.cc = cc  # speed of light in km/s
        self.molecule_name = mol  # Add molecule_name attribute
        self.molecule_path = molpath  # Add molecule_name attribute

    @property
    def model_line_width(self):
        return self.cc / self.fwhm

    @property
    def model_pixel_res(self):
        return (self.model_lam_min + self.model_lam_max) / 2 / self.cc * self.fwhm / self.npix
