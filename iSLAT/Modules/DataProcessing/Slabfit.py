"""
Simplified slab fitting module for iSLAT.

This module provides the SlabModel class that integrates with existing iSLAT components
to perform slab model fitting on molecular spectra.

Example Usage:
-------------
    # Create a molecule object
    mol = Molecule(name="H2O", filepath="path/to/h2o.par", temp=500, n_mol=1e17, radius=1.0)
    
    # Create slab model for fitting
    slab = SlabModel(
        output_folder="DATAFILES/MODELS",
        mol_object=mol,
        input_filename="my_observations.csv",
        distance=150,  # Override distance to 150 pc
        fwhm=10       # Override FWHM to 10 km/s
    )
    
    # Perform fitting
    results = slab.fit_parameters(start_t=300, start_n_mol=5e16, start_r=1.5)
    
    # Update molecule with fitted parameters
    slab.update_molecule_parameters(results)
    
    # Save results
    slab.save_results(results)
"""

import os
import numpy as np
from scipy.optimize import fmin
#from iSLAT.Modules.DataProcessing.Chi2Spectrum import Chi2Spectrum
from iSLAT.Modules.DataProcessing import Chi2Spectrum
#from iSLAT.Modules.DataTypes import FluxMeasurement
from iSLAT.Modules.DataTypes import Intensity, Spectrum
from iSLAT.Modules.DataTypes.Molecule import Molecule
import iSLAT.Constants as c


class SlabModel:
    """
    Simplified slab fitting class that integrates with existing project components.
    
    This class takes an output folder and a mol object, allowing users to override
    molecular parameters through kwargs for fitting optimization.
    """
    def __init__(self, output_folder, mol_object : Molecule, data_field=None, **kwargs):
        """
        Initialize the slab fitting system.
        
        Parameters:
        -----------
        output_folder : str
            Directory containing the target file for chi-squared evaluation
        mol_object : Molecule
            Molecule object containing molecular data and parameters
        data_field : object, optional
            GUI data field for status updates
        **kwargs : dict
            Override parameters for fitting. Can include:
            - distance: Distance to object (pc)
            - fwhm: Full width at half maximum (km/s)
            - min_wavelength, max_wavelength: Wavelength range (μm)
            - model_pixel_res: Model pixel resolution (μm)
            - model_line_width: Model line width (R = λ/Δλ)
            - intrinsic_line_width: Intrinsic line broadening (km/s)
            - input_filename: Name of the input file in output_folder
            - input_file: Full path to the input file (optional, defaults to 'fit_data.csv')
        """
        self.output_folder = output_folder
        self.mol_object = mol_object
        self.data_field = data_field
        
        # Store override parameters
        self.overrides = kwargs
        
        # Set up file path for chi-squared evaluation
        self.input_filename = kwargs.get('input_filename', 'fit_data.csv')
        self.input_file = kwargs.get('input_file', os.path.join(output_folder, self.input_filename))

        # Initialize chi-squared evaluator
        self.chi2_evaluator = Chi2Spectrum()
        
        # Cache for configurations to avoid repeated parameter extraction
        self._config_cache = {}
        
        # Load data if file exists
        if os.path.exists(self.input_file):
            self.chi2_evaluator.load_file(self.input_file)
        else:
            print(f"Warning: Input file {self.input_file} not found. Chi-squared evaluation will not be available.")
        
    def _get_parameter(self, param_name, default_value=None):
        """
        Get parameter value with override precedence.
        
        Order of precedence:
        1. kwargs override
        2. mol_object parameter
        3. default value
        """
        if param_name in self.overrides:
            return self.overrides[param_name]
        elif hasattr(self.mol_object, param_name):
            return getattr(self.mol_object, param_name)
        else:
            return default_value
    
    def _get_fitting_parameters(self):
        """Extract parameters needed for fitting, respecting overrides."""
        if 'fitting_params' in self._config_cache:
            return self._config_cache['fitting_params']
        
        params = {
            'distance': self._get_parameter('distance', c.DEFAULT_DISTANCE),
            'fwhm': self._get_parameter('fwhm', c.FWHM_TOLERANCE),
            'min_wavelength': self._get_parameter('min_wavelength', 
                                                getattr(self.mol_object, 'wavelength_range', c.WAVELENGTH_RANGE)[0]),
            'max_wavelength': self._get_parameter('max_wavelength', 
                                                getattr(self.mol_object, 'wavelength_range', c.WAVELENGTH_RANGE)[1]),
            'model_pixel_res': self._get_parameter('model_pixel_res', 
                                                 getattr(self.mol_object, 'model_pixel_res', c.MODEL_PIXEL_RESOLUTION)),
            'model_line_width': self._get_parameter('model_line_width', 
                                                  getattr(self.mol_object, 'model_line_width', c.MODEL_LINE_WIDTH)),
            'intrinsic_line_width': self._get_parameter('intrinsic_line_width', 
                                                      getattr(self.mol_object, 'broad', c.INTRINSIC_LINE_WIDTH))
        }
        
        self._config_cache['fitting_params'] = params
        return params
    
    def evaluate_model(self, t_kin, n_mol, radius):
        """
        Evaluate the chi-squared for given physical parameters.
        
        Parameters:
        -----------
        t_kin : float
            Kinetic temperature in K
        n_mol : float
            Molecular column density (cm^-2)
        radius : float
            Emitting radius (au)
            
        Returns:
        --------
        float
            Chi-squared value
        """
        params = self._get_fitting_parameters()
        
        # Create intensity calculator using existing molecular line data
        intensity = Intensity(self.mol_object.lines)
        intensity.calc_intensity(
            t_kin=t_kin, 
            n_mol=n_mol, 
            dv=params['intrinsic_line_width']
        )
        
        # Create test spectrum
        test_spectrum = Spectrum(
            lam_min=params['min_wavelength'],
            lam_max=params['max_wavelength'],
            dlambda=params['model_pixel_res'],
            R=params['model_line_width'],
            distance=params['distance']
        )
        
        # Add intensity with area scaling
        test_spectrum.add_intensity(intensity, radius**2 * np.pi)
        
        # Evaluate chi-squared
        self.chi2_evaluator.evaluate_spectrum(test_spectrum)
        
        chi2_total = self.chi2_evaluator.chi2_total
        
        # Print progress if verbose
        print(f"t_kin={t_kin:.1f}K, n_mol={n_mol:.2e}cm⁻², radius={radius:.2f}au → χ²={chi2_total:.3e}")
        
        return chi2_total
    
    def fit_parameters(self, start_t=None, start_n_mol=None, start_r=None):
        """
        Fit the slab model parameters using optimization.
        
        Parameters:
        -----------
        start_t : float, optional
            Starting temperature guess (default: use mol_object.temp)
        start_n_mol : float, optional
            Starting molecular column density guess (default: use mol_object.n_mol)
        start_r : float, optional
            Starting radius guess (default: use mol_object.radius)
            
        Returns:
        --------
        dict
            Dictionary containing fitted parameters and results
        """
        # Set starting values from mol_object if not provided
        if start_t is None:
            start_t = getattr(self.mol_object, 'temp', 500.0)
        if start_n_mol is None:
            start_n_mol = getattr(self.mol_object, 'n_mol', 1e17)
        if start_r is None:
            start_r = getattr(self.mol_object, 'radius', 1.0)
        
        # Update status if GUI field is available
        if self.data_field is not None:
            self.data_field.clear()
            self.data_field.insert_text("Fitting slab model...")
        
        # Define optimization function (log scale for n_mol for better convergence)
        def objective_function(params):
            return self.evaluate_model(params[0], 10**params[1], params[2])
        
        # Set up initial guess
        initial_guess = [start_t, np.log10(start_n_mol), start_r]
        
        print(f"Starting optimization with initial guess:")
        print(f"  Temperature: {start_t:.1f} K")
        print(f"  Column density: {start_n_mol:.2e} cm⁻²")
        print(f"  Radius: {start_r:.2f} au")
        
        # Perform optimization
        result = fmin(objective_function, initial_guess, full_output=True)
        optimal_params, final_chi2, iterations, funcalls, warnflag = result
        
        # Extract results
        fitted_params = {
            'temperature': optimal_params[0],
            'log_n_mol': optimal_params[1],
            'n_mol': 10**optimal_params[1],
            'radius': optimal_params[2],
            'chi2_final': final_chi2,
            'iterations': iterations,
            'function_calls': funcalls,
            'convergence_flag': warnflag
        }
        
        print(f"\nOptimization completed:")
        print(f"  Final temperature: {fitted_params['temperature']:.1f} K")
        print(f"  Final column density: {fitted_params['n_mol']:.2e} cm⁻²")
        print(f"  Final radius: {fitted_params['radius']:.2f} au")
        print(f"  Final χ²: {fitted_params['chi2_final']:.3e}")
        print(f"  Iterations: {fitted_params['iterations']}")
        print(f"  Function calls: {fitted_params['function_calls']}")
        
        # Update status
        if self.data_field is not None:
            self.data_field.clear()
            self.data_field.insert_text(f"Fitting complete. χ² = {fitted_params['chi2_final']:.3e}")
        
        return fitted_params
    
    def update_molecule_parameters(self, fitted_params):
        """
        Update the molecule object with fitted parameters.
        
        Parameters:
        -----------
        fitted_params : dict
            Dictionary containing fitted parameters from fit_parameters()
        """
        if hasattr(self.mol_object, 'temp'):
            self.mol_object.temp = fitted_params['temperature']
        if hasattr(self.mol_object, 'n_mol'):
            self.mol_object.n_mol = fitted_params['n_mol']
        if hasattr(self.mol_object, 'radius'):
            self.mol_object.radius = fitted_params['radius']
        
        print(f"Updated molecule '{self.mol_object.name}' with fitted parameters")
    
    def save_results(self, fitted_params, filename=None):
        """
        Save fitting results to a file in the output folder.
        
        Parameters:
        -----------
        fitted_params : dict
            Dictionary containing fitted parameters
        filename : str, optional
            Name of output file (default: 'slab_fit_results.txt')
        """
        if filename is None:
            filename = f"slab_fit_results_{self.mol_object.name}.txt"
        
        output_path = os.path.join(self.output_folder, filename)
        
        with open(output_path, 'w') as f:
            f.write(f"Slab Model Fitting Results\n")
            f.write(f"==========================\n\n")
            f.write(f"Molecule: {self.mol_object.name}\n")
            f.write(f"Input file: {self.input_file}\n\n")
            f.write(f"Fitted Parameters:\n")
            f.write(f"  Temperature: {fitted_params['temperature']:.2f} K\n")
            f.write(f"  Column density: {fitted_params['n_mol']:.3e} cm⁻²\n")
            f.write(f"  Radius: {fitted_params['radius']:.3f} au\n\n")
            f.write(f"Fitting Statistics:\n")
            f.write(f"  Final χ²: {fitted_params['chi2_final']:.6e}\n")
            f.write(f"  Iterations: {fitted_params['iterations']}\n")
            f.write(f"  Function calls: {fitted_params['function_calls']}\n")
            f.write(f"  Convergence flag: {fitted_params['convergence_flag']}\n")
        
        print(f"Results saved to {output_path}")
        return output_path


# Backward compatibility alias
SlabFit = SlabModel