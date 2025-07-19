iSLAT_version = 'v5.00.00'
print(' ')
print('Loading iSLAT ' + iSLAT_version + ': Please Wait ...')

# Import necessary modules
import numpy as np
import pandas as pd
import os
import time

from .Modules.FileHandling.iSLATFileHandling import load_user_settings, read_default_molecule_parameters, read_initial_molecule_parameters, read_save_data, read_HITRAN_data, read_from_user_csv, read_default_csv, read_spectral_data

import iSLAT.Constants as c
from .Modules.GUI import *
from .Modules.DataTypes.Molecule import Molecule
from .Modules.DataTypes.MoleculeDict import MoleculeDict
from .Modules.Debug.DebugConfig import debug_config

class UpdateCoordinator:
    """Centralized update coordinator to manage and debounce plot updates"""
    
    def __init__(self, islat_instance : 'iSLAT'):
        self.islat = islat_instance
        self._update_after_id = None
        self._pending_updates = set()
        
    def request_update(self, update_type):
        """Request an update of a specific type, with debouncing"""
        self._pending_updates.add(update_type)
        
        # Cancel previous pending update
        if self._update_after_id is not None:
            self.islat.GUI.master.after_cancel(self._update_after_id)
        
        # Schedule new update
        self._update_after_id = self.islat.GUI.master.after(50, self._execute_updates)
    
    def _execute_updates(self):
        """Execute all pending updates in the correct order"""
        if not self._pending_updates:
            return
            
        updates = self._pending_updates.copy()
        self._pending_updates.clear()
        self._update_after_id = None
        
        # Execute updates in dependency order
        if 'model_spectrum' in updates and hasattr(self.islat, 'molecules_dict') and hasattr(self.islat, 'wave_data'):
            self.islat.molecules_dict.update_molecule_fluxes(self.islat.wave_data)
        
        if 'plots' in updates and hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
            self.islat.GUI.plot.update_all_plots()

class iSLAT:
    """
    iSLAT class to handle the iSLAT functionalities.
    This class is used to initialize the iSLAT application, load user settings, and manage the main functionalities.
    """

    def __init__(self):
        """
        Initialize the iSLAT application.
        """
        # Core initialization
        self._active_molecule_change_callbacks = []
        self._active_molecule = None
        
        # Load settings once
        self.user_settings = load_user_settings()
        self.initial_molecule_parameters = read_initial_molecule_parameters()
        self.molecules_parameters_default = read_default_molecule_parameters()
        
        # Define molecule constants (use tuples for immutability and performance)
        self.mols = ("H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N")
        self.basem = ("H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N")
        self.isot = (1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1)
        
        # Physical parameters
        self.wavelength_range = c.WAVELENGTH_RANGE
        self._display_range = (23.52, 25.41)
        self._dist = c.DEFAULT_DISTANCE
        self._star_rv = c.DEFAULT_STELLAR_RV
        self._fwhm = c.DEFAULT_FWHM
        self._intrinsic_line_width = c.INTRINSIC_LINE_WIDTH

        # Initialize containers (lazy loading)
        self.hitran_data = {}
        self._molecules_data_default = None  # Load on demand
        self.deleted_molecules = []
        
        # Lazy load these expensive operations
        self._default_molecule_csv_data = None
        self._user_saved_molecules = None
        
        self.input_line_list = None
        self.output_line_measurements = None

        # Initialize update coordinator (will be set up after GUI init)
        self.update_coordinator = None
        
        # Parallel processing disabled by default for stability
        self._use_parallel_processing = False

    def init_gui(self):
        """
        Initialize the GUI components of iSLAT.
        This function sets up the main window, menus, and other GUI elements.
        """
        # Initialize update coordinator once
        if self.update_coordinator is None:
            self.update_coordinator = UpdateCoordinator(self)

        if not hasattr(self, "GUI"):
            self.GUI = GUI(
                master=None,
                molecule_data=getattr(self, 'molecules_dict', None),
                wave_data=getattr(self, 'wave_data', None),
                flux_data=getattr(self, 'flux_data', None),
                config=self.user_settings,
                islat_class_ref=self
            )
        
        self.GUI.start()

    def init_molecules(self, mole_save_data=None, use_optimized_loading=False):
        """
        Initialize molecules with sequential loading by default.
        
        Parameters
        ----------
        mole_save_data : dict, list, or None
            Molecule data to load. If None, uses user_saved_molecules.
        use_optimized_loading : bool, default False
            Whether to use the ultra-fast optimized loading method (multiprocessing).
            Set to False by default to avoid multiprocessing/threading.
        """
        # Lazy load user_saved_molecules if needed
        if mole_save_data is None:
            if self.user_saved_molecules is None:
                self.user_saved_molecules = read_from_user_csv()
            mole_save_data = self.user_saved_molecules

        # Initialize molecules_dict if needed
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()
            # Set global parameters efficiently
            for attr, value in [
                ('global_dist', self._dist),
                ('global_star_rv', self._star_rv),
                ('global_fwhm', self._fwhm),
                ('global_intrinsic_line_width', self._intrinsic_line_width),
                ('global_wavelength_range', self.wavelength_range)
            ]:
                setattr(self.molecules_dict, attr, value)
        
        # Validate and process input data
        if not mole_save_data:
            print("Warning: No molecule data provided to init_molecules")
            return

        # Convert to list format efficiently
        if isinstance(mole_save_data, dict):
            molecules_list = [mol for mol in mole_save_data.values() 
                            if mol.get("Molecule Name") and mol.get("Molecule Name") not in self.molecules_dict]
        elif isinstance(mole_save_data, list):
            molecules_list = [mol for mol in mole_save_data 
                            if mol.get("Molecule Name") and mol.get("Molecule Name") not in self.molecules_dict]
        else:
            print(f"Warning: Unsupported molecule data format: {type(mole_save_data)}")
            return

        if not molecules_list:
            print("No new molecules to load.")
            return

        # Load molecules using sequential method by default
        try:
            # Use parallel processing only if explicitly enabled
            use_optimized = use_optimized_loading or self.use_parallel_processing
            
            if use_optimized:
                print(f"Loading {len(molecules_list)} molecules using parallel method...")
                start_time = time.time()
                
                results = self.molecules_dict.load_molecules_ultra_fast(
                    molecules_list, 
                    self.initial_molecule_parameters,
                    force_multiprocessing=self.use_parallel_processing
                )
                
                elapsed_time = time.time() - start_time
                print(f"Parallel loading completed in {elapsed_time:.3f}s")
                
                if results["success"] > 0:
                    print(f"Successfully loaded {results['success']} molecules")
                    
                if results["failed"] > 0:
                    print(f"Failed to load {results['failed']} molecules:")
                    for error in results["errors"]:
                        print(f"  - {error}")
            else:
                # Sequential loading (default, no multiprocessing/threading)
                print(f"Loading {len(molecules_list)} molecules sequentially...")
                start_time = time.time()
                
                success_count = 0
                for mol_data in molecules_list:
                    mol_name = mol_data.get("Molecule Name")
                    try:
                        new_molecule = Molecule(
                            user_save_data=mol_data,
                            wavelength_range=self.wavelength_range,
                            initial_molecule_parameters=self.initial_molecule_parameters.get(mol_name, self.molecules_parameters_default)
                        )
                        self.molecules_dict[mol_name] = new_molecule
                        success_count += 1
                        print(f"Successfully loaded molecule: {mol_name}")
                    except Exception as e:
                        print(f"Error creating molecule '{mol_name}': {e}")
                
                elapsed_time = time.time() - start_time
                print(f"Sequential loading completed in {elapsed_time:.3f}s")
                print(f"Successfully loaded {success_count}/{len(molecules_list)} molecules")

            # Initialize the active molecule
            self._set_initial_active_molecule()
                    
        except Exception as e:
            print(f"Error in init_molecules: {e}")
            raise

    def _set_initial_active_molecule(self):
        """Set the initial active molecule based on user settings and available molecules."""
        active_molecule_name = self.user_settings.get("default_active_molecule", "H2O")
        
        if active_molecule_name in self.molecules_dict:
            self._active_molecule = self.molecules_dict[active_molecule_name]
        else:
            # Try to fall back to H2O
            if "H2O" in self.molecules_dict:
                print(f"Active molecule '{active_molecule_name}' not found. Defaulting to 'H2O'.")
                self._active_molecule = self.molecules_dict["H2O"]
            elif len(self.molecules_dict) > 0:
                # Use the first available molecule
                first_molecule = next(iter(self.molecules_dict.values()))
                print(f"Neither '{active_molecule_name}' nor 'H2O' found. Using '{first_molecule.name}'.")
                self._active_molecule = first_molecule
            else:
                print("No molecules available to set as active.")
                self._active_molecule = None

    def run(self):
        """
        Run the iSLAT application.
        This function starts the main event loop of the Tkinter application.
        """
        # Initialize data and components in optimal order
        try:
            # Load save data first (lightweight)
            self.savedata = read_save_data()
            
            # Initialize molecules (can be heavy, done after basic setup)
            self.init_molecules()
            
            # Load spectrum data (usually fast)
            self.load_spectrum()
            
            # Finally initialize GUI (requires other components)
            self.init_gui()
            
        except Exception as e:
            print(f"Error during iSLAT initialization: {e}")
            raise
    
    def add_molecule_from_hitran(self, refresh=True, hitran_files=None, molecule_names=None, 
                                base_molecules=None, isotopes=None, use_parallel=False):
        """
        Adds one or more molecules to the iSLAT instance from HITRAN files with sequential loading by default.
        
        Parameters:
        -----------
        refresh : bool
            Whether to refresh the GUI after adding molecules
        hitran_files : str or list
            Single file path or list of file paths. If None, opens file dialog for multiple selection
        molecule_names : str or list
            Single molecule name or list of molecule names corresponding to files
        base_molecules : str or list
            Single base molecule or list of base molecules (currently unused)
        isotopes : int or list
            Single isotope or list of isotopes (currently unused)
        use_parallel : bool, default False
            Whether to use parallel loading (multiprocessing). Default is False for sequential loading.
        """
        if hitran_files is None:
            hitran_files = GUI.file_selector(title='Choose HITRAN Data Files (select multiple with Ctrl/Cmd)',
                                                  filetypes=[('PAR Files', '*.par')],
                                                  initialdir=os.path.abspath("DATAFILES/HITRANdata"))
            
        if not hitran_files:
            print("No HITRAN files selected.")
            return
        
        # Convert single file to list for consistent processing
        if isinstance(hitran_files, str):
            hitran_files = [hitran_files]
        
        # Convert molecule_names to list if provided as single string
        if molecule_names is not None and isinstance(molecule_names, str):
            molecule_names = [molecule_names]
        
        # Prepare molecule data for parallel processing
        molecules_data = []
        
        for i, hitran_file in enumerate(hitran_files):
            # Get molecule name for this file
            if molecule_names is not None and i < len(molecule_names):
                molecule_name = molecule_names[i]
            else:
                # Extract molecule name from file name
                molecule_file_name = os.path.basename(hitran_file)
                molecule_name = molecule_file_name
                # Clean up the molecule name for use as a Python identifier and display
                molecule_name = molecule_name.translate({ord(i): None for i in '_$^{}'})
                molecule_name = molecule_name.translate({ord(i): "_" for i in ' -'})
                if molecule_name and molecule_name[0].isdigit():
                    molecule_name = 'm_' + molecule_name
                molecule_name = molecule_name.upper()
            
            # Prepare molecule data dictionary for parallel processing
            mol_data = {
                "name": molecule_name,
                "Molecule Name": molecule_name,
                "file": hitran_file,
                "hitran_data": hitran_file,
                "label": molecule_name,
                "Molecule Label": molecule_name
            }
            molecules_data.append(mol_data)
        
        # Use sequential loading by default, parallel only if explicitly enabled
        success_count = 0
        use_parallel_loading = use_parallel or self.use_parallel_processing
        
        if use_parallel_loading and len(molecules_data) > 1:
            print(f"Loading {len(molecules_data)} HITRAN molecules using parallel method...")
            results = self.molecules_dict.load_molecules_ultra_fast(
                molecules_data, 
                self.initial_molecule_parameters,
                force_multiprocessing=True
            )
            success_count = results["success"]
            
            if results["failed"] > 0:
                print(f"Failed to load {results['failed']} molecules:")
                for error in results["errors"]:
                    print(f"  - {error}")
        else:
            # Sequential loading (default)
            print(f"Loading {len(molecules_data)} HITRAN molecule(s) sequentially...")
            start_time = time.time()
            
            for mol_data in molecules_data:
                molecule_name = mol_data["name"]
                hitran_file = mol_data["file"]
                
                print(f"Loading molecule '{molecule_name}' from file: {hitran_file}")
                
                try:
                    new_molecule = Molecule(
                        hitran_data=hitran_file,
                        name=molecule_name,
                        wavelength_range=self.wavelength_range,
                        initial_molecule_parameters=self.initial_molecule_parameters.get(molecule_name, self.molecules_parameters_default)
                    )
                    self.molecules_dict[molecule_name] = new_molecule
                    success_count += 1
                    print(f"Successfully created molecule: {molecule_name}")
                    
                except Exception as e:
                    print(f"Error loading molecule '{molecule_name}' from {hitran_file}: {str(e)}")
                    continue
            
            elapsed_time = time.time() - start_time
            print(f"Sequential loading completed in {elapsed_time:.3f}s")
        
        if success_count > 0:
            print(f"Successfully loaded {success_count} molecules.")
            
            # Use the optimized update system
            if refresh:
                self._update_gui_after_molecule_load()
        else:
            print("No molecules were successfully loaded.")

    def check_HITRAN(self):
        """
        Checks that all expected HITRAN files are present and loads them efficiently.
        Only loads when specifically requested to avoid startup delays.
        """
        if not self.user_settings.get("auto_load_hitran", False):
            print("HITRAN auto-loading disabled. Files will be loaded on demand.")
            return
            
        print("Checking HITRAN files:")

        if self.user_settings.get("first_startup", False) or self.user_settings.get("reload_default_files", False):
            print('First startup or reload_default_files is True. Loading default HITRAN files ...')
            
            for mol, bm, iso in zip(self.mols, self.basem, self.isot):
                hitran_file = f"HITRANdata/data_Hitran_2020_{mol}.par"
                if not os.path.exists(hitran_file):
                    print(f"WARNING: HITRAN file for {mol} not found at {hitran_file}")
                    self.hitran_data[mol] = {"lines": [], "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                    continue

                try:
                    lines = read_HITRAN_data(hitran_file)
                    if lines:
                        self.hitran_data[mol] = {"lines": lines, "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                    else:
                        self.hitran_data[mol] = {"lines": [], "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
                except Exception as e:
                    print(f"ERROR: Failed to load HITRAN file for {mol}: {e}")
                    self.hitran_data[mol] = {"lines": [], "base_molecule": bm, "isotope": iso, "file_path": hitran_file}
        else:
            print('Not the first startup and reload_default_files is False. Skipping HITRAN files loading.')

        print("Finished HITRAN file check.\n")
    
    def load_default_molecules(self, reset=True, use_parallel=False):
        """
        Loads default molecules into the molecules_dict with sequential loading by default.
        
        Parameters
        ----------
        reset : bool, optional
            If True, clears existing molecules before loading defaults. Default is True.
        use_parallel : bool, optional
            If True, uses parallel loading. Default is False for sequential loading.
        """
        print("Loading default molecules...")
        
        # Initialize molecules_dict if needed
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()
            # Set global parameters efficiently
            for attr, value in [
                ('global_dist', self._dist),
                ('global_star_rv', self._star_rv),
                ('global_fwhm', self._fwhm),
                ('global_intrinsic_line_width', self._intrinsic_line_width),
                ('global_wavelength_range', self.wavelength_range)
            ]:
                setattr(self.molecules_dict, attr, value)

        if reset:
            self.molecules_dict.clear()
            print("Resetting molecules_dict to empty.")

        try:
            # Lazy load default molecule data
            if self.default_molecule_csv_data is None:
                print("Loading default molecule CSV data...")
                self.default_molecule_csv_data = read_default_csv()
                
            if not self.default_molecule_csv_data:
                print("Error: Could not load default molecule CSV data.")
                return
            
            # Use sequential loading by default
            use_parallel_loading = use_parallel or self.use_parallel_processing
            self.init_molecules(self.default_molecule_csv_data, use_optimized_loading=use_parallel_loading)
            print(f"Successfully loaded {len(self.molecules_dict)} default molecules.")
            
            # Update GUI components if they exist
            if hasattr(self, "GUI") and self.GUI is not None:
                self._update_gui_after_molecule_load()
                
        except Exception as e:
            print(f"Error loading default molecules: {e}")
            raise

    def _update_gui_after_molecule_load(self):
        """
        Helper method to update GUI components after molecules are loaded.
        """
        try:
            # Update molecule table if it exists
            if (hasattr(self.GUI, "molecule_table") and self.GUI.molecule_table is not None):
                self.GUI.molecule_table.update_table()
            
            # Update control panel dropdown if it exists
            if (hasattr(self.GUI, "control_panel") and self.GUI.control_panel is not None and
                hasattr(self.GUI.control_panel, "reload_molecule_dropdown")):
                self.GUI.control_panel.reload_molecule_dropdown()
            
            # Update plots if they exist
            if (hasattr(self.GUI, "plot") and self.GUI.plot is not None):
                self.GUI.plot.update_all_plots()
                
        except Exception as e:
            print(f"Warning: Error updating GUI after molecule load: {e}")

    def load_spectrum(self, file_path=None):
        #filetypes = [('CSV Files', '*.csv'), ('TXT Files', '*.txt'), ('DAT Files', '*.dat')]
        spectra_directory = os.path.abspath("DATAFILES/EXAMPLE-data")
        if file_path is None:
            file_path = GUI.file_selector(
                title='Choose Spectrum Data File',
                initialdir=spectra_directory
            )

        if file_path:
            # Use the new read_spectral_data function
            df = read_spectral_data(file_path)
            
            if df.empty:
                print(f"Failed to load spectrum from {file_path}")
                return
            
            # Check if required columns exist
            required_columns = ['wave', 'flux']
            optional_columns = ['err', 'cont']
            
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Required columns {required_columns} not found in {file_path}")
                print(f"Available columns: {list(df.columns)}")
                return
            
            # Load required data
            self.wave_data = np.array(df['wave'].values) * self.user_settings.get("wave_data_scalar", 1.0)
            self.wave_data_original = self.wave_data.copy()
            self.flux_data = np.array(df['flux'].values) * self.user_settings.get("flux_data_scalar", 1.0)
            
            # Load optional data with defaults if not present
            if 'err' in df.columns:
                self.err_data = np.array(df['err'].values)
            else:
                # Create default error array (e.g., 10% of flux)
                self.err_data = np.abs(self.flux_data) * 0.1
                print("Warning: No 'err' column found. Using 10% of flux as default error.")
            
            if 'cont' in df.columns:
                self.continuum_data = np.array(df['cont'].values)
            else:
                # Create default continuum array (zeros or ones)
                self.continuum_data = np.ones_like(self.flux_data)
                print("Warning: No 'cont' column found. Using ones as default continuum.")
            
            print(f"Successfully loaded spectrum from {file_path}")
            print(f"  Wavelength range: {self.wave_data.min():.3f} - {self.wave_data.max():.3f}")
            print(f"  Data points: {len(self.wave_data)}")

            # Store the loaded file path and name
            self.loaded_spectrum_file = file_path
            self.loaded_spectrum_name = os.path.basename(file_path)

            # Update any dependent components if spectrum is loaded after first start
            if hasattr(self, "GUI"):
                if hasattr(self.GUI, "plot"):
                    self.GUI.plot.update_all_plots()
                if hasattr(self.GUI, "file_interaction_pane"):
                    self.GUI.file_interaction_pane.update_file_label(self.loaded_spectrum_name)
            # If model spectrum or other calculations depend on spectrum, update them
            if hasattr(self, "update_model_spectrum"):
                self.update_model_spectrum()
        else:
            print("No file selected.")

    def update_model_spectrum(self, force_recalculate=False, use_parallel=False):
        """
        Update model spectrum using sequential calculations by default.
        
        Parameters
        ----------
        force_recalculate : bool, default False
            If True, forces recalculation of all molecule intensities and spectra.
        use_parallel : bool, default False
            If True, uses parallel recalculation. Default is False for sequential processing.
        """
        if not hasattr(self, 'molecules_dict') or not hasattr(self, 'wave_data'):
            self.sum_spectrum_flux = np.array([])
            return
        
        if force_recalculate:
            use_parallel_calc = use_parallel or self.use_parallel_processing
            
            if use_parallel_calc:
                # Use parallel recalculation only if explicitly enabled
                print("Force recalculating all molecule spectra using parallel processing...")
                self.molecules_dict.bulk_recalculate_parallel()
            else:
                # Sequential recalculation (default)
                print("Force recalculating all molecule spectra sequentially...")
                self.molecules_dict.bulk_recalculate_sequential()
        
        try:
            # Use the optimized cached summed flux from MoleculeDict
            self.sum_spectrum_flux = self.molecules_dict.get_summed_flux(self.wave_data, visible_only=True)
            
            # Update individual molecule fluxes if needed
            self.molecules_dict.update_molecule_fluxes(self.wave_data)
            
        except Exception as e:
            print(f"Error updating model spectrum: {e}")
            self.sum_spectrum_flux = np.zeros_like(self.wave_data) if hasattr(self, 'wave_data') else np.array([])
    
    def request_update(self, update_type='plots'):
        """
        Request an update through the optimized coordinator system.
        
        Parameters
        ----------
        update_type : str
            Type of update to request ('plots', 'model_spectrum', etc.)
        """
        if self.update_coordinator:
            self.update_coordinator.request_update(update_type)
        elif update_type == 'model_spectrum':
            # Fallback for direct model spectrum update
            self.update_model_spectrum()
    
    def bulk_update_molecule_parameters(self, parameter_dict, molecule_names=None, update_plots=True):
        """
        Bulk update parameters for multiple molecules using the optimized MoleculeDict methods.
        
        Parameters
        ----------
        parameter_dict : dict
            Dictionary of parameter names and values to update
        molecule_names : list, optional
            List of molecule names to update (None for all molecules)
        update_plots : bool, default True
            Whether to update plots after parameter changes
        """
        if not hasattr(self, 'molecules_dict'):
            print("No molecules_dict available for bulk update")
            return
        
        try:
            # Use the optimized bulk parameter update
            self.molecules_dict.bulk_update_parameters(parameter_dict, molecule_names)
            
            # Update model spectrum and plots if requested
            if update_plots:
                self.update_model_spectrum()
                self.request_update('plots')
                
        except Exception as e:
            print(f"Error in bulk parameter update: {e}")
    
    def set_global_parameters(self, **kwargs):
        """
        Set global parameters that affect all molecules using the optimized system.
        
        Parameters
        ----------
        **kwargs : dict
            Global parameters to set (distance, fwhm, stellar_rv, etc.)
        """
        if not hasattr(self, 'molecules_dict'):
            return
        
        try:
            # Update global parameters using the new system
            for param_name, value in kwargs.items():
                if param_name == 'distance':
                    self.molecules_dict.global_dist = value
                    self._dist = value
                elif param_name == 'fwhm':
                    self.molecules_dict.global_fwhm = value
                    self._fwhm = value
                elif param_name == 'stellar_rv':
                    self.molecules_dict.global_star_rv = value
                    self._star_rv = value
                elif param_name == 'intrinsic_line_width':
                    self.molecules_dict.global_intrinsic_line_width = value
                    self._intrinsic_line_width = value
                elif param_name == 'wavelength_range':
                    self.molecules_dict.global_wavelength_range = value
                    self.wavelength_range = value
            
            # Update model spectrum and plots
            self.update_model_spectrum()
            self.request_update('plots')
            
        except Exception as e:
            print(f"Error setting global parameters: {e}")
    
    @property
    def active_molecule(self):
        return self._active_molecule
    
    @active_molecule.setter
    def active_molecule(self, molecule):
        """
        Sets the active molecule based on the provided name or object.
        """
        old_molecule = getattr(self, '_active_molecule', None)
        old_name = getattr(old_molecule, 'name', old_molecule)
        new_name = getattr(molecule, 'name', molecule) if hasattr(molecule, 'name') else molecule
        
        debug_config.info("active_molecule", f"Setting active molecule from {old_name} to {new_name}")
        
        try:
            if isinstance(molecule, Molecule):
                self._active_molecule = molecule
            elif isinstance(molecule, str):
                if hasattr(self, 'molecules_dict') and molecule in self.molecules_dict:
                    self._active_molecule = self.molecules_dict[molecule]
                else:
                    raise ValueError(f"Molecule '{molecule}' not found in the dictionary.")
            else:
                raise TypeError("Active molecule must be a Molecule object or a string representing the molecule name.")
            
            # Only notify callbacks - let them handle plot updates
            # This prevents double calls to plot update methods
            debug_config.verbose("active_molecule", f"Notifying {len(self._active_molecule_change_callbacks)} callbacks of change")
            self._notify_active_molecule_change(old_molecule, self._active_molecule)
                
        except Exception as e:
            debug_config.error("active_molecule", f"Error setting active molecule: {e}")
            # Don't change the active molecule if there's an error
        
    @property
    def display_range(self):
        """tuple: Display range for the spectrum plot."""
        return self._display_range
    
    @display_range.setter
    def display_range(self, value):
        """
        Sets the display range for the spectrum plot.
        The value should be a tuple of two floats representing the start and end wavelengths.
        """
        if isinstance(value, tuple) and len(value) == 2:
            self._display_range = value
            if hasattr(self, "GUI") and hasattr(self.GUI, "plot"):
                self.GUI.plot.match_display_range()
        else:
            raise ValueError("Display range must be a tuple of two floats (start, end).")
    
    def add_active_molecule_change_callback(self, callback):
        """Add a callback function to be called when active molecule changes"""
        self._active_molecule_change_callbacks.append(callback)
    
    def remove_active_molecule_change_callback(self, callback):
        """Remove a callback function for active molecule changes"""
        if callback in self._active_molecule_change_callbacks:
            self._active_molecule_change_callbacks.remove(callback)
    
    def _notify_active_molecule_change(self, old_molecule, new_molecule):
        """Notify all callbacks that the active molecule has changed"""
        debug_config.verbose("active_molecule", f"Notifying {len(self._active_molecule_change_callbacks)} callbacks")
        for i, callback in enumerate(self._active_molecule_change_callbacks):
            try:
                callback_name = callback.__name__ if hasattr(callback, '__name__') else str(callback)
                debug_config.trace("active_molecule", f"Calling callback {i+1}: {callback_name}")
                callback(old_molecule, new_molecule)
                debug_config.trace("active_molecule", f"Callback {i+1} completed successfully")
            except Exception as e:
                debug_config.error("active_molecule", f"Error in callback {i+1}: {e}")
        debug_config.verbose("active_molecule", "All callbacks completed")
    
    # Lazy loading properties for performance optimization
    @property
    def default_molecule_csv_data(self):
        """Lazy load default molecule CSV data only when needed"""
        if self._default_molecule_csv_data is None:
            self._default_molecule_csv_data = read_default_csv()
        return self._default_molecule_csv_data
    
    @default_molecule_csv_data.setter
    def default_molecule_csv_data(self, value):
        self._default_molecule_csv_data = value
    
    @property
    def user_saved_molecules(self):
        """Lazy load user saved molecules only when needed"""
        if self._user_saved_molecules is None:
            self._user_saved_molecules = read_from_user_csv()
        return self._user_saved_molecules
    
    @user_saved_molecules.setter
    def user_saved_molecules(self, value):
        self._user_saved_molecules = value
    
    @property
    def molecules_data_default(self):
        """Lazy load default molecules data only when needed"""
        if self._molecules_data_default is None:
            self._molecules_data_default = c.MOLECULES_DATA.copy()
        return self._molecules_data_default

    def enable_parallel_processing(self):
        """
        Enable parallel processing for molecule loading and calculations.
        Call this method if you want to use multiprocessing/threading for better performance
        with large datasets. By default, iSLAT uses sequential processing for stability.
        """
        print("Enabling parallel processing for iSLAT operations...")
        self._use_parallel_processing = True
        
    def disable_parallel_processing(self):
        """
        Disable parallel processing and use sequential processing (default behavior).
        """
        print("Disabling parallel processing - using sequential processing...")
        self._use_parallel_processing = False
    
    @property
    def use_parallel_processing(self):
        """Check if parallel processing is enabled"""
        return getattr(self, '_use_parallel_processing', False)
