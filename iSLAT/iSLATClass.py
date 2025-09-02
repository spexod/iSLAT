iSLAT_version = 'v5.00.00'

# Import necessary modules
import numpy as np
import pandas as pd
import os
import time
import sys

from .Modules.FileHandling.iSLATFileHandling import load_user_settings, read_default_molecule_parameters, read_initial_molecule_parameters, read_full_molecule_parameters, read_HITRAN_data, read_from_user_csv, read_default_csv, read_spectral_data
from .Modules.FileHandling.iSLATFileHandling import molsave_file_name, save_folder_path, hitran_data_folder_path, hitran_data_folder_name

import iSLAT.Constants as c
from .Modules.GUI import *
from .Modules.DataTypes.Molecule import Molecule
from .Modules.DataTypes.MoleculeDict import MoleculeDict
from .Modules.Debug.DebugConfig import debug_config

class iSLAT:
    """
    iSLAT class to handle the iSLAT functionalities.
    This class is used to initialize the iSLAT application, load user settings, and manage the main functionalities.
    """

    def __init__(self):
        """
        Initialize the iSLAT application with minimal setup.
        """
        # === CORE STATE ===
        self._user_settings = None
        self._active_molecule = None
        self.GUI = None
        
        # Initialize collections
        self.molecules_dict = MoleculeDict()
        self.callbacks = {}
        
        # === CALLBACK SYSTEM ===
        self._active_molecule_change_callbacks = []
        
        # === LAZY LOADING FLAGS ===
        self._initial_molecule_parameters = None
        self._molecules_parameters_default = None
        self._default_molecule_csv_data = None
        self._user_saved_molecules = None
        self._molecules_data_default = None
        self._molecules_loaded = False  # Track if molecules have been initialized
        
        # === MOLECULE CONSTANTS ===
        # Define molecule constants (use tuples for immutability and performance)
        self.mols = ("H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N")
        self.basem = ("H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N")
        self.isot = (1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1)

        # === DATA CONTAINERS ===
        self.hitran_data = {}
        #self._hitran_file_cache = {}  # Cache for HITRAN file data to avoid re-reading
        self.input_line_list = None
        self.output_line_measurements = None
        
        # === PERFORMANCE FLAGS ===
        self._use_parallel_processing = False

    # === MOLECULE MANAGEMENT METHODS ===
    def _set_initial_active_molecule(self):
        """Set the initial active molecule based on user settings and available molecules."""
        active_molecule_name = self.user_settings.get("default_active_molecule", "H2O")
        
        if active_molecule_name in self.molecules_dict:
            self._active_molecule = self.molecules_dict[active_molecule_name]

    def init_molecules(self, mole_save_data=None, use_parallel=False):
        """
        Initialize molecules with automatic optimization and spectrum-aware loading.
        
        Parameters
        ----------
        mole_save_data : dict, list, or None
            Molecule data to load. If None, loads from user_saved_molecules.
        spectrum_optimized : bool, default False
            If True, optimizes loading for the loaded spectrum's wavelength range.
        use_parallel : bool, default False
            If True, uses parallel processing for loading.
        """
        # Use parallel processing setting if not explicitly provided
        if not use_parallel:
            use_parallel = self._use_parallel_processing if hasattr(self, '_use_parallel_processing') else False
        
        # Lazy load user_saved_molecules if needed
        if mole_save_data is None:
            mole_save_data = self.user_saved_molecules

        # Validate and process input data
        if not mole_save_data:
            print("Warning: No molecule data provided for initialization")
            return False

        # Convert to list format efficiently
        if isinstance(mole_save_data, dict):
            molecules_list = [mol for mol in mole_save_data.values() if mol.get("Molecule Name") and mol.get("Molecule Name") not in self.molecules_dict]
            parms = self.initial_molecule_parameters
        elif isinstance(mole_save_data, list):
            molecules_list = mole_save_data
            parms = self.initial_molecule_parameters
            parms['default'] = self.molecules_parameters_default

        if not molecules_list:
            print("No new molecules to load.")
            return False

        try:
            start_time = time.time()
            results = self.molecules_dict.load_molecules(
                molecules_list, 
                parms,
            )
            
            elapsed_time = time.time() - start_time
            print(f"Loaded {len(results)} molecules in {elapsed_time:.3f}s")
            
            if results["success"] > 0:
                print(f"Loaded {results['success']} molecules")
            
            if results["failed"] > 0:
                print(f"Failed to load {results['failed']} molecules")
                for error in results["errors"]:
                    print(f"  - {error}")

            self._set_initial_active_molecule()

            if hasattr(self, "wavedata"):
                spectrum_range = (self.wave_data.min(), self.wave_data.max())
                self.molecules_dict.global_wavelength_range = spectrum_range

            return True
                    
        except Exception as e:
            print(f"Error loading molecules: {e}")

    def add_molecule_from_hitran(self, refresh=True, hitran_files=None, molecule_names=None, use_parallel=False, name_popups=True):
        """
        Adds one or more molecules to the iSLAT instance from HITRAN files.
        
        Parameters:
        -----------
        refresh : bool
            Whether to refresh the GUI after adding molecules
        hitran_files : str or list
            Single file path or list of file paths. If None, opens file dialog for multiple selection
        molecule_names : str or list
            Single molecule name or list of molecule names corresponding to files
        use_parallel : bool, default False
            Whether to prefer parallel loading when beneficial
        """
        if hitran_files is None:
            hitran_files = GUI.file_selector(title='Choose HITRAN Data Files (select multiple with Ctrl/Cmd)',
                                                  filetypes=[('PAR Files', '*.par')],
                                                  initialdir=hitran_data_folder_path,
                                                  allow_multiple=True,
                                                  use_abspath=True)
            
        if not hitran_files:
            print("No HITRAN files selected.")
            return False
        
        # Convert single file to list for consistent processing
        if isinstance(hitran_files, str):
            hitran_files = [hitran_files]
        
        # Convert molecule_names to list if provided as single string
        if molecule_names is not None and isinstance(molecule_names, str):
            molecule_names = [molecule_names]

        if molecule_names is None and name_popups:
            # Use popups to get names for each molecule
            molecule_names = []
            for i, hitran_file in enumerate(hitran_files):
                molecule_name = GUI.add_molecule_name_popup(title=f"Assign label for {os.path.basename(hitran_file)}")
                if molecule_name:
                    molecule_names.append(molecule_name)
                else:
                    print(f"No name provided for {hitran_file}, using default.")
                    molecule_names.append(None)
        
        # Prepare molecule data
        molecules_data = []
        for i, hitran_file in enumerate(hitran_files):
            # Get molecule name for this file
            molecule_file_name = os.path.basename(hitran_file)
            molecule_file_path = hitran_data_folder_name / molecule_file_name
            if molecule_names is not None and i < len(molecule_names):
                molecule_label = molecule_names[i]
                
                molecule_name = molecule_names[i].translate({ord(i): None for i in '_$^{}'})
                molecule_name = molecule_name.translate({ord(i): "_" for i in ' -'})
            else:
                # Extract and clean molecule name from file name
                molecule_name = molecule_file_name
                # Clean up the molecule name for use as a Python identifier and display
                molecule_name = molecule_name.translate({ord(i): None for i in '_$^{}'})
                molecule_name = molecule_name.translate({ord(i): "_" for i in ' -'})
                if molecule_name and molecule_name[0].isdigit():
                    molecule_name = 'm_' + molecule_name
                molecule_name = molecule_name.upper()
                molecule_label = molecule_name
            
            # Create molecule data dictionary using existing format
            mol_data = {
                "name": molecule_name,
                "Molecule Name": molecule_name,
                "file": molecule_file_path,
                "hitran_data": hitran_file,
                "label": molecule_label,
                "Molecule Label": molecule_label
            }
            molecules_data.append(mol_data)
            print(f"mol_data is: {mol_data}")
        
        print(f"Loading {len(molecules_data)} HITRAN molecule(s)...")
        
        try:
            start_time = time.time()
            results = self.molecules_dict.load_molecules(
                molecules_data, 
                read_full_molecule_parameters(),
                update_global_parameters=False,
                strategy="auto",
                force_multiprocessing=use_parallel or getattr(self, '_use_parallel_processing', False)
            )
            elapsed_time = time.time() - start_time
            
            print(f"Loading completed in {elapsed_time:.3f}s")
            
            success_count = results.get("success", 0)
            if success_count > 0:
                print(f"Successfully loaded {success_count} molecules.")
                
                # Refresh GUI if requested using existing GUI methods
                if refresh and hasattr(self, 'GUI') and self.GUI is not None:
                    try:
                        self.GUI.plot.update_model_plot()
                        self.GUI.control_panel.refresh_from_molecules_dict()
                        print("GUI molecule list and plot refreshed.")
                    except Exception as e:
                        print(f"Warning: Could not refresh GUI: {e}")
            else:
                print("No molecules were successfully loaded.")
            
            if results.get("failed", 0) > 0:
                print(f"Failed to load {results['failed']} molecules:")
                for error in results.get("errors", []):
                    print(f"  - {error}")
            
            return success_count > 0
                    
        except Exception as e:
            print(f"Error during molecule loading: {e}")
            print("No molecules were loaded due to the error.")
            return False

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
    
    def load_default_molecules(self, reset=True, use_parallel=False): # Needs updated
        """
        Loads default molecules into the molecules_dict with sequential loading by default.
        
        Parameters
        ----------
        reset : bool, optional
            If True, clears existing molecules before loading defaults. Default is True.
        use_parallel : bool, optional
            If True, uses parallel loading. Default is False for sequential loading.
        """
        # Initialize molecules_dict if needed
        if not hasattr(self, "molecules_dict"):
            self.molecules_dict = MoleculeDict()

        if reset:
            print("Resetting molecules_dict to empty.")
            self.molecules_dict.clear()

        try:
            # Lazy load default molecule data
            if self.default_molecule_csv_data is None:
                print("Loading default molecule CSV data...")
                self.default_molecule_csv_data = read_default_csv()
                
            if not self.default_molecule_csv_data:
                print("Error: Could not load default molecule CSV data.")
                return
            
            # Use parallel loading setting
            use_parallel_loading = use_parallel or self.use_parallel_processing
            self.init_molecules(self.default_molecule_csv_data, use_parallel=use_parallel_loading)
            # Update GUI components
            if hasattr(self, 'GUI'):
                if hasattr(self.GUI, 'plot'):
                    self.GUI.plot.update_all_plots()
                if hasattr(self.GUI, 'control_panel'):
                    self.GUI.control_panel.refresh_from_molecules_dict()
                if hasattr(self.GUI, 'data_field'):
                    self.GUI.data_field.insert_text(
                        f'Successfully loaded parameters from defaults',
                        clear_after=True
                    )
            print(f"Successfully loaded {len(self.molecules_dict)} default molecules.")
                
        except Exception as e:
            print(f"Error loading default molecules: {e}")
            raise

    def get_mole_save_data(self):
        # Check to see if a save for the current spectrum file exists
        if hasattr(self, 'loaded_spectrum_file') and self.loaded_spectrum_file:
            spectrum_base_name = os.path.splitext(self.loaded_spectrum_name)[0]
            formatted_mol_save_file_name = f"{spectrum_base_name}-{molsave_file_name}"
            molsave_path = save_folder_path
            full_path = os.path.join(molsave_path, formatted_mol_save_file_name)
            alternative_path = os.path.join(molsave_path, f"{spectrum_base_name}.csv-{molsave_file_name}")
            if os.path.exists(full_path):
                print(f"Loading molecules from saved file: {full_path}")
                mole_save_data = read_from_user_csv(molsave_path, formatted_mol_save_file_name)
            elif os.path.exists(alternative_path):
                print(f"Loading molecules from old format saved file: {alternative_path}")
                mole_save_data = read_from_user_csv(molsave_path, f"{spectrum_base_name}.csv-{molsave_file_name}")
            else:
                print(f"Warning: Mole save path does not exist: {molsave_path}")
                mole_save_data = None
        else:
            mole_save_data = None   
        
        return mole_save_data

    def _initialize_molecules_for_spectrum(self):
        """
        Initialize molecules optimized for the loaded spectrum's wavelength range.
        This method is called automatically after a spectrum is loaded.
        """
        if not hasattr(self, 'wave_data'):
            print("Warning: No spectrum data available for molecule optimization")
            return
            
        # Optimize wavelength range for the loaded spectrum
        spectrum_range = (self.wave_data.min(), self.wave_data.max())
        print(f"Initializing molecules for spectrum range: {spectrum_range[0]:.1f} - {spectrum_range[1]:.1f} µm")

        mole_save_data = self.user_saved_molecules
        
        try:
            # Initialize molecules with spectrum-optimized settings
            start_time = time.time()
            
            # Use the most efficient initialization method with spectrum optimization
            self.init_molecules(mole_save_data=mole_save_data)

            elapsed_time = time.time() - start_time
            self._molecules_loaded = True
            
            print(f"Molecule initialization completed in {elapsed_time:.3f}s")
            print(f"Loaded {len(self.molecules_dict)} molecules optimized for spectrum")
            
            # Print performance summary
            self._print_performance_summary(elapsed_time)
                
        except Exception as e:
            print(f"Error initializing molecules for spectrum: {e}")
            self._molecules_loaded = False

    # === SPECTRUM METHODS ===
    def load_spectrum(self, file_path=None):
        """
        Load a spectrum from file or show file dialog.
        
        Parameters
        ----------
        file_path : str, optional
            Path to spectrum file. If None, shows file dialog.
            
        Returns
        -------
        bool
            True if spectrum loaded successfully, False otherwise.
            
        Raises
        ------
        FileNotFoundError
            If file_path doesn't exist.
        ValueError  
            If file format is not supported.
        """
        #filetypes = [('CSV Files', '*.csv'), ('TXT Files', '*.txt'), ('DAT Files', '*.dat')]
        spectra_directory = os.path.abspath("DATAFILES/EXAMPLE-data")
        if file_path is None:
            file_path = GUI.file_selector(
                title='Choose Spectrum Data File',
                initialdir=spectra_directory
            )

        if file_path:
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Spectrum file not found: {file_path}")
                
                # Use the new read_spectral_data function
                df = read_spectral_data(file_path)
                
                if df.empty:
                    print(f"Failed to load spectrum from {file_path}")
                    return False
                
            except FileNotFoundError as e:
                print(f"Error: {e}")
                return False
            except Exception as e:
                print(f"Error loading spectrum from {file_path}: {e}")
                return False
            
            # Check if required columns exist
            required_columns = ['wave', 'flux']
            optional_columns = ['err', 'cont']
            
            if not all(col in df.columns for col in required_columns):
                print(f"Error: Required columns {required_columns} not found in {file_path}")
                print(f"Available columns: {list(df.columns)}")
                return False
            
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

            # Initialize molecules after spectrum is loaded (most efficient approach)
            if not self._molecules_loaded:
                self._initialize_molecules_for_spectrum()
                spectrum_range = (self.wave_data.min(), self.wave_data.max())
                self.molecules_dict.global_wavelength_range = spectrum_range
                self.molecules_dict.global_model_pixel_res = np.median(self.wave_data[1:-1] - self.wave_data[0:-2])
            else:
                # Update existing molecules with new wavelength range if needed
                spectrum_range = (self.wave_data.min(), self.wave_data.max())
                self.molecules_dict.global_wavelength_range = spectrum_range
                self.molecules_dict.global_model_pixel_res = np.median(self.wave_data[1:-1] - self.wave_data[0:-2])
                self.update_model_spectrum()
                print(f"Updated existing molecules for new wavelength range: {spectrum_range[0]:.3f} - {spectrum_range[1]:.3f}")

            if not hasattr(self, "display_range") or self.display_range[0] < self.wave_data.min() or self.display_range[1] > self.wave_data.max():
                # Set display limits based on loaded spectrum
                fig_max_limit = np.nanmax(self.wave_data)
                fig_min_limit = np.nanmin(self.wave_data)

                display_first_entry = round((fig_min_limit + (fig_max_limit - fig_min_limit) / 2), 2)
                display_second_entry = round((display_first_entry + (fig_max_limit - fig_min_limit) / 10), 2)

                self.display_range = (display_first_entry, display_second_entry)

            else:
                # trigger display range setter
                self.display_range = self.display_range

            # Initialize GUI after molecules are loaded
            if not hasattr(self, "GUI") or self.GUI is None:
                pass

            else:
                # GUI already exists, just update the spectrum display
                print("Updating existing GUI with new spectrum...")
                if hasattr(self.GUI, "plot") and self.GUI.plot is not None:
                    self.GUI.plot.update_model_plot()
                    self.GUI.plot.match_display_range(match_y=True)
                    if hasattr(self.GUI.plot, 'canvas'):
                        self.GUI.plot.canvas.draw()
                
                # Update control panel
                if hasattr(self.GUI, "control_panel") and self.GUI.control_panel is not None:
                    self.GUI.control_panel.refresh_from_molecules_dict()

                # Update file label
                if (hasattr(self.GUI, "file_interaction_pane") and 
                    hasattr(self, 'loaded_spectrum_name')):
                    self.GUI.file_interaction_pane.update_file_label(self.loaded_spectrum_name)
                self.update_model_spectrum()
            
            print("Spectrum loaded successfully")
            return True
        else:
            print("No spectrum was loaded")
            return False

    def update_model_spectrum(self, force_recalculate=False):
        """
        Update model spectrum using sequential calculations by default.
        
        Parameters
        ----------
        force_recalculate : bool, default False
            If True, forces recalculation of all molecule intensities and spectra.
        use_parallel : bool, default False
            If True, uses parallel recalculation. Default is False for sequential processing.
            
        Returns
        -------
        None
            Updates self.sum_spectrum_flux with the calculated model spectrum.
        """
        if not hasattr(self, 'molecules_dict') or not hasattr(self, 'wave_data'):
            self.sum_spectrum_flux = np.array([])
            return
        
        if force_recalculate:
            self.molecules_dict.bulk_recalculate()
        
        try:
            # Use the optimized cached summed flux from MoleculeDict - now returns (wavelengths, flux)
            self.sum_spectrum_wavelengths, self.sum_spectrum_flux = self.molecules_dict.get_summed_flux(self.wave_data, visible_only=True)
            
        except Exception as e:
            print(f"Error updating model spectrum: {e}")
            self.sum_spectrum_flux = np.zeros_like(self.wave_data) if hasattr(self, 'wave_data') else np.array([])

    # === CALLBACK SYSTEM ===
    def register_callback(self, event_type, callback_func):
        """Register a callback for specific events."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        if callback_func not in self.callbacks[event_type]:
            self.callbacks[event_type].append(callback_func)

    def unregister_callback(self, event_type, callback_func):
        """Remove a callback for specific events.""" 
        if event_type in self.callbacks:
            self.callbacks[event_type] = [cb for cb in self.callbacks[event_type] if cb != callback_func]

    def _trigger_callbacks(self, event_type, *args, **kwargs):
        """Trigger all callbacks for an event type."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error in {event_type}: {e}")
    
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

    # === UTILITY METHODS ===
    def _safe_load_data(self, loader_func, cache_attr, error_message):
        """Helper method to safely load and cache data."""
        try:
            if not hasattr(self, cache_attr) or getattr(self, cache_attr) is None:
                data = loader_func()
                setattr(self, cache_attr, data)
            return getattr(self, cache_attr)
        except Exception as e:
            print(f"{error_message}: {e}")
            return None

    # === GUI METHODS ===
    def init_gui(self):
        """
        Initialize the GUI components of iSLAT.
        This function sets up the main window, menus, and other GUI elements.
        """
        try:
            if not hasattr(self, "GUI") or self.GUI is None:
                from .Modules.GUI import GUI as GUIClass
                
                self.GUI = GUIClass(
                    master=None,
                    molecule_data=getattr(self, 'molecules_dict', None),
                    wave_data=getattr(self, 'wave_data', None),
                    flux_data=getattr(self, 'flux_data', None),
                    config=self.user_settings,
                    islat_class_ref=self
                )
                
                if self.GUI is None:
                    raise RuntimeError("Failed to create GUI object")
            
            if hasattr(self.GUI, 'start') and callable(self.GUI.start):
                self.GUI.start()
                
                # Immediately display spectrum after GUI starts if we have data
                if (hasattr(self, 'wave_data') and hasattr(self, 'flux_data') and 
                    hasattr(self.GUI, "plot") and self.GUI.plot is not None):
                    try:
                        print("Displaying spectrum in GUI...")
                        self.GUI.plot.update_model_plot()
                        
                        # Force immediate canvas update to ensure spectrum is visible
                        if hasattr(self.GUI.plot, 'canvas'):
                            self.GUI.plot.canvas.draw()
                            
                        print("Spectrum displayed successfully")
                        
                        # Update file label if available
                        if (hasattr(self.GUI, "file_interaction_pane") and 
                            hasattr(self, 'loaded_spectrum_name')):
                            self.GUI.file_interaction_pane.update_file_label(self.loaded_spectrum_name)
                            
                    except Exception as e:
                        print(f"Warning: Error displaying spectrum during GUI init: {e}")
                        
            else:
                raise AttributeError(f"GUI object does not have a callable 'start' method")
                
        except Exception as e:
            print(f"Error initializing GUI: {e}")
            import traceback
            traceback.print_exc()
            print("GUI initialization failed. Running in headless mode.")
            self.GUI = None
            raise

    def run(self):
        """
        Run the iSLAT application.
        """
        print("\n" + "="*60)
        print(f"iSLAT {iSLAT_version} - interactive Spectral-Line Analysis Tool")
        print("="*60)
        
        try:
            print("\n" + "="*60)
            print("Please select a spectrum file to load.")
            print("="*60)
            continue_init = self.load_spectrum()
            if not continue_init:
                print("No spectrum selected. Exiting...")
                sys.exit(0)
            self.init_gui()
            
        except Exception as e:
            print(f"Error during iSLAT initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # === LAZY LOADING PROPERTIES ===
    @property
    def user_settings(self):
        """Lazy load user settings only when needed."""
        if self._user_settings is None:
            self._user_settings = load_user_settings()
        return self._user_settings

    @property
    def initial_molecule_parameters(self):
        """Lazy load initial molecule parameters only when needed."""
        if self._initial_molecule_parameters is None:
            self._initial_molecule_parameters = read_initial_molecule_parameters()
        return self._initial_molecule_parameters

    @property
    def molecules_parameters_default(self):
        """Lazy load default molecule parameters only when needed."""
        if self._molecules_parameters_default is None:
            self._molecules_parameters_default = read_default_molecule_parameters()
        return self._molecules_parameters_default

    @property
    def default_molecule_csv_data(self):
        """Lazy load default molecule CSV data with safe error handling."""
        return self._safe_load_data(read_default_csv,'_default_molecule_csv_data', "Failed to load default molecules")
    
    @default_molecule_csv_data.setter
    def default_molecule_csv_data(self, value):
        self._default_molecule_csv_data = value

    @property
    def molecules_data_default(self):
        """Lazy load default molecules data only when needed."""
        if self._molecules_data_default is None:
            self._molecules_data_default = c.MOLECULES_DATA.copy()
        return self._molecules_data_default
    
    @property
    def user_saved_molecules(self):
        """Lazy load user saved molecules data with safe error handling."""
        return self._safe_load_data(read_from_user_csv, '_user_saved_molecules', "Failed to load user molecules")
        
    @user_saved_molecules.setter 
    def user_saved_molecules(self, value):
        """Set user saved molecules data."""
        self._user_saved_molecules = value

    # === REMAINING PROPERTIES ===
    @property
    def use_parallel_processing(self):
        """Check if parallel processing is enabled"""
        return getattr(self, '_use_parallel_processing', False)
    
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
            
            # Trigger callbacks using the new unified callback system
            self._trigger_callbacks('active_molecule_changed', old_molecule, self._active_molecule)
            
            # Also maintain backwards compatibility with old callback system
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
                self.GUI.plot.match_display_range(match_y=False)
            if hasattr(self, "GUI") and hasattr(self.GUI, "control_panel"):
                self.GUI.control_panel._update_display_range(self._display_range)
        else:
            raise ValueError("Display range must be a tuple of two floats (start, end).")
    
    def _print_performance_summary(self, molecule_load_time):
        """Print a summary of performance optimizations and timings"""
        print(f"\n--- Performance Summary ---")
        print(f"Molecule loading time: {molecule_load_time:.3f}s")
        if hasattr(self, '_use_parallel_processing') and self._use_parallel_processing:
            print("Parallel processing enabled")
        if hasattr(self, '_hitran_file_cache') and len(self._hitran_file_cache) > 0:
            print(f"HITRAN file caching active ({len(self._hitran_file_cache)} files cached)")
        if hasattr(self, 'molecules_dict') and hasattr(self.molecules_dict, 'global_wavelength_range'):
            print(f"Optimized for spectrum range: {self.molecules_dict.global_wavelength_range[0]:.1f} - {self.molecules_dict.global_wavelength_range[1]:.1f} µm")
        print("--- Ready for Analysis ---\n")